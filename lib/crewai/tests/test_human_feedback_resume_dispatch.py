"""Resuming a routed ``@human_feedback`` review continues the flow as in-run.

Each parity case runs the same flow twice: once with every review answered
during the run, and once with some reviews pausing, the flow restored with
``from_pending()`` and resumed with ``resume()`` or ``resume_async()``. The
paused log (pre-pause runs plus every resume) must equal the in-run log in
which methods run, how often, and with which argument, and in the order
between the reviewed method's own listeners, its label listeners and the
label listeners of enclosing routers.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable
from contextlib import closing
import json
import sqlite3
import threading
from typing import Any
from unittest.mock import patch

from crewai.flow import (
    Flow,
    HumanFeedbackResult,
    and_,
    human_feedback,
    listen,
    or_,
    router,
    start,
)
from crewai.flow.async_feedback import HumanFeedbackPending
from crewai.flow.persistence import SQLiteFlowPersistence
import pytest


LLM = "gpt-4o-mini"
WAIT_SECONDS = 10

# Keys a PendingFeedbackContext row had before resume progress was carried.
PRE_PROGRESS_CONTEXT_KEYS = frozenset(
    {
        "flow_id",
        "flow_class",
        "method_name",
        "method_output",
        "message",
        "emit",
        "default_outcome",
        "metadata",
        "llm",
        "requested_at",
        "execution_uuid",
        "trace_context",
    }
)


def _collapse(self: Flow[Any], feedback: str, outcomes: Any, llm: Any) -> str:
    return feedback if feedback in outcomes else outcomes[0]


@pytest.fixture(autouse=True)
def _offline_collapse():
    with patch.object(Flow, "_collapse_to_outcome", _collapse):
        yield


class Reviewer:
    """Answers each review from a script, or pauses the reviews named in ``pause``."""

    def __init__(self, answers: dict[str, list[str]], pause: frozenset[str]) -> None:
        self.answers = {name: list(replies) for name, replies in answers.items()}
        self.pause = pause

    def answer(self, method_name: str) -> str:
        return self.answers[method_name].pop(0)

    def request_feedback(self, context: Any, flow: Any) -> str:
        if context.method_name in self.pause:
            raise HumanFeedbackPending(context=context)
        return self.answer(context.method_name)


def payload(value: Any) -> Any:
    if isinstance(value, HumanFeedbackResult):
        return ("feedback", value.method_name, value.outcome, value.feedback)
    if value is None or isinstance(value, str):
        return value
    return type(value).__name__


Record = tuple[str, Any]
Factory = Callable[[Reviewer, list[Record]], type[Flow[Any]]]
Tamper = Callable[[SQLiteFlowPersistence, str, int], None]


def run_in_run(make: Factory, answers: dict[str, list[str]]) -> list[Record]:
    log: list[Record] = []
    make(Reviewer(answers, frozenset()), log)().kickoff()
    return log


def run_paused(
    make: Factory,
    answers: dict[str, list[str]],
    pause: set[str],
    resume_mode: str,
    db_path: str,
    tamper: Tamper | None = None,
) -> tuple[list[list[Record]], list[str]]:
    """Kick off with the ``pause`` reviews pausing and resume until done.

    Returns the log segments (pre-pause, then one per resume) and the name of
    each method the flow paused on.
    """
    log: list[Record] = []
    reviewer = Reviewer(answers, frozenset(pause))
    flow_class = make(reviewer, log)
    persistence = SQLiteFlowPersistence(db_path)

    result = flow_class(persistence=persistence).kickoff()
    assert isinstance(result, HumanFeedbackPending), "the flow did not pause"
    segments = [list(log)]
    paused_on = [result.context.method_name]

    while isinstance(result, HumanFeedbackPending):
        assert len(paused_on) <= 6, f"too many pauses: {paused_on}"
        flow_id = result.context.flow_id
        if tamper is not None:
            tamper(persistence, flow_id, len(paused_on) - 1)
        restored = flow_class.from_pending(flow_id, persistence)
        feedback = reviewer.answer(result.context.method_name)
        start_at = len(log)
        if resume_mode == "sync":
            result = restored.resume(feedback)
        else:
            result = asyncio.run(restored.resume_async(feedback))
        segments.append(log[start_at:])
        if isinstance(result, HumanFeedbackPending):
            paused_on.append(result.context.method_name)

    return segments, paused_on


def flatten(segments: list[list[Record]]) -> list[Record]:
    return [record for segment in segments for record in segment]


def names(records: list[Record]) -> list[str]:
    return [name for name, _ in records]


def group_sequence(records: list[Record], groups: dict[str, str]) -> list[str]:
    sequence: list[str] = []
    for name, _ in records:
        group = groups.get(name)
        if group is not None and (not sequence or sequence[-1] != group):
            sequence.append(group)
    return sequence


def assert_parity(
    in_run: list[Record],
    paused: list[Record],
    *,
    run_only: frozenset[str] = frozenset(),
    groups: dict[str, str] | None = None,
) -> None:
    """Same methods, counts and arguments; ``run_only`` names skip the argument."""

    def key(record: Record) -> Record:
        name, _ = record
        return (name, None) if name in run_only else record

    assert Counter(map(key, paused)) == Counter(map(key, in_run)), (
        f"in-run: {in_run}\npaused: {paused}"
    )
    if groups is not None:
        assert group_sequence(paused, groups) == group_sequence(in_run, groups), (
            f"in-run: {in_run}\npaused: {paused}"
        )


RESUME_MODES = ["sync", "async"]
ENTRY_PATHS = ["start", "listener"]


@pytest.fixture
def db_path(tmp_path: Any) -> str:
    return str(tmp_path / "pending.db")


def entry_point(entry: str) -> Callable[[Any], Any]:
    """S: an unconditional start. L: a listener of the ``begin`` start."""
    return start() if entry == "start" else listen("begin")


# ---------------------------------------------------------------------------
# V-1, V-6, V-8(a): the reviewed method's own listeners run on resume
# ---------------------------------------------------------------------------


def make_own_listeners(entry: str) -> Factory:
    def make(reviewer: Reviewer, log: list[Record]) -> type[Flow[Any]]:
        class OwnListeners(Flow):
            if entry == "listener":

                @start()
                def begin(self):
                    log.append(("begin", None))

            @entry_point(entry)
            @human_feedback(
                message="Review",
                emit=["approved", "rejected"],
                llm=LLM,
                provider=reviewer,
            )
            def draft(self):
                log.append(("draft", None))
                return "draft-output"

            @listen(or_(draft, "z"))
            def or_draft_z(self, r=None):
                log.append(("or_draft_z", payload(r)))

            @listen(draft)
            def own(self, r=None):
                log.append(("own", payload(r)))

            @router(draft)
            def route_draft(self):
                log.append(("route_draft", None))
                return "routed"

            @listen("routed")
            def on_routed(self, r=None):
                log.append(("on_routed", payload(r)))

            @listen("never_emitted")
            def z(self):
                log.append(("z", None))

            @listen("approved")
            def publish(self, r=None):
                log.append(("publish", payload(r)))

        return OwnListeners

    return make


OWN_GROUPS = {
    "own": "own",
    "route_draft": "own",
    "on_routed": "own",
    "or_draft_z": "own",
    "publish": "label",
}


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
@pytest.mark.parametrize("entry", ENTRY_PATHS)
def test_resume_runs_own_listeners_router_and_or_of_reviewed_method(
    entry: str, resume_mode: str, db_path: str
) -> None:
    """V-1: @listen, @router and or_ on the reviewed method run as in-run."""
    make = make_own_listeners(entry)
    answers = {"draft": ["approved"]}

    in_run = run_in_run(make, answers)
    segments, _ = run_paused(make, answers, {"draft"}, resume_mode, db_path)

    assert_parity(in_run, flatten(segments))


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
@pytest.mark.parametrize("entry", ENTRY_PATHS)
def test_resume_runs_own_group_before_label_group(
    entry: str, resume_mode: str, db_path: str
) -> None:
    """V-6: own-group records come before label-group records, as in-run."""
    make = make_own_listeners(entry)
    answers = {"draft": ["approved"]}

    in_run = run_in_run(make, answers)
    segments, _ = run_paused(make, answers, {"draft"}, resume_mode, db_path)

    assert group_sequence(in_run, OWN_GROUPS) == ["own", "label"]
    assert group_sequence(flatten(segments), OWN_GROUPS) == ["own", "label"]


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
@pytest.mark.parametrize("entry", ENTRY_PATHS)
def test_resume_does_not_execute_reviewed_method_again(
    entry: str, resume_mode: str, db_path: str
) -> None:
    """V-8(a) guard: the reviewed method runs as often as in-run."""
    make = make_own_listeners(entry)
    answers = {"draft": ["approved"]}

    in_run = run_in_run(make, answers)
    segments, _ = run_paused(make, answers, {"draft"}, resume_mode, db_path)

    assert names(in_run).count("draft") == 1
    assert names(flatten(segments)).count("draft") == 1
    assert "draft" not in names(flatten(segments[1:]))


def make_own_review_listener(entry: str) -> Factory:
    """V-1 variant: the own listener is itself a review answered in the run."""

    def make(reviewer: Reviewer, log: list[Record]) -> type[Flow[Any]]:
        class OwnReview(Flow):
            if entry == "listener":

                @start()
                def begin(self):
                    log.append(("begin", None))

            @entry_point(entry)
            @human_feedback(
                message="Review",
                emit=["approved", "rejected"],
                llm=LLM,
                provider=reviewer,
            )
            def draft(self):
                log.append(("draft", None))
                return "draft-output"

            @listen(draft)
            @human_feedback(
                message="Audit", emit=["ok2", "no2"], llm=LLM, provider=reviewer
            )
            def audit(self, r=None):
                log.append(("audit", payload(r)))
                return "audit-output"

            @listen("ok2")
            def after_audit(self, r=None):
                log.append(("after_audit", payload(r)))

            @listen("approved")
            def publish(self, r=None):
                log.append(("publish", payload(r)))

        return OwnReview

    return make


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
@pytest.mark.parametrize("entry", ENTRY_PATHS)
def test_resume_label_payload_is_read_after_own_group(
    entry: str, resume_mode: str, db_path: str
) -> None:
    """V-1 payload: the label listener gets the payload in-run gives it.

    In-run, the label payload is ``last_human_feedback`` after the own group,
    which is the own-listener review's result, not the resumed review's.
    """
    make = make_own_review_listener(entry)
    answers = {"draft": ["approved"], "audit": ["ok2"]}

    in_run = run_in_run(make, answers)
    segments, _ = run_paused(make, answers, {"draft"}, resume_mode, db_path)

    assert ("publish", ("feedback", "audit", "ok2", "ok2")) in in_run
    assert_parity(
        in_run,
        flatten(segments),
        groups={"audit": "own", "after_audit": "own", "publish": "label"},
    )


# ---------------------------------------------------------------------------
# V-2: and_ joins on the outcome label
# ---------------------------------------------------------------------------


def make_label_join(entry: str) -> Factory:
    def make(reviewer: Reviewer, log: list[Record]) -> type[Flow[Any]]:
        delivered = threading.Event()

        if entry == "start":

            class LabelJoinS(Flow):
                @start()
                def pre(self):
                    log.append(("pre", None))
                    return "pre-output"

                @listen(pre)
                def pre_delivered(self):
                    delivered.set()

                @start()
                @human_feedback(
                    message="Review",
                    emit=["approved", "rejected"],
                    llm=LLM,
                    provider=reviewer,
                )
                def draft(self):
                    assert delivered.wait(WAIT_SECONDS)
                    log.append(("draft", None))
                    return "draft-output"

                @listen(and_(pre, "approved"))
                def label_join(self, r=None):
                    log.append(("label_join", payload(r)))

            return LabelJoinS

        class LabelJoinL(Flow):
            @start()
            def begin(self):
                log.append(("begin", None))
                return "begin-output"

            @listen(begin)
            @human_feedback(
                message="Review",
                emit=["approved", "rejected"],
                llm=LLM,
                provider=reviewer,
            )
            def draft(self):
                log.append(("draft", None))
                return "draft-output"

            @listen(and_(begin, "approved"))
            def label_join(self, r=None):
                log.append(("label_join", payload(r)))

        return LabelJoinL

    return make


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
@pytest.mark.parametrize("entry", ENTRY_PATHS)
def test_resume_runs_and_join_on_outcome_label(
    entry: str, resume_mode: str, db_path: str
) -> None:
    """V-2: and_(pre, label) with pre done before the pause runs, as in-run."""
    make = make_label_join(entry)
    answers = {"draft": ["approved"]}

    in_run = run_in_run(make, answers)
    segments, _ = run_paused(make, answers, {"draft"}, resume_mode, db_path)

    assert "label_join" in names(in_run)
    assert_parity(in_run, flatten(segments))


# ---------------------------------------------------------------------------
# V-3: conditional start gated on the outcome label
# ---------------------------------------------------------------------------


def make_label_gated_start(entry: str) -> Factory:
    def make(reviewer: Reviewer, log: list[Record]) -> type[Flow[Any]]:
        class Gated(Flow):
            if entry == "listener":

                @start()
                def begin(self):
                    log.append(("begin", None))

            @entry_point(entry)
            @human_feedback(
                message="Review",
                emit=["approved", "rejected"],
                llm=LLM,
                provider=reviewer,
            )
            def draft(self):
                log.append(("draft", None))
                return "draft-output"

            @start("approved")
            def gated(self):
                log.append(("gated", None))

            @listen("approved")
            def publish(self, r=None):
                log.append(("publish", payload(r)))

        return Gated

    return make


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
def test_resume_listener_review_runs_label_gated_start_once(
    resume_mode: str, db_path: str
) -> None:
    """V-3 (L): a @start gated on the label of a listener review runs once."""
    make = make_label_gated_start("listener")
    answers = {"draft": ["approved"]}

    in_run = run_in_run(make, answers)
    segments, _ = run_paused(make, answers, {"draft"}, resume_mode, db_path)

    assert names(in_run).count("gated") == 1
    assert names(flatten(segments)).count("gated") == 1
    assert_parity(in_run, flatten(segments))


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
def test_resume_start_review_does_not_run_label_gated_start(
    resume_mode: str, db_path: str
) -> None:
    """V-3 (S) guard: a start review's label gets no conditional-start check."""
    make = make_label_gated_start("start")
    answers = {"draft": ["approved"]}

    in_run = run_in_run(make, answers)
    segments, _ = run_paused(make, answers, {"draft"}, resume_mode, db_path)

    assert "gated" not in names(in_run)
    assert "gated" not in names(flatten(segments))
    assert_parity(in_run, flatten(segments))


# ---------------------------------------------------------------------------
# V-4: and_ joins on the reviewed method with members done before the pause
# ---------------------------------------------------------------------------


def make_method_join(shape: str) -> Factory:
    def make(reviewer: Reviewer, log: list[Record]) -> type[Flow[Any]]:
        delivered = threading.Event()

        if shape == "start":

            class MethodJoinS(Flow):
                @start()
                def other(self):
                    log.append(("other", None))
                    return "other-output"

                @listen(other)
                def other_delivered(self):
                    delivered.set()

                @start()
                @human_feedback(
                    message="Review",
                    emit=["approved", "rejected"],
                    llm=LLM,
                    provider=reviewer,
                )
                def draft(self):
                    assert delivered.wait(WAIT_SECONDS)
                    log.append(("draft", None))
                    return "draft-output"

                @listen(and_(draft, other))
                def combine(self, r=None):
                    log.append(("combine", payload(r)))

            return MethodJoinS

        if shape == "listener":

            class MethodJoinL(Flow):
                @start()
                def begin(self):
                    log.append(("begin", None))
                    return "begin-output"

                @listen(begin)
                @human_feedback(
                    message="Review",
                    emit=["approved", "rejected"],
                    llm=LLM,
                    provider=reviewer,
                )
                def draft(self):
                    log.append(("draft", None))
                    return "draft-output"

                @listen(and_(draft, begin))
                def combine(self, r=None):
                    log.append(("combine", payload(r)))

            return MethodJoinL

        class MethodJoinNested(Flow):
            @start()
            def begin(self):
                log.append(("begin", None))
                return "begin-output"

            @listen(begin)
            def mid(self):
                log.append(("mid", None))
                return "mid-output"

            @listen(mid)
            @human_feedback(
                message="Review",
                emit=["approved", "rejected"],
                llm=LLM,
                provider=reviewer,
            )
            def draft(self):
                log.append(("draft", None))
                return "draft-output"

            @listen(and_(draft, mid))
            def join_mid(self, r=None):
                log.append(("join_mid", payload(r)))

            @listen(and_(draft, begin))
            def join_begin(self, r=None):
                log.append(("join_begin", payload(r)))

        return MethodJoinNested

    return make


# A join completed by a cut parent trigger gets the outcome string on resume,
# where in-run it gets the parent's output: compared for its run only.
METHOD_JOIN_RUN_ONLY = {
    "start": frozenset(),
    "listener": frozenset({"combine"}),
    "nested": frozenset({"join_mid", "join_begin"}),
}


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
@pytest.mark.parametrize("shape", ["start", "listener", "nested"])
def test_resume_runs_and_join_on_reviewed_method(
    shape: str, resume_mode: str, db_path: str
) -> None:
    """V-4: and_(draft, x) with x done or cut before the pause runs, as in-run."""
    make = make_method_join(shape)
    answers = {"draft": ["approved"]}

    in_run = run_in_run(make, answers)
    segments, _ = run_paused(make, answers, {"draft"}, resume_mode, db_path)

    assert_parity(in_run, flatten(segments), run_only=METHOD_JOIN_RUN_ONLY[shape])


# ---------------------------------------------------------------------------
# V-5: multi-event or_ fired before the pause
# ---------------------------------------------------------------------------


def make_or_fired(case: str) -> Factory:
    def make(reviewer: Reviewer, log: list[Record]) -> type[Flow[Any]]:
        delivered = threading.Event()

        if case == "start_label":

            class OrLabelS(Flow):
                @start()
                def pre(self):
                    log.append(("pre", None))
                    return "pre-output"

                @listen(pre)
                def pre_delivered(self):
                    delivered.set()

                @start()
                @human_feedback(
                    message="Review",
                    emit=["approved", "rejected"],
                    llm=LLM,
                    provider=reviewer,
                )
                def draft(self):
                    assert delivered.wait(WAIT_SECONDS)
                    log.append(("draft", None))
                    return "draft-output"

                @listen(or_(pre, "approved"))
                def either(self, r=None):
                    log.append(("either", payload(r)))

            return OrLabelS

        if case == "start_method":

            class OrMethodS(Flow):
                @start()
                def other(self):
                    log.append(("other", None))
                    return "other-output"

                @listen(other)
                def other_delivered(self):
                    delivered.set()

                @start()
                @human_feedback(
                    message="Review",
                    emit=["approved", "rejected"],
                    llm=LLM,
                    provider=reviewer,
                )
                def draft(self):
                    assert delivered.wait(WAIT_SECONDS)
                    log.append(("draft", None))
                    return "draft-output"

                @listen(or_(draft, other))
                def either(self, r=None):
                    log.append(("either", payload(r)))

            return OrMethodS

        class OrLabelL(Flow):
            @start()
            def pre(self):
                log.append(("pre", None))
                return "pre-output"

            @listen(pre)
            def pre_delivered(self):
                delivered.set()

            @start()
            def begin(self):
                assert delivered.wait(WAIT_SECONDS)
                log.append(("begin", None))

            @listen(begin)
            @human_feedback(
                message="Review",
                emit=["approved", "rejected"],
                llm=LLM,
                provider=reviewer,
            )
            def draft(self):
                log.append(("draft", None))
                return "draft-output"

            @listen(or_(pre, "approved"))
            def either(self, r=None):
                log.append(("either", payload(r)))

        return OrLabelL

    return make


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
def test_resume_start_review_or_on_label_fired_before_pause_fires_once(
    resume_mode: str, db_path: str
) -> None:
    """V-5(a): S or_(pre, label) fired by pre before the pause fires once."""
    make = make_or_fired("start_label")
    answers = {"draft": ["approved"]}

    in_run = run_in_run(make, answers)
    segments, _ = run_paused(make, answers, {"draft"}, resume_mode, db_path)

    assert names(in_run).count("either") == 1
    assert names(flatten(segments)).count("either") == 1
    assert_parity(in_run, flatten(segments))


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
def test_resume_start_review_or_on_method_fired_before_pause_fires_once(
    resume_mode: str, db_path: str
) -> None:
    """V-5(b) guard: S or_(draft, other) fired by other fires once in total."""
    make = make_or_fired("start_method")
    answers = {"draft": ["approved"]}

    in_run = run_in_run(make, answers)
    segments, _ = run_paused(make, answers, {"draft"}, resume_mode, db_path)

    assert names(in_run).count("either") == 1
    assert names(flatten(segments)).count("either") == 1


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
def test_resume_listener_review_or_on_label_rearms_as_in_run(
    resume_mode: str, db_path: str
) -> None:
    """V-5(c) guard: L or_(pre, label) fires twice, as the in-run re-arm does."""
    make = make_or_fired("listener_label")
    answers = {"draft": ["approved"]}

    in_run = run_in_run(make, answers)
    segments, _ = run_paused(make, answers, {"draft"}, resume_mode, db_path)

    assert names(in_run).count("either") == 2
    assert names(flatten(segments)).count("either") == 2


# ---------------------------------------------------------------------------
# V-7: a re-pause on resume inside the resumed review's own group
# ---------------------------------------------------------------------------


def make_nested_repause(entry: str) -> Factory:
    def make(reviewer: Reviewer, log: list[Record]) -> type[Flow[Any]]:
        delivered = threading.Event()

        class Nested(Flow):
            @start()
            def pre(self):
                log.append(("pre", None))
                return "pre-output"

            @listen(pre)
            def pre_delivered(self):
                delivered.set()

            if entry == "listener":

                @start()
                def begin(self):
                    assert delivered.wait(WAIT_SECONDS)
                    log.append(("begin", None))

                @start("approved")
                def gated(self):
                    log.append(("gated", None))

            @entry_point(entry)
            @human_feedback(
                message="Review",
                emit=["approved", "rejected"],
                llm=LLM,
                provider=reviewer,
            )
            def draft(self):
                if entry == "start":
                    assert delivered.wait(WAIT_SECONDS)
                log.append(("draft", None))
                return "draft-output"

            @listen(draft)
            @human_feedback(
                message="Audit", emit=["ok2", "no2"], llm=LLM, provider=reviewer
            )
            def audit(self, r=None):
                log.append(("audit", payload(r)))
                return "audit-output"

            @listen("approved")
            def publish(self, r=None):
                log.append(("publish", payload(r)))

            @listen("ok2")
            def after_audit(self, r=None):
                log.append(("after_audit", payload(r)))

            @listen(and_(pre, "ok2"))
            def pre_and_ok2(self, r=None):
                log.append(("pre_and_ok2", payload(r)))

        return Nested

    return make


NESTED_GROUPS = {
    "audit": "draft-own",
    "after_audit": "audit-label",
    "pre_and_ok2": "audit-label",
    "publish": "draft-label",
    "gated": "draft-label",
}


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
@pytest.mark.parametrize("entry", ENTRY_PATHS)
def test_resume_carries_outer_label_across_repause_in_own_group(
    entry: str, resume_mode: str, db_path: str
) -> None:
    """V-7: the outer label cut by a re-pause in its own group still runs."""
    make = make_nested_repause(entry)
    answers = {"draft": ["approved"], "audit": ["ok2"]}

    in_run = run_in_run(make, answers)
    segments, paused_on = run_paused(
        make, answers, {"draft", "audit"}, resume_mode, db_path
    )

    assert "publish" in names(in_run)
    assert_parity(in_run, flatten(segments), groups=NESTED_GROUPS)
    assert paused_on == ["draft", "audit"]


# ---------------------------------------------------------------------------
# V-7b: a re-pause that already holds a carried label
# ---------------------------------------------------------------------------


def make_three_levels(entry: str, audit2_on: str) -> Factory:
    def make(reviewer: Reviewer, log: list[Record]) -> type[Flow[Any]]:
        class ThreeLevels(Flow):
            if entry == "listener":

                @start()
                def begin(self):
                    log.append(("begin", None))

                @start("approved")
                def gated(self):
                    log.append(("gated", None))

            @entry_point(entry)
            @human_feedback(
                message="Review",
                emit=["approved", "rejected"],
                llm=LLM,
                provider=reviewer,
            )
            def draft(self):
                log.append(("draft", None))
                return "draft-output"

            @listen(draft)
            @human_feedback(
                message="Audit", emit=["ok2", "no2"], llm=LLM, provider=reviewer
            )
            def audit(self, r=None):
                log.append(("audit", payload(r)))
                return "audit-output"

            @listen(audit2_on)
            @human_feedback(
                message="Audit again",
                emit=["ok3", "no3"],
                llm=LLM,
                provider=reviewer,
            )
            def audit2(self, r=None):
                log.append(("audit2", payload(r)))
                return "audit2-output"

            @listen("approved")
            def publish(self, r=None):
                log.append(("publish", payload(r)))

            @listen("ok2")
            def after_audit(self, r=None):
                log.append(("after_audit", payload(r)))

            @listen("ok3")
            def after_audit2(self, r=None):
                log.append(("after_audit2", payload(r)))

        return ThreeLevels

    return make


THREE_LEVEL_GROUPS = {
    "after_audit2": "audit2-label",
    "after_audit": "audit-label",
    "publish": "draft-label",
    "gated": "draft-label",
}


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
@pytest.mark.parametrize("audit2_on", ["audit", "ok2"])
@pytest.mark.parametrize("entry", ENTRY_PATHS)
def test_resume_keeps_carried_labels_innermost_first_across_repauses(
    entry: str, audit2_on: str, resume_mode: str, db_path: str
) -> None:
    """V-7b: a re-pause keeps the labels it already carries, innermost first."""
    make = make_three_levels(entry, audit2_on)
    answers = {"draft": ["approved"], "audit": ["ok2"], "audit2": ["ok3"]}

    in_run = run_in_run(make, answers)
    segments, paused_on = run_paused(
        make, answers, {"audit", "audit2"}, resume_mode, db_path
    )

    assert "publish" in names(in_run)
    assert_parity(in_run, flatten(segments), groups=THREE_LEVEL_GROUPS)
    assert paused_on == ["audit", "audit2"]


# ---------------------------------------------------------------------------
# V-8(b), V-8(c): the reviewed method re-runs only where routing calls it
# ---------------------------------------------------------------------------


def make_revision_loop(entry: str) -> Factory:
    """S: draft is a start that listens to "revise"; L: or_(begin, "revise")."""
    revise_entry = (
        (lambda method: start()(listen("revise")(method)))
        if entry == "start"
        else listen(or_("begin", "revise"))
    )

    def make(reviewer: Reviewer, log: list[Record]) -> type[Flow[Any]]:
        class RevisionLoop(Flow):
            if entry == "listener":

                @start()
                def begin(self):
                    log.append(("begin", None))

            @revise_entry
            @human_feedback(
                message="Review",
                emit=["approved", "revise"],
                llm=LLM,
                provider=reviewer,
            )
            def draft(self):
                log.append(("draft", None))
                return "draft-output"

            @listen("approved")
            def publish(self, r=None):
                log.append(("publish", payload(r)))

        return RevisionLoop

    return make


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
@pytest.mark.parametrize("entry", ENTRY_PATHS)
def test_resume_revision_loop_reruns_reviewed_method_as_in_run(
    entry: str, resume_mode: str, db_path: str
) -> None:
    """V-8(b) guard: a revision outcome re-runs the reviewed method, once."""
    make = make_revision_loop(entry)
    answers = {"draft": ["revise", "approved"]}

    in_run = run_in_run(make, answers)
    segments, paused_on = run_paused(make, answers, {"draft"}, resume_mode, db_path)

    assert paused_on == ["draft", "draft"]
    assert names(in_run).count("draft") == 2
    assert names(flatten(segments)).count("draft") == 2
    assert names(flatten(segments)).count("publish") == 1


def make_join_revision() -> Factory:
    def make(reviewer: Reviewer, log: list[Record]) -> type[Flow[Any]]:
        delivered = threading.Event()

        class JoinRevision(Flow):
            @start()
            def pre(self):
                log.append(("pre", None))
                return "pre-output"

            @listen(pre)
            def pre_delivered(self):
                delivered.set()

            @start()
            @listen(and_("again", pre))
            @human_feedback(
                message="Review",
                emit=["approved", "again"],
                llm=LLM,
                provider=reviewer,
            )
            def draft(self):
                assert delivered.wait(WAIT_SECONDS)
                log.append(("draft", None))
                return "draft-output"

            @listen("approved")
            def publish(self, r=None):
                log.append(("publish", payload(r)))

        return JoinRevision

    return make


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
def test_resume_reruns_reviewed_method_through_join_with_member_before_pause(
    resume_mode: str, db_path: str
) -> None:
    """V-8(c): draft listening to and_("again", pre) re-runs, as in-run."""
    make = make_join_revision()
    answers = {"draft": ["again", "approved"]}

    in_run = run_in_run(make, answers)
    segments, _ = run_paused(make, answers, {"draft"}, resume_mode, db_path)

    assert names(in_run).count("draft") == 2
    assert_parity(in_run, flatten(segments))


# ---------------------------------------------------------------------------
# V-10: a review without emit keeps join progress
# ---------------------------------------------------------------------------


def make_plain_review_join(entry: str) -> Factory:
    def make(reviewer: Reviewer, log: list[Record]) -> type[Flow[Any]]:
        delivered = threading.Event()

        class PlainReviewJoin(Flow):
            @start()
            def other(self):
                log.append(("other", None))
                return "other-output"

            @listen(other)
            def other_delivered(self):
                delivered.set()

            if entry == "listener":

                @start()
                def begin(self):
                    assert delivered.wait(WAIT_SECONDS)
                    log.append(("begin", None))

            @entry_point(entry)
            @human_feedback(message="Review", provider=reviewer)
            def review(self):
                if entry == "start":
                    assert delivered.wait(WAIT_SECONDS)
                log.append(("review", None))
                return "review-output"

            @listen(and_(review, other))
            def combine(self, r=None):
                log.append(("combine", payload(r)))

        return PlainReviewJoin

    return make


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
@pytest.mark.parametrize("entry", ENTRY_PATHS)
def test_resume_review_without_emit_runs_join_with_member_before_pause(
    entry: str, resume_mode: str, db_path: str
) -> None:
    """V-10: and_(review, other) with other done before the pause runs."""
    make = make_plain_review_join(entry)
    answers = {"review": ["looks good"]}

    in_run = run_in_run(make, answers)
    segments, _ = run_paused(make, answers, {"review"}, resume_mode, db_path)

    assert "combine" in names(in_run)
    assert_parity(in_run, flatten(segments))


# ---------------------------------------------------------------------------
# V-11: pending rows saved without resume progress
# ---------------------------------------------------------------------------


def _rewrite_context(
    persistence: SQLiteFlowPersistence,
    flow_id: str,
    edit: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    with closing(sqlite3.connect(persistence.db_path)) as conn, conn:
        row = conn.execute(
            "SELECT context_json FROM pending_feedback WHERE flow_uuid = ?",
            (flow_id,),
        ).fetchone()
        context = edit(json.loads(row[0]))
        conn.execute(
            "UPDATE pending_feedback SET context_json = ? WHERE flow_uuid = ?",
            (json.dumps(context), flow_id),
        )


def _strip_progress(context: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in context.items() if k in PRE_PROGRESS_CONTEXT_KEYS}


def _progress_not_a_mapping(context: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v if k in PRE_PROGRESS_CONTEXT_KEYS else "not-a-mapping"
        for k, v in context.items()
    }


def _non_string_leaves(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _non_string_leaves(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_non_string_leaves(v) for v in value]
    return 42


def _progress_entries_malformed(context: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v if k in PRE_PROGRESS_CONTEXT_KEYS else _non_string_leaves(v)
        for k, v in context.items()
    }


ROW_EDITS = {
    "old_row": _strip_progress,
    "progress_not_a_mapping": _progress_not_a_mapping,
    "progress_entries_malformed": _progress_entries_malformed,
}


def _edit_every_row(edit: Callable[[dict[str, Any]], dict[str, Any]]) -> Tamper:
    def tamper(persistence: SQLiteFlowPersistence, flow_id: str, index: int) -> None:
        _rewrite_context(persistence, flow_id, edit)

    return tamper


def make_old_row_start_review() -> Factory:
    def make(reviewer: Reviewer, log: list[Record]) -> type[Flow[Any]]:
        delivered = threading.Event()

        class OldRowS(Flow):
            @start()
            def other(self):
                log.append(("other", None))
                return "other-output"

            @listen(other)
            def other_delivered(self):
                delivered.set()

            @start()
            @human_feedback(
                message="Review",
                emit=["approved", "rejected"],
                llm=LLM,
                provider=reviewer,
            )
            def draft(self):
                assert delivered.wait(WAIT_SECONDS)
                log.append(("draft", None))
                return "draft-output"

            @listen(draft)
            def own(self, r=None):
                log.append(("own", payload(r)))

            @listen("approved")
            def publish(self, r=None):
                log.append(("publish", payload(r)))

            @listen(or_(draft, other))
            def either(self, r=None):
                log.append(("either", payload(r)))

            @listen(and_(draft, other))
            def combine(self, r=None):
                log.append(("combine", payload(r)))

        return OldRowS

    return make


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
@pytest.mark.parametrize("row_edit", list(ROW_EDITS))
def test_old_row_loads_and_resumes(
    row_edit: str, resume_mode: str, db_path: str
) -> None:
    """V-11 load guard: a row without (or with malformed) progress restores."""
    segments, paused_on = run_paused(
        make_old_row_start_review(),
        {"draft": ["approved"]},
        {"draft"},
        resume_mode,
        db_path,
        tamper=_edit_every_row(ROW_EDITS[row_edit]),
    )

    assert paused_on == ["draft"]
    assert "publish" in names(segments[1])


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
@pytest.mark.parametrize("row_edit", list(ROW_EDITS))
def test_old_row_runs_own_group_and_refires_or_on_reviewed_method(
    row_edit: str, resume_mode: str, db_path: str
) -> None:
    """V-11(a): an old row gets the full dispatch with empty progress.

    Own and label listeners run once; or_(draft, other), fired by other
    before the pause, fires again because its fired state was not stored.
    """
    segments, _ = run_paused(
        make_old_row_start_review(),
        {"draft": ["approved"]},
        {"draft"},
        resume_mode,
        db_path,
        tamper=_edit_every_row(ROW_EDITS[row_edit]),
    )

    all_names = names(flatten(segments))
    assert all_names.count("own") == 1
    assert all_names.count("publish") == 1
    assert names(segments[0]).count("either") == 1
    assert all_names.count("either") == 2


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
@pytest.mark.parametrize("row_edit", list(ROW_EDITS))
def test_old_row_does_not_run_join_with_member_before_pause(
    row_edit: str, resume_mode: str, db_path: str
) -> None:
    """V-11(b) guard: and_(draft, other) does not run for an old row."""
    segments, _ = run_paused(
        make_old_row_start_review(),
        {"draft": ["approved"]},
        {"draft"},
        resume_mode,
        db_path,
        tamper=_edit_every_row(ROW_EDITS[row_edit]),
    )

    assert "combine" not in names(flatten(segments))


def make_old_row_start_and_listen() -> Factory:
    def make(reviewer: Reviewer, log: list[Record]) -> type[Flow[Any]]:
        class StartAndListen(Flow):
            @start()
            @listen("revise")
            @human_feedback(
                message="Review",
                emit=["approved", "revise"],
                llm=LLM,
                provider=reviewer,
            )
            def draft(self):
                log.append(("draft", None))
                return "draft-output"

            @start("approved")
            def gated(self):
                log.append(("gated", None))

            @listen("approved")
            def publish(self, r=None):
                log.append(("publish", payload(r)))

        return StartAndListen

    return make


def _unknown_markers(known: set[str]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def replace(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: replace(v) for k, v in value.items()}
        if isinstance(value, list):
            return [replace(v) for v in value]
        if isinstance(value, str) and value not in known:
            return "unknown-marker"
        return value

    def edit(context: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v if k in PRE_PROGRESS_CONTEXT_KEYS else replace(v)
            for k, v in context.items()
        }

    return edit


START_AND_LISTEN_EDITS = {
    **ROW_EDITS,
    "unknown_marker": _unknown_markers(
        {"draft", "gated", "publish", "approved", "revise"}
    ),
}


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
@pytest.mark.parametrize("row_edit", list(START_AND_LISTEN_EDITS))
def test_old_row_start_and_listen_review_infers_start_path(
    row_edit: str, resume_mode: str, db_path: str
) -> None:
    """V-11(c) guard: paused on its listener path, an old row infers the start path.

    The label then goes out at index 0, so the @start gated on it does not run.
    """
    edit = START_AND_LISTEN_EDITS[row_edit]

    def tamper(persistence: SQLiteFlowPersistence, flow_id: str, index: int) -> None:
        if index == 1:
            _rewrite_context(persistence, flow_id, edit)

    segments, paused_on = run_paused(
        make_old_row_start_and_listen(),
        {"draft": ["revise", "approved"]},
        {"draft"},
        resume_mode,
        db_path,
        tamper=tamper,
    )

    assert paused_on == ["draft", "draft"]
    assert "publish" in names(segments[2])
    assert "gated" not in names(flatten(segments))


def make_old_row_enclosing() -> Factory:
    def make(reviewer: Reviewer, log: list[Record]) -> type[Flow[Any]]:
        class Enclosing(Flow):
            @start()
            @human_feedback(
                message="Review",
                emit=["approved", "rejected"],
                llm=LLM,
                provider=reviewer,
            )
            def draft(self):
                log.append(("draft", None))
                return "draft-output"

            @listen(draft)
            @human_feedback(
                message="Audit", emit=["ok2", "no2"], llm=LLM, provider=reviewer
            )
            def audit(self, r=None):
                log.append(("audit", payload(r)))
                return "audit-output"

            @listen("approved")
            def publish(self, r=None):
                log.append(("publish", payload(r)))

        return Enclosing

    return make


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
@pytest.mark.parametrize("row_edit", list(ROW_EDITS))
def test_old_row_does_not_deliver_enclosing_label(
    row_edit: str, resume_mode: str, db_path: str
) -> None:
    """V-11(d) guard: an old row carries no enclosing label delivery."""
    segments, paused_on = run_paused(
        make_old_row_enclosing(),
        {"draft": ["approved"], "audit": ["ok2"]},
        {"audit"},
        resume_mode,
        db_path,
        tamper=_edit_every_row(ROW_EDITS[row_edit]),
    )

    assert paused_on == ["audit"]
    assert "publish" not in names(flatten(segments))


# ---------------------------------------------------------------------------
# V-12: listeners a cut parent trigger satisfies alone are left alone
# ---------------------------------------------------------------------------


def make_cut_siblings() -> Factory:
    def make(reviewer: Reviewer, log: list[Record]) -> type[Flow[Any]]:
        delivered = threading.Event()

        class CutSiblings(Flow):
            @start()
            def x(self):
                log.append(("x", None))

            @listen(x)
            def x_delivered(self):
                delivered.set()

            @start()
            def begin(self):
                assert delivered.wait(WAIT_SECONDS)
                log.append(("begin", None))

            @listen(begin)
            @human_feedback(
                message="Review",
                emit=["approved", "rejected"],
                llm=LLM,
                provider=reviewer,
            )
            def draft(self):
                log.append(("draft", None))
                return "draft-output"

            @listen(begin)
            def sibling(self):
                log.append(("sibling", None))

            @listen(and_(begin, x))
            def begin_and_x(self):
                log.append(("begin_and_x", None))

            @listen("approved")
            def publish(self, r=None):
                log.append(("publish", payload(r)))

        return CutSiblings

    return make


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
def test_resume_leaves_listeners_of_cut_parent_trigger_alone(
    resume_mode: str, db_path: str
) -> None:
    """V-12 guard: a sibling and and_(begin, x) cut by the pause do not run."""
    segments, _ = run_paused(
        make_cut_siblings(), {"draft": ["approved"]}, {"draft"}, resume_mode, db_path
    )

    assert "sibling" not in names(segments[0])
    assert "begin_and_x" not in names(segments[0])
    resumed = names(flatten(segments[1:]))
    assert "publish" in resumed
    assert "sibling" not in resumed
    assert "begin_and_x" not in resumed


# ---------------------------------------------------------------------------
# V-13: enclosing router labels cut by a pause at kickoff
# ---------------------------------------------------------------------------


def make_enclosing(shape: str) -> Factory:
    def make(reviewer: Reviewer, log: list[Record]) -> type[Flow[Any]]:
        if shape == "start":

            class EnclosingS(Flow):
                @start()
                @human_feedback(
                    message="Review",
                    emit=["approved", "rejected"],
                    llm=LLM,
                    provider=reviewer,
                )
                def draft(self):
                    log.append(("draft", None))
                    return "draft-output"

                @listen(draft)
                @human_feedback(
                    message="Audit", emit=["ok2", "no2"], llm=LLM, provider=reviewer
                )
                def audit(self, r=None):
                    log.append(("audit", payload(r)))
                    return "audit-output"

                @listen("approved")
                def publish(self, r=None):
                    log.append(("publish", payload(r)))

                @listen("ok2")
                def after_audit(self, r=None):
                    log.append(("after_audit", payload(r)))

            return EnclosingS

        if shape == "listener":

            class EnclosingL(Flow):
                @start()
                def begin(self):
                    log.append(("begin", None))

                @listen(begin)
                @human_feedback(
                    message="Review",
                    emit=["approved", "rejected"],
                    llm=LLM,
                    provider=reviewer,
                )
                def draft(self):
                    log.append(("draft", None))
                    return "draft-output"

                @listen(draft)
                @human_feedback(
                    message="Audit", emit=["ok2", "no2"], llm=LLM, provider=reviewer
                )
                def audit(self, r=None):
                    log.append(("audit", payload(r)))
                    return "audit-output"

                @listen("approved")
                def publish(self, r=None):
                    log.append(("publish", payload(r)))

                @start("approved")
                def gated(self):
                    log.append(("gated", None))

            return EnclosingL

        class EnclosingRouter(Flow):
            @start()
            def begin(self):
                log.append(("begin", None))

            @router(begin)
            def decide(self):
                log.append(("decide", None))
                return "go"

            if shape == "router":

                @listen(decide)
                @human_feedback(
                    message="Audit", emit=["ok2", "no2"], llm=LLM, provider=reviewer
                )
                def audit(self, r=None):
                    log.append(("audit", payload(r)))
                    return "audit-output"

            else:

                @listen(decide)
                @human_feedback(message="Audit", provider=reviewer)
                def audit(self, r=None):
                    log.append(("audit", payload(r)))
                    return "audit-output"

            @listen("go")
            def on_go(self, r=None):
                log.append(("on_go", payload(r)))

        return EnclosingRouter

    return make


ENCLOSING_ANSWERS = {
    "start": {"draft": ["approved"], "audit": ["ok2"]},
    "listener": {"draft": ["approved"], "audit": ["ok2"]},
    "router": {"audit": ["ok2"]},
    "router_plain_review": {"audit": ["looks good"]},
}

ENCLOSING_LABEL = {
    "start": "publish",
    "listener": "publish",
    "router": "on_go",
    "router_plain_review": "on_go",
}


@pytest.mark.parametrize("resume_mode", RESUME_MODES)
@pytest.mark.parametrize("shape", list(ENCLOSING_ANSWERS))
def test_resume_delivers_enclosing_label_cut_at_kickoff(
    shape: str, resume_mode: str, db_path: str
) -> None:
    """V-13: an enclosing router's label cut by the inner pause still runs."""
    make = make_enclosing(shape)
    answers = ENCLOSING_ANSWERS[shape]

    in_run = run_in_run(make, answers)
    segments, paused_on = run_paused(make, answers, {"audit"}, resume_mode, db_path)

    assert paused_on == ["audit"]
    assert ENCLOSING_LABEL[shape] in names(in_run)
    assert_parity(
        in_run,
        flatten(segments),
        groups={"after_audit": "inner-label", "publish": "outer", "gated": "outer"},
    )
