"""Restoring a paused flow keeps only the carried label deliveries it defines.

A pending row carries the label deliveries of enclosing routers that a pause
cut off. On restore, each entry must name a router the flow defines and a
label some listener or conditional start of the flow waits on; any other
entry is dropped, like the other restored progress fields.
"""

from __future__ import annotations

import asyncio
from contextlib import closing
import json
import sqlite3
from typing import Any
from unittest.mock import patch

from crewai.flow import Flow, human_feedback, listen, start
from crewai.flow.async_feedback import HumanFeedbackPending
from crewai.flow.persistence import SQLiteFlowPersistence
import pytest


def _collapse(self: Flow[Any], feedback: str, outcomes: Any, llm: Any) -> str:
    return feedback if feedback in outcomes else outcomes[0]


@pytest.fixture(autouse=True)
def _offline_collapse():
    with patch.object(Flow, "_collapse_to_outcome", _collapse):
        yield


class PausingProvider:
    def request_feedback(self, context: Any, flow: Any) -> str:
        raise HumanFeedbackPending(context=context)


def make_flow(log: list[str]) -> type[Flow[Any]]:
    class CarriedFlow(Flow):
        @start()
        def plain(self):
            log.append("plain")

        @start()
        @human_feedback(
            message="Review",
            emit=["approved", "rejected"],
            llm="gpt-4o-mini",
            provider=PausingProvider(),
        )
        def draft(self):
            log.append("draft")
            return "draft-output"

        @listen("approved")
        def publish(self):
            log.append("publish")

        @listen("kept")
        def on_kept(self):
            log.append("on_kept")

        @listen("from_unknown_router")
        def on_unknown_router(self):
            log.append("on_unknown_router")

        @listen("from_plain_method")
        def on_plain_method(self):
            log.append("on_plain_method")

    return CarriedFlow


TAMPERED_CARRIED = [
    {"method_name": "draft", "label": "kept", "entry_path": "start"},
    {"method_name": "ghost", "label": "from_unknown_router", "entry_path": "start"},
    {"method_name": "plain", "label": "from_plain_method", "entry_path": "start"},
    {"method_name": "draft", "label": "undefined_label", "entry_path": "start"},
]


def _add_carried_labels(db_path: str, flow_id: str) -> None:
    with closing(sqlite3.connect(db_path)) as conn, conn:
        row = conn.execute(
            "SELECT context_json FROM pending_feedback WHERE flow_uuid = ?",
            (flow_id,),
        ).fetchone()
        context = json.loads(row[0])
        context["resume_progress"]["carried_labels"] = TAMPERED_CARRIED
        conn.execute(
            "UPDATE pending_feedback SET context_json = ? WHERE flow_uuid = ?",
            (json.dumps(context), flow_id),
        )


@pytest.mark.parametrize("resume_mode", ["sync", "async"])
def test_restore_drops_carried_labels_the_flow_does_not_define(
    resume_mode: str, tmp_path: Any
) -> None:
    log: list[str] = []
    flow_class = make_flow(log)
    db_path = str(tmp_path / "pending.db")
    persistence = SQLiteFlowPersistence(db_path)

    paused = flow_class(persistence=persistence).kickoff()
    assert isinstance(paused, HumanFeedbackPending)
    flow_id = paused.context.flow_id
    _add_carried_labels(db_path, flow_id)

    restored = flow_class.from_pending(flow_id, persistence)
    context = restored.pending_feedback
    assert context is not None and context.resume_progress is not None
    assert [
        (carried.method_name, carried.label)
        for carried in context.resume_progress.carried_labels
    ] == [("draft", "kept")]

    start_at = len(log)
    if resume_mode == "sync":
        restored.resume("approved")
    else:
        asyncio.run(restored.resume_async("approved"))

    assert log[start_at:] == ["publish", "on_kept"]
