"""A run left behind by ``max_execution_time`` that kicks off a coworker.

The step in flight at the limit may go on to run another agent through
``kickoff`` or ``kickoff_async``. That agent may already be serving a later
call, so the abandoned run must leave it as it was. A coworker kicked off
before the limit finishes only its own call in flight: after it, the
coworker converts no output again, runs no guardrail and saves nothing to
memory.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from pydantic import BaseModel
import pytest

from crewai import Agent
from crewai.tools import tool
from tests.agents.test_agent_execution_timeout import (
    ORPHAN_BLOCK,
    _agent,
    _asking_script,
    _CoworkerState,
    _final,
    _join_timed_runs,
    _RecordingMemory,
    _ScriptedLLM,
    _task,
)


def _kicking_off_coworker_tool(
    coworker: Agent, release: threading.Event, outcome: list[str], entry: str
) -> Any:
    """A tool that waits for the test, then kicks the coworker off."""

    @tool("ask_coworker")
    def ask_coworker() -> str:
        """Ask the coworker."""
        release.wait(ORPHAN_BLOCK)
        try:
            if entry == "kickoff":
                coworker.kickoff("SUB_ONE: answer")
            else:
                asyncio.run(coworker.kickoff_async("SUB_ONE: answer"))
            outcome.append("ran")
        except BaseException as error:
            outcome.append(type(error).__name__)
        return "ASKED_COWORKER"

    return ask_coworker


@pytest.mark.parametrize("entry", ["kickoff", "kickoff_async"])
def test_timed_out_run_leaves_coworker_it_kicks_off_after_the_limit_as_it_was(
    entry: str,
) -> None:
    """A coworker kicked off from the step in flight after the limit keeps its tool failures, last used tool and tool usage counts."""
    coworker = _CoworkerState()
    coworker.record(coworker.agent.execute_task(coworker.pre_task))
    release = threading.Event()
    outcome: list[str] = []
    ask = _kicking_off_coworker_tool(coworker.agent, release, outcome, entry)
    agent = _agent(_ScriptedLLM(_asking_script), tools=[ask])

    before = set(threading.enumerate())
    try:
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "TASK_ONE"))
        assert not release.is_set()
    finally:
        release.set()
        still_alive = _join_timed_runs(before)
    assert still_alive == []
    assert len(outcome) == 1, "the timed-out run's tool never called the coworker"
    coworker.assert_unchanged()


CONVERTER_PROMPT = "Format your final answer according to the following OpenAPI schema"


class _Answer(BaseModel):
    value: str


class _CoworkerFinish:
    """A coworker kicked off before the limit, whose run ends after it.

    The coworker answers in free form, so its answer is converted into the
    response model once its run ends. The test chooses what is in flight at
    the limit: the conversion, or the coworker's guardrail.
    """

    def __init__(self, release: threading.Event, block: str, converted: str) -> None:
        self.release = release
        self.conversions: list[str] = []
        self.guardrail_runs = 0
        self.memory = _RecordingMemory()

        def script(text: str) -> str:
            if CONVERTER_PROMPT in text:
                self.conversions.append(text)
                if block == "conversion" and len(self.conversions) == 1:
                    release.wait(ORPHAN_BLOCK)
                return converted
            return _final("plain words, not the response model")

        def guardrail(output: Any) -> tuple[bool, Any]:
            self.guardrail_runs += 1
            if block == "guardrail":
                release.wait(ORPHAN_BLOCK)
            return True, output

        self.agent = _agent(
            _ScriptedLLM(script),
            role="Coworker",
            max_execution_time=None,
            guardrail=guardrail,
        )
        object.__setattr__(self.agent, "memory", self.memory)


def _kicking_off_coworker_now_tool(
    coworker: Agent, outcome: list[str], entry: str
) -> Any:
    """A tool that kicks the coworker off at once, asking for the response model."""

    @tool("ask_coworker")
    def ask_coworker() -> str:
        """Ask the coworker."""
        try:
            if entry == "kickoff":
                coworker.kickoff("SUB_ONE: answer", response_format=_Answer)
            else:
                asyncio.run(
                    coworker.kickoff_async("SUB_ONE: answer", response_format=_Answer)
                )
            outcome.append("ran")
        except BaseException as error:
            outcome.append(type(error).__name__)
        return "ASKED_COWORKER"

    return ask_coworker


def _time_out_kicking_off(coworker: _CoworkerFinish, entry: str) -> list[str]:
    """Time a run out while the coworker it kicked off is finishing; wait for its end."""
    outcome: list[str] = []
    ask = _kicking_off_coworker_now_tool(coworker.agent, outcome, entry)
    agent = _agent(_ScriptedLLM(_asking_script), tools=[ask])

    before = set(threading.enumerate())
    try:
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "TASK_ONE"))
        assert not coworker.release.is_set(), (
            "the coworker's call in flight ended before the limit"
        )
    finally:
        coworker.release.set()
        still_alive = _join_timed_runs(before)
    assert still_alive == []
    assert len(outcome) == 1, "the timed-out run's tool never called the coworker"
    return outcome


@pytest.mark.parametrize("entry", ["kickoff", "kickoff_async"])
def test_coworker_kicked_off_by_timed_out_run_runs_no_guardrail_and_saves_nothing(
    entry: str,
) -> None:
    """A coworker whose output conversion ends after the limit runs no guardrail and saves nothing to memory."""
    coworker = _CoworkerFinish(
        threading.Event(), block="conversion", converted='{"value": "converted"}'
    )

    _time_out_kicking_off(coworker, entry)

    assert len(coworker.conversions) == 1, "the conversion was not in flight"
    ran = [
        name
        for name, count in (
            ("guardrail", coworker.guardrail_runs),
            ("memory save", len(coworker.memory.saved)),
        )
        if count
    ]
    assert ran == [], f"the coworker ran {ran} after the limit"


@pytest.mark.parametrize("entry", ["kickoff", "kickoff_async"])
def test_coworker_kicked_off_by_timed_out_run_saves_nothing_after_its_guardrail(
    entry: str,
) -> None:
    """A coworker whose guardrail ends after the limit saves nothing to memory."""
    coworker = _CoworkerFinish(
        threading.Event(), block="guardrail", converted='{"value": "converted"}'
    )

    _time_out_kicking_off(coworker, entry)

    assert coworker.guardrail_runs == 1, "the guardrail was not in flight"
    assert coworker.memory.saved == [], "the coworker saved to memory after the limit"


@pytest.mark.parametrize("entry", ["kickoff", "kickoff_async"])
def test_coworker_kicked_off_by_timed_out_run_does_not_convert_again(
    entry: str,
) -> None:
    """A coworker whose output conversion fails after the limit does not try the conversion again."""
    coworker = _CoworkerFinish(
        threading.Event(), block="conversion", converted="not the response model"
    )

    _time_out_kicking_off(coworker, entry)

    assert len(coworker.conversions) == 1, (
        f"the conversion ran {len(coworker.conversions)} times; only the call in "
        "flight at the limit may run"
    )
