"""A run left behind by ``max_execution_time`` that kicks off a coworker.

The step in flight at the limit may go on to run another agent through
``kickoff`` or ``kickoff_async``. That agent may already be serving a later
call, so the abandoned run must leave it as it was.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from crewai import Agent
from crewai.tools import tool
from tests.agents.test_agent_execution_timeout import (
    ORPHAN_BLOCK,
    _agent,
    _asking_script,
    _CoworkerState,
    _join_timed_runs,
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
