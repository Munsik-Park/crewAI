"""A plan whose function-calling request fails after ``max_execution_time``.

Planning asks for the plan through function calling and, when that request
fails, asks again as plain text. When the failing request is the call in
flight at the limit, the run left behind does not make the plain-text
request, and it ends without reporting an error.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from crewai.agent.planning_config import PlanningConfig
from tests.agents.test_agent_execution_timeout import (
    ORPHAN_BLOCK,
    _agent,
    _crewai_error_records,
    _ScriptedLLM,
    _task,
)
from tests.agents.test_agent_execution_timeout_in_step import (
    QUIET_END_NOISE,
    _time_out,
)


class _FunctionCallingPlanner(_ScriptedLLM):
    """Scripted LLM that declares function calling, so planning uses it."""

    def supports_function_calling(self) -> bool:
        return True


def test_timed_out_run_does_not_retry_a_failed_plan_request_as_text(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """A function-calling plan request that fails after the limit is not asked again as plain text."""
    release = threading.Event()
    calls: list[str] = []

    def script(text: str) -> Any:
        calls.append(text)
        if len(calls) == 1:
            release.wait(ORPHAN_BLOCK)
            raise RuntimeError("plan request failed after the limit")
        return "Plan: answer.\nREADY: I am ready to execute the task."

    agent = _agent(
        _FunctionCallingPlanner(script),
        planning_config=PlanningConfig(max_attempts=3),
        verbose=True,
    )

    _time_out(agent, _task(agent, "TASK_ONE"), release)

    assert len(calls) == 1, (
        f"the LLM was called {len(calls)} times; only the call in flight at the "
        "limit may run"
    )
    output = capsys.readouterr()
    printed = output.out + output.err
    assert [text for text in QUIET_END_NOISE if text in printed] == []
    assert _crewai_error_records(caplog) == []
