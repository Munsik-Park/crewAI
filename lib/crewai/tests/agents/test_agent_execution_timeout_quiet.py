"""The run left behind by ``max_execution_time`` stops quietly inside a step.

When the call in flight at the limit ends, the run does not start the call
that would follow it in the same step. Nobody receives the result of that
stop: the caller already has its ``TimeoutError``. So the stop is not printed
as an error, not logged, and not fed back to the LLM as a tool error, even
when the agent is verbose.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from crewai.tools import tool
from tests.agents.test_agent_execution_timeout import (
    ORPHAN_BLOCK,
    _action,
    _agent,
    _crewai_error_records,
    _final,
    _ScriptedLLM,
    _task,
)
from tests.agents.test_agent_execution_timeout_in_step import (
    QUIET_END_NOISE,
    _NativeLLM,
    _time_out,
)


def _assert_quiet(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    output = capsys.readouterr()
    printed = output.out + output.err
    assert [text for text in QUIET_END_NOISE if text in printed] == []
    assert _crewai_error_records(caplog) == []


def test_stop_before_a_tool_retry_is_quiet(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """A verbose agent whose tool fails after the limit stops without reporting the stop."""
    release = threading.Event()
    invocations: list[str] = []

    @tool("flaky_tool")
    def flaky_tool(query: str) -> str:
        """Look a query up."""
        invocations.append(query)
        if len(invocations) == 1:
            release.wait(ORPHAN_BLOCK)
        raise RuntimeError("tool failed after the limit")

    def script(text: str) -> str:
        if "tool failed after the limit" in text:
            return _final("orphan answer")
        return _action("flaky_tool", '{"query": "q"}')

    agent = _agent(_ScriptedLLM(script), tools=[flaky_tool], verbose=True)

    _time_out(agent, _task(agent, "TASK_ONE"), release)

    assert len(invocations) == 1
    _assert_quiet(caplog, capsys)


def test_stop_before_the_tool_of_a_parsed_call_is_quiet(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """A verbose agent whose tool call is parsed after the limit stops without reporting the stop."""
    release = threading.Event()
    parses: list[str] = []
    tool_runs: list[str] = []

    @tool("lookup")
    def lookup() -> str:
        """Look something up."""
        tool_runs.append("lookup")
        return "LOOKUP_RESULT"

    def parsing_script(text: str) -> str:
        parses.append(text)
        release.wait(ORPHAN_BLOCK)
        return json.dumps({"tool_name": "lookup", "arguments": {}})

    agent = _agent(
        _ScriptedLLM(lambda text: _action("lookup", '"look it up"')),
        tools=[lookup],
        function_calling_llm=_ScriptedLLM(parsing_script),
        verbose=True,
    )

    _time_out(agent, _task(agent, "TASK_ONE"), release)

    assert len(parses) == 1
    assert tool_runs == []
    _assert_quiet(caplog, capsys)


@pytest.mark.parametrize("calling", ["react", "native"])
def test_stop_before_an_empty_answer_is_asked_again_is_quiet(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    calling: str,
) -> None:
    """A verbose agent whose LLM answers empty after the limit stops without reporting the stop."""
    release = threading.Event()
    calls: list[str] = []

    def script(text: str) -> Any:
        calls.append(text)
        if len(calls) == 1:
            release.wait(ORPHAN_BLOCK)
            return ""
        return _final("orphan answer")

    @tool("lookup")
    def lookup() -> str:
        """Look something up."""
        return "LOOKUP_RESULT"

    if calling == "native":
        agent = _agent(_NativeLLM(script), tools=[lookup], verbose=True)
    else:
        agent = _agent(_ScriptedLLM(script), verbose=True)

    _time_out(agent, _task(agent, "TASK_ONE"), release)

    assert len(calls) == 1
    _assert_quiet(caplog, capsys)
