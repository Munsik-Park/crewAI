"""Work that one step of a timed-out run would start after the limit.

The run left behind by ``max_execution_time`` finishes only the call in flight
at the limit. Some steps make more than one call: a failed tool is tried
again, an empty LLM answer is asked for again, a free-form final answer is
asked for once more in the response model, a request is preceded by context
summarization, and native tool calls run one after another or queue for a
worker. When the call in flight ends after the limit, none of the calls that
would follow it may start.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
from typing import Any

from pydantic import BaseModel
import pytest

from crewai.tools import BaseTool, tool
from tests.agents.test_agent_execution_timeout import (
    JOIN_BOUND,
    ORPHAN_BLOCK,
    _action,
    _agent,
    _final,
    _join_timed_runs,
    _ScriptedLLM,
    _task,
    _TrackingExecutor,
)


TOOL_FAILURE = "tool failed after the limit"


def _time_out(agent: Any, task: Any, release: threading.Event) -> None:
    """Run ``task`` into its timeout, then release the run and wait for its end."""
    before = set(threading.enumerate())
    try:
        with pytest.raises(TimeoutError):
            agent.execute_task(task)
        assert not release.is_set(), "the call in flight ended before the limit"
    finally:
        release.set()
        still_alive = _join_timed_runs(before)
    assert still_alive == []


def _failing_tool(
    release: threading.Event, invocations: list[str], with_args: bool
) -> BaseTool:
    """A tool whose first call blocks through the limit; every call then fails."""

    def fail() -> str:
        if len(invocations) == 1:
            release.wait(ORPHAN_BLOCK)
        raise RuntimeError(TOOL_FAILURE)

    if with_args:

        @tool("flaky_tool")
        def flaky_with_args(query: str) -> str:
            """Look a query up."""
            invocations.append(query)
            return fail()

        return flaky_with_args

    @tool("flaky_tool")
    def flaky() -> str:
        """Look something up."""
        invocations.append("call")
        return fail()

    return flaky


def _failing_tool_script(with_args: bool) -> Any:
    def script(text: str) -> str:
        if TOOL_FAILURE in text:
            return _final("orphan answer")
        if with_args:
            return _action("flaky_tool", '{"query": "q"}')
        return _action("flaky_tool")

    return script


@pytest.mark.parametrize("with_args", [False, True], ids=["no_args", "args"])
def test_timed_out_run_does_not_call_its_failed_tool_again(with_args: bool) -> None:
    """A tool call that fails after the limit is not tried again."""
    release = threading.Event()
    invocations: list[str] = []
    flaky = _failing_tool(release, invocations, with_args)
    agent = _agent(_ScriptedLLM(_failing_tool_script(with_args)), tools=[flaky])

    _time_out(agent, _task(agent, "TASK_ONE"), release)

    assert len(invocations) == 1, (
        f"the tool ran {len(invocations)} times; only the call in flight at the "
        "limit may run"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("with_args", [False, True], ids=["no_args", "args"])
async def test_cancelled_run_does_not_call_its_failed_tool_again(
    with_args: bool,
) -> None:
    """After an async timeout, a tool call that fails in its worker thread is not tried again."""
    loop = asyncio.get_running_loop()
    pool = _TrackingExecutor()
    loop.set_default_executor(pool)

    release = threading.Event()
    invocations: list[str] = []
    in_flight_at_timeout: list[set[concurrent.futures.Future[Any]]] = []
    flaky = _failing_tool(release, invocations, with_args)
    agent = _agent(_ScriptedLLM(_failing_tool_script(with_args)), tools=[flaky])

    try:
        with pytest.raises(TimeoutError):
            await agent.aexecute_task(_task(agent, "TASK_ONE"))
        in_flight_at_timeout.append(pool.snapshot())
        assert not release.is_set(), "the call in flight ended before the limit"
    finally:
        release.set()
        if in_flight_at_timeout:
            concurrent.futures.wait(in_flight_at_timeout[0], timeout=JOIN_BOUND)
        pool.shutdown(wait=True)

    assert len(invocations) == 1, (
        f"the tool ran {len(invocations)} times; only the call in flight at the "
        "limit may run"
    )


def test_timed_out_run_does_not_ask_again_for_an_empty_answer() -> None:
    """An LLM answer that comes back empty after the limit is not requested again."""
    release = threading.Event()
    calls: list[str] = []

    def script(text: str) -> str:
        calls.append(text)
        if len(calls) == 1:
            release.wait(ORPHAN_BLOCK)
            return ""
        return _final("orphan answer")

    agent = _agent(_ScriptedLLM(script))

    _time_out(agent, _task(agent, "TASK_ONE"), release)

    assert len(calls) == 1, (
        f"the LLM was called {len(calls)} times; only the call in flight at the "
        "limit may run"
    )


class _Answer(BaseModel):
    value: str


def test_timed_out_run_does_not_ask_again_for_a_structured_answer() -> None:
    """A free-form final answer that arrives after the limit is not requested again in the response model."""
    release = threading.Event()
    calls: list[str] = []

    @tool("lookup")
    def lookup() -> str:
        """Look something up."""
        return "LOOKUP_RESULT"

    def script(text: str) -> str:
        calls.append(text)
        if len(calls) == 1:
            release.wait(ORPHAN_BLOCK)
            return _final("plain words, not the response model")
        return json.dumps({"value": "structured"})

    agent = _agent(_ScriptedLLM(script), tools=[lookup])
    task = _task(agent, "TASK_ONE")
    task.output_pydantic = _Answer

    _time_out(agent, task, release)

    assert len(calls) == 1, (
        f"the LLM was called {len(calls)} times; only the call in flight at the "
        "limit may run"
    )


class _SummarizingLLM(_ScriptedLLM):
    """Scripted LLM whose context always needs summarizing; the summary blocks."""

    def __init__(self, script: Any, release: threading.Event) -> None:
        super().__init__(script)
        self._release = release
        self.summaries = 0

    def is_context_window_overflowing_threshold(
        self, *args: Any, **kwargs: Any
    ) -> bool:
        return True

    def summarize_messages(self, *args: Any, **kwargs: Any) -> None:
        self.summaries += 1
        self._release.wait(ORPHAN_BLOCK)


def test_timed_out_run_does_not_send_the_request_it_was_summarizing_for() -> None:
    """A context summarization that ends after the limit is not followed by the request it prepared."""
    release = threading.Event()
    requests: list[str] = []

    @tool("lookup")
    def lookup() -> str:
        """Look something up."""
        return "LOOKUP_RESULT"

    def script(text: str) -> str:
        requests.append(text)
        if "LOOKUP_RESULT" in text:
            return _final("orphan answer")
        return _action("lookup")

    llm = _SummarizingLLM(script, release)
    agent = _agent(llm, tools=[lookup])

    _time_out(agent, _task(agent, "TASK_ONE"), release)

    assert llm.summaries == 1, "the summarization was not the call in flight"
    sent_after = [text for text in requests if "LOOKUP_RESULT" in text]
    assert sent_after == [], "the request after the summarization was sent"


class _NativeLLM(_ScriptedLLM):
    """Scripted LLM that declares function calling, so tool calls are native."""

    def supports_function_calling(self) -> bool:
        return True

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        return self._script(str(messages))


def _native_call(index: int, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _native_script(calls: list[dict[str, Any]]) -> Any:
    def script(text: str) -> Any:
        if "'role': 'tool'" in text:
            return "orphan answer"
        return calls

    return script


def test_timed_out_run_does_not_start_its_next_native_tool_call() -> None:
    """Of native tool calls run one after another, the one after the call in flight at the limit does not start."""
    release = threading.Event()
    ran: list[str] = []

    @tool("slow_tool")
    def slow_tool() -> str:
        """Block until the test releases it."""
        ran.append("slow_tool")
        release.wait(ORPHAN_BLOCK)
        return "SLOW_RESULT"

    # A usage limit makes the executor run the calls one after another.
    slow_tool.max_usage_count = 5

    @tool("next_tool")
    def next_tool() -> str:
        """Record that it ran."""
        ran.append("next_tool")
        return "NEXT_RESULT"

    script = _native_script(
        [_native_call(0, "slow_tool", {}), _native_call(1, "next_tool", {})]
    )
    agent = _agent(_NativeLLM(script), tools=[slow_tool, next_tool])

    _time_out(agent, _task(agent, "TASK_ONE"), release)

    assert ran == ["slow_tool"], f"tools run: {ran}"


def test_timed_out_run_does_not_start_its_queued_native_tool_calls() -> None:
    """Native tool calls still queued for a worker at the limit do not start."""
    release = threading.Event()
    started: list[int] = []
    lock = threading.Lock()

    @tool("slow_tool")
    def slow_tool(n: int) -> str:
        """Block until the test releases it."""
        with lock:
            started.append(n)
        release.wait(ORPHAN_BLOCK)
        return f"SLOW_RESULT {n}"

    # One call more than the executor runs at once.
    calls = [_native_call(i, "slow_tool", {"n": i}) for i in range(9)]
    agent = _agent(_NativeLLM(_native_script(calls)), tools=[slow_tool])
    before = set(threading.enumerate())
    in_flight: list[int] = []
    try:
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "TASK_ONE"))
        assert not release.is_set(), "the calls in flight ended before the limit"
        with lock:
            in_flight.extend(started)
    finally:
        release.set()
        still_alive = _join_timed_runs(before)
    assert still_alive == []

    assert len(in_flight) == 8, f"calls in flight at the limit: {in_flight}"
    assert sorted(started) == sorted(in_flight), (
        f"calls started after the limit: {sorted(set(started) - set(in_flight))}"
    )
