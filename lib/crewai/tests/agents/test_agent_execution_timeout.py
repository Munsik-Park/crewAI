"""Behavior of ``max_execution_time`` when an agent run exceeds it.

A timed-out call must raise ``TimeoutError`` at the limit, without waiting for
the run still in progress, and the run left behind must not disturb a later
call on the same agent (result, executor state, tool bookkeeping, event scope)
or save its discarded result to memory.

Every test is offline (scripted LLM, local tools). Runs left behind block on
events the test releases, and each test waits for their end before returning
so nothing leaks into later tests on the same xdist worker:

* sync path: the run's worker thread, named with the ``CrewAITimedRun``
  prefix, is joined with a bound;
* async path: the test installs its own default executor on the loop and
  waits for the work items in flight.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import threading
import time
from typing import Any

import pytest

from crewai import Agent, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.event_context import (
    EventContextConfig,
    MismatchBehavior,
    _event_context_config,
)
from crewai.events.types.agent_events import AgentExecutionStartedEvent
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool


LIMIT = 1
IN_TIME = 2.0
ORPHAN_BLOCK = 5.0
JOIN_BOUND = 20.0
TIMED_RUN_PREFIX = "CrewAITimedRun"
REPEATED_USAGE = "I tried reusing the same input"


def _action(name: str, args: str = "{}") -> str:
    return f"Thought: I will use a tool\nAction: {name}\nAction Input: {args}"


def _final(answer: str) -> str:
    return f"Thought: I know the answer\nFinal Answer: {answer}"


class _ScriptedLLM(BaseLLM):
    """ReAct-only LLM whose reply is decided by ``script(messages_text)``.

    It does not declare function calling, so tools go through ``ToolUsage``.
    """

    def __init__(self, script: Any) -> None:
        super().__init__(model="scripted-timeout-test")
        self._script = script

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
        return self._script(str(messages))

    async def acall(self, messages: Any, *args: Any, **kwargs: Any) -> str:
        return self._script(str(messages))


class _RecordingMemory:
    """Writable memory stub recording what the executor tries to save."""

    read_only = False
    root_scope = None

    def __init__(self) -> None:
        self.saved: list[str] = []

    def recall(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def extract_memories(self, content: str) -> list[str]:
        self.saved.append(content)
        return []

    def remember_many(self, *args: Any, **kwargs: Any) -> None:
        return None


class _TrackingExecutor(concurrent.futures.ThreadPoolExecutor):
    """Default loop executor that records the work items still in flight."""

    def __init__(self) -> None:
        super().__init__(max_workers=8, thread_name_prefix="timeout-test-loop")
        self._lock = threading.Lock()
        self.in_flight: set[concurrent.futures.Future[Any]] = set()

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        future = super().submit(fn, *args, **kwargs)
        with self._lock:
            self.in_flight.add(future)
        future.add_done_callback(self._done)
        return future

    def _done(self, future: concurrent.futures.Future[Any]) -> None:
        with self._lock:
            self.in_flight.discard(future)

    def snapshot(self) -> set[concurrent.futures.Future[Any]]:
        with self._lock:
            return set(self.in_flight)


def _join_timed_runs(before: set[threading.Thread]) -> list[threading.Thread]:
    """Join the timed-run threads started since ``before``; return those still alive."""
    started = [
        t
        for t in threading.enumerate()
        if t not in before and t.name.startswith(TIMED_RUN_PREFIX)
    ]
    deadline = time.monotonic() + JOIN_BOUND
    for thread in started:
        thread.join(max(0.0, deadline - time.monotonic()))
    return [t for t in started if t.is_alive()]


def _raise_on_scope_mismatch() -> contextvars.Token[EventContextConfig | None]:
    return _event_context_config.set(
        EventContextConfig(
            mismatch_behavior=MismatchBehavior.RAISE,
            empty_pop_behavior=MismatchBehavior.RAISE,
        )
    )


def _agent(llm: BaseLLM, tools: list[Any] | None = None) -> Agent:
    return Agent(
        role="Timeout tester",
        goal="answer",
        backstory="b",
        llm=llm,
        tools=tools or [],
        max_execution_time=LIMIT,
        max_retry_limit=0,
    )


def _task(agent: Agent, marker: str) -> Task:
    return Task(description=f"{marker}: answer", expected_output="an answer", agent=agent)


def test_sync_timeout_raises_without_waiting_for_the_run() -> None:
    """V1 (C1, C3): TimeoutError arrives in time while the run is still blocked."""
    release = threading.Event()

    def script(text: str) -> str:
        release.wait(ORPHAN_BLOCK)
        return _final("late answer")

    agent = _agent(_ScriptedLLM(script))
    task = _task(agent, "TASK_ONE")

    before = set(threading.enumerate())
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            task.execute_sync(agent=agent)
        elapsed = time.monotonic() - start

        assert elapsed < IN_TIME, (
            f"TimeoutError took {elapsed:.2f}s; the limit is {LIMIT}s"
        )
        assert not release.is_set()
    finally:
        release.set()
        still_alive = _join_timed_runs(before)
    assert still_alive == []


def test_later_sync_call_runs_while_timed_out_run_is_blocked() -> None:
    """V2 (C2): a later call on the same agent starts at once and returns its own answer."""
    release = threading.Event()

    def script(text: str) -> str:
        if "TASK_ONE" in text:
            release.wait(ORPHAN_BLOCK)
            return _final("orphan answer")
        return _final("second answer")

    token = _raise_on_scope_mismatch()
    agent = _agent(_ScriptedLLM(script))

    before = set(threading.enumerate())
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "TASK_ONE"))
        elapsed = time.monotonic() - start
        assert elapsed < IN_TIME, (
            f"TimeoutError took {elapsed:.2f}s; the limit is {LIMIT}s"
        )

        later_start = time.monotonic()
        result = agent.execute_task(_task(agent, "TASK_TWO"))
        later_elapsed = time.monotonic() - later_start

        assert result == "second answer"
        assert later_elapsed < 1.0, f"later call took {later_elapsed:.2f}s"
        assert not release.is_set()
    finally:
        release.set()
        still_alive = _join_timed_runs(before)
        _event_context_config.reset(token)
    assert still_alive == []


def test_later_sync_call_is_not_blocked_by_timed_out_runs_tool_use() -> None:
    """V2b (C2): the timed-out run's tool call does not mark the later identical call as repeated."""
    release = threading.Event()
    orphan_used_tool = threading.Event()
    later_messages: list[str] = []
    args = '{"query": "same"}'

    @tool("lookup")
    def lookup(query: str) -> str:
        """Look a query up."""
        return f"LOOKUP_RESULT for {query}"

    def script(text: str) -> str:
        if "TASK_ONE" in text:
            if "LOOKUP_RESULT" in text:
                orphan_used_tool.set()
                return _final("orphan answer")
            release.wait(ORPHAN_BLOCK)
            return _action("lookup", args)
        if "TASK_TWO" in text:
            if "LOOKUP_RESULT" not in text and REPEATED_USAGE not in text:
                release.set()
                orphan_used_tool.wait(JOIN_BOUND)
                return _action("lookup", args)
            later_messages.append(text)
            return _final("second answer")
        return _final("unexpected")

    agent = _agent(_ScriptedLLM(script), tools=[lookup])

    before = set(threading.enumerate())
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "TASK_ONE"))
        elapsed = time.monotonic() - start
        assert elapsed < IN_TIME, (
            f"TimeoutError took {elapsed:.2f}s; the limit is {LIMIT}s"
        )

        result = agent.execute_task(_task(agent, "TASK_TWO"))

        assert orphan_used_tool.is_set()
        assert result == "second answer"
        assert later_messages, "the later call never reached its second LLM step"
        assert "LOOKUP_RESULT for same" in later_messages[-1]
        assert REPEATED_USAGE not in later_messages[-1]
    finally:
        release.set()
        still_alive = _join_timed_runs(before)
    assert still_alive == []


@pytest.mark.asyncio
async def test_later_async_call_is_isolated_from_cancelled_run() -> None:
    """V3 (C1 async, C2): own result, own state and the caller's event scope after an async timeout."""
    loop = asyncio.get_running_loop()
    pool = _TrackingExecutor()
    loop.set_default_executor(pool)
    token = _raise_on_scope_mismatch()

    release = threading.Event()
    in_flight_at_timeout: list[set[concurrent.futures.Future[Any]]] = []
    later_messages: list[str] = []

    @tool("slow_tool")
    def slow_tool() -> str:
        """Block until the test releases it."""
        release.wait(ORPHAN_BLOCK)
        return "ORPHAN_TOOL_OUTPUT"

    @tool("echo_tool")
    def echo_tool() -> str:
        """Echo."""
        return "ECHO_OUTPUT"

    def script(text: str) -> str:
        if "TASK_ONE" in text:
            if "ORPHAN_TOOL_OUTPUT" in text:
                return _final("orphan answer")
            return _action("slow_tool")
        if "TASK_TWO" in text:
            if "ECHO_OUTPUT" not in text:
                release.set()
                if in_flight_at_timeout:
                    concurrent.futures.wait(
                        in_flight_at_timeout[0], timeout=JOIN_BOUND
                    )
                return _action("echo_tool")
            later_messages.append(text)
            return _final("second answer")
        return _final("unexpected")

    agent = _agent(_ScriptedLLM(script), tools=[slow_tool, echo_tool])
    started: list[AgentExecutionStartedEvent] = []

    try:
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(AgentExecutionStartedEvent)
            def _record(source: Any, event: AgentExecutionStartedEvent) -> None:
                started.append(event)

            start = time.monotonic()
            with pytest.raises(TimeoutError):
                await agent.aexecute_task(_task(agent, "TASK_ONE"))
            elapsed = time.monotonic() - start
            in_flight_at_timeout.append(pool.snapshot())
            assert elapsed < IN_TIME, (
                f"TimeoutError took {elapsed:.2f}s; the limit is {LIMIT}s"
            )

            agent.max_execution_time = 30
            result = await agent.aexecute_task(_task(agent, "TASK_TWO"))
            crewai_event_bus.flush(timeout=10.0)

        assert result == "second answer"
        assert later_messages, "the later call never reached its second LLM step"
        assert "ORPHAN_TOOL_OUTPUT" not in later_messages[-1]
        first, later = (
            next(e for e in started if marker in e.task_prompt)
            for marker in ("TASK_ONE", "TASK_TWO")
        )
        assert later.parent_event_id == first.parent_event_id
    finally:
        release.set()
        if in_flight_at_timeout:
            concurrent.futures.wait(in_flight_at_timeout[0], timeout=JOIN_BOUND)
        _event_context_config.reset(token)
        pool.shutdown(wait=True)


def test_timed_out_run_does_not_save_to_memory() -> None:
    """V4 (C2): the discarded result of a timed-out run is not saved to memory."""
    release = threading.Event()
    orphan_answered = threading.Event()

    def script(text: str) -> str:
        if "TASK_ONE" in text:
            release.wait(ORPHAN_BLOCK)
            orphan_answered.set()
            return _final("orphan answer")
        return _final("second answer")

    agent = _agent(_ScriptedLLM(script))
    memory = _RecordingMemory()
    object.__setattr__(agent, "memory", memory)

    before = set(threading.enumerate())
    try:
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "TASK_ONE"))
    finally:
        release.set()
        still_alive = _join_timed_runs(before)
    assert still_alive == []
    assert orphan_answered.is_set(), "the timed-out run never reached its answer"
    assert not [s for s in memory.saved if "TASK_ONE" in s], (
        "the timed-out run saved its result to memory"
    )

    agent.max_execution_time = None
    assert agent.execute_task(_task(agent, "TASK_TWO")) == "second answer"
    assert len([s for s in memory.saved if "TASK_TWO" in s]) == 1
