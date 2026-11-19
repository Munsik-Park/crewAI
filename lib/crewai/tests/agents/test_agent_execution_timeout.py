"""Behavior of ``max_execution_time`` when an agent run exceeds it.

A timed-out call must raise ``TimeoutError`` at the limit, without waiting for
the run still in progress. The run left behind finishes only the step in flight
at the limit: it runs no further step, asks for no human input and saves
nothing to memory, and it ends without printing or logging an error. It must
not disturb a later call on the same agent, or on a coworker it was delegating
to (result, executor state, retry budget, tool bookkeeping, event scope).

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
import json
import logging
import threading
import time
from typing import Any

from pydantic import Field
import pytest

from crewai import Agent, Task
from crewai.core.providers.human_input import (
    SyncHumanInputProvider,
    reset_provider,
    set_provider,
)
from crewai.events.event_bus import crewai_event_bus
from crewai.events.event_context import (
    EventContextConfig,
    MismatchBehavior,
    _event_context_config,
)
from crewai.events.types.agent_events import AgentExecutionStartedEvent
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool
from crewai.tools.agent_tools.agent_tools import AgentTools


LIMIT = 1
IN_TIME = 2.0
ORPHAN_BLOCK = 5.0
JOIN_BOUND = 20.0
TIMED_RUN_PREFIX = "CrewAITimedRun"
REPEATED_USAGE = "I tried reusing the same input"
EXECUTOR_BUSY = "already running"
PAIRING_WARNING = "Event pairing mismatch"
UNKNOWN_ERROR_PRINT = "An unknown error occurred"
DELEGATE = "Delegate work to coworker"


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


class _RecordingFeedbackProvider(SyncHumanInputProvider):
    """Human-input provider that counts feedback requests and approves the answer."""

    def __init__(self) -> None:
        self.calls = 0

    def handle_feedback(self, formatted_answer: Any, context: Any) -> Any:
        self.calls += 1
        return formatted_answer

    async def handle_feedback_async(self, formatted_answer: Any, context: Any) -> Any:
        return self.handle_feedback(formatted_answer, context)


class _BlockingFeedbackProvider(_RecordingFeedbackProvider):
    """Recording provider whose feedback blocks until the test releases it."""

    def __init__(self, release: threading.Event) -> None:
        super().__init__()
        self._release = release

    def handle_feedback(self, formatted_answer: Any, context: Any) -> Any:
        answer = super().handle_feedback(formatted_answer, context)
        self._release.wait(ORPHAN_BLOCK)
        return answer


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


def _agent(llm: BaseLLM, tools: list[Any] | None = None, **kwargs: Any) -> Agent:
    settings: dict[str, Any] = {
        "role": "Timeout tester",
        "goal": "answer",
        "backstory": "b",
        "llm": llm,
        "tools": tools or [],
        "max_execution_time": LIMIT,
        "max_retry_limit": 0,
    }
    settings.update(kwargs)
    return Agent(**settings)


def _task(agent: Agent, marker: str, human_input: bool = False) -> Task:
    return Task(
        description=f"{marker}: answer",
        expected_output="an answer",
        agent=agent,
        human_input=human_input,
    )


def _delegate(subtask: str) -> str:
    return _action(
        DELEGATE,
        json.dumps({"task": subtask, "context": "c", "coworker": "Coworker"}),
    )


def _after_last_delegation(messages_text: str) -> str:
    """The part of an LLM's input after its last delegation action: the tool's result."""
    return messages_text.rpartition(DELEGATE)[2]


def _crewai_error_records(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        f"{r.name}: {r.getMessage()}"
        for r in caplog.records
        if r.name.startswith("crewai") and r.levelno >= logging.ERROR
    ]


def test_sync_timeout_raises_without_waiting_for_the_run() -> None:
    """TimeoutError arrives in time while the run is still blocked."""
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
    """A later call on the same agent starts at once and returns its own answer."""
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
    """The tool call in flight at the timeout does not mark the later identical call as repeated."""
    release = threading.Event()
    lookups: list[str] = []
    orphan_ended: list[bool] = []
    later_messages: list[str] = []
    args = '{"query": "same"}'

    @tool("lookup")
    def lookup(query: str) -> str:
        """Look a query up."""
        lookups.append(query)
        if len(lookups) == 1:
            release.wait(ORPHAN_BLOCK)
        return f"LOOKUP_RESULT for {query}"

    before = set(threading.enumerate())

    def script(text: str) -> str:
        if "TASK_ONE" in text:
            if "LOOKUP_RESULT" in text:
                return _final("orphan answer")
            return _action("lookup", args)
        if "TASK_TWO" in text:
            if "LOOKUP_RESULT" not in text and REPEATED_USAGE not in text:
                release.set()
                orphan_ended.append(_join_timed_runs(before) == [])
                return _action("lookup", args)
            later_messages.append(text)
            return _final("second answer")
        return _final("unexpected")

    agent = _agent(_ScriptedLLM(script), tools=[lookup])

    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "TASK_ONE"))
        elapsed = time.monotonic() - start
        assert elapsed < IN_TIME, (
            f"TimeoutError took {elapsed:.2f}s; the limit is {LIMIT}s"
        )
        assert lookups == ["same"], "the timed-out run's tool was not in flight"

        agent.max_execution_time = None
        result = agent.execute_task(_task(agent, "TASK_TWO"))

        assert orphan_ended == [True], "the timed-out run did not end"
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
    """Own result, own state and the caller's event scope after an async timeout."""
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
    """The discarded result of a timed-out run is not saved to memory.

    The run's answer is ready before the limit, and the human feedback on it
    is still in progress when the limit passes.
    """
    release = threading.Event()
    provider = _BlockingFeedbackProvider(release)

    def script(text: str) -> str:
        if "TASK_ONE" in text:
            return _final("orphan answer")
        return _final("second answer")

    agent = _agent(_ScriptedLLM(script))
    memory = _RecordingMemory()
    object.__setattr__(agent, "memory", memory)

    provider_token = set_provider(provider)
    before = set(threading.enumerate())
    try:
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "TASK_ONE", human_input=True))
    finally:
        release.set()
        still_alive = _join_timed_runs(before)
        reset_provider(provider_token)
    assert still_alive == []
    assert provider.calls == 1, "the timed-out run never reached human feedback"
    assert not [s for s in memory.saved if "TASK_ONE" in s], (
        "the timed-out run saved its result to memory"
    )

    agent.max_execution_time = None
    assert agent.execute_task(_task(agent, "TASK_TWO")) == "second answer"
    assert len([s for s in memory.saved if "TASK_TWO" in s]) == 1


def test_timed_out_run_does_not_ask_for_human_input() -> None:
    """A run whose last LLM step is in flight at the limit never asks for human feedback."""
    release = threading.Event()
    provider = _RecordingFeedbackProvider()

    def script(text: str) -> str:
        release.wait(ORPHAN_BLOCK)
        return _final("late answer")

    agent = _agent(_ScriptedLLM(script))

    provider_token = set_provider(provider)
    before = set(threading.enumerate())
    try:
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "TASK_ONE", human_input=True))
        assert not release.is_set()
    finally:
        release.set()
        still_alive = _join_timed_runs(before)
        reset_provider(provider_token)
    assert still_alive == []
    assert provider.calls == 0, "the timed-out run asked for human feedback"


@pytest.mark.parametrize(
    "coworker_limit",
    [None, 30],
    ids=["coworker-without-limit", "coworker-with-own-limit"],
)
def test_later_delegation_to_coworker_left_running_by_timeout(
    coworker_limit: int | None, capsys: pytest.CaptureFixture[str]
) -> None:
    """A coworker still busy in a timed-out delegation serves a later delegation at once.

    The timed-out run makes no further LLM call once the coworker's step
    ends, its stop does not use up the coworker's retry budget, and nothing
    reports an unclosed event scope.
    """
    release = threading.Event()
    coworker_calls: dict[str, int] = {}
    agent_calls: dict[str, int] = {}
    observed: dict[str, str] = {}

    def coworker_script(text: str) -> str:
        for marker in ("SUB_ONE", "SUB_TWO", "SUB_THREE"):
            if marker in text:
                break
        else:
            return _final("unexpected")
        coworker_calls[marker] = coworker_calls.get(marker, 0) + 1
        if marker == "SUB_ONE":
            release.wait(ORPHAN_BLOCK)
            return _final("COWORKER_FIRST")
        if marker == "SUB_THREE" and coworker_calls[marker] == 1:
            raise RuntimeError("transient coworker failure")
        return _final(f"COWORKER_ANSWER_{marker}")

    coworker = _agent(
        _ScriptedLLM(coworker_script),
        role="Coworker",
        max_execution_time=coworker_limit,
        max_retry_limit=1,
    )
    delegation = AgentTools(agents=[coworker]).tools()

    def script(text: str) -> str:
        marker, subtask = (
            ("TASK_ONE", "SUB_ONE") if "TASK_ONE" in text else ("TASK_TWO", "SUB_TWO")
        )
        agent_calls[marker] = agent_calls.get(marker, 0) + 1
        if agent_calls[marker] == 1:
            return _delegate(subtask)
        observed[marker] = text
        return _final(f"done {marker}")

    agent = _agent(_ScriptedLLM(script), tools=delegation)

    before = set(threading.enumerate())
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "TASK_ONE"), tools=delegation)
        elapsed = time.monotonic() - start
        assert elapsed < IN_TIME, (
            f"TimeoutError took {elapsed:.2f}s; the limit is {LIMIT}s"
        )

        agent.max_execution_time = None
        result = agent.execute_task(_task(agent, "TASK_TWO"), tools=delegation)
        assert not release.is_set(), "the coworker's first step ended too early"

        assert result == "done TASK_TWO"
        delegation_result = _after_last_delegation(observed.get("TASK_TWO", ""))
        assert EXECUTOR_BUSY not in delegation_result, delegation_result
        assert "COWORKER_ANSWER_SUB_TWO" in delegation_result, delegation_result
    finally:
        release.set()
        still_alive = _join_timed_runs(before)
    assert still_alive == []
    assert agent_calls.get("TASK_ONE") == 1, (
        "the timed-out run made another LLM call after the coworker's step"
    )

    third = coworker.execute_task(_task(coworker, "SUB_THREE"))
    assert third == "COWORKER_ANSWER_SUB_THREE"

    out = capsys.readouterr().out
    assert PAIRING_WARNING not in out


@pytest.mark.asyncio
async def test_later_async_delegation_to_coworker_left_running_by_timeout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """After an async timeout, a coworker still busy in the cancelled run's delegation step serves a later delegation at once."""
    loop = asyncio.get_running_loop()
    pool = _TrackingExecutor()
    loop.set_default_executor(pool)

    release = threading.Event()
    in_flight_at_timeout: list[set[concurrent.futures.Future[Any]]] = []
    agent_calls: dict[str, int] = {}
    observed: dict[str, str] = {}

    def coworker_script(text: str) -> str:
        if "SUB_ONE" in text:
            release.wait(ORPHAN_BLOCK)
            return _final("COWORKER_FIRST")
        if "SUB_TWO" in text:
            return _final("COWORKER_ANSWER_SUB_TWO")
        return _final("unexpected")

    coworker = _agent(
        _ScriptedLLM(coworker_script),
        role="Coworker",
        max_execution_time=None,
        max_retry_limit=1,
    )
    delegation = AgentTools(agents=[coworker]).tools()

    def script(text: str) -> str:
        marker, subtask = (
            ("TASK_ONE", "SUB_ONE") if "TASK_ONE" in text else ("TASK_TWO", "SUB_TWO")
        )
        agent_calls[marker] = agent_calls.get(marker, 0) + 1
        if agent_calls[marker] == 1:
            return _delegate(subtask)
        observed[marker] = text
        return _final(f"done {marker}")

    agent = _agent(_ScriptedLLM(script), tools=delegation)

    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            await agent.aexecute_task(_task(agent, "TASK_ONE"), tools=delegation)
        elapsed = time.monotonic() - start
        in_flight_at_timeout.append(pool.snapshot())
        assert elapsed < IN_TIME, (
            f"TimeoutError took {elapsed:.2f}s; the limit is {LIMIT}s"
        )

        agent.max_execution_time = None
        result = await agent.aexecute_task(_task(agent, "TASK_TWO"), tools=delegation)
        assert not release.is_set(), "the coworker's first step ended too early"

        assert result == "done TASK_TWO"
        delegation_result = _after_last_delegation(observed.get("TASK_TWO", ""))
        assert EXECUTOR_BUSY not in delegation_result, delegation_result
        assert "COWORKER_ANSWER_SUB_TWO" in delegation_result, delegation_result
    finally:
        release.set()
        if in_flight_at_timeout:
            concurrent.futures.wait(in_flight_at_timeout[0], timeout=JOIN_BOUND)
        pool.shutdown(wait=True)

    out = capsys.readouterr().out
    assert PAIRING_WARNING not in out


def test_timed_out_runs_result_as_answer_tool_does_not_become_later_answer() -> None:
    """A result_as_answer tool in flight at the timeout does not replace a later call's answer."""
    release = threading.Event()
    lookups: list[str] = []
    orphan_ended: list[bool] = []

    @tool("lookup", result_as_answer=True)
    def lookup() -> str:
        """Look the answer up."""
        lookups.append("called")
        release.wait(ORPHAN_BLOCK)
        return "ORPHAN_RESULT"

    before = set(threading.enumerate())

    def script(text: str) -> str:
        if "TASK_TWO" in text:
            release.set()
            orphan_ended.append(_join_timed_runs(before) == [])
            return _final("SECOND_RESULT")
        return _action("lookup")

    agent = _agent(_ScriptedLLM(script), tools=[lookup])

    try:
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "TASK_ONE"))
        assert lookups == ["called"], "the timed-out run's tool was not in flight"

        agent.max_execution_time = None
        result = agent.execute_task(_task(agent, "TASK_TWO"))
    finally:
        release.set()
        still_alive = _join_timed_runs(before)
    assert still_alive == []
    assert orphan_ended == [True], "the timed-out run did not end"
    assert result == "SECOND_RESULT"


def test_earlier_result_as_answer_does_not_become_later_answer() -> None:
    """Without a limit, a task's result_as_answer output is not the answer of the next task on the agent."""

    @tool("lookup", result_as_answer=True)
    def lookup() -> str:
        """Look the answer up."""
        return "FIRST_TOOL_RESULT"

    def script(text: str) -> str:
        if "TASK_TWO" in text:
            return _final("SECOND_RESULT")
        return _action("lookup")

    agent = _agent(_ScriptedLLM(script), tools=[lookup], max_execution_time=None)

    assert agent.execute_task(_task(agent, "TASK_ONE")) == "FIRST_TOOL_RESULT"
    assert agent.execute_task(_task(agent, "TASK_TWO")) == "SECOND_RESULT"


def test_other_agents_result_as_answer_in_a_tool_does_not_become_the_answer() -> None:
    """A tool that runs another agent's kickoff does not take that agent's result_as_answer output as the answer."""

    @tool("final_lookup", result_as_answer=True)
    def final_lookup() -> str:
        """Look the other agent's answer up."""
        return "OTHER_TOOL_RESULT"

    def other_script(text: str) -> str:
        if "OTHER_TOOL_RESULT" in text:
            return _final("other done")
        return _action("final_lookup")

    other = _agent(
        _ScriptedLLM(other_script),
        tools=[final_lookup],
        role="Other",
        max_execution_time=None,
    )

    @tool("ask_other")
    def ask_other() -> str:
        """Ask the other agent."""
        other.kickoff("OTHER_QUESTION")
        return "ASKED_OTHER"

    def script(text: str) -> str:
        if "ASKED_OTHER" in text:
            return _final("OWN_ANSWER")
        return _action("ask_other")

    agent = _agent(_ScriptedLLM(script), tools=[ask_other], max_execution_time=None)

    assert agent.execute_task(_task(agent, "TASK_ONE")) == "OWN_ANSWER"


def test_timed_out_run_stops_before_its_next_step_quietly(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """The step after the one in flight at the limit never runs, and the run ends without an error print or log."""
    release = threading.Event()
    tool_runs: list[str] = []

    @tool("side_effect")
    def side_effect() -> str:
        """Do something visible."""
        tool_runs.append("ran")
        return "SIDE_EFFECT_DONE"

    def script(text: str) -> str:
        if "SIDE_EFFECT_DONE" in text:
            return _final("orphan answer")
        release.wait(ORPHAN_BLOCK)
        return _action("side_effect")

    agent = _agent(_ScriptedLLM(script), tools=[side_effect], verbose=True)

    caplog.set_level(logging.WARNING)
    before = set(threading.enumerate())
    try:
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "TASK_ONE"))
        assert not release.is_set()
    finally:
        release.set()
        still_alive = _join_timed_runs(before)
    assert still_alive == []
    assert tool_runs == [], "the timed-out run ran its next step"
    assert _crewai_error_records(caplog) == []
    assert UNKNOWN_ERROR_PRINT not in capsys.readouterr().out


class _CoworkerState:
    """A coworker that has already run a task, and what that run left on it.

    The earlier task used ``lookup`` twice and a failing tool once, so the
    coworker holds tool failures, a last used tool and a usage count.
    """

    def __init__(self) -> None:
        self.llm_calls: list[str] = []

        @tool("lookup")
        def lookup(query: str) -> str:
            """Look a query up."""
            return f"LOOKUP_RESULT for {query}"

        @tool("broken")
        def broken() -> str:
            """Always fails."""
            raise RuntimeError("broken tool")

        self.lookup = lookup

        def script(text: str) -> str:
            if "SUB_ONE" in text:
                self.llm_calls.append("SUB_ONE")
                return _final("COWORKER_ORPHAN_ANSWER")
            if "LOOKUP_RESULT for b" not in text:
                if "LOOKUP_RESULT for a" not in text:
                    return _action("lookup", '{"query": "a"}')
                return _action("lookup", '{"query": "b"}')
            if "broken tool" not in text:
                return _action("broken")
            return _final("COWORKER_PRE_ANSWER")

        self.agent = _agent(
            _ScriptedLLM(script),
            tools=[lookup, broken],
            role="Coworker",
            max_execution_time=None,
        )
        self.pre_task = _task(self.agent, "PRE")

    def record(self, pre_answer: Any) -> None:
        """Keep what the earlier task left on the coworker."""
        assert pre_answer == "COWORKER_PRE_ANSWER"
        self.failures = self.agent.last_tool_failures
        self.handler = self.agent.tools_handler
        self.last_used_tool = self.handler.last_used_tool if self.handler else None
        self.usage_count = self.lookup.current_usage_count
        assert self.failures, "the coworker's earlier task recorded no tool failure"
        assert self.last_used_tool is not None
        assert self.usage_count > 0

    def assert_unchanged(self) -> None:
        handler = self.agent.tools_handler
        reset = [
            name
            for name, unchanged in (
                ("tool failures", self.agent.last_tool_failures == self.failures),
                (
                    "last used tool",
                    handler is self.handler
                    and handler is not None
                    and handler.last_used_tool is self.last_used_tool,
                ),
                (
                    "tool usage count",
                    self.lookup.current_usage_count == self.usage_count,
                ),
            )
            if not unchanged
        ]
        assert reset == [], f"the timed-out run reset the coworker's {reset}"
        assert self.llm_calls == []


def _calling_coworker_tool(
    coworker: Agent, release: threading.Event, outcome: list[str], entry: str
) -> Any:
    """A tool that waits for the test, then runs a task on ``coworker`` itself."""

    @tool("ask_coworker")
    def ask_coworker() -> str:
        """Ask the coworker."""
        release.wait(ORPHAN_BLOCK)
        sub_task = _task(coworker, "SUB_ONE")
        try:
            if entry == "execute_task":
                coworker.execute_task(sub_task)
            else:
                asyncio.run(coworker.aexecute_task(sub_task))
            outcome.append("ran")
        except BaseException as error:
            outcome.append(type(error).__name__)
        return "ASKED_COWORKER"

    return ask_coworker


def _asking_script(text: str) -> str:
    if "ASKED_COWORKER" in text:
        return _final("orphan answer")
    return _action("ask_coworker")


@pytest.mark.parametrize("entry", ["execute_task", "aexecute_task"])
def test_timed_out_run_leaves_coworker_it_calls_after_the_limit_as_it_was(
    entry: str,
) -> None:
    """A coworker task started from the step in flight after the limit leaves the coworker's tool failures, last used tool and tool usage counts as they were."""
    coworker = _CoworkerState()
    coworker.record(coworker.agent.execute_task(coworker.pre_task))
    release = threading.Event()
    outcome: list[str] = []
    ask = _calling_coworker_tool(coworker.agent, release, outcome, entry)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["execute_task", "aexecute_task"])
async def test_cancelled_run_leaves_coworker_it_calls_after_the_limit_as_it_was(
    entry: str,
) -> None:
    """After an async timeout, a coworker task started from the step in flight leaves the coworker's tool failures, last used tool and tool usage counts as they were."""
    loop = asyncio.get_running_loop()
    pool = _TrackingExecutor()
    loop.set_default_executor(pool)

    coworker = _CoworkerState()
    coworker.record(await coworker.agent.aexecute_task(coworker.pre_task))
    release = threading.Event()
    outcome: list[str] = []
    in_flight_at_timeout: list[set[concurrent.futures.Future[Any]]] = []
    ask = _calling_coworker_tool(coworker.agent, release, outcome, entry)
    agent = _agent(_ScriptedLLM(_asking_script), tools=[ask])

    try:
        with pytest.raises(TimeoutError):
            await agent.aexecute_task(_task(agent, "TASK_ONE"))
        in_flight_at_timeout.append(pool.snapshot())
        assert not release.is_set()
    finally:
        release.set()
        if in_flight_at_timeout:
            concurrent.futures.wait(in_flight_at_timeout[0], timeout=JOIN_BOUND)
        pool.shutdown(wait=True)
    assert len(outcome) == 1, "the timed-out run's tool never called the coworker"
    coworker.assert_unchanged()


class _RecordingKnowledgeStorage(BaseKnowledgeStorage):
    """Knowledge storage stub recording each search."""

    searches: list[Any] = Field(default_factory=list)

    def search(self, query: Any, *args: Any, **kwargs: Any) -> list[Any]:
        self.searches.append(query)
        return []

    async def asearch(self, query: Any, *args: Any, **kwargs: Any) -> list[Any]:
        return self.search(query)

    def save(self, documents: list[str]) -> None:
        return None

    async def asave(self, documents: list[str]) -> None:
        return None

    def reset(self) -> None:
        return None

    async def areset(self) -> None:
        return None


class _BlockingRecallMemory(_RecordingMemory):
    """Recording memory whose first recall blocks until the test releases it."""

    def __init__(self, release: threading.Event) -> None:
        super().__init__()
        self._release = release
        self.recalls = 0

    def recall(self, *args: Any, **kwargs: Any) -> list[Any]:
        self.recalls += 1
        if self.recalls == 1:
            self._release.wait(ORPHAN_BLOCK)
        return []


@pytest.mark.parametrize("entry", ["execute_task", "aexecute_task"])
def test_coworker_task_started_before_the_limit_queries_no_knowledge_after_it(
    entry: str,
) -> None:
    """A coworker task whose memory recall ends after the limit makes no knowledge query and no LLM call."""
    release = threading.Event()
    coworker_calls: list[str] = []
    storage = _RecordingKnowledgeStorage()
    memory = _BlockingRecallMemory(release)

    def coworker_script(text: str) -> str:
        coworker_calls.append(text)
        return _final("COWORKER_ORPHAN_ANSWER")

    coworker = _agent(
        _ScriptedLLM(coworker_script),
        role="Coworker",
        max_execution_time=None,
        knowledge=Knowledge(collection_name="coworker", sources=[], storage=storage),
    )
    object.__setattr__(coworker, "memory", memory)
    outcome: list[str] = []

    @tool("ask_coworker")
    def ask_coworker() -> str:
        """Ask the coworker."""
        sub_task = _task(coworker, "SUB_ONE")
        try:
            if entry == "execute_task":
                coworker.execute_task(sub_task)
            else:
                asyncio.run(coworker.aexecute_task(sub_task))
            outcome.append("ran")
        except BaseException as error:
            outcome.append(type(error).__name__)
        return "ASKED_COWORKER"

    agent = _agent(_ScriptedLLM(_asking_script), tools=[ask_coworker])

    before = set(threading.enumerate())
    try:
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "TASK_ONE"))
        assert not release.is_set(), "the coworker's recall ended before the limit"
    finally:
        release.set()
        still_alive = _join_timed_runs(before)
    assert still_alive == []
    assert len(outcome) == 1, "the timed-out run's tool never called the coworker"
    assert memory.recalls == 1, "the coworker's recall was not in flight"
    ran = [
        name
        for name, happened in (
            ("knowledge search", storage.searches),
            ("LLM call", coworker_calls),
            ("knowledge query written", coworker.knowledge_search_query),
        )
        if happened
    ]
    assert ran == [], f"after the limit the coworker made: {ran}"
