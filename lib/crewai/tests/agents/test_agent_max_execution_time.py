"""Regression tests for ``Agent.max_execution_time`` (issue #6).

The limit must raise ``TimeoutError`` at the deadline without waiting for the
work still running, a later call must not be affected by the timed-out run,
and the timed-out run must start no new tool call or model call through
crewAI once the error is raised.

Every test uses a scripted ``BaseLLM`` subclass and local tools (no network,
no cassette) and waits, with a bound, for the work it abandoned to end before
it returns.
"""

from __future__ import annotations

import asyncio
import builtins
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from crewai import Agent, Task
from crewai.hooks import (
    register_before_tool_call_hook,
    unregister_before_tool_call_hook,
)
from crewai.llms.base_llm import BaseLLM, LLMCallBlockedError
from crewai.tools import tool
from crewai.utilities import converter as converter_module
from crewai.utilities.converter import Converter, ConverterError
from pydantic import BaseModel

LIMIT = 1
# The raise must come well before the abandoned work could have ended.
RAISE_BOUND = 3.0
# Bound on waiting for abandoned work to end before a test returns.
DRAIN_BOUND = 15.0


class _Recorder:
    """Thread-safe log of timestamped events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[tuple[float, str]] = []

    def add(self, label: str) -> float:
        ts = time.monotonic()
        with self._lock:
            self.events.append((ts, label))
        return ts

    def first(self, label: str) -> float:
        with self._lock:
            return next(ts for ts, lb in self.events if lb == label)

    def after(self, ts: float) -> list[str]:
        with self._lock:
            return [lb for t, lb in self.events if t > ts]


def _text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    return "\n".join(str(m.get("content", "")) for m in messages)


def _act(tool_name: str, arg: str) -> str:
    return f"Thought: I will use the tool\nAction: {tool_name}\nAction Input: {arg}"


def _final(text: str) -> str:
    return f"Thought: I now know the final answer\nFinal Answer: {text}"


class ScriptedLLM(BaseLLM):
    """Returns scripted replies, each after a scripted delay.

    Each call start is recorded inside the call body, below the wrapper that
    ``BaseLLM.__init_subclass__`` installs, so a refused call records nothing.
    """

    def __init__(
        self,
        script: list[tuple[float, str]],
        name: str,
        recorder: _Recorder,
    ) -> None:
        super().__init__(model="test-model")
        self.script = list(script)
        self.name = name
        self.recorder = recorder
        self.inputs: list[str] = []

    def _next(self, messages: Any) -> tuple[float, str]:
        self.recorder.add(f"llm:{self.name}")
        self.inputs.append(_text(messages))
        if self.script:
            return self.script.pop(0)
        return 0.0, _final("DONE")

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
        **kwargs: Any,
    ) -> str:
        delay, reply = self._next(messages)
        time.sleep(delay)
        return reply

    async def acall(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
        **kwargs: Any,
    ) -> str:
        delay, reply = self._next(messages)
        await asyncio.sleep(delay)
        return reply

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192


class FunctionCallingLLM(ScriptedLLM):
    """Makes the converter take its instructor path."""

    def supports_function_calling(self) -> bool:
        return True


class _RecordingMemory:
    """Stands in for unified memory and records saves."""

    read_only = False
    root_scope = None

    def __init__(self, recorder: _Recorder) -> None:
        self.recorder = recorder

    def drain_writes(self) -> None:
        pass

    def recall(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def extract_memories(self, *_args: Any, **_kwargs: Any) -> list[str]:
        self.recorder.add("memory:save")
        return []

    def remember_many(self, *_args: Any, **_kwargs: Any) -> None:
        self.recorder.add("memory:save")


def _agent(llm: BaseLLM, tools: list[Any], **kwargs: Any) -> Agent:
    kwargs.setdefault("max_execution_time", LIMIT)
    return Agent(
        role="Researcher",
        goal="Answer the task",
        backstory="You answer tasks.",
        llm=llm,
        tools=tools,
        allow_delegation=False,
        max_retry_limit=0,
        **kwargs,
    )


def _task(agent: Agent, description: str = "Do the work", **kwargs: Any) -> Task:
    return Task(
        description=description,
        expected_output="An answer",
        agent=agent,
        **kwargs,
    )


@pytest.fixture
def recorder() -> _Recorder:
    return _Recorder()


@pytest.fixture
def prompts(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder) -> list[str]:
    """Counts every console prompt (hook approvals and ``human_input``)."""
    seen: list[str] = []

    def fake_input(*_args: Any) -> str:
        seen.append(threading.current_thread().name)
        recorder.add("prompt")
        return ""

    monkeypatch.setattr(builtins, "input", fake_input)
    return seen


@pytest.fixture
def approval_gate(prompts: list[str]) -> Iterator[None]:
    """A global before-tool hook that asks for approval, removed on teardown."""

    def gate(context: Any) -> bool | None:
        context.request_human_input(prompt=f"Allow {context.tool_name}?")
        return None

    register_before_tool_call_hook(gate)
    try:
        yield
    finally:
        unregister_before_tool_call_hook(gate)


def _side_effect_tool(recorder: _Recorder) -> Any:
    @tool("delete_file")
    def delete_file(path: str) -> str:
        """Delete a file (a side effect)."""
        recorder.add("side-effect")
        return "deleted"

    return delete_file


def _raise_time(call: Any) -> tuple[float, float]:
    """Runs ``call`` and returns (seconds until TimeoutError, monotonic raise time)."""
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        call()
    raised_at = time.monotonic()
    return raised_at - start, raised_at


# --- C1 ---------------------------------------------------------------------


def test_v1_sync_timeout_raises_at_the_deadline(recorder: _Recorder) -> None:
    """V1: ``execute_task`` raises at the limit, not when the running tool ends."""
    done = threading.Event()

    @tool("slow_tool")
    def slow_tool(q: str) -> str:
        """A tool that runs for about 6 seconds."""
        try:
            time.sleep(6)
            return "slow result"
        finally:
            done.set()

    llm = ScriptedLLM([(0, _act("slow_tool", '{"q": "x"}'))], "run1", recorder)
    agent = _agent(llm, [slow_tool])

    try:
        elapsed, _ = _raise_time(lambda: agent.execute_task(_task(agent)))
        assert elapsed < RAISE_BOUND, f"TimeoutError took {elapsed:.2f}s"
    finally:
        assert done.wait(DRAIN_BOUND)


# --- C2b --------------------------------------------------------------------


def test_v2_sync_later_call_returns_its_own_result_and_is_not_held(
    recorder: _Recorder,
) -> None:
    """V2: a later ``execute_task`` on the same agent, while run 1's tool runs."""
    run1_tool_done = threading.Event()
    uses: list[str] = []

    @tool("lookup")
    def lookup(x: str) -> str:
        """Look a value up. The first use is slow."""
        first = not uses
        uses.append(x)
        if first:
            try:
                time.sleep(4)
                recorder.add("run1-tool-end")
                return "RUN1-TOOL-RESULT"
            finally:
                run1_tool_done.set()
        recorder.add("run2-tool-ran")
        return "RUN2-TOOL-RESULT"

    llm1 = ScriptedLLM(
        [(0, 'Thought: RUN1-MARKER\nAction: lookup\nAction Input: {"x": "a"}')],
        "run1",
        recorder,
    )
    agent = _agent(llm1, [lookup])

    try:
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "RUN1-TASK"))

        llm2 = ScriptedLLM(
            [(3.5, _act("lookup", '{"x": "a"}')), (0, _final("SECOND"))],
            "run2",
            recorder,
        )
        agent.llm = llm2
        agent.max_execution_time = 30
        result = agent.execute_task(_task(agent, "RUN2-TASK"))

        # 1. Run 2 returns its own result.
        assert str(result) == "SECOND"
        # 2. Not held: run 2's first LLM call starts before run 1's tool ends.
        assert recorder.first("llm:run2") < recorder.first("run1-tool-end")
        # 3. Run 2 received nothing from run 1.
        for text in llm2.inputs:
            assert "RUN1-" not in text
        # 4. No repeated-usage refusal, and run 2's tool ran.
        assert "run2-tool-ran" in [lb for _, lb in recorder.events]
        for text in llm2.inputs:
            assert "RUN1-TOOL-RESULT" not in text
            assert "already used" not in text.lower()
        # 5. Run 1's LLM received nothing from run 2.
        for text in llm1.inputs:
            assert "RUN2-" not in text and "SECOND" not in text
    finally:
        assert run1_tool_done.wait(DRAIN_BOUND)


def test_v3_async_later_calls_after_timeout_are_independent(
    recorder: _Recorder,
) -> None:
    """V3: in the coroutine that caught the error, later calls are not affected."""
    run1_tool_done = threading.Event()
    uses: list[str] = []

    @tool("lookup")
    def lookup(x: str) -> str:
        """Look a value up. The first use is slow."""
        first = not uses
        uses.append(x)
        if first:
            try:
                time.sleep(4)
                recorder.add("run1-tool-end")
                return "RUN1-TOOL-RESULT"
            finally:
                run1_tool_done.set()
        recorder.add("same-tool-ran")
        return "SAME-TOOL-RESULT"

    llm1 = ScriptedLLM(
        [(0, 'Thought: RUN1-MARKER\nAction: lookup\nAction Input: {"x": "a"}')],
        "run1",
        recorder,
    )
    agent_a = _agent(llm1, [lookup])

    async def scenario() -> dict[str, Any]:
        out: dict[str, Any] = {}
        start = time.monotonic()
        try:
            await agent_a.aexecute_task(_task(agent_a, "RUN1-TASK"))
        except TimeoutError:
            out["raise_after"] = time.monotonic() - start
        out["run1_step_running"] = not run1_tool_done.is_set()

        # (i) the same agent, a new scripted LLM, a raised limit. Its first
        # reply comes after run 1's step has ended, inside this run.
        llm_same = ScriptedLLM(
            [(3.5, _act("lookup", '{"x": "a"}')), (0, _final("SAME-OWN"))],
            "same",
            recorder,
        )
        agent_a.llm = llm_same
        agent_a.max_execution_time = 30
        try:
            out["same"] = str(await agent_a.aexecute_task(_task(agent_a, "RUN2-TASK")))
        except Exception as e:  # noqa: BLE001 - recorded and asserted below
            out["same"] = e
        out["same_inputs"] = llm_same.inputs
        executor = agent_a.agent_executor
        out["same_messages"] = (
            [str(m.get("content", "")) for m in executor.state.messages]
            if executor is not None
            else []
        )

        # (ii) another agent with no limit.
        agent_b = _agent(
            ScriptedLLM([(0, _final("OTHER-OWN"))], "other", recorder),
            [],
            max_execution_time=None,
        )
        try:
            out["other"] = str(await agent_b.aexecute_task(_task(agent_b)))
        except Exception as e:  # noqa: BLE001 - recorded and asserted below
            out["other"] = e

        # (iii) a direct model call on a fresh LLM.
        direct = ScriptedLLM([(0, "DIRECT-OWN")], "direct", recorder)
        try:
            out["direct"] = await direct.acall([{"role": "user", "content": "hi"}])
        except Exception as e:  # noqa: BLE001 - recorded and asserted below
            out["direct"] = e
        return out

    try:
        out = asyncio.run(scenario())

        assert "raise_after" in out, "run 1 did not raise TimeoutError"
        assert out["raise_after"] < RAISE_BOUND, f"took {out['raise_after']:.2f}s"
        assert out["run1_step_running"], "run 1's step ended before the later calls"

        assert not isinstance(out["same"], Exception), repr(out["same"])
        assert out["same"] == "SAME-OWN"
        assert recorder.first("llm:same") < recorder.first("run1-tool-end")
        ran = [lb for _, lb in recorder.events]
        assert "same-tool-ran" in ran, out["same_inputs"][-1:]
        for text in out["same_inputs"] + out["same_messages"]:
            assert "RUN1-" not in text
            assert "already used" not in text.lower()

        assert not isinstance(out["other"], Exception), repr(out["other"])
        assert out["other"] == "OTHER-OWN"

        assert not isinstance(out["direct"], LLMCallBlockedError), repr(out["direct"])
        assert out["direct"] == "DIRECT-OWN"
    finally:
        assert run1_tool_done.wait(DRAIN_BOUND)


# --- I7: the timed-out run starts nothing new after the error -----------------


def test_v6a_sync_gated_tool_after_deadline_is_not_started(
    recorder: _Recorder, prompts: list[str], approval_gate: None
) -> None:
    """V6a: after the error, no approval prompt, no side effect, no LLM call."""
    llm = ScriptedLLM(
        [(2, _act("delete_file", '{"path": "x"}')), (0, _final("DONE"))],
        "run1",
        recorder,
    )
    agent = _agent(llm, [_side_effect_tool(recorder)])

    _, raised_at = _raise_time(lambda: agent.execute_task(_task(agent)))
    time.sleep(3)  # let the abandoned run reach its next boundaries

    after = recorder.after(raised_at)
    assert after.count("prompt") == 0, after
    assert after.count("side-effect") == 0, after
    assert [lb for lb in after if lb.startswith("llm:")] == [], after


def test_v6c_async_tool_in_flight_cannot_call_an_llm(recorder: _Recorder) -> None:
    """V6c: a tool in flight at the deadline gets its own LLM call refused."""
    tool_done = threading.Event()
    outcome: dict[str, Any] = {}
    inner_llm = ScriptedLLM([(0, "INNER")], "tool-inner", recorder)

    @tool("research")
    def research(q: str) -> str:
        """A slow tool that asks an LLM after 2 seconds."""
        try:
            time.sleep(2)
            try:
                outcome["inner"] = inner_llm.call([{"role": "user", "content": q}])
            except Exception as e:  # noqa: BLE001 - recorded and asserted below
                outcome["inner"] = e
            return "researched"
        finally:
            tool_done.set()

    llm = ScriptedLLM([(0, _act("research", '{"q": "x"}'))], "run1", recorder)
    agent = _agent(llm, [research])

    async def scenario() -> tuple[float, float]:
        start = time.monotonic()
        try:
            await agent.aexecute_task(_task(agent))
        except TimeoutError:
            return time.monotonic() - start, time.monotonic()
        pytest.fail("aexecute_task did not raise TimeoutError")

    try:
        elapsed, raised_at = asyncio.run(scenario())
        assert elapsed < RAISE_BOUND, f"took {elapsed:.2f}s"
    finally:
        assert tool_done.wait(DRAIN_BOUND)

    assert isinstance(outcome.get("inner"), LLMCallBlockedError), outcome
    after = recorder.after(raised_at)
    assert [lb for lb in after if lb.startswith("llm:")] == [], after


def test_v6d_sync_delegated_coworker_starts_nothing_after_deadline(
    recorder: _Recorder, prompts: list[str], approval_gate: None
) -> None:
    """V6d: a co-worker running inside the timed-out run is stopped too."""
    delegation_done = threading.Event()
    coworker = Agent(
        role="Coworker",
        goal="Help",
        backstory="You help.",
        llm=ScriptedLLM(
            [(2, _act("delete_file", '{"path": "x"}')), (0, _final("CO-DONE"))],
            "coworker",
            recorder,
        ),
        tools=[_side_effect_tool(recorder)],
        allow_delegation=False,
        max_retry_limit=0,
    )

    @tool("ask_coworker")
    def ask_coworker(q: str) -> str:
        """Delegate the question to a co-worker."""
        try:
            return str(coworker.execute_task(_task(coworker, "Sub task")))
        finally:
            delegation_done.set()

    outer = _agent(
        ScriptedLLM([(0, _act("ask_coworker", '{"q": "x"}'))], "outer", recorder),
        [ask_coworker],
    )

    try:
        _, raised_at = _raise_time(lambda: outer.execute_task(_task(outer)))
    finally:
        assert delegation_done.wait(DRAIN_BOUND)
    time.sleep(1)  # let the outer run reach its next boundary

    after = recorder.after(raised_at)
    assert after.count("prompt") == 0, after
    assert after.count("side-effect") == 0, after
    assert [lb for lb in after if lb.startswith("llm:")] == [], after


def test_v6e_sync_final_answer_after_deadline_asks_no_feedback_saves_no_memory(
    recorder: _Recorder, prompts: list[str]
) -> None:
    """V6e: a final answer after the deadline: no ``human_input`` prompt, no save."""
    llm = ScriptedLLM([(2, _final("LATE ANSWER"))], "run1", recorder)
    agent = _agent(llm, [])
    agent.memory = _RecordingMemory(recorder)

    _, raised_at = _raise_time(
        lambda: agent.execute_task(_task(agent, human_input=True))
    )
    time.sleep(3)  # the final answer arrives at about 2 s

    after = recorder.after(raised_at)
    assert after.count("prompt") == 0, after
    assert after.count("memory:save") == 0, after


class _Extracted(BaseModel):
    v: str


def test_v6f_async_converter_in_tool_in_flight_builds_no_client(
    recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """V6f: the structured-output converter makes no provider request after the error.

    ``InternalInstructor`` builds the provider client; every construction is
    recorded instead of building a real one.
    """
    tool_done = threading.Event()
    outcome: dict[str, Any] = {}

    class RecordingInstructor:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            recorder.add("instructor-built")

        def to_json(self) -> str:
            return '{"v": "x"}'

    monkeypatch.setattr(converter_module, "InternalInstructor", RecordingInstructor)

    @tool("convert")
    def convert(q: str) -> str:
        """A slow tool that uses the structured-output converter after 2 seconds."""
        try:
            time.sleep(2)
            outcome["result"] = Converter(
                llm=FunctionCallingLLM([], "conv", recorder),
                text="v is x",
                model=_Extracted,
                instructions="extract v",
                max_attempts=1,
            ).to_json()
            return "converted"
        finally:
            tool_done.set()

    llm = ScriptedLLM([(0, _act("convert", '{"q": "x"}'))], "run1", recorder)
    agent = _agent(llm, [convert])

    async def scenario() -> float:
        try:
            await agent.aexecute_task(_task(agent))
        except TimeoutError:
            return time.monotonic()
        pytest.fail("aexecute_task did not raise TimeoutError")

    try:
        raised_at = asyncio.run(scenario())
    finally:
        assert tool_done.wait(DRAIN_BOUND)

    after = recorder.after(raised_at)
    assert after.count("instructor-built") == 0, after
    assert isinstance(outcome.get("result"), ConverterError), outcome
