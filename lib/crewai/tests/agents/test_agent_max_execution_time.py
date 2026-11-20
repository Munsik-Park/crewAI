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
import contextvars
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from crewai import Agent, Task
from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import AgentExecutionStartedEvent
from crewai.hooks import (
    register_after_llm_call_hook,
    register_after_tool_call_hook,
    register_before_llm_call_hook,
    register_before_tool_call_hook,
    unregister_after_llm_call_hook,
    unregister_after_tool_call_hook,
    unregister_before_llm_call_hook,
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


def _run_end(agent: Agent, monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """An event set when the agent's (possibly abandoned) execution ends."""
    ended = threading.Event()
    execute = agent._execute_without_timeout

    def tracked(*args: Any, **kwargs: Any) -> Any:
        try:
            return execute(*args, **kwargs)
        finally:
            ended.set()

    monkeypatch.setattr(agent, "_execute_without_timeout", tracked)
    return ended


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
    recorder: _Recorder,
    prompts: list[str],
    approval_gate: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V6a: after the error, no approval prompt, no side effect, no LLM call."""
    llm = ScriptedLLM(
        [(2, _act("delete_file", '{"path": "x"}')), (0, _final("DONE"))],
        "run1",
        recorder,
    )
    agent = _agent(llm, [_side_effect_tool(recorder)])
    run_ended = _run_end(agent, monkeypatch)

    _, raised_at = _raise_time(lambda: agent.execute_task(_task(agent)))
    # The abandoned run passes its tool and model boundaries before it ends.
    assert run_ended.wait(DRAIN_BOUND)

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
    recorder: _Recorder, prompts: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """V6e: a final answer after the deadline: no ``human_input`` prompt, no save."""
    llm = ScriptedLLM([(2, _final("LATE ANSWER"))], "run1", recorder)
    agent = _agent(llm, [])
    agent.memory = _RecordingMemory(recorder)
    run_ended = _run_end(agent, monkeypatch)

    _, raised_at = _raise_time(
        lambda: agent.execute_task(_task(agent, human_input=True))
    )
    # The feedback step and the memory save come after the final answer.
    assert run_ended.wait(DRAIN_BOUND)

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


# --- I7: no hook runs for a call the stop refuses ------------------------------


@pytest.fixture
def model_and_tool_hooks(prompts: list[str], recorder: _Recorder) -> Iterator[None]:
    """A global before-LLM approval gate and an after-tool hook, removed on teardown."""

    def model_gate(context: Any) -> bool | None:
        recorder.add("hook:before-llm")
        context.request_human_input(prompt="Allow this model call?")
        return None

    def after_tool(context: Any) -> str | None:
        recorder.add(f"hook:after-tool:{context.tool_name}")
        return None

    register_before_llm_call_hook(model_gate)
    register_after_tool_call_hook(after_tool)
    try:
        yield
    finally:
        unregister_before_llm_call_hook(model_gate)
        unregister_after_tool_call_hook(after_tool)


def test_v6g_sync_refused_calls_run_no_hooks(
    recorder: _Recorder,
    prompts: list[str],
    model_and_tool_hooks: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the error, the refused tool call and model call run no hooks.

    The model answers after the deadline and asks for a tool; the stop refuses
    that tool call and the next model call. Neither the after-tool hook nor the
    before-LLM approval gate may run for them, so no prompt appears.
    """
    llm = ScriptedLLM(
        [(2, _act("delete_file", '{"path": "x"}')), (0, _final("DONE"))],
        "run1",
        recorder,
    )
    agent = _agent(llm, [_side_effect_tool(recorder)])
    run_ended = _run_end(agent, monkeypatch)

    _, raised_at = _raise_time(lambda: agent.execute_task(_task(agent)))
    assert run_ended.wait(DRAIN_BOUND)

    after = recorder.after(raised_at)
    assert [lb for lb in after if lb.startswith("hook:")] == [], after
    assert after.count("prompt") == 0, after
    assert after.count("side-effect") == 0, after
    assert [lb for lb in after if lb.startswith("llm:")] == [], after


def test_v6h_async_refused_model_call_in_flight_step_runs_no_hooks(
    recorder: _Recorder, prompts: list[str], model_and_tool_hooks: None
) -> None:
    """On async, a co-worker started by the step in flight runs no model hook.

    The tool in flight at the deadline delegates to a co-worker after 2 s. The
    co-worker's model call is refused, and its before-LLM approval gate must
    not run, so no prompt appears.
    """
    tool_done = threading.Event()
    outcome: dict[str, Any] = {}
    coworker = Agent(
        role="Coworker",
        goal="Help",
        backstory="You help.",
        llm=ScriptedLLM([(0, _final("CO-DONE"))], "coworker", recorder),
        allow_delegation=False,
        max_retry_limit=0,
    )

    @tool("ask_coworker")
    def ask_coworker(q: str) -> str:
        """Delegate the question to a co-worker after 2 seconds."""
        try:
            time.sleep(2)
            try:
                outcome["co"] = str(coworker.execute_task(_task(coworker, "Sub")))
            except Exception as e:  # noqa: BLE001 - recorded and asserted below
                outcome["co"] = e
            return "asked"
        finally:
            tool_done.set()

    outer = _agent(
        ScriptedLLM([(0, _act("ask_coworker", '{"q": "x"}'))], "outer", recorder),
        [ask_coworker],
    )

    async def scenario() -> float:
        try:
            await outer.aexecute_task(_task(outer))
        except TimeoutError:
            return time.monotonic()
        pytest.fail("aexecute_task did not raise TimeoutError")

    try:
        raised_at = asyncio.run(scenario())
    finally:
        assert tool_done.wait(DRAIN_BOUND)

    # The tool in flight runs to its end with its own hooks; only the model
    # call it starts after the deadline is refused.
    after = recorder.after(raised_at)
    assert after.count("hook:before-llm") == 0, after
    assert after.count("prompt") == 0, after
    assert [lb for lb in after if lb.startswith("llm:")] == [], after
    assert isinstance(outcome.get("co"), Exception), outcome


# --- I7 / I8: the hook boundary around the stop, case by case -----------------
#
# The rule: a call checks the stop where it would begin. A call the stop refuses
# before its before-hooks runs no hook, before or after. A call still held in
# its before-hooks at the deadline does not execute afterwards, whatever those
# hooks decide: a tool call is treated as blocked (its after-tool hooks run), a
# model call is refused (no after-hooks). Late work in these tests waits for
# the caller to catch the TimeoutError, never for a fixed time.

TARGET = "Target"
HOLD_BOUND = 5.0  # bound on late work waiting for the caller to catch the error
AFTER_ERROR = -1.0  # a scripted delay that waits for the caller to catch the error


class _GatedLLM(ScriptedLLM):
    """A ScriptedLLM whose ``AFTER_ERROR`` delay waits on the ``caught`` event."""

    def __init__(
        self,
        script: list[tuple[float, str]],
        name: str,
        recorder: _Recorder,
        caught: threading.Event,
    ) -> None:
        super().__init__(script, name, recorder)
        self.caught = caught

    def _next(self, messages: Any) -> tuple[float, str]:
        delay, reply = super()._next(messages)
        if delay == AFTER_ERROR:
            self.caught.wait(HOLD_BOUND)
            return 0.0, reply
        return delay, reply


@pytest.fixture
def caught() -> threading.Event:
    """Set by the test once its caller has caught the TimeoutError."""
    return threading.Event()


# Events counted after the TimeoutError, per kind of call.
_BOUNDARY_EXPECTED = {
    ("tool", "refused"): {
        "before-tool:start": 0,
        "before-tool:end": 0,
        "tool-ran": 0,
        "after-tool": 0,
    },
    ("tool", "allowed"): {
        "before-tool:start": 0,
        "before-tool:end": 1,
        "tool-ran": 0,
        "after-tool": 1,
    },
    ("tool", "blocked"): {
        "before-tool:start": 0,
        "before-tool:end": 1,
        "tool-ran": 0,
        "after-tool": 1,
    },
    ("model", "refused"): {
        "before-llm:start": 0,
        "before-llm:end": 0,
        "llm:target": 0,
        "after-llm": 0,
    },
    ("model", "allowed"): {
        "before-llm:start": 0,
        "before-llm:end": 1,
        "llm:target": 0,
        "after-llm": 0,
    },
    ("model", "blocked"): {
        "before-llm:start": 0,
        "before-llm:end": 1,
        "llm:target": 0,
        "after-llm": 0,
    },
}


def _is_target(context: Any) -> bool:
    return getattr(getattr(context, "agent", None), "role", None) == TARGET


@pytest.fixture
def boundary_hooks(
    recorder: _Recorder, caught: threading.Event, request: pytest.FixtureRequest
) -> Iterator[None]:
    """Global hooks around the target agent's calls, removed on teardown.

    For a held case the matching before-hook holds its first call until the
    caller has caught the TimeoutError, then allows or blocks that call.
    """
    kind, case = request.param
    held = case in ("allowed", "blocked")
    held_tool = [kind == "tool" and held]
    held_model = [kind == "model" and held]

    def before_tool(context: Any) -> bool | None:
        if not _is_target(context) or context.tool_name != "probe_tool":
            return None
        recorder.add("before-tool:start")
        hold, held_tool[0] = held_tool[0], False
        if hold:
            caught.wait(HOLD_BOUND)
        recorder.add("before-tool:end")
        return False if hold and case == "blocked" else None

    def after_tool(context: Any) -> str | None:
        if _is_target(context) and context.tool_name == "probe_tool":
            recorder.add("after-tool")
        return None

    def before_llm(context: Any) -> bool | None:
        if not _is_target(context):
            return None
        recorder.add("before-llm:start")
        hold, held_model[0] = held_model[0], False
        if hold:
            caught.wait(HOLD_BOUND)
        recorder.add("before-llm:end")
        return False if hold and case == "blocked" else None

    def after_llm(context: Any) -> str | None:
        if _is_target(context):
            recorder.add("after-llm")
        return None

    register_before_tool_call_hook(before_tool)
    register_after_tool_call_hook(after_tool)
    register_before_llm_call_hook(before_llm)
    register_after_llm_call_hook(after_llm)
    try:
        yield
    finally:
        unregister_before_tool_call_hook(before_tool)
        unregister_after_tool_call_hook(after_tool)
        unregister_before_llm_call_hook(before_llm)
        unregister_after_llm_call_hook(after_llm)


def _boundary_target(
    kind: str,
    case: str,
    recorder: _Recorder,
    caught: threading.Event,
    **kwargs: Any,
) -> Agent:
    """The agent whose calls the hooks watch, scripted for one case."""

    @tool("probe_tool")
    def probe_tool(x: str) -> str:
        """The tool call under test."""
        recorder.add("tool-ran")
        return "probed"

    @tool("slow_tool")
    def slow_tool(x: str) -> str:
        """Runs until the error is caught, so the next model call starts after it."""
        caught.wait(HOLD_BOUND)
        return "slow"

    if kind == "tool":
        delay = AFTER_ERROR if case == "refused" else 0
        first = (delay, _act("probe_tool", '{"x": "a"}'))
        script = [first, (0, _final("DONE"))]
    elif case == "refused":
        script = [(0, _act("slow_tool", '{"x": "a"}')), (0, _final("DONE"))]
    else:
        script = [(0, _final("DONE"))]
    return Agent(
        role=TARGET,
        goal="Answer the task",
        backstory="You answer tasks.",
        llm=_GatedLLM(script, "target", recorder, caught),
        tools=[probe_tool, slow_tool],
        allow_delegation=False,
        max_retry_limit=0,
        **kwargs,
    )


_BOUNDARY_CASES = [
    pytest.param((kind, case), path, id=f"{kind}-{case}-{path}")
    for kind in ("tool", "model")
    for case in ("refused", "allowed", "blocked")
    for path in ("sync", "async")
]


@pytest.mark.parametrize(
    ("boundary_hooks", "path"), _BOUNDARY_CASES, indirect=["boundary_hooks"]
)
def test_v6i_hooks_around_the_stop_follow_the_rule(
    recorder: _Recorder,
    prompts: list[str],
    caught: threading.Event,
    boundary_hooks: None,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    path: str,
) -> None:
    """Which before- and after-hooks run for a call refused, allowed or blocked.

    Sync: the target agent itself has the limit. Async: the target is a
    co-worker run by the outer agent's tool, the step in flight at the deadline.
    """
    kind, case = request.node.callspec.params["boundary_hooks"]

    if path == "sync":
        target = _boundary_target(
            kind, case, recorder, caught, max_execution_time=LIMIT
        )
        run_ended = _run_end(target, monkeypatch)
        try:
            _, raised_at = _raise_time(lambda: target.execute_task(_task(target)))
        finally:
            caught.set()
        assert run_ended.wait(DRAIN_BOUND)
    else:
        coworker = _boundary_target(kind, case, recorder, caught)
        delegated = threading.Event()

        @tool("delegate")
        def delegate(q: str) -> str:
            """Run the co-worker."""
            try:
                if kind == "model" and case == "refused":
                    caught.wait(HOLD_BOUND)  # the co-worker's first call is after it
                try:
                    return str(coworker.execute_task(_task(coworker, "Sub")))
                except Exception as e:  # noqa: BLE001 - a refused run may raise
                    return f"failed: {type(e).__name__}"
            finally:
                delegated.set()

        outer = Agent(
            role="Outer",
            goal="Delegate",
            backstory="You delegate.",
            llm=ScriptedLLM([(0, _act("delegate", '{"q": "x"}'))], "outer", recorder),
            tools=[delegate],
            allow_delegation=False,
            max_retry_limit=0,
            max_execution_time=LIMIT,
        )

        async def scenario() -> float:
            try:
                await outer.aexecute_task(_task(outer))
            except TimeoutError:
                raised = time.monotonic()
                caught.set()
                return raised
            pytest.fail("aexecute_task did not raise TimeoutError")

        try:
            raised_at = asyncio.run(scenario())
        finally:
            caught.set()
            assert delegated.wait(DRAIN_BOUND)

    after = recorder.after(raised_at)
    expected = _BOUNDARY_EXPECTED[(kind, case)]
    observed = {label: after.count(label) for label in expected}
    assert observed == expected, after
    assert after.count("prompt") == 0, after


# --- I7 / I8: work that starts after the error inside the work in flight -------
#
# Each row starts one piece of late work from inside the call executing at the
# deadline, after the caller has caught the TimeoutError, and checks that its
# model request is not sent. Requests are recorded inside the fake model's body,
# below every check, so a refused request records nothing.


class _LateEvent(BaseEvent):
    type: str = "max_execution_time_late_work"


class _FnLLM(BaseLLM):
    """A fake model whose reply to its n-th request comes from ``reply(n)``.

    With ``dispatch_hooks`` it dispatches ``PRE_MODEL_CALL`` inside its own
    call, as the native providers do.
    """

    def __init__(
        self,
        name: str,
        recorder: _Recorder,
        reply: Any,
        areply: Any = None,
        dispatch_hooks: bool = False,
    ) -> None:
        super().__init__(model="test-model")
        self.name = name
        self.recorder = recorder
        self.reply = reply
        self.areply = areply
        self.dispatch_hooks = dispatch_hooks
        self.requests = 0

    def _start(self, messages: Any, from_agent: Any) -> int:
        if self.dispatch_hooks:
            self._invoke_before_llm_call_hooks(messages, from_agent)
        self.requests += 1
        self.recorder.add(f"llm:{self.name}")
        return self.requests

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
        return self.reply(self._start(messages, from_agent))

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
        n = self._start(messages, from_agent)
        if self.areply is not None:
            return await self.areply(n)
        return self.reply(n)

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192


_LATE_ROWS = {
    "j1": "a sync event listener, for an event emitted inside the model request",
    "j2": "an async event listener, for an event emitted inside the model request",
    "j3": "a thread started with copy_context() inside the model request",
    "j4": "a custom model in flight that calls another model after the error",
    "j5": "the empty-answer retry of the model request in flight",
    "j6": "the rate-limit retry of the model request in flight",
    "j7": "an event listener, for an event emitted inside the tool in flight",
    "j8": "a thread started with copy_context() inside the tool in flight",
    "j9": "a task created inside an acall the tool in flight is awaiting",
    "j10": "a model call from the tool in flight, held in its own before-hook",
}

_LATE_CASES = [
    pytest.param(row, path, id=f"{row}-{path}")
    for row in _LATE_ROWS
    for path in ("sync", "async")
    if not (row == "j9" and path == "sync")
]


@pytest.mark.parametrize(("row", "path"), _LATE_CASES)
def test_v6j_work_started_after_the_error_sends_no_request(
    recorder: _Recorder,
    prompts: list[str],
    caught: threading.Event,
    monkeypatch: pytest.MonkeyPatch,
    row: str,
    path: str,
) -> None:
    """No model request is sent by late work started inside the work in flight.

    Sync: the target agent itself has the limit. Async: the target is a
    co-worker run by the outer agent's tool, the step in flight at the deadline.
    """
    late_done = threading.Event()
    retry_row = row in ("j5", "j6")
    probe = "llm:target" if retry_row else "llm:side"
    side = _FnLLM("side", recorder, lambda n: "SIDE", dispatch_hooks=row == "j10")

    def side_request() -> None:
        try:
            side.call([{"role": "user", "content": "late"}])
        except LLMCallBlockedError:
            pass
        finally:
            late_done.set()

    def late_after_error() -> None:
        caught.wait(HOLD_BOUND)
        recorder.add("late:start")
        side_request()

    def listener(source: Any, event: Any) -> None:
        recorder.add("late:start")
        side_request()

    async def alistener(source: Any, event: Any) -> None:
        recorder.add("late:start")
        side_request()

    def held_side_hook(context: Any) -> bool | None:
        if context.llm is not side:
            return None
        caught.wait(HOLD_BOUND)
        recorder.add("late:start")
        return None

    async def late_task() -> None:
        await asyncio.to_thread(caught.wait, HOLD_BOUND)
        recorder.add("late:start")
        try:
            await side.acall([{"role": "user", "content": "late"}])
        except LLMCallBlockedError:
            pass
        finally:
            late_done.set()

    async def inflight_areply(n: int) -> str:
        await asyncio.get_running_loop().create_task(late_task())
        return "IN-FLIGHT"

    inflight = _FnLLM("inflight", recorder, lambda n: "", areply=inflight_areply)

    def target_reply(n: int) -> str:
        if row in ("j7", "j8", "j9", "j10"):
            return _act("worker", '{"x": "a"}') if n == 1 else _final("DONE")
        if n > 1:
            return _final("DONE")
        if row in ("j1", "j2"):
            caught.wait(HOLD_BOUND)
            crewai_event_bus.emit(target, _LateEvent())
            late_done.wait(DRAIN_BOUND)
        elif row == "j3":
            ctx = contextvars.copy_context()
            threading.Thread(target=ctx.run, args=(late_after_error,)).start()
            caught.wait(HOLD_BOUND)
        elif row == "j4":
            late_after_error()
        elif row == "j5":
            caught.wait(HOLD_BOUND)
            recorder.add("late:start")
            return ""
        elif row == "j6":
            caught.wait(HOLD_BOUND)
            recorder.add("late:start")
            raise RuntimeError("rate limit exceeded")
        return _final("DONE")

    @tool("worker")
    def worker(x: str) -> str:
        """The tool in flight at the deadline."""
        if row == "j7":
            caught.wait(HOLD_BOUND)
            crewai_event_bus.emit(None, _LateEvent())
            late_done.wait(DRAIN_BOUND)
        elif row == "j8":
            ctx = contextvars.copy_context()
            threading.Thread(target=ctx.run, args=(late_after_error,)).start()
            caught.wait(HOLD_BOUND)
        elif row == "j9":
            asyncio.run(inflight.acall([{"role": "user", "content": "work"}]))
        elif row == "j10":
            side_request()
        return "worked"

    if row in ("j1", "j7"):
        crewai_event_bus.on(_LateEvent)(listener)
        handler: Any = listener
    elif row == "j2":
        crewai_event_bus.on(_LateEvent)(alistener)
        handler = alistener
    else:
        handler = None
    if row == "j10":
        register_before_llm_call_hook(held_side_hook)

    target = Agent(
        role=TARGET,
        goal="Answer the task",
        backstory="You answer tasks.",
        llm=_FnLLM("target", recorder, target_reply),
        tools=[worker],
        allow_delegation=False,
        max_retry_limit=0,
        **({"max_execution_time": LIMIT} if path == "sync" else {}),
    )

    try:
        if path == "sync":
            run_ended = _run_end(target, monkeypatch)
            try:
                _, raised_at = _raise_time(lambda: target.execute_task(_task(target)))
            finally:
                caught.set()
            assert run_ended.wait(DRAIN_BOUND)
        else:
            run_ended = threading.Event()

            @tool("delegate")
            def delegate(q: str) -> str:
                """Run the co-worker."""
                try:
                    return str(target.execute_task(_task(target, "Sub")))
                except Exception as e:  # noqa: BLE001 - a refused run may raise
                    return f"failed: {type(e).__name__}"
                finally:
                    run_ended.set()

            outer = Agent(
                role="Outer",
                goal="Delegate",
                backstory="You delegate.",
                llm=ScriptedLLM(
                    [(0, _act("delegate", '{"q": "x"}'))], "outer", recorder
                ),
                tools=[delegate],
                allow_delegation=False,
                max_retry_limit=0,
                max_execution_time=LIMIT,
            )

            async def scenario() -> float:
                try:
                    await outer.aexecute_task(_task(outer))
                except TimeoutError:
                    raised = time.monotonic()
                    caught.set()
                    return raised
                pytest.fail("aexecute_task did not raise TimeoutError")

            try:
                raised_at = asyncio.run(scenario())
            finally:
                caught.set()
            assert run_ended.wait(DRAIN_BOUND)
        if not retry_row:
            assert late_done.wait(DRAIN_BOUND)
    finally:
        if handler is not None:
            crewai_event_bus.off(_LateEvent, handler)
        if row == "j10":
            unregister_before_llm_call_hook(held_side_hook)

    after = recorder.after(raised_at)
    assert after.count("late:start") == 1, (_LATE_ROWS[row], after)
    assert after.count(probe) == 0, (_LATE_ROWS[row], after)
    assert after.count("prompt") == 0, after


# --- C2b, I3, I9: later and overlapping calls (V2-L) -------------------------
#
# V2 and V3 above are rows L1 to L3. Each row below is a call that meets work
# a timed-out run still holds, or a call next to a live call on one agent.
# Every ordering is set by an Event, each wait bounded; each run's replies are
# chosen by the task named in its request, so no run can consume another's.

SUB_TASK = "SUB-TASK-FIRST"
LATER_TASK = "LATER-TASK"


class _ByTaskLLM(BaseLLM):
    """A fake model that answers each request with the reply of the task in it."""

    def __init__(
        self, name: str, recorder: _Recorder, replies: dict[str, Callable[[], str]]
    ) -> None:
        super().__init__(model="test-model")
        self.name = name
        self.recorder = recorder
        self.replies = replies
        self.inputs: list[str] = []
        self.answered = {marker: threading.Event() for marker in replies}
        self._lock = threading.Lock()

    def _reply(self, messages: Any) -> str:
        text = _text(messages)
        with self._lock:
            self.inputs.append(text)
        self.recorder.add(f"llm:{self.name}")
        for marker, reply in self.replies.items():
            if marker in text:
                try:
                    return reply()
                finally:
                    self.answered[marker].set()
        return _final("UNEXPECTED")

    def inputs_for(self, marker: str) -> list[str]:
        with self._lock:
            return [text for text in self.inputs if marker in text]

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
        return self._reply(messages)

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
        return await asyncio.to_thread(self._reply, messages)

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192


def _held(event: threading.Event, reply: str) -> Callable[[], str]:
    """A reply given once ``event`` is set (bounded)."""

    def answer() -> str:
        event.wait(HOLD_BOUND)
        return reply

    return answer


def _outcome(call: Callable[[], Any]) -> Any:
    try:
        return call()
    except Exception as e:  # noqa: BLE001 - the row asserts on it
        return e


async def _aoutcome(awaitable: Any) -> Any:
    try:
        return await awaitable
    except Exception as e:  # noqa: BLE001 - the row asserts on it
        return e


def _already_running(outcome: Any) -> bool:
    return isinstance(outcome, Exception) and "Executor is already running" in str(
        outcome
    )


def _messages_text(messages: Any) -> str:
    return " ".join(str(m.get("content", "")) for m in messages or [])


class _CoworkerRun:
    """An outer agent with the limit whose tool delegates to a co-worker."""

    def __init__(
        self,
        recorder: _Recorder,
        monkeypatch: pytest.MonkeyPatch,
        replies: dict[str, Callable[[], str]],
        tools: list[Any] | None = None,
        max_retry_limit: int = 0,
    ) -> None:
        self.llm = _ByTaskLLM("coworker", recorder, replies)
        self.coworker = Agent(
            role="Coworker",
            goal="Help",
            backstory="You help.",
            llm=self.llm,
            tools=tools or [],
            allow_delegation=False,
            max_retry_limit=max_retry_limit,
        )
        self.delegation_errors: list[Exception] = []
        self._idle = threading.Condition()
        self._attempts = 0
        self._in_flight = 0

        @tool("ask_coworker")
        def ask_coworker(q: str) -> str:
            """Delegate the question to a co-worker."""
            with self._idle:
                self._attempts += 1
                self._in_flight += 1
            try:
                return str(self.coworker.execute_task(_task(self.coworker, SUB_TASK)))
            except Exception as e:
                self.delegation_errors.append(e)
                raise
            finally:
                with self._idle:
                    self._in_flight -= 1
                    self._idle.notify_all()

        self.outer = _agent(
            ScriptedLLM([(0, _act("ask_coworker", '{"q": "x"}'))], "outer", recorder),
            [ask_coworker],
        )
        self.outer_ended = _run_end(self.outer, monkeypatch)
        self.path = "sync"

    def delegation_running(self) -> bool:
        with self._idle:
            return self._in_flight > 0

    def drained(self) -> bool:
        """Waits for the abandoned work to end (bounded).

        On sync the timed worker ends with it. On async the timed task is
        cancelled at the deadline while its step thread runs on, so the end
        is that of the delegation attempts.
        """
        if self.path == "sync":
            return self.outer_ended.wait(DRAIN_BOUND)
        with self._idle:
            return self._idle.wait_for(
                lambda: self._attempts > 0 and self._in_flight == 0, DRAIN_BOUND
            )

    def time_out_then(self, path: str, caught: threading.Event, later: Any) -> Any:
        """Runs the outer agent into its limit, then ``later`` in the caller.

        ``later`` takes the co-worker and returns its outcome (a coroutine
        function on the async path).
        """
        self.path = path
        if path == "sync":
            try:
                with pytest.raises(TimeoutError):
                    self.outer.execute_task(_task(self.outer))
            finally:
                caught.set()
            return later(self.coworker)

        async def scenario() -> Any:
            try:
                with pytest.raises(TimeoutError):
                    await self.outer.aexecute_task(_task(self.outer))
            finally:
                caught.set()
            return await later(self.coworker)

        return asyncio.run(scenario())


def _row_coworker_later_call(
    recorder: _Recorder,
    caught: threading.Event,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    """L4, L5: a later call on the co-worker whose delegated call still runs."""
    later_done = threading.Event()
    run = _CoworkerRun(
        recorder,
        monkeypatch,
        {
            SUB_TASK: _held(later_done, _final("CO-FIRST")),
            LATER_TASK: lambda: _final("LATER-OWN"),
        },
    )
    still_running: list[bool] = []

    def later(coworker: Agent) -> Any:
        still_running.append(run.delegation_running())
        try:
            return _outcome(lambda: coworker.execute_task(_task(coworker, LATER_TASK)))
        finally:
            later_done.set()

    async def alater(coworker: Agent) -> Any:
        still_running.append(run.delegation_running())
        try:
            return await _aoutcome(coworker.aexecute_task(_task(coworker, LATER_TASK)))
        finally:
            later_done.set()

    try:
        outcome = run.time_out_then(path, caught, later if path == "sync" else alater)
    finally:
        later_done.set()
        assert run.drained()

    assert still_running == [True], "the delegated call ended before the later call"
    assert not isinstance(outcome, Exception), repr(outcome)
    assert str(outcome) == "LATER-OWN"
    later_inputs = run.llm.inputs_for(LATER_TASK)
    assert later_inputs, "the later call sent no request"
    for text in later_inputs:
        assert SUB_TASK not in text and "CO-FIRST" not in text


def _same_agent_overlap(
    recorder: _Recorder,
    caught: threading.Event,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    a_step_ended: bool,
) -> None:
    """C is prepared before A's deadline and starts running after it."""
    a_started = threading.Event()
    c_returned = threading.Event()
    a_release = caught if a_step_ended else c_returned

    def a_reply() -> str:
        a_started.set()
        a_release.wait(HOLD_BOUND)
        return _final("A-OWN")

    llm = _ByTaskLLM(
        "shared", recorder, {"A-TASK": a_reply, "C-TASK": lambda: _final("C-OWN")}
    )
    agent = _agent(llm, [])
    a_task, c_task = _task(agent, "A-TASK"), _task(agent, "C-TASK")
    a_step_done = llm.answered["A-TASK"]
    c_prepared_at: list[float] = []
    a_out: dict[str, Any] = {}

    if path == "sync":
        timed = agent._execute_with_timeout

        def gated(task_prompt: str, task: Task, timeout: int) -> Any:
            if task is c_task:
                c_prepared_at.append(time.monotonic())
                caught.wait(HOLD_BOUND)
                if a_step_ended:
                    a_step_done.wait(HOLD_BOUND)
            return timed(task_prompt, task, timeout)

        monkeypatch.setattr(agent, "_execute_with_timeout", gated)

        def run_a() -> None:
            try:
                agent.execute_task(a_task)
            except Exception as e:  # noqa: BLE001 - asserted below
                a_out["A"] = e
                a_out["raised_at"] = time.monotonic()
            finally:
                caught.set()

        thread_a = threading.Thread(target=run_a)
        thread_a.start()
        try:
            assert a_started.wait(HOLD_BOUND)
            c_out = _outcome(lambda: agent.execute_task(c_task))
        finally:
            c_returned.set()
            thread_a.join(DRAIN_BOUND)
    else:
        atimed = agent._aexecute_with_timeout

        async def agated(task_prompt: str, task: Task, timeout: int) -> Any:
            if task is c_task:
                c_prepared_at.append(time.monotonic())
                await asyncio.to_thread(caught.wait, HOLD_BOUND)
                if a_step_ended:
                    await asyncio.to_thread(a_step_done.wait, HOLD_BOUND)
            return await atimed(task_prompt, task, timeout)

        monkeypatch.setattr(agent, "_aexecute_with_timeout", agated)

        async def scenario() -> Any:
            async def run_a() -> None:
                try:
                    await agent.aexecute_task(a_task)
                except Exception as e:  # noqa: BLE001 - asserted below
                    a_out["A"] = e
                    a_out["raised_at"] = time.monotonic()
                finally:
                    caught.set()

            async def run_c() -> Any:
                try:
                    assert await asyncio.to_thread(a_started.wait, HOLD_BOUND)
                    return await _aoutcome(agent.aexecute_task(c_task))
                finally:
                    c_returned.set()

            _, c = await asyncio.gather(run_a(), run_c())
            return c

        try:
            c_out = asyncio.run(scenario())
        finally:
            c_returned.set()
    assert a_step_done.wait(DRAIN_BOUND)

    assert isinstance(a_out.get("A"), TimeoutError), a_out
    assert c_prepared_at and c_prepared_at[0] < a_out["raised_at"], (
        "C was not prepared before A's deadline"
    )
    assert not isinstance(c_out, Exception), repr(c_out)
    assert str(c_out) == "C-OWN"
    for text in llm.inputs_for("C-TASK"):
        assert "A-TASK" not in text and "A-OWN" not in text
    for text in llm.inputs_for("A-TASK"):
        assert "C-TASK" not in text


def _row_prepared_before_step_running(
    recorder: _Recorder,
    caught: threading.Event,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    """L6, L8: A's step still runs when C starts running."""
    _same_agent_overlap(recorder, caught, monkeypatch, path, a_step_ended=False)


def _row_prepared_before_step_ended(
    recorder: _Recorder,
    caught: threading.Event,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    """L7, L9: A's step has ended before C starts running."""
    _same_agent_overlap(recorder, caught, monkeypatch, path, a_step_ended=True)


def _row_stopped_coworker_reentry(
    recorder: _Recorder,
    caught: threading.Event,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    """L10: stopped work calls the co-worker again while a later call runs."""

    @tool("noop")
    def noop(x: str) -> str:
        """Do nothing."""
        return "NOOP"

    outer_done = threading.Event()
    run = _CoworkerRun(
        recorder,
        monkeypatch,
        {
            SUB_TASK: _held(caught, _act("noop", '{"x": "a"}')),
            LATER_TASK: _held(outer_done, _final("LATER-OWN")),
        },
        tools=[noop],
        max_retry_limit=2,
    )
    starts: list[str] = []
    starts_lock = threading.Lock()

    def on_start(source: Any, event: AgentExecutionStartedEvent) -> None:
        if getattr(event.agent, "role", None) == "Coworker":
            with starts_lock:
                starts.append(str(event.task.description))

    def watch_outer() -> None:
        run.drained()
        outer_done.set()

    def later(coworker: Agent) -> Any:
        threading.Thread(target=watch_outer, daemon=True).start()
        return _outcome(lambda: coworker.execute_task(_task(coworker, LATER_TASK)))

    async def alater(coworker: Agent) -> Any:
        threading.Thread(target=watch_outer, daemon=True).start()
        return await _aoutcome(coworker.aexecute_task(_task(coworker, LATER_TASK)))

    crewai_event_bus.on(AgentExecutionStartedEvent)(on_start)
    try:
        try:
            outcome = run.time_out_then(
                path, caught, later if path == "sync" else alater
            )
        finally:
            outer_done.set()
            assert run.drained()
        assert crewai_event_bus.flush(timeout=10)
    finally:
        crewai_event_bus.off(AgentExecutionStartedEvent, on_start)

    assert not isinstance(outcome, Exception), repr(outcome)
    assert str(outcome) == "LATER-OWN"
    with starts_lock:
        begun = list(starts)
    # One co-worker call began before the error; after it, only the later one.
    assert begun.count(SUB_TASK) == 1, begun
    assert begun.count(LATER_TASK) == 1, begun


def _row_coworker_end_of_call_state(
    recorder: _Recorder,
    caught: threading.Event,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    """L11: the abandoned co-worker call ends after a later one has ended."""
    later_done = threading.Event()
    run = _CoworkerRun(
        recorder,
        monkeypatch,
        {
            SUB_TASK: _held(later_done, _final("CO-FIRST")),
            LATER_TASK: lambda: _final("LATER-OWN"),
        },
    )

    def later(coworker: Agent) -> Any:
        try:
            return _outcome(
                lambda: _task(coworker, LATER_TASK).execute_sync(agent=coworker)
            )
        finally:
            later_done.set()

    async def alater(coworker: Agent) -> Any:
        try:
            return await _aoutcome(
                _task(coworker, LATER_TASK).aexecute_sync(agent=coworker)
            )
        finally:
            later_done.set()

    try:
        output = run.time_out_then(path, caught, later if path == "sync" else alater)
    finally:
        later_done.set()
        assert run.drained()

    assert not isinstance(output, Exception), repr(output)
    assert output.raw == "LATER-OWN"
    for label, messages in (
        ("task output", output.messages),
        ("co-worker", run.coworker.last_messages),
    ):
        text = _messages_text(messages)
        assert LATER_TASK in text, (label, text)
        assert SUB_TASK not in text, (label, text)


def _row_live_calls_on_one_agent(
    recorder: _Recorder,
    caught: threading.Event,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    """L12: C starts while A runs and has not timed out."""
    a_started = threading.Event()
    c_returned = threading.Event()

    def a_reply() -> str:
        a_started.set()
        c_returned.wait(HOLD_BOUND)
        return _final("A-OWN")

    @tool("tool_a")
    def tool_a(x: str) -> str:
        """A's tool."""
        return "A"

    @tool("tool_b")
    def tool_b(x: str) -> str:
        """C's tool."""
        return "B"

    llm = _ByTaskLLM(
        "shared", recorder, {"A-TASK": a_reply, "C-TASK": lambda: _final("C-OWN")}
    )
    agent = _agent(llm, [], max_execution_time=30)
    a_task = _task(agent, "A-TASK")
    c_task = _task(agent, "C-TASK", output_json=_Extracted)
    seen: dict[str, Any] = {}

    def held_state() -> dict[str, Any]:
        executor = seen["held"]
        return {
            "task": executor.task,
            "response_model": executor.response_model,
            "tools": sorted(t.name for t in executor.tools),
        }

    if path == "sync":
        a_out: dict[str, Any] = {}
        thread_a = threading.Thread(
            target=lambda: a_out.__setitem__(
                "A", _outcome(lambda: agent.execute_task(a_task, tools=[tool_a]))
            )
        )
        thread_a.start()
        try:
            assert a_started.wait(HOLD_BOUND)
            seen["held"] = agent.agent_executor
            c_out = _outcome(lambda: agent.execute_task(c_task, tools=[tool_b]))
            seen["after_c"] = held_state()
        finally:
            c_returned.set()
            thread_a.join(DRAIN_BOUND)
        a_result = a_out.get("A")
    else:

        async def scenario() -> tuple[Any, Any]:
            async def run_c() -> Any:
                try:
                    assert await asyncio.to_thread(a_started.wait, HOLD_BOUND)
                    seen["held"] = agent.agent_executor
                    out = await _aoutcome(agent.aexecute_task(c_task, tools=[tool_b]))
                    seen["after_c"] = held_state()
                    return out
                finally:
                    c_returned.set()

            return await asyncio.gather(
                _aoutcome(agent.aexecute_task(a_task, tools=[tool_a])), run_c()
            )

        try:
            a_result, c_out = asyncio.run(scenario())
        finally:
            c_returned.set()

    assert _already_running(c_out), repr(c_out)
    assert seen["after_c"] == {
        "task": a_task,
        "response_model": None,
        "tools": ["tool_a"],
    }, seen["after_c"]
    assert not isinstance(a_result, Exception), repr(a_result)
    assert str(a_result) == "A-OWN"
    for text in llm.inputs_for("A-TASK"):
        assert "C-TASK" not in text


def _row_two_calls_choose_after_one_retirement(
    recorder: _Recorder,
    caught: threading.Event,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    """L13: C and D, prepared before A's deadline, start together after it."""
    a_started = threading.Event()
    one_returned = threading.Event()
    both_returned = threading.Event()

    def a_reply() -> str:
        a_started.set()
        both_returned.wait(HOLD_BOUND)
        return _final("A-OWN")

    llm = _ByTaskLLM(
        "shared",
        recorder,
        {
            "A-TASK": a_reply,
            "C-TASK": _held(one_returned, _final("C-OWN")),
            "D-TASK": _held(one_returned, _final("D-OWN")),
        },
    )
    agent = _agent(llm, [])
    a_task = _task(agent, "A-TASK")
    late = {"C": _task(agent, "C-TASK"), "D": _task(agent, "D-TASK")}
    late_tasks = list(late.values())
    outs: dict[str, Any] = {}
    a_out: dict[str, Any] = {}

    def returned(name: str, outcome: Any) -> None:
        outs[name] = outcome
        one_returned.set()
        if len(outs) == 2:
            both_returned.set()

    if path == "sync":
        barrier = threading.Barrier(2)
        timed = agent._execute_with_timeout

        def gated(task_prompt: str, task: Task, timeout: int) -> Any:
            if task in late_tasks:
                caught.wait(HOLD_BOUND)
                barrier.wait(HOLD_BOUND)
            return timed(task_prompt, task, timeout)

        monkeypatch.setattr(agent, "_execute_with_timeout", gated)

        def run_a() -> None:
            try:
                agent.execute_task(a_task)
            except Exception as e:  # noqa: BLE001 - asserted below
                a_out["A"] = e
            finally:
                caught.set()

        def run_late(name: str) -> None:
            returned(name, _outcome(lambda: agent.execute_task(late[name])))

        thread_a = threading.Thread(target=run_a)
        thread_a.start()
        threads: list[threading.Thread] = []
        try:
            assert a_started.wait(HOLD_BOUND)
            threads = [threading.Thread(target=run_late, args=(n,)) for n in late]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(DRAIN_BOUND)
        finally:
            both_returned.set()
            thread_a.join(DRAIN_BOUND)
    else:
        atimed = agent._aexecute_with_timeout
        arrived: list[str] = []

        async def agated(task_prompt: str, task: Task, timeout: int) -> Any:
            if task in late_tasks:
                await asyncio.to_thread(caught.wait, HOLD_BOUND)
                arrived.append(task.description)
                while len(arrived) < 2:
                    await asyncio.sleep(0)
            return await atimed(task_prompt, task, timeout)

        monkeypatch.setattr(agent, "_aexecute_with_timeout", agated)

        async def scenario() -> None:
            async def run_a() -> None:
                try:
                    await agent.aexecute_task(a_task)
                except Exception as e:  # noqa: BLE001 - asserted below
                    a_out["A"] = e
                finally:
                    caught.set()

            async def run_late(name: str) -> None:
                assert await asyncio.to_thread(a_started.wait, HOLD_BOUND)
                returned(name, await _aoutcome(agent.aexecute_task(late[name])))

            await asyncio.wait_for(
                asyncio.gather(run_a(), run_late("C"), run_late("D")), DRAIN_BOUND
            )

        try:
            asyncio.run(scenario())
        finally:
            both_returned.set()
    assert llm.answered["A-TASK"].wait(DRAIN_BOUND)

    assert isinstance(a_out.get("A"), TimeoutError), a_out
    assert set(outs) == {"C", "D"}, outs
    own = [n for n, o in outs.items() if str(o) == f"{n}-OWN"]
    refused = [n for n, o in outs.items() if _already_running(o)]
    assert len(own) == 1 and len(refused) == 1, outs


def _row_coworker_chooses_in_stopped_run(
    recorder: _Recorder,
    caught: threading.Event,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    """L14: a co-worker call whose preparation spans the outer deadline."""
    from crewai.utilities.execution_stop import EXECUTION_STOPPED_MESSAGE

    run = _CoworkerRun(recorder, monkeypatch, {SUB_TASK: lambda: _final("CO")})
    coworker = run.coworker
    finalize = coworker._finalize_task_prompt
    prepared: list[Any] = []

    def slow_preparation(task_prompt: str, tools: Any, task: Task) -> str:
        prompt = finalize(task_prompt, tools, task)
        prepared.append(coworker.agent_executor)
        caught.wait(HOLD_BOUND)
        return prompt

    monkeypatch.setattr(coworker, "_finalize_task_prompt", slow_preparation)

    def later(_: Agent) -> None:
        return None

    async def alater(_: Agent) -> None:
        return None

    try:
        run.time_out_then(path, caught, later if path == "sync" else alater)
    finally:
        assert run.drained()

    assert len(prepared) == 1, prepared
    assert [str(e) for e in run.delegation_errors][:1] == [EXECUTION_STOPPED_MESSAGE]
    assert coworker.agent_executor is prepared[0]
    assert prepared[0] is not None and list(prepared[0].messages) == []
    assert run.llm.inputs == []


_OVERLAP_ROWS: dict[str, Callable[..., None]] = {
    "L4-L5": _row_coworker_later_call,
    "L6-L8": _row_prepared_before_step_running,
    "L7-L9": _row_prepared_before_step_ended,
    "L10": _row_stopped_coworker_reentry,
    "L11": _row_coworker_end_of_call_state,
    "L12": _row_live_calls_on_one_agent,
    "L13": _row_two_calls_choose_after_one_retirement,
    "L14": _row_coworker_chooses_in_stopped_run,
}


@pytest.mark.parametrize("path", ["sync", "async"])
@pytest.mark.parametrize("row", list(_OVERLAP_ROWS))
def test_v2l_later_and_overlapping_calls(
    recorder: _Recorder,
    caught: threading.Event,
    monkeypatch: pytest.MonkeyPatch,
    row: str,
    path: str,
) -> None:
    """V2-L: a call that starts after the error is independent of the run.

    Two live calls on one agent are still refused, and the stopped run
    enters, retries and writes nothing on an agent after the error.
    """
    _OVERLAP_ROWS[row](recorder, caught, monkeypatch, path)
