import asyncio
from collections.abc import Callable, Iterator
from concurrent.futures import Future
import contextvars
from dataclasses import dataclass
import json
import threading
import time
from typing import Any

import pytest

from crewai import Agent, Crew, Task
import crewai.agent.core as agent_core
from crewai.hooks.dispatch import HookAborted
from crewai.llms.base_llm import BaseLLM
from crewai.state.checkpoint_config import CheckpointConfig
from crewai.state.provider.json_provider import JsonProvider
from crewai.state.runtime import RuntimeState
from crewai.tools import tool
from crewai.tools.agent_tools.agent_tools import AgentTools
from crewai.utilities.agent_utils import message_content_text


# A gate the test never opens still opens after VALVE, so a run on code that
# waits for the worker ends instead of hanging until pytest's timeout.
VALVE = 10.0
LIMIT = 1
REPEATED = "I tried reusing the same input"


class Scene:
    def __init__(self) -> None:
        self.gates: list[Gate] = []
        self.threads: set[threading.Thread] = set()

    def gate(self) -> "Gate":
        gate = Gate(self)
        self.gates.append(gate)
        return gate

    def background(self, fn: Callable[[], Any]) -> "Background":
        run = Background(fn)
        self.threads.add(run.thread)
        return run

    def close(self) -> None:
        for gate in self.gates:
            gate.open()
        deadline = time.monotonic() + 3 * VALVE
        alive = []
        for thread in list(self.threads):
            if thread is threading.current_thread():
                continue
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                alive.append(thread.name)
        assert not alive, f"threads still running after the test released them: {alive}"


class Gate:
    def __init__(self, scene: Scene) -> None:
        self._scene = scene
        self.reached = threading.Event()
        self._open = threading.Event()

    def wait(self) -> None:
        self._scene.threads.add(threading.current_thread())
        self.reached.set()
        self._open.wait(timeout=VALVE)

    def open(self) -> None:
        self._open.set()

    @property
    def is_open(self) -> bool:
        return self._open.is_set()


class Background:
    def __init__(self, fn: Callable[[], Any]) -> None:
        self._future: Future[Any] = Future()
        self.thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self.thread.start()

    def _run(self, fn: Callable[[], Any]) -> None:
        try:
            self._future.set_result(fn())
        except BaseException as e:  # noqa: BLE001
            self._future.set_exception(e)

    def result(self) -> Any:
        return self._future.result(timeout=3 * VALVE)


@pytest.fixture
def scene() -> Iterator[Scene]:
    current = Scene()
    yield current
    current.close()


@dataclass
class Call:
    task: str | None
    text: str


class StubLLM(BaseLLM):
    def __init__(self, reply: Callable[[Call], str]) -> None:
        super().__init__(model="stub-model")
        self._reply = reply
        self.calls: list[Call] = []

    def call(  # type: ignore[override]
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
        text = (
            messages
            if isinstance(messages, str)
            else "\n".join(message_content_text(m) for m in messages)
        )
        call = Call(task=getattr(from_task, "description", None), text=text)
        self.calls.append(call)
        return self._reply(call)

    async def acall(  # type: ignore[override]
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
        return await asyncio.to_thread(
            self.call, messages, tools, callbacks, available_functions, from_task
        )

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192

    def calls_for(self, marker: str) -> list[Call]:
        return [c for c in self.calls if c.task is not None and marker in c.task]


def final(answer: str) -> str:
    return f"Thought: done\nFinal Answer: {answer}"


def act(tool_name: str, arguments: dict[str, Any] | None = None) -> str:
    return (
        f"Thought: use the tool\nAction: {tool_name}\n"
        f"Action Input: {json.dumps(arguments or {})}"
    )


def delegate(task: str, coworker: str) -> str:
    return act(
        "Delegate work to coworker",
        {"task": task, "context": "none", "coworker": coworker},
    )


def recording_tool(name: str, runs: list[str], result: str | None = None) -> Any:
    def run() -> str:
        runs.append(name)
        return result or f"{name}-result"

    run.__doc__ = f"The {name} tool."
    return tool(name)(run)


def make_agent(llm: StubLLM, role: str = "worker", **fields: Any) -> Agent:
    fields.setdefault("max_retry_limit", 0)
    return Agent(role=role, goal="g", backstory="b", llm=llm, verbose=False, **fields)


def make_task(description: str, agent: Agent, **fields: Any) -> Task:
    return Task(description=description, expected_output="x", agent=agent, **fields)


def messages_text(messages: list[Any]) -> str:
    return "\n".join(message_content_text(m) for m in messages)


def restored_agent(tmp_path: Any, llm: StubLLM, **fields: Any) -> Agent:
    source = make_agent(StubLLM(lambda call: final("pre")), role="restorable")
    source.execute_task(make_task("TASK-PRE", source))
    state = RuntimeState(root=[source])
    state._provider = JsonProvider()
    location = state.checkpoint(str(tmp_path))
    agent = Agent.from_checkpoint(CheckpointConfig(restore_from=location))
    assert agent.agent_executor is not None and agent.agent_executor._resuming, (
        "precondition: the restored agent's executor resumes"
    )
    agent.llm = llm
    for name, value in fields.items():
        setattr(agent, name, value)
    return agent


def raises_timeout_at_the_limit(call: Callable[[], Any]) -> float:
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        call()
    elapsed = time.monotonic() - started
    assert elapsed < LIMIT + 2, (
        f"TimeoutError came {elapsed:.2f}s after the call, past the limit of {LIMIT}s"
    )
    return elapsed


def test_timeout_is_raised_at_the_limit_counted_from_the_end_of_preparation(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = scene.gate()
    worker_returned = threading.Event()

    def reply(call: Call) -> str:
        gate.wait()
        worker_returned.set()
        return final("one")

    agent = make_agent(StubLLM(reply), max_execution_time=LIMIT)
    prepared: list[float] = []
    original_prompt = Task.prompt

    def slow_prompt(self: Task) -> str:
        time.sleep(1.0)
        prepared.append(time.monotonic())
        return original_prompt(self)

    monkeypatch.setattr(Task, "prompt", slow_prompt)

    with pytest.raises(TimeoutError):
        agent.execute_task(make_task("TASK-ONE wait", agent))
    raised = time.monotonic()

    assert not worker_returned.is_set(), (
        "TimeoutError reached the caller only after the worker's model call returned"
    )
    since_preparation = raised - prepared[0]
    assert LIMIT <= since_preparation <= LIMIT + 2, (
        f"TimeoutError came {since_preparation:.2f}s after preparation ended, "
        f"not at the {LIMIT}s limit"
    )


def _blocked_worker_agent(
    scene: Scene, runs: list[str]
) -> tuple[Agent, StubLLM, Gate, threading.Event, threading.Event]:
    gate = scene.gate()
    returned = threading.Event()
    finished = threading.Event()

    def reply(call: Call) -> str:
        if call.task and "TASK-ONE" in call.task:
            if len(llm.calls_for("TASK-ONE")) == 1:
                gate.wait()
                returned.set()
                return act("tool_a")
            finished.set()
            return final("answer-one")
        if call.task and "TASK-TWO" in call.task:
            return final("answer-two")
        return final("answer-other")

    llm = StubLLM(reply)
    agent = make_agent(
        llm, tools=[recording_tool("tool_a", runs)], max_execution_time=LIMIT
    )
    raises_timeout_at_the_limit(
        lambda: agent.execute_task(make_task("TASK-ONE use a", agent))
    )
    assert not returned.is_set(), "precondition: the worker is still blocked"
    return agent, llm, gate, returned, finished


def test_later_execute_task_on_the_same_agent_returns_its_own_answer_while_the_worker_runs(
    scene: Scene,
) -> None:
    runs: list[str] = []
    agent, llm, gate, returned, _ = _blocked_worker_agent(scene, runs)

    result = agent.execute_task(
        make_task("TASK-TWO say two", agent), tools=[recording_tool("tool_b", runs)]
    )

    assert result == "answer-two"
    assert not returned.is_set(), "the later call returned only after the worker's"


def test_later_aexecute_task_on_the_same_agent_returns_its_own_answer_while_the_worker_runs(
    scene: Scene,
) -> None:
    runs: list[str] = []
    agent, llm, gate, returned, _ = _blocked_worker_agent(scene, runs)

    result = asyncio.run(
        agent.aexecute_task(
            make_task("TASK-TWO say two async", agent),
            tools=[recording_tool("tool_b", runs)],
        )
    )

    assert result == "answer-two"
    assert not returned.is_set(), "the later call returned only after the worker's"


def test_the_worker_goes_on_with_its_own_task_and_tools_after_later_preparations(
    scene: Scene,
) -> None:
    runs: list[str] = []
    agent, llm, gate, returned, finished = _blocked_worker_agent(scene, runs)
    tool_b = recording_tool("tool_b", runs)

    while_running = agent.execute_task(make_task("TASK-TWO say two", agent), tools=[tool_b])
    asyncio.run(
        agent.aexecute_task(make_task("TASK-THREE async", agent), tools=[tool_b])
    )
    Crew(agents=[agent], tasks=[make_task("TASK-FOUR crew", agent)]).kickoff()
    assert not returned.is_set(), "the later calls returned only after the worker's"

    gate.open()
    assert finished.wait(VALVE), "the worker did not finish after its release"

    after_release = llm.calls_for("TASK-ONE")[1:]
    assert after_release, "the worker made no model call after its release"
    assert all(c.task == "TASK-ONE use a" for c in after_release)
    for marker in ("TASK-TWO", "TASK-THREE", "TASK-FOUR"):
        assert not any(marker in c.text for c in after_release), (
            f"the worker's messages carry the later {marker}"
        )
    assert runs == ["tool_a"], f"tools run: {runs}"

    after_end = agent.execute_task(make_task("TASK-TWO say two", agent), tools=[tool_b])
    assert after_end == while_running == "answer-two"


def test_later_kickoff_on_a_restored_agent_returns_while_its_timed_out_worker_runs(
    scene: Scene, tmp_path: Any
) -> None:
    gate = scene.gate()

    def reply(call: Call) -> str:
        if call.task and "TASK-R1" in call.task:
            gate.wait()
            return final("r-one")
        return final("r-kickoff")

    agent = restored_agent(tmp_path, StubLLM(reply), max_execution_time=LIMIT)
    raises_timeout_at_the_limit(
        lambda: agent.execute_task(make_task("TASK-R1 block", agent))
    )
    assert not gate.is_open, "precondition: the worker is still blocked"

    output = agent.kickoff("TASK-K say kickoff")

    assert output.raw == "r-kickoff"
    assert not gate.is_open


def test_later_execute_task_on_a_coworker_the_timed_out_worker_delegated_to(
    scene: Scene,
) -> None:
    gate = scene.gate()
    runs: list[str] = []

    delegated_finished = threading.Event()

    def coworker_reply(call: Call) -> str:
        if call.task and "TASK-DELEG" in call.task:
            if len(coworker_llm.calls_for("TASK-DELEG")) == 1:
                gate.wait()
                return act("tool_c")
            delegated_finished.set()
            return final("coworker-one")
        return final("coworker-two")

    coworker_llm = StubLLM(coworker_reply)
    coworker = make_agent(
        coworker_llm, role="coworker", tools=[recording_tool("tool_c", runs)]
    )

    def manager_reply(call: Call) -> str:
        if "coworker-one" in call.text:
            return final("manager-done")
        return delegate("TASK-DELEG do it", "coworker")

    manager = make_agent(
        StubLLM(manager_reply),
        role="manager",
        tools=AgentTools(agents=[coworker]).tools(),
        max_execution_time=LIMIT,
    )
    raises_timeout_at_the_limit(
        lambda: manager.execute_task(make_task("TASK-M delegate", manager))
    )
    assert gate.reached.is_set() and not gate.is_open, (
        "precondition: the coworker's delegated model call started and is blocked"
    )

    result = coworker.execute_task(make_task("TASK-C2 say two", coworker))

    assert result == "coworker-two"
    gate.open()
    assert delegated_finished.wait(VALVE), "the delegated run did not finish"
    after_release = coworker_llm.calls_for("TASK-DELEG")[1:]
    assert after_release and all(c.task == "TASK-DELEG do it" for c in after_release)
    assert not any("TASK-C2" in c.text for c in after_release)
    assert runs == ["tool_c"], f"tools run: {runs}"


def test_abandoned_delegation_reaching_a_coworker_whose_later_execution_runs(
    scene: Scene,
) -> None:
    manager_gate = scene.gate()
    later_gate = scene.gate()
    delegated_done = threading.Event()
    runs: list[str] = []

    def coworker_reply(call: Call) -> str:
        if call.task and "TASK-DELEG" in call.task:
            delegated_done.set()
            return final("deleg-answer")
        if call.task and "TASK-Y" in call.task:
            if len(coworker_llm.calls_for("TASK-Y")) == 1:
                later_gate.wait()
                return act("tool_y")
            return final("y-answer")
        return final("other")

    coworker_llm = StubLLM(coworker_reply)
    coworker = make_agent(coworker_llm, role="coworker")

    manager_done = threading.Event()

    def manager_reply(call: Call) -> str:
        if len(manager_llm.calls) > 1:
            manager_done.set()
            return final("manager-done")
        manager_gate.wait()
        return delegate("TASK-DELEG do it", "coworker")

    manager_llm = StubLLM(manager_reply)
    manager = make_agent(
        manager_llm,
        role="manager",
        tools=AgentTools(agents=[coworker]).tools(),
        max_execution_time=LIMIT,
    )
    raises_timeout_at_the_limit(
        lambda: manager.execute_task(make_task("TASK-M delegate", manager))
    )
    assert not manager_gate.is_open, "precondition: the manager's worker is blocked"

    later = scene.background(
        lambda: coworker.execute_task(
            make_task("TASK-Y later", coworker), tools=[recording_tool("tool_y", runs)]
        )
    )
    assert later_gate.reached.wait(VALVE), "the later execution never reached its model call"
    manager_gate.open()
    assert delegated_done.wait(VALVE), "the abandoned delegation never ran"
    later_gate.open()

    assert later.result() == "y-answer"
    assert runs == ["tool_y"], f"tools run: {runs}"
    assert all(c.task == "TASK-DELEG do it" for c in coworker_llm.calls_for("TASK-DELEG"))
    assert all(c.task == "TASK-Y later" for c in coworker_llm.calls_for("TASK-Y"))
    assert manager_done.wait(VALVE), "the abandoned worker never got its delegation back"
    assert any("deleg-answer" in c.text for c in manager_llm.calls), (
        "the abandoned delegation was refused"
    )


def _hold_after_preparation(
    monkeypatch: pytest.MonkeyPatch, marker: str, gate: Gate
) -> None:
    original = Agent._emit_skill_usage

    def held(self: Agent, task: Task) -> None:
        original(self, task)
        if marker in task.description:
            gate.wait()

    monkeypatch.setattr(Agent, "_emit_skill_usage", held)


@pytest.mark.parametrize("held", ["delegation", "later"])
def test_two_preparations_on_a_coworker_inside_each_others_span(
    scene: Scene, monkeypatch: pytest.MonkeyPatch, held: str
) -> None:
    manager_gate = scene.gate()
    hold = scene.gate()
    runs: list[str] = []

    def coworker_reply(call: Call) -> str:
        for marker, tool_name, answer in (
            ("TASK-DELEG", "tool_d", "deleg-answer"),
            ("TASK-Y", "tool_y", "y-answer"),
        ):
            if call.task and marker in call.task:
                if len(coworker_llm.calls_for(marker)) == 1:
                    return act(tool_name)
                return final(answer)
        return final("other")

    coworker_llm = StubLLM(coworker_reply)
    coworker = make_agent(
        coworker_llm, role="coworker", tools=[recording_tool("tool_d", runs)]
    )
    tool_y = recording_tool("tool_y", runs)

    manager_done = threading.Event()

    def manager_reply(call: Call) -> str:
        if len(manager_llm.calls) > 1:
            manager_done.set()
            return final("manager-done")
        if held == "later":
            manager_gate.wait()
        return delegate("TASK-DELEG do it", "coworker")

    manager_llm = StubLLM(manager_reply)
    manager = make_agent(
        manager_llm,
        role="manager",
        tools=AgentTools(agents=[coworker]).tools(),
        max_execution_time=LIMIT,
    )
    _hold_after_preparation(
        monkeypatch, "TASK-DELEG" if held == "delegation" else "TASK-Y", hold
    )
    later_task = make_task("TASK-Y later", coworker)

    if held == "delegation":
        raises_timeout_at_the_limit(
            lambda: manager.execute_task(make_task("TASK-M delegate", manager))
        )
        assert hold.reached.is_set() and not hold.is_open, (
            "precondition: the abandoned delegation is held after its preparation"
        )
        later_result = coworker.execute_task(later_task, tools=[tool_y])
        hold.open()
    else:
        raises_timeout_at_the_limit(
            lambda: manager.execute_task(make_task("TASK-M delegate", manager))
        )
        later = scene.background(
            lambda: coworker.execute_task(later_task, tools=[tool_y])
        )
        assert hold.reached.wait(VALVE), "the later execution never reached its hold"
        manager_gate.open()
        deadline = time.monotonic() + VALVE
        while len(coworker_llm.calls_for("TASK-DELEG")) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        hold.open()
        later_result = later.result()

    assert manager_done.wait(VALVE), "the abandoned worker never got its delegation back"
    assert later_result == "y-answer"
    assert sorted(runs) == ["tool_d", "tool_y"], f"tools run: {runs}"
    assert all(c.task == "TASK-DELEG do it" for c in coworker_llm.calls_for("TASK-DELEG"))
    assert all(c.task == "TASK-Y later" for c in coworker_llm.calls_for("TASK-Y"))
    assert any("deleg-answer" in c.text for c in manager_llm.calls), (
        "the abandoned delegation did not return its own answer"
    )


def test_crew_with_two_async_tasks_on_one_agent_returns_each_tasks_own_output(
    scene: Scene,
) -> None:
    runs: list[str] = []
    in_flight = {"TASK-A": threading.Event(), "TASK-B": threading.Event()}

    def reply(call: Call) -> str:
        for marker, other, tool_name in (
            ("TASK-A", "TASK-B", "tool_a"),
            ("TASK-B", "TASK-A", "tool_b"),
        ):
            if call.task and marker in call.task:
                calls = llm.calls_for(marker)
                if len(calls) == 1:
                    in_flight[marker].set()
                    in_flight[other].wait(VALVE)
                    return act(tool_name)
                return final(f"answer-{marker}")
        return final("summary")

    llm = StubLLM(reply)
    agent = make_agent(llm)
    task_a = make_task(
        "TASK-A first", agent, async_execution=True, tools=[recording_tool("tool_a", runs)]
    )
    task_b = make_task(
        "TASK-B second", agent, async_execution=True, tools=[recording_tool("tool_b", runs)]
    )
    summary = make_task("SUMMARY", agent, context=[task_a, task_b])

    Crew(agents=[agent], tasks=[task_a, task_b, summary]).kickoff()

    assert in_flight["TASK-A"].is_set() and in_flight["TASK-B"].is_set()
    assert task_a.output is not None and task_a.output.raw == "answer-TASK-A"
    assert task_b.output is not None and task_b.output.raw == "answer-TASK-B"
    assert sorted(runs) == ["tool_a", "tool_b"], f"tools run: {runs}"
    a_text = messages_text(task_a.output.messages)
    b_text = messages_text(task_b.output.messages)
    assert "TASK-A" in a_text and "TASK-B" not in a_text
    assert "TASK-B" in b_text and "TASK-A" not in b_text


def test_execution_after_another_one_invoked_but_before_it_ended(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    hold = scene.gate()
    two_gate = scene.gate()
    runs: list[str] = []

    def reply(call: Call) -> str:
        for marker, tool_name in (("TASK-ONE", "tool_one"), ("TASK-TWO", "tool_two")):
            if call.task and marker in call.task:
                calls = llm.calls_for(marker)
                if len(calls) == 1:
                    if marker == "TASK-TWO":
                        two_gate.wait()
                    return act(tool_name)
                return final(f"answer-{marker}")
        return final("other")

    llm = StubLLM(reply)
    agent = make_agent(llm)
    original_save = agent_core.save_last_messages
    one_thread: list[threading.Thread] = []

    def held_save(*args: Any, **kwargs: Any) -> None:
        if one_thread and threading.current_thread() is one_thread[0]:
            hold.wait()
        original_save(*args, **kwargs)

    monkeypatch.setattr(agent_core, "save_last_messages", held_save)
    task_one = make_task("TASK-ONE first", agent, tools=[recording_tool("tool_one", runs)])
    task_two = make_task("TASK-TWO second", agent, tools=[recording_tool("tool_two", runs)])

    one = scene.background(lambda: task_one.execute_sync(agent=agent))
    one_thread.append(one.thread)
    assert hold.reached.wait(VALVE), "TASK-ONE never reached the end of its execution"
    two = scene.background(lambda: task_two.execute_sync(agent=agent))
    assert two_gate.reached.wait(VALVE), "TASK-TWO never reached its model call"
    hold.open()
    one_output = one.result()
    two_gate.open()
    two_output = two.result()

    assert one_output.raw == "answer-TASK-ONE"
    one_text = messages_text(one_output.messages)
    assert "TASK-ONE" in one_text and "TASK-TWO" not in one_text
    assert two_output.raw == "answer-TASK-TWO"
    assert sorted(runs) == ["tool_one", "tool_two"], f"tools run: {runs}"
    assert all(c.task == "TASK-TWO second" for c in llm.calls_for("TASK-TWO"))
    assert all(c.task == "TASK-ONE first" for c in llm.calls_for("TASK-ONE"))


def test_kickoff_guardrail_reinvoke_keeps_its_executor_while_another_execution_runs(
    scene: Scene, tmp_path: Any
) -> None:
    guardrail_gate = scene.gate()
    runs: list[str] = []
    verdicts: list[bool] = []

    def reply(call: Call) -> str:
        if "TASK-L" in call.text:
            if call.task and "TASK-L" in call.task and len(llm.calls_for("TASK-L")) == 1:
                return act("tool_l")
            return final("later-answer")
        return final("kick-answer")

    def guardrail(output: Any) -> tuple[bool, Any]:
        if not verdicts:
            verdicts.append(False)
            guardrail_gate.wait()
            return False, "try again"
        verdicts.append(True)
        return True, output.raw

    llm = StubLLM(reply)
    agent = restored_agent(tmp_path, llm, guardrail=guardrail)

    kickoff = scene.background(lambda: agent.kickoff("TASK-K kick"))
    assert guardrail_gate.reached.wait(VALVE), "the kickoff never reached its guardrail"
    later = agent.execute_task(
        make_task("TASK-L later", agent), tools=[recording_tool("tool_l", runs)]
    )
    calls_before_reinvoke = len(llm.calls)
    guardrail_gate.open()
    output = kickoff.result()

    assert later == "later-answer"
    assert runs == ["tool_l"], f"tools run: {runs}"
    reinvoke_calls = llm.calls[calls_before_reinvoke:]
    assert reinvoke_calls, "the kickoff's guardrail retry made no model call"
    assert not any(c.task == "TASK-L later" for c in reinvoke_calls), (
        "the kickoff's re-invoke ran under the later execution's task"
    )
    assert output.raw == "kick-answer"
    kick_text = messages_text(output.messages)
    assert "TASK-K" in kick_text and "TASK-L" not in kick_text


def test_crew_setup_while_an_execution_holds_the_executor_before_its_invoke(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    hold = scene.gate()
    runs: list[str] = []
    llm = StubLLM(lambda call: _own_tool_reply(llm, call))
    agent = make_agent(llm, tools=[recording_tool("tool_agent", runs)])
    _hold_after_preparation(monkeypatch, "TASK-E", hold)

    execution = scene.background(
        lambda: agent.execute_task(
            make_task("TASK-E own", agent), tools=[recording_tool("tool_e", runs)]
        )
    )
    assert hold.reached.wait(VALVE), "TASK-E never reached its hold"
    Crew(agents=[agent], tasks=[make_task("TASK-CREW run", agent)]).kickoff()
    hold.open()

    assert execution.result() == "answer-TASK-E"
    assert "tool_e" in runs, f"tools run: {runs}"
    assert all(c.task == "TASK-E own" for c in llm.calls_for("TASK-E"))


def test_execution_preparing_while_crew_setup_rebinds_the_executor(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    rebind_gate = scene.gate()
    e_gate = scene.gate()
    runs: list[str] = []

    def reply(call: Call) -> str:
        if call.task and "TASK-E" in call.task and len(llm.calls_for("TASK-E")) == 1:
            e_gate.wait()
        return _own_tool_reply(llm, call)

    llm = StubLLM(reply)
    agent = make_agent(llm, tools=[recording_tool("tool_agent", runs)])
    agent.execute_task(make_task("TASK-PRE warm", agent))
    original = Agent._update_executor_parameters
    crew_thread: list[threading.Thread] = []

    def held_rebind(self: Agent, *args: Any, **kwargs: Any) -> None:
        if (
            kwargs.get("task") is None
            and crew_thread
            and threading.current_thread() is crew_thread[0]
            and not rebind_gate.reached.is_set()
        ):
            rebind_gate.wait()
        original(self, *args, **kwargs)

    monkeypatch.setattr(Agent, "_update_executor_parameters", held_rebind)
    crew = Crew(agents=[agent], tasks=[make_task("TASK-CREW run", agent)])
    crew_run = scene.background(crew.kickoff)
    crew_thread.append(crew_run.thread)
    assert rebind_gate.reached.wait(VALVE), "crew setup never reached its rebind"

    execution = scene.background(
        lambda: agent.execute_task(
            make_task("TASK-E own", agent), tools=[recording_tool("tool_e", runs)]
        )
    )
    assert e_gate.reached.wait(VALVE), "TASK-E never reached its model call"
    rebind_gate.open()
    crew_run.result()
    e_gate.open()

    assert execution.result() == "answer-TASK-E"
    assert "tool_e" in runs, f"tools run: {runs}"
    assert all(c.task == "TASK-E own" for c in llm.calls_for("TASK-E"))


def _own_tool_reply(llm: StubLLM, call: Call) -> str:
    if call.task and "TASK-E" in call.task:
        if len(llm.calls_for("TASK-E")) == 1:
            return act("tool_e")
        return final("answer-TASK-E")
    return final("answer-other")


class AbortingMemory:
    read_only = True

    def recall(self, *args: Any, **kwargs: Any) -> Any:
        raise HookAborted("recall denied")


@pytest.mark.parametrize("run", ["sync", "async"])
def test_kickoff_failing_after_its_preparation_leaves_the_restored_executor_reusable(
    tmp_path: Any, run: str
) -> None:
    agent = restored_agent(tmp_path, StubLLM(lambda call: final("answer")))
    restored_executor = agent.agent_executor
    agent.memory = AbortingMemory()  # type: ignore[assignment]

    with pytest.raises(HookAborted):
        if run == "sync":
            agent.kickoff("TASK-K kick")
        else:
            asyncio.run(agent.kickoff_async("TASK-K kick"))
    agent.memory = None

    assert agent.execute_task(make_task("TASK-AFTER", agent)) == "answer"
    assert agent.agent_executor is restored_executor, (
        "the failed kickoff left the restored executor taken; a fresh one replaced it"
    )


@pytest.mark.parametrize("run", ["sync", "async", "timed"])
def test_failed_attempt_and_finished_execution_leave_the_executor_reusable(
    run: str,
) -> None:
    failures: list[str] = []

    def reply(call: Call) -> str:
        if call.task and "TASK-RETRY" in call.task and not failures:
            failures.append("raised")
            raise ValueError("transient model error")
        return final("answer")

    timed = {"max_execution_time": 30} if run == "timed" else {}
    agent = make_agent(StubLLM(reply), max_retry_limit=1, **timed)
    agent.execute_task(make_task("TASK-PRE warm", agent))
    held = agent.agent_executor

    def execute(task: Task) -> Any:
        if run == "async":
            return asyncio.run(agent.aexecute_task(task))
        return agent.execute_task(task)

    assert execute(make_task("TASK-RETRY once", agent)) == "answer"
    assert failures == ["raised"], "precondition: the first attempt failed"
    assert agent.agent_executor is held, "the retry built a fresh executor"
    assert execute(make_task("TASK-NEXT", agent)) == "answer"
    assert agent.agent_executor is held, (
        "the finished execution left its executor taken; a fresh one replaced it"
    )


def test_timed_worker_held_before_its_invoke_keeps_its_executor(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    hold = scene.gate()
    runs: list[str] = []

    worker_finished = threading.Event()

    def reply(call: Call) -> str:
        for marker, tool_name in (("TASK-ONE", "tool_a"), ("TASK-TWO", "tool_b")):
            if call.task and marker in call.task:
                if len(llm.calls_for(marker)) == 1:
                    return act(tool_name)
                if marker == "TASK-ONE":
                    worker_finished.set()
                return final(f"answer-{marker}")
        return final("other")

    llm = StubLLM(reply)
    agent = make_agent(
        llm, tools=[recording_tool("tool_a", runs)], max_execution_time=LIMIT
    )
    original = Agent._execute_without_timeout

    def held(self: Agent, task_prompt: str, task: Task) -> Any:
        if "TASK-ONE" in task.description:
            hold.wait()
        return original(self, task_prompt, task)

    monkeypatch.setattr(Agent, "_execute_without_timeout", held)
    raises_timeout_at_the_limit(
        lambda: agent.execute_task(make_task("TASK-ONE use a", agent))
    )
    assert hold.reached.is_set() and not hold.is_open, (
        "precondition: the worker is held before its invoke"
    )

    later = agent.execute_task(
        make_task("TASK-TWO use b", agent), tools=[recording_tool("tool_b", runs)]
    )

    assert later == "answer-TASK-TWO"
    assert runs == ["tool_b"], f"tools run: {runs}"
    hold.open()
    assert worker_finished.wait(VALVE), "the worker did not finish after its release"
    assert all(c.task == "TASK-ONE use a" for c in llm.calls_for("TASK-ONE"))
    assert not any("TASK-TWO" in c.text for c in llm.calls_for("TASK-ONE"))
    assert runs == ["tool_b", "tool_a"], f"tools run: {runs}"


def test_timed_execution_whose_worker_thread_cannot_start_leaves_the_executor_reusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_agent(StubLLM(lambda call: final("answer")), max_execution_time=30)
    agent.execute_task(make_task("TASK-PRE warm", agent))
    held = agent.agent_executor

    class UnstartableThread(threading.Thread):
        def start(self) -> None:
            raise RuntimeError("can't start new thread")

    with monkeypatch.context() as patch:
        patch.setattr(agent_core, "Thread", UnstartableThread)
        with pytest.raises(RuntimeError, match="can't start new thread"):
            agent.execute_task(make_task("TASK-NO-THREAD", agent))

    assert agent.execute_task(make_task("TASK-NEXT", agent)) == "answer"
    assert agent.agent_executor is held, (
        "the execution whose worker thread did not start left the executor "
        "held; a fresh one replaced it"
    )


def _guardrail_failing_once() -> Callable[[Any], tuple[bool, Any]]:
    verdicts: list[bool] = []

    def guardrail(output: Any) -> tuple[bool, Any]:
        verdicts.append(bool(verdicts))
        return (True, output.raw) if verdicts[-1] else (False, "try again")

    return guardrail


@pytest.mark.parametrize(
    "construction", ["sync", "async", "sync-guardrail-retry", "async-guardrail-retry"]
)
def test_task_output_carries_its_own_executions_messages_when_another_ends_first(
    scene: Scene, monkeypatch: pytest.MonkeyPatch, construction: str
) -> None:
    hold = scene.gate()
    llm = StubLLM(
        lambda call: final("answer-a") if "TASK-A" in call.text else final("answer-b")
    )
    agent = make_agent(llm)
    retry = construction.endswith("guardrail-retry")
    task_a = make_task(
        "TASK-A first", agent, **({"guardrail": _guardrail_failing_once()} if retry else {})
    )
    task_b = make_task("TASK-B second", agent)

    def hold_for_a(original: Callable[..., Any]) -> Callable[..., Any]:
        def held(self: Task, *args: Any, **kwargs: Any) -> Any:
            if self is task_a:
                hold.wait()
            return original(self, *args, **kwargs)

        return held

    if not retry:
        monkeypatch.setattr(
            Task, "_post_agent_execution", hold_for_a(Task._post_agent_execution)
        )
    elif construction.startswith("sync"):
        monkeypatch.setattr(Task, "_export_output", hold_for_a(Task._export_output))
    else:
        original_aexport = Task._aexport_output

        async def held_aexport(self: Task, *args: Any, **kwargs: Any) -> Any:
            if self is task_a:
                hold.wait()
            return await original_aexport(self, *args, **kwargs)

        monkeypatch.setattr(Task, "_aexport_output", held_aexport)

    if construction.startswith("sync"):
        run_a = scene.background(lambda: task_a.execute_sync(agent=agent))
    else:
        run_a = scene.background(lambda: asyncio.run(task_a.aexecute_sync(agent=agent)))
    assert hold.reached.wait(VALVE), "task A's output build was never reached"
    output_b = task_b.execute_sync(agent=agent)
    hold.open()
    output_a = run_a.result()

    assert output_a.raw == "answer-a" and output_b.raw == "answer-b"
    a_text = messages_text(output_a.messages)
    assert "TASK-A" in a_text and "TASK-B" not in a_text, (
        "task A's output carries task B's messages"
    )
    b_text = messages_text(output_b.messages)
    assert "TASK-B" in b_text and "TASK-A" not in b_text


def test_task_output_on_a_coworker_is_not_filled_by_the_abandoned_delegated_execution(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    deleg_gate = scene.gate()
    hold = scene.gate()

    def coworker_reply(call: Call) -> str:
        if call.task and "TASK-DELEG" in call.task:
            deleg_gate.wait()
            return final("deleg-answer")
        return final("c2-answer")

    coworker = make_agent(StubLLM(coworker_reply), role="coworker")

    delegation_returned = threading.Event()

    def manager_reply(call: Call) -> str:
        if "deleg-answer" in call.text:
            delegation_returned.set()
            return final("manager-done")
        return delegate("TASK-DELEG do it", "coworker")

    manager = make_agent(
        StubLLM(manager_reply),
        role="manager",
        tools=AgentTools(agents=[coworker]).tools(),
        max_execution_time=LIMIT,
    )
    raises_timeout_at_the_limit(
        lambda: manager.execute_task(make_task("TASK-M delegate", manager))
    )
    assert deleg_gate.reached.is_set() and not deleg_gate.is_open, (
        "precondition: the delegated execution is blocked in its model call"
    )
    task_c2 = make_task("TASK-C2 later", coworker)
    original = Task._post_agent_execution

    def held(self: Task, agent: Any) -> None:
        if self is task_c2:
            hold.wait()
        original(self, agent)

    monkeypatch.setattr(Task, "_post_agent_execution", held)

    run_c2 = scene.background(lambda: task_c2.execute_sync(agent=coworker))
    assert hold.reached.wait(VALVE), "the later task's output build was never reached"
    deleg_gate.open()
    assert delegation_returned.wait(VALVE), "the abandoned delegated execution did not end"
    hold.open()
    output = run_c2.result()

    assert output.raw == "c2-answer"
    text = messages_text(output.messages)
    assert "TASK-C2" in text and "TASK-DELEG" not in text


@pytest.mark.parametrize("nested", ["another-agent", "the-tasks-own-agent"])
def test_task_output_is_not_filled_by_an_execution_in_an_abandoned_worker_below_it(
    scene: Scene, monkeypatch: pytest.MonkeyPatch, nested: str
) -> None:
    nested_gate = scene.gate()
    nested_returned = threading.Event()
    hold = scene.gate()
    nested_role = "third" if nested == "another-agent" else "manager"
    nested_running_at_delegation_result: list[bool] = []

    def nested_reply(call: Call) -> str:
        nested_gate.wait()
        nested_returned.set()
        return final("nested-answer")

    def manager_reply(call: Call) -> str:
        if call.task and "TASK-NESTED" in call.task:
            return nested_reply(call)
        if len(manager_llm.calls_for("TASK-T")) > 1:
            nested_running_at_delegation_result.append(not nested_returned.is_set())
            return final("answer-t")
        return delegate("TASK-C do it", "coworker")

    nested_returned_to_coworker = threading.Event()

    def coworker_reply(call: Call) -> str:
        if "nested-answer" in call.text:
            nested_returned_to_coworker.set()
            return final("c-answer")
        return delegate("TASK-NESTED below", nested_role)

    manager_llm = StubLLM(manager_reply)
    manager = make_agent(manager_llm, role="manager")
    if nested == "another-agent":
        nested_agent = make_agent(StubLLM(nested_reply), role="third")
    else:
        nested_agent = manager
    coworker = make_agent(
        StubLLM(coworker_reply),
        role="coworker",
        tools=AgentTools(agents=[nested_agent]).tools(),
        max_execution_time=LIMIT,
    )
    manager.tools = AgentTools(agents=[coworker]).tools()
    task_t = make_task("TASK-T top", manager)
    original = Task._post_agent_execution

    def held(self: Task, agent: Any) -> None:
        if self is task_t:
            hold.wait()
        original(self, agent)

    monkeypatch.setattr(Task, "_post_agent_execution", held)

    run_t = scene.background(lambda: task_t.execute_sync(agent=manager))
    assert hold.reached.wait(3 * VALVE), "task T's output build was never reached"
    assert nested_running_at_delegation_result == [True], (
        "precondition: the coworker's timeout reached task T while the nested "
        "execution was still blocked in the abandoned worker"
    )
    nested_gate.open()
    assert nested_returned_to_coworker.wait(VALVE), "the nested execution did not end"
    hold.open()
    output = run_t.result()

    assert output.raw == "answer-t"
    text = messages_text(output.messages)
    assert "TASK-T" in text and "TASK-NESTED" not in text, (
        "task T's output carries the nested execution's conversation"
    )


def test_task_output_is_not_filled_by_another_agents_execution_started_inside_it(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    side_gate = scene.gate()
    side_runs: list[Background] = []

    def side_reply(call: Call) -> str:
        side_gate.wait()
        return final("side-answer")

    side_agent = make_agent(StubLLM(side_reply), role="side")
    side_task = make_task("TASK-SIDE aside", side_agent)

    @tool("start_side_job")
    def start_side_job() -> str:
        """Start a side job and return at once."""
        ctx = contextvars.copy_context()
        side_runs.append(
            scene.background(lambda: ctx.run(side_agent.execute_task, side_task))
        )
        return "side job started"

    def main_reply(call: Call) -> str:
        if "side job started" in call.text:
            return final("main-answer")
        return act("start_side_job")

    main_agent = make_agent(StubLLM(main_reply), role="main", tools=[start_side_job])
    main_task = make_task("TASK-MAIN run", main_agent)
    original = Task._post_agent_execution

    def after_main_saved(self: Task, agent: Any) -> None:
        if self is main_task:
            side_gate.open()
            assert side_runs[0].result() == "side-answer", (
                "precondition: the side execution ended before the task's output build"
            )
        original(self, agent)

    monkeypatch.setattr(Task, "_post_agent_execution", after_main_saved)

    output = main_task.execute_sync(agent=main_agent)

    assert output.raw == "main-answer"
    text = messages_text(output.messages)
    assert "TASK-MAIN" in text and "TASK-SIDE" not in text, (
        "the task's output carries the side agent's conversation"
    )


@pytest.mark.parametrize("cache", [False, True])
def test_later_execution_repeating_the_workers_tool_call_gets_the_tool_result(
    scene: Scene, cache: bool
) -> None:
    tool_gate = scene.gate()
    worker_gate = scene.gate()
    worker_after_tool = threading.Event()
    lookups: list[str] = []
    action = act("lookup", {"q": "x"})

    @tool("lookup")
    def lookup(q: str) -> str:
        """Look something up."""
        lookups.append(q)
        if len(lookups) == 1:
            tool_gate.wait()
        return f"lookup-result-for-{q}"

    def reply(call: Call) -> str:
        if call.task and "TASK-ONE" in call.task:
            if len(llm.calls_for("TASK-ONE")) == 1:
                return action
            worker_after_tool.set()
            worker_gate.wait()
            return final("one")
        if len(llm.calls_for("TASK-TWO")) == 1:
            tool_gate.open()
            worker_after_tool.wait(VALVE)
            return action
        if REPEATED in call.text:
            return final("two was told it repeated itself")
        if "lookup-result-for-x" in call.text:
            return final("two got the tool's result")
        return final("two got neither")

    llm = StubLLM(reply)
    agent = make_agent(llm, tools=[lookup], max_execution_time=LIMIT, cache=cache)
    raises_timeout_at_the_limit(
        lambda: agent.execute_task(make_task("TASK-ONE look up x", agent))
    )
    assert tool_gate.reached.is_set() and not tool_gate.is_open, (
        "precondition: the worker is inside its tool call"
    )
    lookups_before = len(lookups)

    result = agent.execute_task(make_task("TASK-TWO look up x", agent))

    assert worker_after_tool.is_set(), "precondition: the worker's tool call ended"
    assert result == "two got the tool's result"
    if not cache:
        assert len(lookups) - lookups_before == 1, "the tool did not run for it"


class StubKnowledge:
    def __init__(self, name: str, hold: Callable[[str], None] | None = None) -> None:
        self.name = name
        self.queries: list[str] = []
        self._hold = hold

    def query(self, queries: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        self.queries.extend(queries)
        if self._hold is not None:
            self._hold(queries[0])
        return [{"content": f"{self.name}-snippet-for-{queries[0]}"}]

    async def aquery(self, queries: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        self.queries.extend(queries)
        if self._hold is not None:
            await asyncio.to_thread(self._hold, queries[0])
        return [{"content": f"{self.name}-snippet-for-{queries[0]}"}]


@pytest.mark.parametrize("run", ["sync", "async"])
def test_overlapping_preparations_use_their_own_knowledge_query_and_contexts(
    scene: Scene, run: str
) -> None:
    hold = scene.gate()

    def reply(call: Call) -> str:
        for marker, query in (("TASK-P1", "query-p1"), ("TASK-P2", "query-p2")):
            if call.task is None and marker in call.text:
                return query
            if call.task and marker in call.task:
                return final(f"answer-{marker}")
        return final("other")

    def hold_p1(query: str) -> None:
        if query == "query-p1":
            hold.wait()

    llm = StubLLM(reply)
    agent = make_agent(llm)
    agent_knowledge = StubKnowledge("agent", hold_p1)
    crew_knowledge = StubKnowledge("crew")
    crew = Crew(agents=[agent], tasks=[make_task("unused", agent)])
    crew.knowledge = crew_knowledge  # type: ignore[assignment]
    agent.crew = crew
    agent.knowledge = agent_knowledge  # type: ignore[assignment]
    p1 = make_task("TASK-P1 first", agent)
    p2 = make_task("TASK-P2 second", agent)

    if run == "sync":
        first = scene.background(lambda: agent.execute_task(p1))
        assert hold.reached.wait(VALVE), "P1 never reached its knowledge search"
        second = agent.execute_task(p2)
        hold.open()
        first_result = first.result()
    else:

        async def both() -> tuple[Any, Any]:
            first_run = asyncio.create_task(agent.aexecute_task(p1))
            await asyncio.to_thread(hold.reached.wait, VALVE)
            second_result = await agent.aexecute_task(p2)
            hold.open()
            return await first_run, second_result

        first_result, second = asyncio.run(both())

    assert first_result == "answer-TASK-P1" and second == "answer-TASK-P2"
    assert crew_knowledge.queries.count("query-p1") == 1, (
        f"P1's crew search received {crew_knowledge.queries}"
    )
    assert crew_knowledge.queries.count("query-p2") == 1
    for marker, own, other in (
        ("TASK-P1", "query-p1", "query-p2"),
        ("TASK-P2", "query-p2", "query-p1"),
    ):
        prompt = llm.calls_for(marker)[0].text
        assert f"agent-snippet-for-{own}" in prompt and f"crew-snippet-for-{own}" in prompt
        assert other not in prompt, f"{marker}'s prompt carries the other's knowledge"
