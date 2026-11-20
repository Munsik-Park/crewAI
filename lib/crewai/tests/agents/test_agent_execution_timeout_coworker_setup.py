"""A coworker whose setup is still running when the limit passes.

The step in flight at the limit may have started a task on a coworker, by
delegation or through ``kickoff``, before the limit. That coworker first
converts its tools and, when it holds no executor, builds one. That setup may
be slow, and it runs in user code a test can override. The ``TimeoutError``
must still arrive in time, and once the setup ends, the abandoned run must
leave the coworker's tool usage counts and executor as they were and call
no LLM.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import time
from typing import Any

import pytest

from crewai import Agent
from crewai.agents.agent_executor import AgentExecutor
from crewai.tools import BaseTool, tool
from crewai.tools.agent_tools.agent_tools import AgentTools
from crewai.tools.structured_tool import CrewStructuredTool
from tests.agents.test_agent_execution_timeout import (
    DELEGATE,
    IN_TIME,
    JOIN_BOUND,
    LIMIT,
    _action,
    _agent,
    _final,
    _join_timed_runs,
    _ScriptedLLM,
    _task,
    _TrackingExecutor,
)


SETUP_BLOCK = 3.0
CASES = [
    ("delegation", "conversion"),
    ("delegation", "construction"),
    ("kickoff", "conversion"),
]
CASE_IDS = [f"{route}-{block}" for route, block in CASES]


class _SlowSetup:
    """Gates the coworker's setup: once armed, each setup call waits for release."""

    def __init__(self) -> None:
        self.armed = threading.Event()
        self.release = threading.Event()
        self.entered: list[float] = []

    def wait_if_armed(self) -> None:
        if self.armed.is_set():
            self.entered.append(time.monotonic())
            self.release.wait(SETUP_BLOCK)


def _slow_lookup(gate: _SlowSetup | None) -> BaseTool:
    """A ``lookup`` tool whose conversion waits once ``gate`` is armed."""

    class SlowLookup(BaseTool):
        name: str = "lookup"
        description: str = "Look a query up."

        def _run(self, query: str) -> str:
            return f"LOOKUP_RESULT for {query}"

        def to_structured_tool(self) -> CrewStructuredTool:
            if gate is not None:
                gate.wait_if_armed()
            return super().to_structured_tool()

    return SlowLookup()


def _slow_executor_class(gate: _SlowSetup) -> type[AgentExecutor]:
    """An executor class whose construction waits once ``gate`` is armed."""

    class SlowExecutor(AgentExecutor):
        def __init__(self, **data: Any) -> None:
            gate.wait_if_armed()
            super().__init__(**data)

    return SlowExecutor


class _Coworker:
    """A coworker whose ``lookup`` was used before, and what that left on it.

    In the conversion case the coworker ran the earlier task itself and holds
    an executor. In the construction case another agent sharing ``lookup``
    ran it, so the coworker holds no executor and its next task builds one.
    """

    def __init__(self, route: str, block: str) -> None:
        self.route = route
        self.gate = _SlowSetup()
        self.llm_calls: list[str] = []
        self.outcome: list[str] = []
        self.lookup = _slow_lookup(self.gate if block == "conversion" else None)

        def script(text: str) -> str:
            if "SUB_ONE" in text:
                self.llm_calls.append("SUB_ONE")
                return _final("COWORKER_ORPHAN_ANSWER")
            if "LOOKUP_RESULT for a" not in text:
                return _action("lookup", '{"query": "a"}')
            return _final("COWORKER_PRE_ANSWER")

        settings: dict[str, Any] = {}
        if block == "construction":
            settings["executor_class"] = _slow_executor_class(self.gate)
        self.agent = _agent(
            _ScriptedLLM(script),
            tools=[self.lookup],
            role="Coworker",
            max_execution_time=None,
            **settings,
        )
        self.pre_agent = (
            _agent(
                _ScriptedLLM(script),
                tools=[self.lookup],
                role="Earlier user",
                max_execution_time=None,
            )
            if block == "construction"
            else self.agent
        )
        self.pre_task = _task(self.pre_agent, "PRE")

    def record(self, pre_answer: Any) -> None:
        """Keep what the earlier task left on the coworker."""
        assert pre_answer == "COWORKER_PRE_ANSWER"
        self.usage_count = self.lookup.current_usage_count
        self.executor = self.agent.agent_executor
        assert self.usage_count > 0, "the earlier task did not use lookup"
        assert (self.executor is None) == (self.pre_agent is not self.agent)

    def timed_agent(self) -> Agent:
        """The timed agent, which reaches the coworker from its first step."""
        if self.route == "delegation":
            tools = AgentTools(agents=[self.agent]).tools()

            def script(text: str) -> str:
                if not self.gate.armed.is_set():
                    self.gate.armed.set()
                    return _action(
                        DELEGATE,
                        json.dumps(
                            {"task": "SUB_ONE", "context": "c", "coworker": "Coworker"}
                        ),
                    )
                return _final("orphan answer")

        else:
            coworker = self.agent

            @tool("ask_coworker")
            def ask_coworker() -> str:
                """Ask the coworker."""
                try:
                    coworker.kickoff("SUB_ONE: answer")
                    self.outcome.append("ran")
                except BaseException as error:
                    self.outcome.append(type(error).__name__)
                return "ASKED_COWORKER"

            tools = [ask_coworker]

            def script(text: str) -> str:
                if self.gate.armed.is_set():
                    return _final("orphan answer")
                self.gate.armed.set()
                return _action("ask_coworker")

        return _agent(_ScriptedLLM(script), tools=tools)

    def assert_in_time(self, elapsed: float, timed_out_at: float) -> None:
        assert elapsed < IN_TIME, (
            f"TimeoutError took {elapsed:.2f}s; the limit is {LIMIT}s"
        )
        assert self.gate.entered, "the coworker's setup was never reached"
        assert self.gate.entered[0] < timed_out_at, (
            "the coworker's setup started after the limit"
        )

    def assert_unchanged(self) -> None:
        changed = [
            name
            for name, unchanged in (
                (
                    "tool usage count",
                    self.lookup.current_usage_count == self.usage_count,
                ),
                (
                    "executor",
                    self.route != "delegation"
                    or self.agent.agent_executor is self.executor,
                ),
            )
            if not unchanged
        ]
        assert changed == [], f"the timed-out run changed the coworker's {changed}"
        assert self.llm_calls == [], "the coworker called its LLM after the limit"


@pytest.mark.parametrize(("route", "block"), CASES, ids=CASE_IDS)
def test_coworker_setup_across_the_limit_does_not_delay_the_timeout(
    route: str, block: str
) -> None:
    """A coworker's slow setup across the limit neither delays the TimeoutError nor changes the coworker."""
    coworker = _Coworker(route, block)
    coworker.record(coworker.pre_agent.execute_task(coworker.pre_task))
    agent = coworker.timed_agent()

    before = set(threading.enumerate())
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "TASK_ONE"))
        timed_out_at = time.monotonic()
        assert not coworker.gate.release.is_set()
    finally:
        coworker.gate.release.set()
        still_alive = _join_timed_runs(before)
    assert still_alive == []
    coworker.assert_in_time(timed_out_at - start, timed_out_at)
    coworker.assert_unchanged()


@pytest.mark.asyncio
@pytest.mark.parametrize(("route", "block"), CASES, ids=CASE_IDS)
async def test_coworker_setup_across_the_limit_does_not_delay_the_async_timeout(
    route: str, block: str
) -> None:
    """After an async timeout, a coworker's slow setup across the limit neither delayed the TimeoutError nor changes the coworker."""
    loop = asyncio.get_running_loop()
    pool = _TrackingExecutor()
    loop.set_default_executor(pool)
    coworker = _Coworker(route, block)
    coworker.record(await coworker.pre_agent.aexecute_task(coworker.pre_task))
    agent = coworker.timed_agent()
    in_flight_at_timeout: list[set[concurrent.futures.Future[Any]]] = []

    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            await agent.aexecute_task(_task(agent, "TASK_ONE"))
        timed_out_at = time.monotonic()
        in_flight_at_timeout.append(pool.snapshot())
        assert not coworker.gate.release.is_set()
    finally:
        coworker.gate.release.set()
        if in_flight_at_timeout:
            concurrent.futures.wait(in_flight_at_timeout[0], timeout=JOIN_BOUND)
        pool.shutdown(wait=True)
    coworker.assert_in_time(timed_out_at - start, timed_out_at)
    coworker.assert_unchanged()
