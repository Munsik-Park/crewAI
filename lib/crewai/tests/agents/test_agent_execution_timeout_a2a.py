"""A run left behind by ``max_execution_time`` that reaches a coworker with ``a2a`` set.

Such a coworker talks to a remote agent: it fetches the remote agent cards
when it starts, and it sends the remote agent a message when its answer is a
delegation. That is work on a further agent, so the run left behind must not
start it: a coworker called after the limit fetches no card, and a coworker
whose own call in flight ends after the limit sends no message.

The remote side is replaced by recording stubs, so nothing goes on the
network. The tests need the optional ``a2a-sdk`` package.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest


pytest.importorskip("a2a")

from a2a.types import AgentCapabilities, AgentCard

from crewai.a2a import wrapper as a2a_wrapper
from crewai.a2a.config import A2AClientConfig
from crewai.core.providers.human_input import reset_provider, set_provider
from crewai.tools import tool
from tests.agents.test_agent_execution_timeout import (
    ORPHAN_BLOCK,
    _action,
    _agent,
    _BlockingFeedbackProvider,
    _final,
    _join_timed_runs,
    _ScriptedLLM,
    _task,
)


ENDPOINT = "http://remote-agent.invalid/.well-known/agent-card.json"


class _RemoteAgent:
    """Stands in for the remote agent and records every request made to it."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self._lock = threading.Lock()

    def _record(self, request: str) -> None:
        with self._lock:
            self.requests.append(request)

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self.requests)

    def fetch_card(self, endpoint: str, *args: Any, **kwargs: Any) -> AgentCard:
        self._record("fetch card")
        return AgentCard(
            name="Remote",
            description="remote agent",
            url="http://remote-agent.invalid",
            version="1",
            capabilities=AgentCapabilities(),
            default_input_modes=["text"],
            default_output_modes=["text"],
            skills=[],
        )

    async def afetch_card(self, endpoint: str, *args: Any, **kwargs: Any) -> AgentCard:
        return self.fetch_card(endpoint)

    def send(self, **kwargs: Any) -> Any:
        self._record("send message")
        raise RuntimeError("the remote agent is a stub")

    async def asend(self, **kwargs: Any) -> Any:
        return self.send(**kwargs)


@pytest.fixture
def remote(monkeypatch: pytest.MonkeyPatch) -> _RemoteAgent:
    stub = _RemoteAgent()
    monkeypatch.setattr(a2a_wrapper, "fetch_agent_card", stub.fetch_card)
    monkeypatch.setattr(a2a_wrapper, "afetch_agent_card", stub.afetch_card)
    monkeypatch.setattr(a2a_wrapper, "execute_a2a_delegation", stub.send)
    monkeypatch.setattr(a2a_wrapper, "aexecute_a2a_delegation", stub.asend)
    return stub


def _a2a_coworker(answer: str) -> Any:
    return _agent(
        _ScriptedLLM(lambda text: _final(answer)),
        role="Coworker",
        max_execution_time=None,
        a2a=A2AClientConfig(endpoint=ENDPOINT, fail_fast=False),
    )


def _run_on(coworker: Any, entry: str, human_input: bool = False) -> None:
    if entry == "execute_task":
        coworker.execute_task(_task(coworker, "SUB_ONE", human_input=human_input))
    elif entry == "aexecute_task":
        asyncio.run(
            coworker.aexecute_task(_task(coworker, "SUB_ONE", human_input=human_input))
        )
    elif entry == "kickoff":
        coworker.kickoff("SUB_ONE: answer")
    else:
        asyncio.run(coworker.kickoff_async("SUB_ONE: answer"))


def _asking_agent(ask: Any) -> Any:
    return _agent(
        _ScriptedLLM(
            lambda text: (
                _final("orphan answer")
                if "ASKED_COWORKER" in text
                else _action("ask_coworker")
            )
        ),
        tools=[ask],
    )


def _time_out(agent: Any, release: threading.Event, remote: _RemoteAgent) -> list[str]:
    """Run the agent into its timeout, release its run, wait for its end.

    Returns the remote requests made by the time ``TimeoutError`` was raised.
    """
    before = set(threading.enumerate())
    at_timeout: list[str] = []
    try:
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "TASK_ONE"))
        at_timeout.extend(remote.snapshot())
        assert not release.is_set(), "the call in flight ended before the limit"
    finally:
        release.set()
        still_alive = _join_timed_runs(before)
    assert still_alive == []
    return at_timeout


@pytest.mark.parametrize(
    "entry", ["execute_task", "aexecute_task", "kickoff", "kickoff_async"]
)
def test_a2a_coworker_called_after_the_limit_makes_no_remote_request(
    remote: _RemoteAgent, entry: str
) -> None:
    """An a2a coworker that the timed-out run calls after the limit fetches no remote agent card."""
    coworker = _a2a_coworker("COWORKER_ANSWER")
    release = threading.Event()
    outcome: list[str] = []

    @tool("ask_coworker")
    def ask_coworker() -> str:
        """Ask the coworker."""
        release.wait(ORPHAN_BLOCK)
        try:
            _run_on(coworker, entry)
            outcome.append("ran")
        except BaseException as error:
            outcome.append(type(error).__name__)
        return "ASKED_COWORKER"

    at_timeout = _time_out(_asking_agent(ask_coworker), release, remote)

    assert at_timeout == []
    assert len(outcome) == 1, "the timed-out run's tool never called the coworker"
    assert remote.requests == [], (
        f"remote requests made after the limit: {remote.requests}"
    )


@pytest.mark.parametrize("entry", ["execute_task", "aexecute_task"])
def test_a2a_coworker_sends_no_delegation_after_its_call_in_flight(
    remote: _RemoteAgent, entry: str
) -> None:
    """An a2a coworker whose human feedback ends after the limit does not send its delegation to the remote agent."""
    delegation = json.dumps(
        {"a2a_ids": [ENDPOINT], "message": "please do it", "is_a2a": True}
    )
    coworker = _a2a_coworker(delegation)
    release = threading.Event()
    provider = _BlockingFeedbackProvider(release)
    outcome: list[str] = []

    @tool("ask_coworker")
    def ask_coworker() -> str:
        """Ask the coworker."""
        try:
            _run_on(coworker, entry, human_input=True)
            outcome.append("ran")
        except BaseException as error:
            outcome.append(type(error).__name__)
        return "ASKED_COWORKER"

    token = set_provider(provider)
    try:
        at_timeout = _time_out(_asking_agent(ask_coworker), release, remote)
    finally:
        reset_provider(token)

    assert provider.calls == 1, "the coworker's human feedback was not in flight"
    assert at_timeout == ["fetch card"], f"remote requests by the limit: {at_timeout}"
    after = remote.requests[len(at_timeout) :]
    assert after == [], f"remote requests made after the limit: {after}"
