"""Tool results of one agent execution, isolated from other executions."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


_active_results: ContextVar[tuple[Any, list[dict[str, Any]]] | None] = ContextVar(
    "crewai_tool_results", default=None
)


@contextmanager
def tool_result_collector(agent: Any) -> Generator[list[dict[str, Any]], None, None]:
    """Collect the tool results of one execution of ``agent``.

    ``agent.tools_results`` accumulates over every execution, including runs
    still in progress after their caller timed out, so reading it lets one
    execution take another's ``result_as_answer`` output as its answer. The
    collector is a ContextVar, which asyncio tasks and threads copy, so each
    execution sees only its own results. It records only results of tools
    used by the agent that opened it: a run of another agent started inside
    this execution (for example ``Agent.kickoff`` called from a tool) does
    not contribute to it.
    """
    results: list[dict[str, Any]] = []
    token = _active_results.set((agent, results))
    try:
        yield results
    finally:
        _active_results.reset(token)


def record_tool_result(agent: Any, data: dict[str, Any]) -> None:
    """Store a tool result on the active collector of ``agent`` and on the agent."""
    active = _active_results.get()
    if active is not None and active[0] is agent:
        active[1].append(data)
    agent.tools_results.append(data)


def collected_tool_results(agent: Any) -> list[dict[str, Any]] | None:
    """Results of the execution of ``agent`` in progress, or None outside one."""
    active = _active_results.get()
    if active is not None and active[0] is agent:
        return active[1]
    return None
