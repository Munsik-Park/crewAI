"""The messages a task's own execution ended with, kept apart from others."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from crewai.utilities.types import LLMMessage


class ExecutionMessages:
    """The messages a task execution of ``agent`` ended with."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.messages: list[LLMMessage] | None = None

    def read(self) -> list[LLMMessage]:
        """The recorded messages, or the agent's last ones when no execution of
        the agent filled the record (an agent that does not record them)."""
        if self.messages is not None:
            return self.messages
        return self.agent.last_messages  # type: ignore[no-any-return]


_active: ContextVar[ExecutionMessages | None] = ContextVar(
    "crewai_execution_messages", default=None
)


@contextmanager
def execution_messages(agent: Any) -> Iterator[ExecutionMessages]:
    """Record the messages of the execution of ``agent`` run inside the block.

    ``agent.last_messages`` is one field per agent, which an execution
    overlapping this one on the same agent overwrites; the record is this
    block's own. The ContextVar is copied into threads and asyncio tasks, so an
    execution run from a copied context fills it too.
    """
    record = ExecutionMessages(agent)
    token = _active.set(record)
    try:
        yield record
    finally:
        _active.reset(token)


def record_execution_messages(agent: Any, messages: list[LLMMessage]) -> None:
    """Fill the record open in this context when it is for ``agent``."""
    record = _active.get()
    if record is not None and record.agent is agent:
        record.messages = messages


def clear_execution_messages() -> None:
    """Detach the current context from the record open in it."""
    _active.set(None)
