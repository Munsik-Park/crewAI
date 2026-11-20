"""Abandonment of agent runs that exceeded ``max_execution_time``.

A timed-out run cannot be stopped while a step is in progress: Python has no
way to stop a running thread. What can be done is to end its reach once the
caller has given up on it. Each timed call makes a ``TimedRunScope`` current
for the work it hands off. The scope travels in a ``ContextVar``, so it
reaches every executor the run starts, including coworkers reached by
delegation. On timeout the scope is abandoned: the executors it holds are
detached from their agents, and the run stops at its next step boundary and
refuses to start on any further agent.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
import threading
from typing import Any


class RunAbandonedError(Exception):
    """Ends a run whose caller already received ``TimeoutError``.

    It is a deliberate stop, not a failure: it is not retried, not counted
    against the retry budget, and nothing is printed or logged as an error
    for it. The events that close what the run opened are still emitted:
    ``AgentExecutionErrorEvent`` for an attempt that emitted its start event,
    and ``FlowFailedEvent`` for the executor's flow.
    """

    def __init__(
        self, message: str = "The run was abandoned after its timeout."
    ) -> None:
        super().__init__(message)


class TimedRunScope:
    """The executors one timed call's run is using, and whether it was abandoned.

    A scope opened while another is current belongs to it, so abandoning the
    outer one also abandons runs nested in it. Nested scopes share the
    outer scope's lock, so a check and the write it guards can never
    interleave with an abandonment anywhere in the tree.
    """

    def __init__(self, parent: TimedRunScope | None = None) -> None:
        self._parent = parent
        self._lock: threading.RLock = (
            parent._lock if parent is not None else threading.RLock()
        )
        self._abandoned = False
        self._executors: set[Any] = set()
        self._children: set[TimedRunScope] = set()
        if parent is not None:
            with self._lock:
                parent._children.add(self)

    @property
    def abandoned(self) -> bool:
        scope: TimedRunScope | None = self
        while scope is not None:
            if scope._abandoned:
                return True
            scope = scope._parent
        return False

    def register(self, executor: Any) -> None:
        """Record an executor that starts a run under this scope.

        Raises:
            RunAbandonedError: The scope was abandoned.
        """
        with self._lock:
            if self.abandoned:
                raise RunAbandonedError()
            self._executors.add(executor)

    def unregister(self, executor: Any) -> None:
        with self._lock:
            self._executors.discard(executor)

    @contextmanager
    def guard(self) -> Generator[None, None, None]:
        """Hold the scope while writing shared agent state; refuse if abandoned.

        Raises:
            RunAbandonedError: The scope was abandoned.
        """
        with self._lock:
            if self.abandoned:
                raise RunAbandonedError()
            yield

    def close(self) -> None:
        """Drop this scope from its parent once its timed call has returned."""
        if self._parent is not None:
            with self._lock:
                self._parent._children.discard(self)

    def abandon(self) -> None:
        """Abandon the run and detach every executor it holds, nested ones included."""
        with self._lock:
            self._abandoned = True
            pending: list[TimedRunScope] = [self]
            while pending:
                scope = pending.pop()
                pending.extend(scope._children)
                for executor in list(scope._executors):
                    detach_abandoned_executor(executor)


def detach_abandoned_executor(executor: Any) -> None:
    """Leave an abandoned run its executor and tools handler.

    The step in progress keeps using both, so the agent gets fresh ones for
    its next call; the tool cache stays shared.
    """
    from crewai.agents.tools_handler import ToolsHandler

    executor.abandon()
    agent = getattr(executor, "agent", None)
    if agent is None:
        return
    if getattr(agent, "agent_executor", None) is executor:
        agent.agent_executor = None
    handler = getattr(executor, "tools_handler", None)
    if handler is not None and getattr(agent, "tools_handler", None) is handler:
        agent.tools_handler = ToolsHandler(cache=handler.cache)


_current_scope: ContextVar[TimedRunScope | None] = ContextVar(
    "crewai_timed_run_scope", default=None
)


def current_run_scope() -> TimedRunScope | None:
    return _current_scope.get()


def set_run_scope(scope: TimedRunScope | None) -> Any:
    return _current_scope.set(scope)


def reset_run_scope(token: Any) -> None:
    _current_scope.reset(token)


def run_scope_abandoned() -> bool:
    scope = _current_scope.get()
    return scope is not None and scope.abandoned


def refuse_abandoned_run() -> None:
    """Refuse to start work on an agent for a run abandoned at its timeout.

    A no-op when no timed run is current.

    Raises:
        RunAbandonedError: The current scope was abandoned.
    """
    if run_scope_abandoned():
        raise RunAbandonedError()


@contextmanager
def run_scope_write_guard() -> Generator[None, None, None]:
    """Guard a write to agent state against the current scope's abandonment.

    A no-op when no timed run is current.

    Raises:
        RunAbandonedError: The current scope was abandoned.
    """
    scope = _current_scope.get()
    if scope is None:
        yield
        return
    with scope.guard():
        yield


def join_run_scope(executor: Any) -> TimedRunScope | None:
    """Register an executor with the current scope, if any.

    Raises:
        RunAbandonedError: The current scope was abandoned.
    """
    scope = _current_scope.get()
    if scope is not None:
        scope.register(executor)
    return scope


def leave_run_scope(scope: TimedRunScope | None, executor: Any) -> None:
    if scope is not None:
        scope.unregister(executor)
