"""Stop marks for agent runs abandoned when ``max_execution_time`` is reached.

A run that times out cannot be interrupted: the call in progress keeps running
in its thread. Each timed run therefore carries a stop ``Event`` in a context
variable that is set only inside that run's own context. Once the deadline sets
the event, the run's next tool call, model call or converter request is refused.

The marks are a tuple so a run nested inside a timed run (a delegated
co-worker, for example) also stops when any run enclosing it times out.

This module imports nothing from crewAI so any layer can check the mark.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import contextvars
import threading
from typing import Final


_stop_events: contextvars.ContextVar[tuple[threading.Event, ...]] = (
    contextvars.ContextVar("crewai_execution_stop_events", default=())
)

_admitted_call: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "crewai_execution_stop_admitted_call", default=False
)

EXECUTION_STOPPED_MESSAGE: Final[str] = (
    "The agent run exceeded max_execution_time and was stopped; "
    "no further model calls are made for it."
)


def enter_timed_run(stop: threading.Event) -> None:
    """Add ``stop`` to the marks of the current context.

    Call it inside the context the timed run executes in, never in the
    caller's, so that nothing outside the run sees the mark.
    """
    _stop_events.set((*_stop_events.get(), stop))


def execution_stopped() -> bool:
    """Whether the run executing in the current context has been stopped."""
    return any(event.is_set() for event in _stop_events.get())


@contextmanager
def admitted_call() -> Iterator[None]:
    """Mark a model call that passed the stop check before its hooks ran.

    A call already in progress at the deadline, even one still held in its
    own before-hook, runs to its end; only calls started after the stop are
    refused.
    """
    token = _admitted_call.set(True)
    try:
        yield
    finally:
        _admitted_call.reset(token)


def model_call_refused() -> bool:
    """Whether a model call starting now must be refused because its run stopped."""
    return execution_stopped() and not _admitted_call.get()
