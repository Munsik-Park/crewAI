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

import contextvars
import threading
from typing import Final


_stop_events: contextvars.ContextVar[tuple[threading.Event, ...]] = (
    contextvars.ContextVar("crewai_execution_stop_events", default=())
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
