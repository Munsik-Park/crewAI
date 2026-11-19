"""Core types for async human feedback in Flows.

This module defines the protocol, exception, and context types used for
non-blocking human-in-the-loop workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from crewai.flow.flow import Flow


ENTRY_PATH_START = "start"
ENTRY_PATH_LISTENER = "listener"
_ENTRY_PATHS = frozenset({ENTRY_PATH_START, ENTRY_PATH_LISTENER})


class _MalformedProgressError(ValueError):
    """Raised while parsing a persisted progress field that is not well formed."""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise _MalformedProgressError
    return list(value)


def _entry_path(value: Any) -> str | None:
    """Return a known entry-path marker, or ``None`` for any other value."""
    return value if isinstance(value, str) and value in _ENTRY_PATHS else None


@dataclass
class CarriedLabelDelivery:
    """An enclosing router's label delivery cut by a pause.

    Attributes:
        method_name: The router that produced the label.
        label: The router's result as a string.
        entry_path: How the router was reached: ``"start"`` for a start
            method, ``"listener"`` for a listener. ``None`` when unknown; the
            path is then inferred from the flow definition.
    """

    method_name: str
    label: str
    entry_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "label": self.label,
            "entry_path": self.entry_path,
        }

    @classmethod
    def from_dict(cls, data: Any) -> CarriedLabelDelivery:
        if not isinstance(data, dict):
            raise _MalformedProgressError
        method_name = data.get("method_name")
        label = data.get("label")
        if not isinstance(method_name, str) or not isinstance(label, str):
            raise _MalformedProgressError
        return cls(
            method_name=method_name,
            label=label,
            entry_path=_entry_path(data.get("entry_path")),
        )


@dataclass
class ResumeProgress:
    """Execution progress of a flow at the moment a review paused it.

    Restored on resume so the flow continues as it would have if the review
    had been answered during the run.

    Attributes:
        completed_methods: Methods that had completed.
        pending_events: Events already seen by each listener or conditional
            start waiting on a multi-event condition, keyed by subscription.
        fired_or_listeners: Multi-event ``or_`` listeners that had fired.
        entry_path: How the paused method was reached (``"start"`` or
            ``"listener"``). ``None`` when unknown; the path is then inferred
            from the flow definition.
        carried_labels: Label deliveries of enclosing routers that the pause
            cut, innermost first.
    """

    completed_methods: list[str] = field(default_factory=list)
    pending_events: dict[str, list[str]] = field(default_factory=dict)
    fired_or_listeners: list[str] = field(default_factory=list)
    entry_path: str | None = None
    carried_labels: list[CarriedLabelDelivery] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_methods": list(self.completed_methods),
            "pending_events": {
                key: list(events) for key, events in self.pending_events.items()
            },
            "fired_or_listeners": list(self.fired_or_listeners),
            "entry_path": self.entry_path,
            "carried_labels": [carried.to_dict() for carried in self.carried_labels],
        }

    @classmethod
    def from_dict(cls, data: Any) -> ResumeProgress | None:
        """Parse a persisted progress field; anything malformed yields ``None``."""
        if isinstance(data, ResumeProgress):
            return data
        if not isinstance(data, dict):
            return None
        try:
            pending = data.get("pending_events", {})
            if not isinstance(pending, dict) or not all(
                isinstance(key, str) for key in pending
            ):
                raise _MalformedProgressError
            carried = data.get("carried_labels", [])
            if not isinstance(carried, list):
                raise _MalformedProgressError
            return cls(
                completed_methods=_string_list(data.get("completed_methods", [])),
                pending_events={
                    key: _string_list(events) for key, events in pending.items()
                },
                fired_or_listeners=_string_list(data.get("fired_or_listeners", [])),
                entry_path=_entry_path(data.get("entry_path")),
                carried_labels=[CarriedLabelDelivery.from_dict(c) for c in carried],
            )
        except _MalformedProgressError:
            return None


@dataclass
class PendingFeedbackContext:
    """Context capturing everything needed to resume a paused flow.

    When a flow is paused waiting for async human feedback, this dataclass
    stores all the information needed to:
    1. Identify which flow execution is waiting
    2. What method triggered the feedback request
    3. What was shown to the human
    4. How to route the response when it arrives

    Attributes:
        flow_id: Unique identifier for the flow instance (from state.id)
        flow_class: Fully qualified class name (e.g., "myapp.flows.ReviewFlow")
        method_name: Name of the method that triggered feedback request
        method_output: The output that was shown to the human for review
        message: The message displayed when requesting feedback
        emit: Optional list of outcome strings for routing
        default_outcome: Outcome to use when no feedback is provided
        metadata: Optional metadata for external system integration
        llm: LLM model string for outcome collapsing
        requested_at: When the feedback was requested
        execution_uuid: Outermost kickoff uuid to restore on resume so traces
            stay on the same run after HITL pause. ``None`` only for pending
            rows persisted before this field existed; resume then creates a
            new uuid. New pauses always store a value.
        trace_context: Previous trace and root span IDs, used to link the resumed
            execution segment. Absent when execution tracing was disabled.
        resume_progress: Execution progress at the pause (completed methods,
            join and ``or_`` progress, how the paused method was reached, and
            cut label deliveries of enclosing routers). ``None`` when the
            stored context has no such key or its value is malformed; resume
            then starts from empty progress.

    Example:
        ```python
        context = PendingFeedbackContext(
            flow_id="abc-123",
            flow_class="myapp.ReviewFlow",
            method_name="review_content",
            method_output={"title": "Draft", "body": "..."},
            message="Please review and approve or reject:",
            emit=["approved", "rejected"],
            llm="gpt-5.4-mini",
        )
        ```
    """

    flow_id: str
    flow_class: str
    method_name: str
    method_output: Any
    message: str
    emit: list[str] | None = None
    default_outcome: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] | str | None = None
    requested_at: datetime = field(default_factory=datetime.now)
    execution_uuid: str | None = None
    trace_context: tuple[int, int] | None = None
    resume_progress: ResumeProgress | None = None

    @staticmethod
    def _make_json_safe(value: Any) -> Any:
        """Convert a value to a JSON-serializable form.

        Handles Pydantic models, dataclasses, and arbitrary objects by
        progressively falling back to string representation.
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [PendingFeedbackContext._make_json_safe(v) for v in value]
        if isinstance(value, dict):
            return {
                k: PendingFeedbackContext._make_json_safe(v) for k, v in value.items()
            }

        from pydantic import BaseModel

        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        import dataclasses

        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return PendingFeedbackContext._make_json_safe(dataclasses.asdict(value))
        return str(value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize context to a dictionary for persistence.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "flow_id": self.flow_id,
            "flow_class": self.flow_class,
            "method_name": self.method_name,
            "method_output": self._make_json_safe(self.method_output),
            "message": self.message,
            "emit": self.emit,
            "default_outcome": self.default_outcome,
            "metadata": self._make_json_safe(self.metadata),
            "llm": self.llm,
            "requested_at": self.requested_at.isoformat(),
            "execution_uuid": self.execution_uuid,
            "trace_context": list(self.trace_context) if self.trace_context else None,
            "resume_progress": self.resume_progress.to_dict()
            if self.resume_progress is not None
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingFeedbackContext:
        """Deserialize context from a dictionary.

        Args:
            data: Dictionary representation of the context.

        Returns:
            Reconstructed PendingFeedbackContext instance.
        """
        requested_at = data.get("requested_at")
        if isinstance(requested_at, str):
            requested_at = datetime.fromisoformat(requested_at)
        elif requested_at is None:
            requested_at = datetime.now()

        trace_context = data.get("trace_context")
        if not (
            isinstance(trace_context, (list, tuple))
            and len(trace_context) == 2
            and all(isinstance(value, int) for value in trace_context)
        ):
            trace_context = None

        return cls(
            flow_id=data["flow_id"],
            flow_class=data["flow_class"],
            method_name=data["method_name"],
            method_output=data.get("method_output"),
            message=data.get("message", ""),
            emit=data.get("emit"),
            default_outcome=data.get("default_outcome"),
            metadata=data.get("metadata", {}),
            llm=data.get("llm"),
            requested_at=requested_at,
            execution_uuid=data.get("execution_uuid"),
            trace_context=(trace_context[0], trace_context[1])
            if trace_context
            else None,
            resume_progress=ResumeProgress.from_dict(data.get("resume_progress")),
        )


class HumanFeedbackPending(Exception):  # noqa: N818 - Not an error, a control flow signal
    """Signal that flow execution should pause for async human feedback.

    When raised by a provider, the flow framework will:
    1. Stop execution at the current method
    2. Automatically persist state and context (if persistence is configured)
    3. Return this object to the caller (not re-raise it)

    The caller receives this as a return value from `flow.kickoff()`, enabling
    graceful handling of the paused state without try/except blocks:

        ```python
        result = flow.kickoff()
        if isinstance(result, HumanFeedbackPending):
            # Flow is paused, handle async feedback
            print(f"Waiting for feedback: {result.context.flow_id}")
        else:
            # Normal completion
            print(f"Flow completed: {result}")
        ```

    Note:
        The flow framework automatically saves pending feedback when this
        exception is raised. Providers do NOT need to call `save_pending_feedback`
        manually - just raise this exception and the framework handles persistence.

    Attributes:
        context: The PendingFeedbackContext with all details needed to resume
        callback_info: Optional dict with information for external systems
            (e.g., webhook URL, ticket ID, Slack thread ID)

    Example:
        ```python
        class SlackProvider(HumanFeedbackProvider):
            def request_feedback(self, context, flow):
                # Send notification to external system
                ticket_id = self.create_slack_thread(context)

                # Raise to pause - framework handles persistence automatically
                raise HumanFeedbackPending(
                    context=context,
                    callback_info={
                        "slack_channel": "#reviews",
                        "thread_id": ticket_id,
                    },
                )
        ```
    """

    def __init__(
        self,
        context: PendingFeedbackContext,
        callback_info: dict[str, Any] | None = None,
        message: str | None = None,
    ):
        """Initialize the pending feedback exception.

        Args:
            context: The pending feedback context with flow details
            callback_info: Optional information for external system callbacks
            message: Optional custom message (defaults to descriptive message)
        """
        self.context = context
        self.callback_info = callback_info or {}

        if message is None:
            message = (
                f"Human feedback pending for flow '{context.flow_id}' "
                f"at method '{context.method_name}'"
            )
        super().__init__(message)


@runtime_checkable
class HumanFeedbackProvider(Protocol):
    """Protocol for human feedback collection strategies.

    Implement this protocol to create custom feedback providers that integrate
    with external systems like Slack, Teams, email, or custom APIs.

    Providers can be either:
    - **Synchronous (blocking)**: Return feedback string directly
    - **Asynchronous (non-blocking)**: Raise HumanFeedbackPending to pause

    The default ConsoleProvider is synchronous and blocks waiting for input.
    For async workflows, implement a provider that raises HumanFeedbackPending.

    Note:
        The flow framework automatically handles state persistence when
        HumanFeedbackPending is raised. Providers only need to:
        1. Notify the external system (Slack, email, webhook, etc.)
        2. Raise HumanFeedbackPending with the context and callback info

    Example synchronous provider:
        ```python
        class ConsoleProvider(HumanFeedbackProvider):
            def request_feedback(self, context, flow):
                print(context.method_output)
                return input("Your feedback: ")
        ```

    Example async provider:
        ```python
        class SlackProvider(HumanFeedbackProvider):
            def __init__(self, channel: str):
                self.channel = channel

            def request_feedback(self, context, flow):
                # Send notification to Slack
                thread_id = self.post_to_slack(
                    channel=self.channel,
                    message=context.message,
                    content=context.method_output,
                )

                # Raise to pause - framework handles persistence automatically
                raise HumanFeedbackPending(
                    context=context,
                    callback_info={
                        "channel": self.channel,
                        "thread_id": thread_id,
                    },
                )
        ```
    """

    def request_feedback(
        self,
        context: PendingFeedbackContext,
        flow: Flow[Any],
    ) -> str:
        """Request feedback from a human.

        For synchronous providers, block and return the feedback string.
        For async providers, notify the external system and raise
        HumanFeedbackPending to pause the flow.

        Args:
            context: The pending feedback context containing all details
                about what feedback is needed and how to route the response.
            flow: The Flow instance, providing access to state and name.

        Returns:
            The human's feedback as a string (synchronous providers only).

        Raises:
            HumanFeedbackPending: To signal that the flow should pause and
                wait for external feedback. The framework will automatically
                persist state when this is raised.
        """
        ...
