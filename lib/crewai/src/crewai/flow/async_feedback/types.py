"""Core types for async human feedback in Flows.

This module defines the protocol, exception, and context types used for
non-blocking human-in-the-loop workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from crewai.flow.flow import Flow


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
        pending_events: Each listener's join record when the flow paused - the
            triggers it has seen since it last ran - restored on resume.
        fired_or_listeners: Multi-event ``or_`` listeners that had already fired
            when the flow paused, restored on resume.
        completed_methods: Methods completed when the flow paused, restored on
            resume.
        cut_triggers: Triggers whose listener pass the pause cut short; on
            resume they are dispatched to the joins of the reviewed method.
        carried_cut_triggers: Triggers still cut short by an earlier pause, each
            mapped to the reviewed methods whose joins were already given it;
            on resume they are dispatched to the joins of the reviewed method
            that name none of those methods.
        cut_trigger_payloads: The payload each cut or carried trigger's listener
            pass would have passed, handed to the joins it completes on resume.
            Restored from JSON: a ``HumanFeedbackResult`` is rebuilt, any other
            value arrives in its JSON form.

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
    pending_events: dict[str, list[str]] = field(default_factory=dict)
    fired_or_listeners: list[str] = field(default_factory=list)
    completed_methods: list[str] = field(default_factory=list)
    cut_triggers: list[str] = field(default_factory=list)
    carried_cut_triggers: dict[str, list[str]] = field(default_factory=dict)
    cut_trigger_payloads: dict[str, Any] = field(default_factory=dict)

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

    @staticmethod
    def _keyed_json_safe(value: Any) -> Any:
        """``_make_json_safe`` with each dict key JSON cannot hold made its ``str``."""
        if isinstance(value, dict):
            return {
                key
                if key is None or isinstance(key, (str, int, float, bool))
                else str(key): PendingFeedbackContext._keyed_json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [PendingFeedbackContext._keyed_json_safe(item) for item in value]
        return PendingFeedbackContext._make_json_safe(value)

    @staticmethod
    def _payload_to_row(payload: Any) -> dict[str, Any]:
        """Row form of a cut trigger's payload; never raises, so a pause always persists."""
        from crewai.flow.human_feedback import HumanFeedbackResult

        safe = PendingFeedbackContext._keyed_json_safe
        try:
            if isinstance(payload, HumanFeedbackResult):
                row: dict[str, Any] = {
                    "human_feedback_result": {
                        "output": safe(payload.output),
                        "feedback": payload.feedback,
                        "outcome": payload.outcome,
                        "timestamp": payload.timestamp.isoformat(),
                        "method_name": payload.method_name,
                        "metadata": safe(payload.metadata),
                    }
                }
            else:
                row = {"value": safe(payload)}
            json.dumps(row)
        except Exception:
            try:
                text = str(payload)
            except Exception:
                text = object.__repr__(payload)
            row = {"value": text}
        return row

    @staticmethod
    def _payload_from_row(row: dict[str, Any]) -> Any:
        from crewai.flow.human_feedback import HumanFeedbackResult

        fields = row.get("human_feedback_result")
        if fields is None:
            return row.get("value")
        return HumanFeedbackResult(
            **{**fields, "timestamp": datetime.fromisoformat(fields["timestamp"])}
        )

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
            "pending_events": {
                key: list(seen) for key, seen in self.pending_events.items()
            },
            "fired_or_listeners": list(self.fired_or_listeners),
            "completed_methods": list(self.completed_methods),
            "cut_triggers": list(self.cut_triggers),
            "carried_cut_triggers": {
                trigger: list(credited)
                for trigger, credited in self.carried_cut_triggers.items()
            },
            "cut_trigger_payloads": {
                trigger: self._payload_to_row(payload)
                for trigger, payload in self.cut_trigger_payloads.items()
            },
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
            pending_events={
                key: list(seen)
                for key, seen in (data.get("pending_events") or {}).items()
            },
            fired_or_listeners=list(data.get("fired_or_listeners") or []),
            completed_methods=list(data.get("completed_methods") or []),
            cut_triggers=list(data.get("cut_triggers") or []),
            carried_cut_triggers={
                trigger: list(credited)
                for trigger, credited in (
                    data.get("carried_cut_triggers") or {}
                ).items()
            },
            cut_trigger_payloads={
                trigger: cls._payload_from_row(row)
                for trigger, row in (data.get("cut_trigger_payloads") or {}).items()
            },
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
