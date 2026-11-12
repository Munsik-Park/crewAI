"""Tests for async human feedback functionality.

This module tests the async/non-blocking human feedback flow, including:
- PendingFeedbackContext creation and serialization
- HumanFeedbackPending exception handling
- HumanFeedbackProvider protocol
- ConsoleProvider
- Flow.from_pending() and Flow.resume()
- SQLite persistence with pending feedback
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from crewai.flow import (
    Flow,
    HumanFeedbackResult,
    and_,
    human_feedback,
    listen,
    or_,
    router,
    start,
)
from crewai.flow.async_feedback import (
    ConsoleProvider,
    HumanFeedbackPending,
    HumanFeedbackProvider,
    PendingFeedbackContext,
)
from crewai.flow.persistence import SQLiteFlowPersistence


# PendingFeedbackContext Tests


class TestPendingFeedbackContext:
    """Tests for PendingFeedbackContext dataclass."""

    def test_create_basic_context(self) -> None:
        """Test creating a basic pending feedback context."""
        context = PendingFeedbackContext(
            flow_id="test-flow-123",
            flow_class="myapp.flows.ReviewFlow",
            method_name="review_content",
            method_output="Content to review",
            message="Please review this content:",
        )

        assert context.flow_id == "test-flow-123"
        assert context.flow_class == "myapp.flows.ReviewFlow"
        assert context.method_name == "review_content"
        assert context.method_output == "Content to review"
        assert context.message == "Please review this content:"
        assert context.emit is None
        assert context.default_outcome is None
        assert context.metadata == {}
        assert isinstance(context.requested_at, datetime)

    def test_create_context_with_emit(self) -> None:
        """Test creating context with routing outcomes."""
        context = PendingFeedbackContext(
            flow_id="test-flow-456",
            flow_class="myapp.flows.ApprovalFlow",
            method_name="submit_for_approval",
            method_output={"document": "content"},
            message="Approve or reject:",
            emit=["approved", "rejected", "needs_revision"],
            default_outcome="needs_revision",
            llm="gpt-4o-mini",
        )

        assert context.emit == ["approved", "rejected", "needs_revision"]
        assert context.default_outcome == "needs_revision"
        assert context.llm == "gpt-4o-mini"

    def test_to_dict_serialization(self) -> None:
        """Test serializing context to dictionary."""
        context = PendingFeedbackContext(
            flow_id="test-flow-789",
            flow_class="myapp.flows.TestFlow",
            method_name="test_method",
            method_output={"key": "value"},
            message="Test message",
            emit=["yes", "no"],
            metadata={"channel": "#reviews"},
        )

        result = context.to_dict()

        assert result["flow_id"] == "test-flow-789"
        assert result["flow_class"] == "myapp.flows.TestFlow"
        assert result["method_name"] == "test_method"
        assert result["method_output"] == {"key": "value"}
        assert result["message"] == "Test message"
        assert result["emit"] == ["yes", "no"]
        assert result["metadata"] == {"channel": "#reviews"}
        assert "requested_at" in result

    def test_from_dict_deserialization(self) -> None:
        """Test deserializing context from dictionary."""
        data = {
            "flow_id": "test-flow-abc",
            "flow_class": "myapp.flows.TestFlow",
            "method_name": "my_method",
            "method_output": "output value",
            "message": "Feedback message",
            "emit": ["option_a", "option_b"],
            "default_outcome": "option_a",
            "metadata": {"user_id": "123"},
            "llm": "gpt-4o-mini",
            "requested_at": "2024-01-15T10:30:00",
        }

        context = PendingFeedbackContext.from_dict(data)

        assert context.flow_id == "test-flow-abc"
        assert context.flow_class == "myapp.flows.TestFlow"
        assert context.method_name == "my_method"
        assert context.emit == ["option_a", "option_b"]
        assert context.default_outcome == "option_a"
        assert context.llm == "gpt-4o-mini"

    def test_roundtrip_serialization(self) -> None:
        """Test that to_dict/from_dict roundtrips correctly."""
        original = PendingFeedbackContext(
            flow_id="roundtrip-test",
            flow_class="test.TestFlow",
            method_name="test",
            method_output={"nested": {"data": [1, 2, 3]}},
            message="Test",
            emit=["a", "b"],
            metadata={"key": "value"},
        )

        serialized = original.to_dict()
        restored = PendingFeedbackContext.from_dict(serialized)

        assert restored.flow_id == original.flow_id
        assert restored.flow_class == original.flow_class
        assert restored.method_name == original.method_name
        assert restored.method_output == original.method_output
        assert restored.emit == original.emit
        assert restored.metadata == original.metadata


# HumanFeedbackPending Exception Tests


class TestHumanFeedbackPending:
    """Tests for HumanFeedbackPending exception."""

    def test_basic_exception(self) -> None:
        """Test creating basic pending exception."""
        context = PendingFeedbackContext(
            flow_id="exc-test",
            flow_class="test.Flow",
            method_name="method",
            method_output="output",
            message="message",
        )

        exc = HumanFeedbackPending(context=context)

        assert exc.context == context
        assert exc.callback_info == {}
        assert "exc-test" in str(exc)
        assert "method" in str(exc)

    def test_exception_with_callback_info(self) -> None:
        """Test pending exception with callback information."""
        context = PendingFeedbackContext(
            flow_id="callback-test",
            flow_class="test.Flow",
            method_name="method",
            method_output="output",
            message="message",
        )

        exc = HumanFeedbackPending(
            context=context,
            callback_info={
                "webhook_url": "https://example.com/webhook",
                "slack_thread": "123456",
            },
        )

        assert exc.callback_info["webhook_url"] == "https://example.com/webhook"
        assert exc.callback_info["slack_thread"] == "123456"

    def test_exception_with_custom_message(self) -> None:
        """Test pending exception with custom message."""
        context = PendingFeedbackContext(
            flow_id="msg-test",
            flow_class="test.Flow",
            method_name="method",
            method_output="output",
            message="message",
        )

        exc = HumanFeedbackPending(
            context=context,
            message="Custom pending message",
        )

        assert str(exc) == "Custom pending message"

    def test_exception_is_catchable(self) -> None:
        """Test that exception can be caught and handled."""
        context = PendingFeedbackContext(
            flow_id="catch-test",
            flow_class="test.Flow",
            method_name="method",
            method_output="output",
            message="message",
        )

        with pytest.raises(HumanFeedbackPending) as exc_info:
            raise HumanFeedbackPending(context=context)

        assert exc_info.value.context.flow_id == "catch-test"


# HumanFeedbackProvider Protocol Tests


class TestHumanFeedbackProvider:
    """Tests for HumanFeedbackProvider protocol."""

    def test_protocol_compliance_sync_provider(self) -> None:
        """Test that sync provider complies with protocol."""

        class SyncProvider:
            def request_feedback(
                self, context: PendingFeedbackContext, flow: Flow
            ) -> str:
                return "sync feedback"

        provider = SyncProvider()
        assert isinstance(provider, HumanFeedbackProvider)

    def test_protocol_compliance_async_provider(self) -> None:
        """Test that async provider complies with protocol."""

        class AsyncProvider:
            def request_feedback(
                self, context: PendingFeedbackContext, flow: Flow
            ) -> str:
                raise HumanFeedbackPending(context=context)

        provider = AsyncProvider()
        assert isinstance(provider, HumanFeedbackProvider)


# ConsoleProvider Tests


class TestConsoleProvider:
    """Tests for ConsoleProvider."""

    def test_provider_initialization(self) -> None:
        """Test console provider initialization."""
        provider = ConsoleProvider()
        assert provider.verbose is True

        quiet_provider = ConsoleProvider(verbose=False)
        assert quiet_provider.verbose is False



# SQLite Persistence Tests for Async Feedback


class TestSQLitePendingFeedback:
    """Tests for SQLite persistence with pending feedback."""

    def test_save_and_load_pending_feedback(self) -> None:
        """Test saving and loading pending feedback context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            context = PendingFeedbackContext(
                flow_id="persist-test-123",
                flow_class="test.TestFlow",
                method_name="review",
                method_output={"data": "test"},
                message="Review this:",
                emit=["approved", "rejected"],
                llm="gpt-4o-mini",
            )

            state_data = {"counter": 10, "items": ["a", "b"]}

            persistence.save_pending_feedback(
                flow_uuid="persist-test-123",
                context=context,
                state_data=state_data,
            )

            result = persistence.load_pending_feedback("persist-test-123")

            assert result is not None
            loaded_state, loaded_context = result
            assert loaded_state["counter"] == 10
            assert loaded_state["items"] == ["a", "b"]
            assert loaded_context.flow_id == "persist-test-123"
            assert loaded_context.emit == ["approved", "rejected"]

    def test_load_nonexistent_pending_feedback(self) -> None:
        """Test loading pending feedback that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            result = persistence.load_pending_feedback("nonexistent-id")
            assert result is None

    def test_clear_pending_feedback(self) -> None:
        """Test clearing pending feedback after resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            context = PendingFeedbackContext(
                flow_id="clear-test",
                flow_class="test.Flow",
                method_name="method",
                method_output="output",
                message="message",
            )

            persistence.save_pending_feedback(
                flow_uuid="clear-test",
                context=context,
                state_data={"key": "value"},
            )

            assert persistence.load_pending_feedback("clear-test") is not None

            persistence.clear_pending_feedback("clear-test")

            assert persistence.load_pending_feedback("clear-test") is None

    def test_replace_existing_pending_feedback(self) -> None:
        """Test that saving pending feedback replaces existing entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            flow_id = "replace-test"

            context1 = PendingFeedbackContext(
                flow_id=flow_id,
                flow_class="test.Flow",
                method_name="method1",
                method_output="output1",
                message="message1",
            )
            persistence.save_pending_feedback(
                flow_uuid=flow_id,
                context=context1,
                state_data={"version": 1},
            )

            context2 = PendingFeedbackContext(
                flow_id=flow_id,
                flow_class="test.Flow",
                method_name="method2",
                method_output="output2",
                message="message2",
            )
            persistence.save_pending_feedback(
                flow_uuid=flow_id,
                context=context2,
                state_data={"version": 2},
            )

            result = persistence.load_pending_feedback(flow_id)
            assert result is not None
            state, context = result
            assert state["version"] == 2
            assert context.method_name == "method2"


# Custom Async Provider Tests


class TestCustomAsyncProvider:
    """Tests for custom async providers."""

    def test_provider_raises_pending_exception(self) -> None:
        """Test that async provider raises HumanFeedbackPending."""

        class WebhookProvider:
            def __init__(self, webhook_url: str):
                self.webhook_url = webhook_url

            def request_feedback(
                self, context: PendingFeedbackContext, flow: Flow
            ) -> str:
                raise HumanFeedbackPending(
                    context=context,
                    callback_info={"url": f"{self.webhook_url}/{context.flow_id}"},
                )

        provider = WebhookProvider("https://example.com/api")
        context = PendingFeedbackContext(
            flow_id="webhook-test",
            flow_class="test.Flow",
            method_name="method",
            method_output="output",
            message="message",
        )
        mock_flow = MagicMock()

        with pytest.raises(HumanFeedbackPending) as exc_info:
            provider.request_feedback(context, mock_flow)

        assert exc_info.value.callback_info["url"] == (
            "https://example.com/api/webhook-test"
        )


# Flow.from_pending and resume Tests


class TestFlowResumeWithFeedback:
    """Tests for Flow.from_pending and resume."""

    def test_from_pending_uses_default_persistence(self) -> None:
        """Test that from_pending uses SQLiteFlowPersistence by default."""

        class TestFlow(Flow):
            @start()
            def begin(self):
                return "started"

        with pytest.raises(ValueError, match="No pending feedback found"):
            TestFlow.from_pending("nonexistent-id")

    def test_from_pending_raises_for_missing_flow(self) -> None:
        """Test that from_pending raises error for nonexistent flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                @start()
                def begin(self):
                    return "started"

            with pytest.raises(ValueError, match="No pending feedback found"):
                TestFlow.from_pending("nonexistent-id", persistence)

    def test_from_pending_restores_state(self) -> None:
        """Test that from_pending correctly restores flow state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestState(BaseModel):
                id: str = "test-restore-123"
                counter: int = 0

            class TestFlow(Flow[TestState]):
                @start()
                def begin(self):
                    return "started"

            # Manually save pending feedback
            context = PendingFeedbackContext(
                flow_id="test-restore-123",
                flow_class="test.TestFlow",
                method_name="review",
                method_output="content",
                message="Review:",
            )
            persistence.save_pending_feedback(
                flow_uuid="test-restore-123",
                context=context,
                state_data={"id": "test-restore-123", "counter": 42},
            )

            flow = TestFlow.from_pending("test-restore-123", persistence)

            assert flow._pending_feedback_context is not None
            assert flow._pending_feedback_context.flow_id == "test-restore-123"
            assert flow._is_execution_resuming is True
            assert flow.state.counter == 42

    def test_suppressed_resume_emits_a_matched_lifecycle_pair(self) -> None:
        """A suppressed resume emits exactly one start and one finish.

        Both lifecycle events used to be gated on ``suppress_flow_events`` here,
        while the kickoff path emits them either way -- so a suppressed resume was
        invisible in flow telemetry. Ungating only the start would have been worse:
        a start with no terminal event breaks pairing in the other direction.

        The flag is not a gate for emission. It asks for console quiet -- see
        ``_flow_origin`` in ``events/event_listener.py``, which says so and notes it
        "can legitimately be set on a caller's own flow" -- and the listener already
        honours it wherever it prints.

        Counts are asserted exactly, and the resume is a real one that runs to
        completion, so neither an unpaired event nor a swallowed failure can pass.
        """
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.flow_events import (
            FlowFinishedEvent,
            FlowStartedEvent,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SQLiteFlowPersistence(os.path.join(tmpdir, "flows.db"))

            class TestFlow(Flow):
                suppress_flow_events = True

                @start()
                @human_feedback(message="Review this:")
                def generate(self):
                    return "generated content"

                @listen(generate)
                def process(self, feedback_result):
                    return f"Processed: {feedback_result.feedback}"

            context = PendingFeedbackContext(
                flow_id="suppressed-resume-1",
                flow_class="test.TestFlow",
                method_name="generate",
                method_output="generated content",
                message="Review this:",
            )
            persistence.save_pending_feedback(
                flow_uuid="suppressed-resume-1",
                context=context,
                state_data={"id": "suppressed-resume-1"},
            )

            flow = TestFlow.from_pending("suppressed-resume-1", persistence)

            seen: list[str] = []
            original_emit = crewai_event_bus.emit

            def capture_emit(source, event):
                if isinstance(event, FlowStartedEvent):
                    seen.append("started")
                elif isinstance(event, FlowFinishedEvent):
                    seen.append("finished")
                return original_emit(source, event)

            with patch.object(crewai_event_bus, "emit", side_effect=capture_emit):
                result = flow.resume("looks good!")

            assert result is not None
            assert seen.count("started") == 1, (
                f"expected exactly one flow_started on a suppressed resume, got {seen}"
            )
            assert seen.count("finished") == 1, (
                f"expected exactly one flow_finished to pair with it, got {seen} -- "
                f"an unpaired event breaks started/finished pairing and every "
                f"duration built on it"
            )

    def test_resume_without_pending_raises_error(self) -> None:
        """Test that resume raises error without pending context."""

        class TestFlow(Flow):
            @start()
            def begin(self):
                return "started"

        flow = TestFlow()

        with pytest.raises(ValueError, match="No pending feedback context"):
            flow.resume("some feedback")

    def test_resume_from_async_context_raises_error(self) -> None:
        """Test that resume() raises RuntimeError when called from async context."""
        import asyncio

        class TestFlow(Flow):
            @start()
            def begin(self):
                return "started"

        async def call_resume_from_async():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = os.path.join(tmpdir, "test.db")
                persistence = SQLiteFlowPersistence(db_path)

                context = PendingFeedbackContext(
                    flow_id="async-context-test",
                    flow_class="TestFlow",
                    method_name="begin",
                    method_output="output",
                    message="Review:",
                )
                persistence.save_pending_feedback(
                    flow_uuid="async-context-test",
                    context=context,
                    state_data={"id": "async-context-test"},
                )

                flow = TestFlow.from_pending("async-context-test", persistence)

                # This should raise RuntimeError because we're in an async context
                with pytest.raises(RuntimeError, match="cannot be called from within an async context"):
                    flow.resume("feedback")

        asyncio.run(call_resume_from_async())

    @pytest.mark.asyncio
    async def test_resume_async_direct(self) -> None:
        """Test resume_async() can be called directly in async context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                @start()
                @human_feedback(message="Review:")
                def generate(self):
                    return "content"

                @listen(generate)
                def process(self, result):
                    return f"processed: {result.feedback}"

            context = PendingFeedbackContext(
                flow_id="async-direct-test",
                flow_class="TestFlow",
                method_name="generate",
                method_output="content",
                message="Review:",
            )
            persistence.save_pending_feedback(
                flow_uuid="async-direct-test",
                context=context,
                state_data={"id": "async-direct-test"},
            )

            flow = TestFlow.from_pending("async-direct-test", persistence)

            with patch("crewai.flow.runtime.crewai_event_bus.emit"):
                result = await flow.resume_async("async feedback")

            assert flow.last_human_feedback is not None
            assert flow.last_human_feedback.feedback == "async feedback"

    @patch("crewai.flow.runtime.crewai_event_bus.emit")
    def test_resume_basic(self, mock_emit: MagicMock) -> None:
        """Test basic resume functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                @start()
                @human_feedback(message="Review this:")
                def generate(self):
                    return "generated content"

                @listen(generate)
                def process(self, feedback_result):
                    return f"Processed: {feedback_result.feedback}"

            # Manually save pending feedback (simulating async pause)
            context = PendingFeedbackContext(
                flow_id="resume-test-123",
                flow_class="test.TestFlow",
                method_name="generate",
                method_output="generated content",
                message="Review this:",
            )
            persistence.save_pending_feedback(
                flow_uuid="resume-test-123",
                context=context,
                state_data={"id": "resume-test-123"},
            )

            flow = TestFlow.from_pending("resume-test-123", persistence)
            result = flow.resume("looks good!")

            assert flow.last_human_feedback is not None
            assert flow.last_human_feedback.feedback == "looks good!"
            assert flow.last_human_feedback.output == "generated content"

            assert persistence.load_pending_feedback("resume-test-123") is None

    @patch("crewai.flow.runtime.crewai_event_bus.emit")
    def test_terminal_resume_without_emit_returns_feedback_result(
        self, mock_emit: MagicMock
    ) -> None:
        """Terminal resumed non-emit methods return the full feedback result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                @start()
                @human_feedback(message="Review this:", metadata={"stage": "draft"})
                def generate(self):
                    return {"content": "generated content"}

            context = PendingFeedbackContext(
                flow_id="terminal-non-emit-test-123",
                flow_class="test.TestFlow",
                method_name="generate",
                method_output={"content": "generated content"},
                message="Review this:",
                metadata={"stage": "draft"},
            )
            persistence.save_pending_feedback(
                flow_uuid="terminal-non-emit-test-123",
                context=context,
                state_data={"id": "terminal-non-emit-test-123"},
            )

            flow = TestFlow.from_pending("terminal-non-emit-test-123", persistence)
            result = flow.resume("looks good!")

            assert isinstance(result, HumanFeedbackResult)
            assert result.output == {"content": "generated content"}
            assert result.feedback == "looks good!"
            assert result.outcome is None
            assert result.metadata == {"stage": "draft"}
            assert flow.method_outputs == [result]

    @patch("crewai.flow.runtime.crewai_event_bus.emit")
    def test_resume_routing(self, mock_emit: MagicMock) -> None:
        """Test resume with routing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                result_path: str = ""

                @start()
                @human_feedback(
                    message="Approve?",
                    emit=["approved", "rejected"],
                    llm="gpt-4o-mini",
                )
                def review(self):
                    return "content"

                @listen("approved")
                def handle_approved(self):
                    self.result_path = "approved"
                    return "Approved!"

                @listen("rejected")
                def handle_rejected(self):
                    self.result_path = "rejected"
                    return "Rejected!"

            context = PendingFeedbackContext(
                flow_id="route-test-123",
                flow_class="test.TestFlow",
                method_name="review",
                method_output="content",
                message="Approve?",
                emit=["approved", "rejected"],
                llm="gpt-4o-mini",
            )
            persistence.save_pending_feedback(
                flow_uuid="route-test-123",
                context=context,
                state_data={"id": "route-test-123"},
            )

            flow = TestFlow.from_pending("route-test-123", persistence)

            with patch.object(flow, "_collapse_to_outcome", return_value="approved"):
                result = flow.resume("yes, this looks great")

            assert flow.last_human_feedback.outcome == "approved"
            assert flow.result_path == "approved"

    @patch("crewai.flow.runtime.crewai_event_bus.emit")
    def test_terminal_resume_with_emit_returns_method_output(
        self, mock_emit: MagicMock
    ) -> None:
        """Terminal resumed emit methods return the original method output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)
            method_output = {"content": "original content", "status": "ready"}

            class TestFlow(Flow):
                @start()
                @human_feedback(
                    message="Approve?",
                    emit=["approved", "rejected"],
                    llm="gpt-4o-mini",
                )
                def review(self):
                    return method_output

            context = PendingFeedbackContext(
                flow_id="terminal-route-test-123",
                flow_class="test.TestFlow",
                method_name="review",
                method_output=method_output,
                message="Approve?",
                emit=["approved", "rejected"],
                llm="gpt-4o-mini",
            )
            persistence.save_pending_feedback(
                flow_uuid="terminal-route-test-123",
                context=context,
                state_data={"id": "terminal-route-test-123"},
            )

            flow = TestFlow.from_pending("terminal-route-test-123", persistence)

            with patch.object(flow, "_collapse_to_outcome", return_value="approved"):
                result = flow.resume("yes, this looks great")

            assert result == method_output
            assert flow.method_outputs == [method_output]
            assert flow.last_human_feedback.outcome == "approved"

    @patch("crewai.flow.runtime.crewai_event_bus.emit")
    def test_resume_records_method_output_before_downstream_listeners(
        self, mock_emit: MagicMock
    ) -> None:
        """Downstream listeners can read outputs from the resumed method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                @start()
                @human_feedback(message="Review:")
                def review(self):
                    return "generated content"

                @listen(review)
                def downstream(self, result):
                    self.state["seen_outputs"] = self.method_outputs
                    return f"downstream:{result.output}"

            context = PendingFeedbackContext(
                flow_id="listener-output-test-123",
                flow_class="test.TestFlow",
                method_name="review",
                method_output="generated content",
                message="Review:",
            )
            persistence.save_pending_feedback(
                flow_uuid="listener-output-test-123",
                context=context,
                state_data={"id": "listener-output-test-123"},
            )

            flow = TestFlow.from_pending("listener-output-test-123", persistence)
            result = flow.resume("looks good")

            assert result == "downstream:generated content"
            assert len(flow.state["seen_outputs"]) == 1
            seen_output = flow.state["seen_outputs"][0]
            assert isinstance(seen_output, HumanFeedbackResult)
            assert seen_output.output == "generated content"
            assert seen_output.feedback == "looks good"


# Integration Tests with @human_feedback decorator


class TestAsyncHumanFeedbackIntegration:
    """Integration tests for async human feedback with decorator."""

    def test_decorator_with_provider_parameter(self) -> None:
        """Test that decorator accepts provider parameter."""

        class MockProvider:
            def request_feedback(
                self, context: PendingFeedbackContext, flow: Flow
            ) -> str:
                raise HumanFeedbackPending(context=context)

        class TestFlow(Flow):
            @start()
            @human_feedback(
                message="Review:",
                provider=MockProvider(),
            )
            def review(self):
                return "content"

        flow = TestFlow()
        method = getattr(flow, "review")
        assert hasattr(method, "__human_feedback_config__")
        assert method.__human_feedback_config__.provider is not None

    @patch("crewai.flow.runtime.crewai_event_bus.emit")
    def test_async_provider_pauses_flow(self, mock_emit: MagicMock) -> None:
        """Test that async provider pauses flow execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class PausingProvider:
                def __init__(self, persistence: SQLiteFlowPersistence):
                    self.persistence = persistence

                def request_feedback(
                    self, context: PendingFeedbackContext, flow: Flow
                ) -> str:
                    self.persistence.save_pending_feedback(
                        flow_uuid=context.flow_id,
                        context=context,
                        state_data=flow.state if isinstance(flow.state, dict) else flow.state.model_dump(),
                    )
                    raise HumanFeedbackPending(
                        context=context,
                        callback_info={"saved": True},
                    )

            class TestFlow(Flow):
                @start()
                @human_feedback(
                    message="Review:",
                    provider=PausingProvider(persistence),
                )
                def generate(self):
                    return "generated content"

            flow = TestFlow(persistence=persistence)

            # kickoff now returns HumanFeedbackPending instead of raising it
            result = flow.kickoff()

            assert isinstance(result, HumanFeedbackPending)
            assert result.callback_info["saved"] is True

            flow_id = result.context.flow_id

            persisted = persistence.load_pending_feedback(flow_id)
            assert persisted is not None

    @patch("crewai.flow.runtime.crewai_event_bus.emit")
    def test_full_async_flow_cycle(self, mock_emit: MagicMock) -> None:
        """Test complete async flow: start -> pause -> resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            flow_id_holder: list[str] = []

            class SaveAndPauseProvider:
                def __init__(self, persistence: SQLiteFlowPersistence):
                    self.persistence = persistence

                def request_feedback(
                    self, context: PendingFeedbackContext, flow: Flow
                ) -> str:
                    flow_id_holder.append(context.flow_id)
                    self.persistence.save_pending_feedback(
                        flow_uuid=context.flow_id,
                        context=context,
                        state_data=flow.state if isinstance(flow.state, dict) else flow.state.model_dump(),
                    )
                    raise HumanFeedbackPending(context=context)

            class ReviewFlow(Flow):
                processed_feedback: str = ""

                @start()
                @human_feedback(
                    message="Review this content:",
                    provider=SaveAndPauseProvider(persistence),
                )
                def generate(self):
                    return "AI generated content"

                @listen(generate)
                def process(self, feedback_result):
                    self.processed_feedback = feedback_result.feedback
                    return f"Final: {feedback_result.feedback}"

            flow1 = ReviewFlow(persistence=persistence)
            result = flow1.kickoff()

            # kickoff now returns HumanFeedbackPending instead of raising it
            assert isinstance(result, HumanFeedbackPending)
            assert len(flow_id_holder) == 1
            paused_flow_id = flow_id_holder[0]

            flow2 = ReviewFlow.from_pending(paused_flow_id, persistence)
            result = flow2.resume("This is my feedback")

            assert flow2.last_human_feedback.feedback == "This is my feedback"
            assert flow2.processed_feedback == "This is my feedback"


# Edge Case Tests


class TestAutoPersistence:
    """Tests for automatic persistence when no persistence is provided."""

    @patch("crewai.flow.runtime.crewai_event_bus.emit")
    def test_auto_persistence_when_none_provided(self, mock_emit: MagicMock) -> None:
        """Test that persistence is auto-created when HumanFeedbackPending is raised."""

        class PausingProvider:
            def request_feedback(
                self, context: PendingFeedbackContext, flow: Flow
            ) -> str:
                raise HumanFeedbackPending(
                    context=context,
                    callback_info={"paused": True},
                )

        class TestFlow(Flow):
            @start()
            @human_feedback(
                message="Review:",
                provider=PausingProvider(),
            )
            def generate(self):
                return "content"

        flow = TestFlow()
        assert flow.persistence is None

        # kickoff should auto-create persistence when HumanFeedbackPending is raised
        result = flow.kickoff()

        assert isinstance(result, HumanFeedbackPending)

        # Persistence should have been auto-created
        assert flow.persistence is not None

        flow_id = result.context.flow_id
        loaded = flow.persistence.load_pending_feedback(flow_id)
        assert loaded is not None


class TestCollapseToOutcomeJsonParsing:
    """Tests for _collapse_to_outcome JSON parsing edge cases."""

    def test_json_string_response_is_parsed(self) -> None:
        """Test that JSON string response from LLM is correctly parsed."""
        flow = Flow()

        with patch("crewai.llm.LLM") as MockLLM:
            mock_llm = MagicMock()
            # Simulate LLM returning JSON string (the bug we fixed)
            mock_llm.call.return_value = '{"outcome": "approved"}'
            MockLLM.return_value = mock_llm

            result = flow._collapse_to_outcome(
                feedback="I approve this",
                outcomes=["approved", "rejected"],
                llm="gpt-4o-mini",
            )

        assert result == "approved"

    def test_plain_string_response_is_matched(self) -> None:
        """Test that plain string response is correctly matched."""
        flow = Flow()

        with patch("crewai.llm.LLM") as MockLLM:
            mock_llm = MagicMock()
            # Simulate LLM returning plain outcome string
            mock_llm.call.return_value = "rejected"
            MockLLM.return_value = mock_llm

            result = flow._collapse_to_outcome(
                feedback="This is not good",
                outcomes=["approved", "rejected"],
                llm="gpt-4o-mini",
            )

        assert result == "rejected"

    def test_invalid_json_falls_back_to_matching(self) -> None:
        """Test that invalid JSON falls back to string matching."""
        flow = Flow()

        with patch("crewai.llm.LLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.call.return_value = "{invalid json but says approved"
            MockLLM.return_value = mock_llm

            result = flow._collapse_to_outcome(
                feedback="looks good",
                outcomes=["approved", "rejected"],
                llm="gpt-4o-mini",
            )

        assert result == "approved"

    def test_json_without_outcome_raises(self) -> None:
        """Parsed JSON missing a usable outcome fails closed."""
        from crewai.flow.human_feedback import HumanFeedbackCollapseError

        flow = Flow()

        with patch("crewai.llm.LLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.call.return_value = '{"foo": "bar"}'
            MockLLM.return_value = mock_llm

            with pytest.raises(HumanFeedbackCollapseError, match="Could not match"):
                flow._collapse_to_outcome(
                    feedback="looks good",
                    outcomes=["approved", "rejected"],
                    llm="gpt-4o-mini",
                )

    def test_non_object_json_does_not_attribute_error(self) -> None:
        """JSON that is not an object is matched as raw text, not parsed.get."""
        from crewai.flow.human_feedback import HumanFeedbackCollapseError

        flow = Flow()

        with patch("crewai.llm.LLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.call.return_value = "null"
            MockLLM.return_value = mock_llm

            with pytest.raises(HumanFeedbackCollapseError, match="Could not match"):
                flow._collapse_to_outcome(
                    feedback="looks good",
                    outcomes=["approved", "rejected"],
                    llm="gpt-4o-mini",
                )

    def test_llm_exception_falls_back_to_simple_prompting(self) -> None:
        """Test that LLM exception triggers fallback to simple prompting."""
        flow = Flow()

        with patch("crewai.llm.LLM") as MockLLM:
            mock_llm = MagicMock()
            # First call raises, second call succeeds (fallback)
            mock_llm.call.side_effect = [
                Exception("Structured output failed"),
                "approved",
            ]
            MockLLM.return_value = mock_llm

            result = flow._collapse_to_outcome(
                feedback="I approve",
                outcomes=["approved", "rejected"],
                llm="gpt-4o-mini",
            )

        assert result == "approved"
        # Verify it was called twice (initial + fallback)
        assert mock_llm.call.call_count == 2


class TestLLMObjectPreservedInContext:
    """Tests that BaseLLM objects have their model string preserved in PendingFeedbackContext."""

    @patch("crewai.flow.runtime.crewai_event_bus.emit")
    def test_basellm_object_model_string_survives_roundtrip(self, mock_emit: MagicMock) -> None:
        """Test that when llm is a BaseLLM object, its model string is stored in context
        so that outcome collapsing works after async pause/resume.

        This is the exact bug: locally the sync path keeps the LLM object in memory,
        but in production the async path serializes the context and the LLM object was
        discarded (stored as None), causing resume to skip classification and always
        fall back to emit[0].
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            from crewai.llm import LLM
            mock_llm_obj = LLM(
                model="llama3",
                provider="ollama",
                base_url="http://localhost:11434",
            )

            class PausingProvider:
                def __init__(self, persistence: SQLiteFlowPersistence):
                    self.persistence = persistence
                    self.captured_context: PendingFeedbackContext | None = None

                def request_feedback(
                    self, context: PendingFeedbackContext, flow: Flow
                ) -> str:
                    self.captured_context = context
                    self.persistence.save_pending_feedback(
                        flow_uuid=context.flow_id,
                        context=context,
                        state_data=flow.state if isinstance(flow.state, dict) else flow.state.model_dump(),
                    )
                    raise HumanFeedbackPending(context=context)

            provider = PausingProvider(persistence)

            class TestFlow(Flow):
                result_path: str = ""

                @start()
                @human_feedback(
                    message="Approve?",
                    emit=["needs_changes", "approved"],
                    llm=mock_llm_obj,
                    default_outcome="approved",
                    provider=provider,
                )
                def review(self):
                    return "content for review"

                @listen("approved")
                def handle_approved(self):
                    self.result_path = "approved"
                    return "Approved!"

                @listen("needs_changes")
                def handle_changes(self):
                    self.result_path = "needs_changes"
                    return "Changes needed"

            flow1 = TestFlow(persistence=persistence)
            result = flow1.kickoff()
            assert isinstance(result, HumanFeedbackPending)

            assert provider.captured_context is not None
            assert isinstance(provider.captured_context.llm, dict)
            assert provider.captured_context.llm["model"] == "ollama/llama3"

            flow_id = result.context.flow_id
            loaded = persistence.load_pending_feedback(flow_id)
            assert loaded is not None
            _, loaded_context = loaded
            assert isinstance(loaded_context.llm, dict)
            assert loaded_context.llm["model"] == "ollama/llama3"

            flow2 = TestFlow.from_pending(flow_id, persistence)
            assert flow2._pending_feedback_context is not None
            assert isinstance(flow2._pending_feedback_context.llm, dict)
            assert flow2._pending_feedback_context.llm["model"] == "ollama/llama3"

            with patch.object(flow2, "_collapse_to_outcome", return_value="approved") as mock_collapse:
                flow2.resume("this looks good, proceed!")

            # The key assertion: _collapse_to_outcome was called (not skipped due to llm=None)
            mock_collapse.assert_called_once()
            call_kwargs = mock_collapse.call_args
            assert call_kwargs.kwargs["feedback"] == "this looks good, proceed!"
            assert call_kwargs.kwargs["outcomes"] == ["needs_changes", "approved"]
            # LLM should be a live object (from _human_feedback_llm) or reconstructed, not None
            assert call_kwargs.kwargs["llm"] is not None
            assert getattr(call_kwargs.kwargs["llm"], "model", None) == "llama3"
            assert flow2.last_human_feedback.outcome == "approved"
            assert flow2.result_path == "approved"

    def test_string_llm_still_works(self) -> None:
        """Test that passing llm as a string still works correctly."""
        context = PendingFeedbackContext(
            flow_id="str-llm-test",
            flow_class="test.Flow",
            method_name="review",
            method_output="output",
            message="Review:",
            emit=["approved", "rejected"],
            llm="gpt-4o-mini",
        )

        serialized = context.to_dict()
        restored = PendingFeedbackContext.from_dict(serialized)
        assert restored.llm == "gpt-4o-mini"

    def test_none_llm_when_no_model_attr(self) -> None:
        """Test that llm is None when object has no model attribute."""
        from crewai.flow.human_feedback import _serialize_llm_for_context

        mock_obj = MagicMock(spec=[])
        assert _serialize_llm_for_context(mock_obj) is None

    def test_provider_prefix_added_to_bare_model(self) -> None:
        """Test that provider prefix is added when model has no slash."""
        from crewai.flow.human_feedback import _serialize_llm_for_context
        from crewai.llm import LLM

        llm = LLM(
            model="llama3",
            provider="ollama",
            base_url="http://localhost:11434",
        )
        result = _serialize_llm_for_context(llm)
        assert isinstance(result, dict)
        assert result["model"] == "ollama/llama3"

    def test_provider_prefix_not_doubled_when_already_present(self) -> None:
        """Test that provider prefix is not added when model already has a slash."""
        from crewai.flow.human_feedback import _serialize_llm_for_context
        from crewai.llm import LLM

        llm = LLM(model="ollama/llama3", base_url="http://localhost:11434")
        result = _serialize_llm_for_context(llm)
        assert isinstance(result, dict)
        assert result["model"] == "ollama/llama3"

    def test_no_provider_attr_falls_back_to_bare_model(self) -> None:
        """Test that objects without to_config_dict fall back to model string."""
        from crewai.flow.human_feedback import _serialize_llm_for_context

        mock_obj = MagicMock(spec=[])
        mock_obj.model = "gpt-4o-mini"
        assert _serialize_llm_for_context(mock_obj) == "gpt-4o-mini"


class TestAsyncHumanFeedbackEdgeCases:
    """Edge case tests for async human feedback."""

    def test_pending_context_with_complex_output(self) -> None:
        """Test context with complex nested output."""
        complex_output = {
            "items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}],
            "metadata": {"total": 2, "page": 1},
            "nested": {"deep": {"value": "test"}},
        }

        context = PendingFeedbackContext(
            flow_id="complex-test",
            flow_class="test.Flow",
            method_name="method",
            method_output=complex_output,
            message="Review:",
        )

        # Serialize and deserialize
        serialized = context.to_dict()
        json_str = json.dumps(serialized)
        restored = PendingFeedbackContext.from_dict(json.loads(json_str))

        assert restored.method_output == complex_output

    def test_empty_feedback_uses_default_outcome(self) -> None:
        """Test that empty feedback uses default outcome during resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                @start()
                def generate(self):
                    return "content"

            context = PendingFeedbackContext(
                flow_id="default-test",
                flow_class="test.Flow",
                method_name="generate",
                method_output="content",
                message="Review:",
                emit=["approved", "rejected"],
                default_outcome="approved",
                llm="gpt-4o-mini",
            )
            persistence.save_pending_feedback(
                flow_uuid="default-test",
                context=context,
                state_data={"id": "default-test"},
            )

            flow = TestFlow.from_pending("default-test", persistence)

            with patch("crewai.flow.runtime.crewai_event_bus.emit"):
                result = flow.resume("")

            assert flow.last_human_feedback.outcome == "approved"

    def test_resume_without_feedback_uses_default(self) -> None:
        """Test that resume() can be called without feedback argument."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                @start()
                def step(self):
                    return "output"

            context = PendingFeedbackContext(
                flow_id="no-feedback-test",
                flow_class="TestFlow",
                method_name="step",
                method_output="test output",
                message="Review:",
                emit=["approved", "rejected"],
                default_outcome="approved",
                llm="gpt-4o-mini",
            )
            persistence.save_pending_feedback(
                flow_uuid="no-feedback-test",
                context=context,
                state_data={"id": "no-feedback-test"},
            )

            flow = TestFlow.from_pending("no-feedback-test", persistence)

            with patch("crewai.flow.runtime.crewai_event_bus.emit"):
                result = flow.resume()

            assert flow.last_human_feedback.outcome == "approved"
            assert flow.last_human_feedback.feedback == ""




class TestResumeLLMFromSerializedContext:
    """Resume rebuilds the collapse LLM from the serialized context alone."""

    @patch("crewai.flow.runtime.crewai_event_bus.emit")
    def test_resume_builds_llm_from_serialized_context(
        self, mock_emit: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                @start()
                @human_feedback(
                    message="Approve?",
                    emit=["approved", "rejected"],
                    llm="gpt-4o-mini",
                )
                def review(self):
                    return "content"

            context = PendingFeedbackContext(
                flow_id="fallback-test",
                flow_class="TestFlow",
                method_name="review",
                method_output="content",
                message="Approve?",
                emit=["approved", "rejected"],
                llm="gpt-4o-mini",
            )
            persistence.save_pending_feedback(
                flow_uuid="fallback-test",
                context=context,
                state_data={"id": "fallback-test"},
            )

            flow = TestFlow.from_pending("fallback-test", persistence)

            captured_llm = []

            def capture_llm(feedback, outcomes, llm):
                captured_llm.append(llm)
                return "approved"

            with patch.object(flow, "_collapse_to_outcome", side_effect=capture_llm):
                flow.resume("looks good!")

            assert len(captured_llm) == 1
            from crewai.llms.base_llm import BaseLLM as BaseLLMClass
            assert isinstance(captured_llm[0], BaseLLMClass)
            assert captured_llm[0].model == "gpt-4o-mini"


# learn=True on resumed feedback


def _pause(tmpdir: str, flow_cls: type[Flow], flow_id: str, **context: Any) -> Flow:
    """Persist a pending review for ``flow_cls.draft`` and restore it, as AMP does."""
    persistence = SQLiteFlowPersistence(os.path.join(tmpdir, "flows.db"))
    persistence.save_pending_feedback(
        flow_uuid=flow_id,
        context=PendingFeedbackContext(
            flow_id=flow_id,
            flow_class=f"test.{flow_cls.__name__}",
            method_name="draft",
            method_output="reviewed draft",
            message="Review:",
            llm="gpt-4o-mini",
            **context,
        ),
        state_data={"id": flow_id},
    )
    flow = flow_cls.from_pending(flow_id, persistence)
    flow.memory = MagicMock()
    flow.memory.recall.return_value = []
    return flow


def _distill_llm(*responses: Any) -> MagicMock:
    llm = MagicMock()
    llm.supports_function_calling.return_value = True
    llm.call.side_effect = list(responses)
    return llm


class TestLearnOnResume:
    """A reply that resumes a paused flow teaches lessons like an in-run reply."""

    def test_resume_stores_lessons_distilled_from_the_reply(self) -> None:
        from crewai.flow.human_feedback import DistilledLessons

        class LearnFlow(Flow):
            @start()
            @human_feedback(message="Review:", llm="gpt-4o-mini", learn=True)
            def draft(self):
                return "draft"

        with tempfile.TemporaryDirectory() as tmpdir:
            flow = _pause(tmpdir, LearnFlow, "learn-resume-1")
            llm = _distill_llm(DistilledLessons(lessons=["Cite a source for every claim"]))
            with patch("crewai.llm.LLM", return_value=llm):
                flow.resume("Always cite your sources")

        flow.memory.remember_many.assert_called_once_with(
            ["Cite a source for every claim"], source="hitl"
        )
        distill_prompt = llm.call.call_args.args[0][1]["content"]
        assert "reviewed draft" in distill_prompt
        assert "Always cite your sources" in distill_prompt
        assert flow.last_human_feedback.feedback == "Always cite your sources"

    def test_resume_async_stores_lessons(self) -> None:
        import asyncio

        from crewai.flow.human_feedback import DistilledLessons

        class LearnFlow(Flow):
            @start()
            @human_feedback(message="Review:", llm="gpt-4o-mini", learn=True)
            def draft(self):
                return "draft"

        with tempfile.TemporaryDirectory() as tmpdir:
            flow = _pause(tmpdir, LearnFlow, "learn-resume-async")
            llm = _distill_llm(DistilledLessons(lessons=["Keep intros under 50 words"]))
            with patch("crewai.llm.LLM", return_value=llm):
                asyncio.run(flow.resume_async("The intro is too long"))

        flow.memory.remember_many.assert_called_once_with(
            ["Keep intros under 50 words"], source="hitl"
        )

    def test_resume_routes_on_the_reply_and_stores_lessons(self) -> None:
        from crewai.flow.human_feedback import DistilledLessons

        class RoutedFlow(Flow):
            published: str = ""

            @start()
            @human_feedback(
                message="Review:",
                emit=["approved", "needs_revision"],
                llm="gpt-4o-mini",
                learn=True,
            )
            def draft(self):
                return "draft"

            @listen("approved")
            def publish(self):
                self.published = self.last_human_feedback.output

        with tempfile.TemporaryDirectory() as tmpdir:
            flow = _pause(
                tmpdir, RoutedFlow, "learn-resume-emit", emit=["approved", "needs_revision"]
            )
            llm = _distill_llm(DistilledLessons(lessons=["Prefer active voice"]))
            with (
                patch.object(flow, "_collapse_to_outcome", return_value="approved"),
                patch("crewai.llm.LLM", return_value=llm),
            ):
                flow.resume("Approved, but prefer active voice next time")

        assert flow.published == "reviewed draft"
        flow.memory.remember_many.assert_called_once_with(
            ["Prefer active voice"], source="hitl"
        )

    def test_resume_stores_lessons_under_the_configured_source(self) -> None:
        from crewai.flow.human_feedback import DistilledLessons

        class SourcedFlow(Flow):
            @start()
            @human_feedback(
                message="Review:",
                llm="gpt-4o-mini",
                learn=True,
                learn_source="editorial",
            )
            def draft(self):
                return "draft"

        with tempfile.TemporaryDirectory() as tmpdir:
            flow = _pause(tmpdir, SourcedFlow, "learn-resume-source")
            llm = _distill_llm(DistilledLessons(lessons=["Use the house style guide"]))
            with patch("crewai.llm.LLM", return_value=llm):
                flow.resume("Follow the house style")

        assert flow.memory.remember_many.call_args.kwargs == {"source": "editorial"}

    def test_resume_with_empty_reply_stores_nothing(self) -> None:
        class LearnFlow(Flow):
            @start()
            @human_feedback(message="Review:", llm="gpt-4o-mini", learn=True)
            def draft(self):
                return "draft"

        with tempfile.TemporaryDirectory() as tmpdir:
            flow = _pause(tmpdir, LearnFlow, "learn-resume-empty")
            llm = _distill_llm()
            with patch("crewai.llm.LLM", return_value=llm):
                flow.resume("")

        llm.call.assert_not_called()
        flow.memory.remember_many.assert_not_called()

    def test_resume_without_learn_leaves_memory_untouched(self) -> None:
        class PlainFlow(Flow):
            @start()
            @human_feedback(message="Review:", llm="gpt-4o-mini")
            def draft(self):
                return "draft"

        with tempfile.TemporaryDirectory() as tmpdir:
            flow = _pause(tmpdir, PlainFlow, "learn-resume-off")
            llm = _distill_llm()
            with patch("crewai.llm.LLM", return_value=llm):
                flow.resume("Always cite your sources")

        llm.call.assert_not_called()
        flow.memory.remember_many.assert_not_called()

    def test_failed_distillation_still_completes_the_resume(self) -> None:
        class LearnFlow(Flow):
            @start()
            @human_feedback(message="Review:", llm="gpt-4o-mini", learn=True)
            def draft(self):
                return "draft"

        with tempfile.TemporaryDirectory() as tmpdir:
            flow = _pause(tmpdir, LearnFlow, "learn-resume-lenient")
            llm = _distill_llm(RuntimeError("distillation model unavailable"))
            with patch("crewai.llm.LLM", return_value=llm):
                flow.resume("Always cite your sources")

        assert flow.last_human_feedback.feedback == "Always cite your sources"
        flow.memory.remember_many.assert_not_called()

    def test_strict_distillation_failure_can_be_retried_cleanly(self) -> None:
        class StrictFlow(Flow):
            @start()
            @human_feedback(
                message="Review:", llm="gpt-4o-mini", learn=True, learn_strict=True
            )
            def draft(self):
                return "draft"

        from crewai.flow.human_feedback import DistilledLessons

        with tempfile.TemporaryDirectory() as tmpdir:
            flow = _pause(tmpdir, StrictFlow, "learn-resume-strict")
            llm = _distill_llm(
                RuntimeError("distillation model unavailable"),
                DistilledLessons(lessons=["Cite a source for every claim"]),
            )
            with patch("crewai.llm.LLM", return_value=llm):
                with pytest.raises(
                    RuntimeError, match="distillation model unavailable"
                ):
                    flow.resume("Always cite your sources")

                assert (
                    flow.persistence.load_pending_feedback("learn-resume-strict")
                    is not None
                )
                assert flow.human_feedback_history == []
                assert flow.last_human_feedback is None
                flow.memory.remember_many.assert_not_called()

                flow.resume("Always cite your sources")

        assert [r.feedback for r in flow.human_feedback_history] == [
            "Always cite your sources"
        ]
        flow.memory.remember_many.assert_called_once_with(
            ["Cite a source for every claim"], source="hitl"
        )

    def test_retry_after_a_failed_completion_offers_the_lessons_again(self) -> None:
        # Lessons are stored before the review is durably completed, which strict
        # mode needs, so a retry after a later persistence failure stores them again
        # and memory consolidation absorbs the repeat.
        from crewai.flow.human_feedback import DistilledLessons

        class LearnFlow(Flow):
            @start()
            @human_feedback(message="Review:", llm="gpt-4o-mini", learn=True)
            def draft(self):
                return "draft"

        lesson = ["Cite a source for every claim"]
        with tempfile.TemporaryDirectory() as tmpdir:
            flow = _pause(tmpdir, LearnFlow, "learn-resume-retry")
            persistence = flow.persistence
            with (
                patch(
                    "crewai.llm.LLM",
                    return_value=_distill_llm(DistilledLessons(lessons=lesson)),
                ),
                patch.object(
                    SQLiteFlowPersistence,
                    "clear_pending_feedback",
                    side_effect=RuntimeError("database is locked"),
                ),
                pytest.raises(RuntimeError, match="database is locked"),
            ):
                flow.resume("Always cite your sources")

            flow.memory.remember_many.assert_called_once_with(lesson, source="hitl")
            assert persistence.load_pending_feedback("learn-resume-retry") is not None

            retried = LearnFlow.from_pending("learn-resume-retry", persistence)
            retried.memory = MagicMock()
            with patch(
                "crewai.llm.LLM",
                return_value=_distill_llm(DistilledLessons(lessons=lesson)),
            ):
                retried.resume("Always cite your sources")

            retried.memory.remember_many.assert_called_once_with(lesson, source="hitl")
            assert persistence.load_pending_feedback("learn-resume-retry") is None

    @patch("crewai.flow.runtime.crewai_event_bus.emit")
    def test_resumed_pydantic_output_arrives_as_a_dict(self, mock_emit: MagicMock) -> None:
        # The pending review is persisted as JSON, so the docs and the
        # HumanFeedbackResult.output docstring say a model comes back as a dict.
        class Draft(BaseModel):
            title: str

        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SQLiteFlowPersistence(os.path.join(tmpdir, "flows.db"))

            class PauseProvider:
                def request_feedback(
                    self, context: PendingFeedbackContext, flow: Flow
                ) -> str:
                    persistence.save_pending_feedback(
                        flow_uuid=context.flow_id,
                        context=context,
                        state_data=flow.state
                        if isinstance(flow.state, dict)
                        else flow.state.model_dump(),
                    )
                    raise HumanFeedbackPending(context=context)

            class DraftFlow(Flow):
                @start()
                @human_feedback(message="Review:", provider=PauseProvider())
                def draft(self):
                    return Draft(title="Launch day")

            pending = DraftFlow(persistence=persistence).kickoff()
            assert isinstance(pending.context.method_output, Draft)

            resumed = DraftFlow.from_pending(pending.context.flow_id, persistence)
            resumed.resume("looks good")

        assert resumed.last_human_feedback.output == {"title": "Launch day"}

    @patch("crewai.flow.runtime.crewai_event_bus.emit")
    def test_reply_after_a_paused_kickoff_is_learned(self, mock_emit: MagicMock) -> None:
        from crewai.flow.human_feedback import DistilledLessons

        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SQLiteFlowPersistence(os.path.join(tmpdir, "flows.db"))

            class EmailProvider:
                def request_feedback(
                    self, context: PendingFeedbackContext, flow: Flow
                ) -> str:
                    persistence.save_pending_feedback(
                        flow_uuid=context.flow_id,
                        context=context,
                        state_data=flow.state
                        if isinstance(flow.state, dict)
                        else flow.state.model_dump(),
                    )
                    raise HumanFeedbackPending(context=context)

            class ReviewFlow(Flow):
                @start()
                @human_feedback(
                    message="Review:",
                    llm="gpt-4o-mini",
                    learn=True,
                    provider=EmailProvider(),
                )
                def draft(self):
                    return "first draft"

            paused = ReviewFlow(persistence=persistence)
            paused.memory = MagicMock()
            paused.memory.recall.return_value = []
            pending = paused.kickoff()
            assert isinstance(pending, HumanFeedbackPending)
            paused.memory.remember_many.assert_not_called()

            resumed = ReviewFlow.from_pending(pending.context.flow_id, persistence)
            resumed.memory = MagicMock()
            llm = _distill_llm(DistilledLessons(lessons=["Lead with the customer outcome"]))
            with patch("crewai.llm.LLM", return_value=llm):
                resumed.resume("Lead with what the customer gets")

        resumed.memory.remember_many.assert_called_once_with(
            ["Lead with the customer outcome"], source="hitl"
        )
        assert "first draft" in llm.call.call_args.args[0][1]["content"]


class _ReviewProvider:
    """Raises ``HumanFeedbackPending`` when ``pause`` is set, else replies at once."""

    def __init__(self, pause: bool, ready: threading.Event | None = None) -> None:
        self.pause = pause
        self.ready = ready

    def request_feedback(self, context: PendingFeedbackContext, flow: Flow) -> str:
        if not self.pause:
            return "reply"
        if self.ready is not None:
            self.ready.wait(timeout=10)
        raise HumanFeedbackPending(context=context)


def _review(provider: _ReviewProvider, emit: list[str] | None) -> Any:
    return human_feedback(
        message="Review:", emit=emit, llm="gpt-4o-mini", provider=provider
    )


def _run_answered(build: Any, outcomes: list[str]) -> list[str]:
    """Run list of ``build(False)``'s flow, each review routed to the next outcome."""
    flow_cls, ran = build(False)
    replies = iter(outcomes)
    with patch.object(
        flow_cls, "_collapse_to_outcome", side_effect=lambda *a, **k: next(replies)
    ):
        flow_cls().kickoff()
    return ran


def _run_paused(
    build: Any,
    outcomes: list[str],
    tmpdir: str,
    *,
    use_async: bool = False,
    restore_each: bool = True,
) -> tuple[list[str], Any]:
    """Run list and last result of ``build(True)``'s flow, resumed once per outcome."""
    flow_cls, ran = build(True)
    persistence = SQLiteFlowPersistence(os.path.join(tmpdir, "flows.db"))
    result: Any = flow_cls(persistence=persistence).kickoff()
    flow = None
    for outcome in outcomes:
        assert isinstance(result, HumanFeedbackPending), (
            f"expected a pause before resuming with {outcome!r}, got {result!r}"
        )
        if flow is None or restore_each:
            flow = flow_cls.from_pending(result.context.flow_id, persistence)
        with patch.object(flow_cls, "_collapse_to_outcome", return_value=outcome):
            result = (
                asyncio.run(flow.resume_async("reply"))
                if use_async
                else flow.resume("reply")
            )
    return ran, result


def _start_under_review(also_listens: bool) -> Any:
    def build(pause: bool) -> tuple[type[Flow], list[str]]:
        ran: list[str] = []
        provider = _ReviewProvider(pause)

        def draft(self: Flow) -> str:
            ran.append("draft")
            return "content"

        reviewed = _review(provider, ["approved", "rejected"])(draft)
        if also_listens:
            reviewed = listen("never")(reviewed)

        class StartReviewFlow(Flow):
            draft = start()(reviewed)

            @listen("draft")
            def on_draft(self, payload: Any) -> None:
                ran.append(f"on_draft:{type(payload).__name__}:{payload}")

            @listen(or_("draft", "never"))
            def or_draft(self) -> None:
                ran.append("or_draft")

            @router("draft")
            def gate(self) -> str:
                ran.append("gate")
                return "gated"

            @listen("gated")
            def after_gate(self) -> None:
                ran.append("after_gate")

            @listen("approved")
            def on_approved(self, payload: Any) -> None:
                ran.append(f"on_approved:{type(payload).__name__}")

            @start("approved")
            def cond_start(self) -> None:
                ran.append("cond_start")

            @listen(or_("draft", "approved"))
            def or_both(self) -> None:
                ran.append("or_both")

        return StartReviewFlow, ran

    return build


def _listener_under_review(pause: bool) -> tuple[type[Flow], list[str]]:
    ran: list[str] = []
    provider = _ReviewProvider(pause)

    class ListenerReviewFlow(Flow):
        @start()
        def begin(self) -> None:
            ran.append("begin")

        @listen("begin")
        @_review(provider, ["approved", "rejected"])
        def draft(self) -> str:
            ran.append("draft")
            return "content"

        @listen("draft")
        def on_draft(self, payload: Any) -> None:
            ran.append(f"on_draft:{type(payload).__name__}:{payload}")

        @listen(or_("draft", "never"))
        def or_draft(self) -> None:
            ran.append("or_draft")

        @router("draft")
        def gate(self) -> str:
            ran.append("gate")
            return "gated"

        @listen("gated")
        def after_gate(self) -> None:
            ran.append("after_gate")

        @listen("approved")
        def on_approved(self, payload: Any) -> None:
            ran.append(f"on_approved:{type(payload).__name__}")

        @start("approved")
        def cond_start(self) -> None:
            ran.append("cond_start")

        @listen("cond_start")
        def after_cond_start(self) -> None:
            ran.append("after_cond_start")

        @listen(or_("draft", "approved"))
        def or_both(self) -> None:
            ran.append("or_both")

    return ListenerReviewFlow, ran


def _join_on_other_start(pause: bool) -> tuple[type[Flow], list[str]]:
    ran: list[str] = []
    other_done = threading.Event()
    # The pause waits for other_start's listener, so the join's other input is
    # recorded before the pause on every run, never by timing.
    provider = _ReviewProvider(pause, ready=other_done)

    class TwoStartsFlow(Flow):
        @start()
        @_review(provider, ["approved", "rejected"])
        def draft(self) -> str:
            ran.append("draft")
            return "content"

        @start()
        def other_start(self) -> None:
            ran.append("other_start")

        @listen("other_start")
        def after_other(self) -> None:
            ran.append("after_other")
            other_done.set()

        @listen(and_("draft", "other_start"))
        def join(self) -> None:
            ran.append("join")

        @listen("approved")
        def on_approved(self) -> None:
            ran.append("on_approved")

    return TwoStartsFlow, ran


def _join_on_own_trigger(pause: bool) -> tuple[type[Flow], list[str]]:
    ran: list[str] = []
    provider = _ReviewProvider(pause)

    class OwnTriggerJoinFlow(Flow):
        @start()
        def begin(self) -> None:
            ran.append("begin")

        @listen("begin")
        @_review(provider, ["approved", "rejected"])
        def draft(self) -> str:
            ran.append("draft")
            return "content"

        @listen(and_("draft", "begin"))
        def join(self) -> None:
            ran.append("join")

        @listen("approved")
        def on_approved(self) -> None:
            ran.append("on_approved")

    return OwnTriggerJoinFlow, ran


def _review_without_emit(pause: bool) -> tuple[type[Flow], list[str]]:
    ran: list[str] = []
    provider = _ReviewProvider(pause)

    class PlainReviewFlow(Flow):
        @start()
        def begin(self) -> None:
            ran.append("begin")

        @listen("begin")
        @_review(provider, None)
        def draft(self) -> str:
            ran.append("draft")
            return "content"

        @router("begin")
        def gate(self) -> str:
            ran.append("gate")
            return "ok"

        @listen(and_("draft", "begin"))
        def join_begin(self) -> None:
            ran.append("join_begin")

        @listen(and_("draft", "ok"))
        def join_ok(self) -> None:
            ran.append("join_ok")

    return PlainReviewFlow, ran


def _fired_or_before_review(pause: bool) -> tuple[type[Flow], list[str]]:
    ran: list[str] = []
    provider = _ReviewProvider(pause)

    class FiredOrFlow(Flow):
        @start()
        def x(self) -> None:
            ran.append("x")

        @listen(or_("review", "x"))
        def logger(self) -> None:
            ran.append("logger")

        @listen("logger")
        @_review(provider, ["approved", "rejected"])
        def review(self) -> str:
            ran.append("review")
            return "content"

        @listen("approved")
        def done(self) -> None:
            ran.append("done")

    return FiredOrFlow, ran


def _two_reviews(pause: bool) -> tuple[type[Flow], list[str]]:
    ran: list[str] = []
    provider = _ReviewProvider(pause)

    class TwoReviewsFlow(Flow):
        @start()
        def begin(self) -> None:
            ran.append("begin")

        @listen("begin")
        @_review(provider, ["go", "stop"])
        def review1(self) -> str:
            ran.append("review1")
            return "plan"

        @listen(or_("go", "again"))
        def work(self) -> None:
            ran.append("work")

        @listen("work")
        @_review(provider, ["again", "done"])
        def review2(self) -> str:
            ran.append("review2")
            return "result"

        @listen("review2")
        def audit2(self) -> None:
            ran.append("audit2")

        @listen("done")
        def finish(self) -> None:
            ran.append("finish")

    return TwoReviewsFlow, ran


def _revision_loop(pause: bool) -> tuple[type[Flow], list[str]]:
    ran: list[str] = []
    provider = _ReviewProvider(pause)

    class RevisionLoopFlow(Flow):
        @start()
        def generate(self) -> None:
            ran.append("generate")

        @listen(or_("generate", "needs_revision"))
        def x(self) -> None:
            ran.append("x")

        @listen(or_("x", "never"))
        def y(self) -> None:
            ran.append("y")

        @listen("y")
        @_review(provider, ["needs_revision", "approved"])
        def review(self) -> str:
            ran.append("review")
            return "content"

        @listen("approved")
        def publish(self) -> None:
            ran.append("publish")

    return RevisionLoopFlow, ran


def _self_loop_with_join(pause: bool) -> tuple[type[Flow], list[str]]:
    ran: list[str] = []
    provider = _ReviewProvider(pause)

    class SelfLoopFlow(Flow):
        @start()
        def generate(self) -> None:
            ran.append("generate")

        @listen(or_("generate", "revise"))
        @_review(provider, ["revise", "approved"])
        def review(self) -> str:
            ran.append("review")
            return "content"

        @listen("approved")
        def publish(self) -> None:
            ran.append("publish")

        @listen(and_("review", "generate"))
        def join(self) -> None:
            ran.append("join")

    return SelfLoopFlow, ran


def _router_before_review(pause: bool) -> tuple[type[Flow], list[str]]:
    ran: list[str] = []
    provider = _ReviewProvider(pause)

    class RoutedReviewFlow(Flow):
        @start()
        def begin(self) -> None:
            ran.append("begin")

        @router("begin")
        def gate(self) -> str:
            ran.append("gate")
            return "go"

        @listen("go")
        @_review(provider, ["approved", "rejected"])
        def draft(self) -> str:
            ran.append("draft")
            return "content"

        @listen(and_("draft", "begin"))
        def join_begin(self) -> None:
            ran.append("join_begin")

        @listen(and_("draft", "go"))
        def join_go(self) -> None:
            ran.append("join_go")

    return RoutedReviewFlow, ran


def _chained_router_on_outcome(pause: bool) -> tuple[type[Flow], list[str]]:
    ran: list[str] = []
    provider = _ReviewProvider(pause)

    class ChainedRouterFlow(Flow):
        @start()
        def begin(self) -> None:
            ran.append("begin")

        @listen("begin")
        @_review(provider, ["approved", "rejected"])
        def draft(self) -> str:
            ran.append("draft")
            return "content"

        @router("approved")
        def chain(self) -> str:
            ran.append("chain")
            return "shipped"

        @listen("shipped")
        def ship(self) -> None:
            ran.append("ship")

    return ChainedRouterFlow, ran


def _join_input_recurring(pause: bool) -> tuple[type[Flow], list[str]]:
    ran: list[str] = []
    provider = _ReviewProvider(pause)

    class RecurringInputFlow(Flow):
        @start()
        def setup(self) -> None:
            ran.append("setup")

        @listen(or_("setup", "revise"))
        def generate(self) -> None:
            ran.append("generate")

        @listen("generate")
        @_review(provider, ["revise", "approved"])
        def review(self) -> str:
            ran.append("review")
            return "content"

        @listen(and_("review", "generate"))
        def summarize(self) -> None:
            ran.append("summarize")

        @listen("approved")
        def publish(self) -> None:
            ran.append("publish")

    return RecurringInputFlow, ran


def _review_of_a_review(pause: bool) -> tuple[type[Flow], list[str]]:
    ran: list[str] = []
    provider = _ReviewProvider(pause)

    class ReviewOfAReviewFlow(Flow):
        @start()
        def begin(self) -> None:
            ran.append("begin")

        @listen("begin")
        @_review(provider, ["approved", "rejected"])
        def draft(self) -> str:
            ran.append("draft")
            return "content"

        @listen("draft")
        @_review(provider, ["ok", "no"])
        def review2(self) -> str:
            ran.append("review2")
            return "second"

        @listen("approved")
        def on_approved(self) -> None:
            ran.append("on_approved")

        @listen("ok")
        def on_ok(self) -> None:
            ran.append("on_ok")

        @listen(and_("review2", "begin"))
        def join_r2_begin(self) -> None:
            ran.append("join_r2_begin")

    return ReviewOfAReviewFlow, ran


def _three_chained_reviews(pause: bool) -> tuple[type[Flow], list[str]]:
    ran: list[str] = []
    provider = _ReviewProvider(pause)

    class ThreeChainedReviewsFlow(Flow):
        @start()
        def begin(self) -> None:
            ran.append("begin")

        @listen("begin")
        @_review(provider, ["a1", "x1"])
        def r1(self) -> str:
            ran.append("r1")
            return "one"

        @listen("r1")
        @_review(provider, ["a2", "x2"])
        def r2(self) -> str:
            ran.append("r2")
            return "two"

        @listen("r2")
        @_review(provider, ["a3", "x3"])
        def r3(self) -> str:
            ran.append("r3")
            return "three"

        @listen(and_("r3", "begin"))
        def join_r3_begin(self) -> None:
            ran.append("join_r3_begin")

        @listen(and_("r3", "r1"))
        def join_r3_r1(self) -> None:
            ran.append("join_r3_r1")

    return ThreeChainedReviewsFlow, ran


def _review_on_a_reviews_outcome(pause: bool) -> tuple[type[Flow], list[str]]:
    ran: list[str] = []
    provider = _ReviewProvider(pause)

    class ReviewOnOutcomeFlow(Flow):
        @start()
        def setup(self) -> None:
            ran.append("setup")

        @listen("setup")
        def gen(self) -> None:
            ran.append("gen")

        @listen("gen")
        @_review(provider, ["go", "stop"])
        def r1(self) -> str:
            ran.append("r1")
            return "c"

        @listen("go")
        @_review(provider, ["fin", "no"])
        def r2(self) -> str:
            ran.append("r2")
            return "d"

        @listen(and_("r2", "gen"))
        def join_r2_gen(self) -> None:
            ran.append("join_r2_gen")

        @listen("r1")
        def on_r1(self) -> None:
            ran.append("on_r1")

        @listen("go")
        def on_go(self) -> None:
            ran.append("on_go")

        @listen("fin")
        def on_fin(self) -> None:
            ran.append("on_fin")

    return ReviewOnOutcomeFlow, ran


class TestResumeRunsWhatTheReviewRuns:
    """A resumed review runs the listeners, joins and routes it runs unpaused."""

    @pytest.mark.parametrize("use_async", [False, True])
    @pytest.mark.parametrize("also_listens", [False, True])
    def test_start_under_review(self, also_listens: bool, use_async: bool) -> None:
        build = _start_under_review(also_listens)
        expected = sorted(
            [
                "draft",
                "gate",
                "after_gate",
                "on_draft:str:approved",
                "or_draft",
                "or_both",
                "on_approved:HumanFeedbackResult",
            ]
        )
        assert sorted(_run_answered(build, ["approved"])) == expected
        with tempfile.TemporaryDirectory() as tmpdir:
            ran, _ = _run_paused(build, ["approved"], tmpdir, use_async=use_async)
        assert sorted(ran) == expected, f"resumed run list: {ran}"

    @pytest.mark.parametrize("use_async", [False, True])
    def test_listener_under_review(self, use_async: bool) -> None:
        expected = sorted(
            [
                "begin",
                "draft",
                "gate",
                "after_gate",
                "on_draft:str:approved",
                "or_draft",
                "or_both",
                "on_approved:HumanFeedbackResult",
                "or_both",
                "cond_start",
                "after_cond_start",
            ]
        )
        assert sorted(_run_answered(_listener_under_review, ["approved"])) == expected
        with tempfile.TemporaryDirectory() as tmpdir:
            ran, _ = _run_paused(
                _listener_under_review, ["approved"], tmpdir, use_async=use_async
            )
        assert sorted(ran) == expected, f"resumed run list: {ran}"

    @pytest.mark.parametrize("use_async", [False, True])
    def test_join_on_a_start_completed_before_the_pause(self, use_async: bool) -> None:
        assert _run_answered(_join_on_other_start, ["approved"]).count("join") == 1
        with tempfile.TemporaryDirectory() as tmpdir:
            ran, _ = _run_paused(
                _join_on_other_start, ["approved"], tmpdir, use_async=use_async
            )
        assert ran.count("join") == 1, f"resumed run list: {ran}"
        assert ran.count("draft") == 1, f"resumed run list: {ran}"

    @pytest.mark.parametrize("use_async", [False, True])
    def test_join_on_the_reviewed_listeners_trigger(self, use_async: bool) -> None:
        assert _run_answered(_join_on_own_trigger, ["approved"]).count("join") == 1
        with tempfile.TemporaryDirectory() as tmpdir:
            ran, _ = _run_paused(
                _join_on_own_trigger, ["approved"], tmpdir, use_async=use_async
            )
        assert sorted(ran) == ["begin", "draft", "join", "on_approved"], (
            f"resumed run list: {ran}"
        )

    def test_joins_of_a_review_without_emit(self) -> None:
        answered = _run_answered(_review_without_emit, [])
        assert (answered.count("join_begin"), answered.count("join_ok")) == (1, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            ran, _ = _run_paused(_review_without_emit, [""], tmpdir)
        assert (ran.count("join_begin"), ran.count("join_ok")) == (1, 1), (
            f"resumed run list: {ran}"
        )

    def test_or_listener_fired_before_the_pause_runs_once(self) -> None:
        assert _run_answered(_fired_or_before_review, ["approved"]).count("logger") == 1
        with tempfile.TemporaryDirectory() as tmpdir:
            ran, _ = _run_paused(_fired_or_before_review, ["approved"], tmpdir)
        assert ran == ["x", "logger", "review", "done"], f"resumed run list: {ran}"

    def test_second_pause_resumed_on_the_same_instance(self) -> None:
        expected = [
            "begin",
            "review1",
            "work",
            "review2",
            "audit2",
            "work",
            "review2",
            "audit2",
            "finish",
        ]
        outcomes = ["go", "again", "done"]
        assert _run_answered(_two_reviews, outcomes) == expected
        with tempfile.TemporaryDirectory() as tmpdir:
            ran, result = _run_paused(
                _two_reviews, outcomes, tmpdir, restore_each=False
            )
        assert ran == expected, f"resumed run list: {ran}"
        assert not isinstance(result, HumanFeedbackPending)

    def test_revision_loop_through_or_listeners(self) -> None:
        expected = ["generate", "x", "y", "review", "x", "y", "review", "publish"]
        outcomes = ["needs_revision", "approved"]
        assert _run_answered(_revision_loop, outcomes) == expected
        with tempfile.TemporaryDirectory() as tmpdir:
            ran, _ = _run_paused(_revision_loop, outcomes, tmpdir)
        assert ran == expected, f"resumed run list: {ran}"

    def test_self_loop_reruns_the_review_and_runs_its_join_once(self) -> None:
        outcomes = ["revise", "approved"]
        answered = _run_answered(_self_loop_with_join, outcomes)
        assert answered.count("join") == 1
        with tempfile.TemporaryDirectory() as tmpdir:
            ran, _ = _run_paused(_self_loop_with_join, outcomes, tmpdir)
        assert [name for name in ran if name != "join"] == [
            "generate",
            "review",
            "review",
            "publish",
        ], f"resumed run list: {ran}"
        assert ran.count("join") == 1, f"resumed run list: {ran}"

    def test_fired_listener_the_flow_no_longer_defines_is_dropped(self) -> None:
        ran: list[str] = []

        class RedeployedFlow(Flow):
            @start()
            def begin(self) -> None:
                ran.append("begin")

            @listen("begin")
            @_review(_ReviewProvider(True), ["approved", "rejected"])
            def draft(self) -> str:
                return "content"

            @listen("approved")
            def on_approved(self) -> None:
                ran.append("on_approved")

        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SQLiteFlowPersistence(os.path.join(tmpdir, "flows.db"))
            context = PendingFeedbackContext.from_dict(
                {
                    "flow_id": "redeployed-1",
                    "flow_class": "test.RedeployedFlow",
                    "method_name": "draft",
                    "method_output": "content",
                    "message": "Review:",
                    "emit": ["approved", "rejected"],
                    "llm": "gpt-4o-mini",
                    "fired_or_listeners": ["removed_step"],
                }
            )
            persistence.save_pending_feedback(
                flow_uuid="redeployed-1",
                context=context,
                state_data={"id": "redeployed-1"},
            )
            flow = RedeployedFlow.from_pending("redeployed-1", persistence)
            with patch.object(flow, "_collapse_to_outcome", return_value="approved"):
                flow.resume("reply")

        assert ran == ["on_approved"]

    def test_joins_on_a_router_label_before_the_reviewed_listener(self) -> None:
        answered = _run_answered(_router_before_review, ["approved"])
        assert (answered.count("join_begin"), answered.count("join_go")) == (1, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            ran, _ = _run_paused(_router_before_review, ["approved"], tmpdir)
        assert (ran.count("join_begin"), ran.count("join_go")) == (1, 1), (
            f"resumed run list: {ran}"
        )

    @pytest.mark.parametrize("use_async", [False, True])
    def test_chained_router_on_the_outcome(self, use_async: bool) -> None:
        expected = ["begin", "draft", "chain", "ship"]
        assert _run_answered(_chained_router_on_outcome, ["approved"]) == expected
        with tempfile.TemporaryDirectory() as tmpdir:
            ran, _ = _run_paused(
                _chained_router_on_outcome, ["approved"], tmpdir, use_async=use_async
            )
        assert ran == expected, f"resumed run list: {ran}"

    def test_join_runs_again_when_its_input_recurs(self) -> None:
        expected = [
            "setup",
            "generate",
            "review",
            "summarize",
            "generate",
            "review",
            "summarize",
            "publish",
        ]
        outcomes = ["revise", "approved"]
        assert _run_answered(_join_input_recurring, outcomes) == expected
        with tempfile.TemporaryDirectory() as tmpdir:
            ran, _ = _run_paused(_join_input_recurring, outcomes, tmpdir)
        assert ran == expected, f"resumed run list: {ran}"

    @pytest.mark.parametrize("restore_each", [True, False])
    def test_join_of_a_review_paused_inside_a_resumed_review(
        self, restore_each: bool
    ) -> None:
        counted = ["draft", "review2", "on_ok", "join_r2_begin"]
        outcomes = ["approved", "ok"]
        answered = _run_answered(_review_of_a_review, outcomes)
        assert [answered.count(name) for name in counted] == [1, 1, 1, 1]
        with tempfile.TemporaryDirectory() as tmpdir:
            ran, result = _run_paused(
                _review_of_a_review, outcomes, tmpdir, restore_each=restore_each
            )
        assert [ran.count(name) for name in counted] == [1, 1, 1, 1], (
            f"resumed run list: {ran}"
        )
        assert not isinstance(result, HumanFeedbackPending)

    def test_joins_of_the_third_review_in_a_chain_of_paused_reviews(self) -> None:
        counted = ["join_r3_begin", "join_r3_r1"]
        outcomes = ["a1", "a2", "a3"]
        answered = _run_answered(_three_chained_reviews, outcomes)
        assert [answered.count(name) for name in counted] == [1, 1]
        with tempfile.TemporaryDirectory() as tmpdir:
            ran, result = _run_paused(_three_chained_reviews, outcomes, tmpdir)
        assert [ran.count(name) for name in counted] == [1, 1], (
            f"resumed run list: {ran}"
        )
        assert not isinstance(result, HumanFeedbackPending)

    def test_join_of_a_review_paused_on_a_resumed_reviews_outcome(self) -> None:
        counted = ["r1", "r2", "on_r1", "on_fin", "join_r2_gen"]
        outcomes = ["go", "fin"]
        answered = _run_answered(_review_on_a_reviews_outcome, outcomes)
        assert [answered.count(name) for name in counted] == [1, 1, 1, 1, 1]
        with tempfile.TemporaryDirectory() as tmpdir:
            ran, result = _run_paused(_review_on_a_reviews_outcome, outcomes, tmpdir)
        assert [ran.count(name) for name in counted] == [1, 1, 1, 1, 1], (
            f"resumed run list: {ran}"
        )
        assert not isinstance(result, HumanFeedbackPending)
