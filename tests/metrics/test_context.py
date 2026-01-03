"""Tests for Context metric."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from pydantic import SecretStr
from fair_forge.metrics.context import Context
from fair_forge.schemas import ContextMetric
from tests.fixtures.mock_retriever import MockRetriever, ContextDatasetRetriever
from tests.fixtures.mock_data import create_sample_batch


class TestContextMetric:
    """Test suite for Context metric."""

    def test_initialization_default(self):
        """Test Context initialization with default parameters."""
        context = Context(retriever=MockRetriever)

        assert context.judge_url == "https://api.groq.com/openai/v1"
        assert context.judge_model == "deepseek-r1-distill-llama-70b"
        assert context.judge_temperature == 0
        assert context.judge_bos_think_token == "<think>"
        assert context.judge_eos_think_token == "</think>"
        assert context.judge_bos_json_clause == "```json"
        assert context.judge_eos_json_clause == "```"

    def test_initialization_custom(self):
        """Test Context initialization with custom parameters."""
        context = Context(
            retriever=MockRetriever,
            judge_bos_think_token="<reasoning>",
            judege_eos_think_token="</reasoning>",
            judge_base_url="https://custom-api.com",
            judge_api_key=SecretStr("test-key"),
            judge_model="custom-model",
            judge_temperature=0.5,
            judge_bos_json_clause="<json>",
            judge_eos_json_clause="</json>",
        )

        assert context.judge_url == "https://custom-api.com"
        assert context.judge_model == "custom-model"
        assert context.judge_temperature == 0.5
        assert context.judge_bos_think_token == "<reasoning>"
        assert context.judge_eos_think_token == "</reasoning>"
        assert context.judge_bos_json_clause == "<json>"
        assert context.judge_eos_json_clause == "</json>"

    @patch('fair_forge.metrics.context.Judge')
    def test_batch_processing(self, mock_judge_class):
        """Test batch method processes interactions correctly."""
        # Setup mock judge
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "I analyzed the context...",
            {"insight": "Good context awareness", "score": 0.85}
        )

        context = Context(retriever=MockRetriever)

        batches = [
            create_sample_batch(qa_id="qa_001"),
            create_sample_batch(qa_id="qa_002"),
        ]

        context.batch(
            session_id="test_session",
            context="Healthcare AI context",
            assistant_id="test_assistant",
            batch=batches,
            language="english"
        )

        assert len(context.metrics) == 2
        assert all(isinstance(m, ContextMetric) for m in context.metrics)
        assert context.metrics[0].session_id == "test_session"
        assert context.metrics[0].qa_id == "qa_001"
        assert context.metrics[0].context_awareness == 0.85
        assert context.metrics[0].context_insight == "Good context awareness"
        assert context.metrics[0].context_thinkings == "I analyzed the context..."

    @patch('fair_forge.metrics.context.Judge')
    def test_batch_with_observation(self, mock_judge_class):
        """Test batch method handles observation correctly."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Observation analysis...",
            {"insight": "With observation", "score": 0.9}
        )

        context = Context(retriever=MockRetriever)

        batch = create_sample_batch(
            qa_id="qa_001",
            observation="The user seems confused"
        )

        context.batch(
            session_id="test_session",
            context="Test context",
            assistant_id="test_assistant",
            batch=[batch]
        )

        # Verify judge was called with observation data
        assert mock_judge.check.called
        call_args = mock_judge.check.call_args
        assert "observation" in call_args[0][2]

    @patch('fair_forge.metrics.context.Judge')
    def test_batch_without_observation(self, mock_judge_class):
        """Test batch method handles case without observation."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Analysis without observation...",
            {"insight": "Without observation", "score": 0.8}
        )

        context = Context(retriever=MockRetriever)

        batch = create_sample_batch(qa_id="qa_001", observation=None)

        context.batch(
            session_id="test_session",
            context="Test context",
            assistant_id="test_assistant",
            batch=[batch]
        )

        # Verify judge was called with ground_truth_assistant
        call_args = mock_judge.check.call_args
        assert "ground_truth_assistant" in call_args[0][2]

    @patch('fair_forge.metrics.context.Judge')
    def test_batch_raises_on_no_json(self, mock_judge_class):
        """Test batch raises ValueError when no JSON is returned."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("Some thinking", None)

        context = Context(retriever=MockRetriever)

        batch = create_sample_batch(qa_id="qa_001")

        with pytest.raises(ValueError, match="No JSON found"):
            context.batch(
                session_id="test_session",
                context="Test context",
                assistant_id="test_assistant",
                batch=[batch]
            )

    @patch('fair_forge.metrics.context.Judge')
    def test_run_method(self, mock_judge_class):
        """Test the run class method."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Thinking...",
            {"insight": "Test insight", "score": 0.75}
        )

        metrics = Context.run(
            ContextDatasetRetriever,
            judge_api_key=SecretStr("test-key"),
            verbose=False
        )

        assert len(metrics) > 0
        assert all(isinstance(m, ContextMetric) for m in metrics)

    @patch('fair_forge.metrics.context.Judge')
    def test_judge_initialization_params(self, mock_judge_class):
        """Test that Judge is initialized with correct parameters."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", {"insight": "test", "score": 0.5})

        context = Context(
            retriever=MockRetriever,
            judge_bos_think_token="<t>",
            judege_eos_think_token="</t>",
            judge_base_url="https://test.com",
            judge_api_key=SecretStr("key"),
            judge_model="model",
            judge_temperature=0.1,
            judge_bos_json_clause="[",
            judge_eos_json_clause="]",
        )

        batch = create_sample_batch(qa_id="qa_001")
        context.batch(
            session_id="s",
            context="c",
            assistant_id="a",
            batch=[batch]
        )

        mock_judge_class.assert_called_once_with(
            bos_think_token="<t>",
            eos_think_token="</t>",
            base_url="https://test.com",
            api_key=SecretStr("key"),
            model="model",
            temperature=0.1,
            bos_json_clause="[",
            eos_json_clause="]",
        )

    @patch('fair_forge.metrics.context.Judge')
    def test_verbose_mode(self, mock_judge_class):
        """Test that verbose mode doesn't break processing."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Thinking...",
            {"insight": "Test", "score": 0.8}
        )

        context = Context(retriever=MockRetriever, verbose=True)

        batch = create_sample_batch(qa_id="qa_001")
        context.batch(
            session_id="s",
            context="c",
            assistant_id="a",
            batch=[batch]
        )

        assert len(context.metrics) == 1
