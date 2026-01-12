"""Tests for Context metric."""
from unittest.mock import MagicMock, patch

import pytest

from fair_forge.metrics.context import Context
from fair_forge.schemas.context import ContextMetric
from tests.fixtures.mock_data import create_sample_batch
from tests.fixtures.mock_retriever import ContextDatasetRetriever, MockRetriever


class TestContextMetric:
    """Test suite for Context metric."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock BaseChatModel."""
        return MagicMock()

    def test_initialization_default(self, mock_model):
        """Test Context initialization with default parameters."""
        context = Context(retriever=MockRetriever, model=mock_model)

        assert context.model == mock_model
        assert context.use_structured_output is False
        assert context.bos_json_clause == "```json"
        assert context.eos_json_clause == "```"

    def test_initialization_custom(self, mock_model):
        """Test Context initialization with custom parameters."""
        context = Context(
            retriever=MockRetriever,
            model=mock_model,
            use_structured_output=True,
            bos_json_clause="<json>",
            eos_json_clause="</json>",
        )

        assert context.model == mock_model
        assert context.use_structured_output is True
        assert context.bos_json_clause == "<json>"
        assert context.eos_json_clause == "</json>"

    @patch("fair_forge.metrics.context.Judge")
    def test_batch_processing(self, mock_judge_class, mock_model):
        """Test batch method processes interactions correctly."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "I analyzed the context...",
            {"insight": "Good context awareness", "score": 0.85},
        )

        context = Context(retriever=MockRetriever, model=mock_model)

        batches = [
            create_sample_batch(qa_id="qa_001"),
            create_sample_batch(qa_id="qa_002"),
        ]

        context.batch(
            session_id="test_session",
            context="Healthcare AI context",
            assistant_id="test_assistant",
            batch=batches,
            language="english",
        )

        assert len(context.metrics) == 2
        assert all(isinstance(m, ContextMetric) for m in context.metrics)
        assert context.metrics[0].session_id == "test_session"
        assert context.metrics[0].qa_id == "qa_001"
        assert context.metrics[0].context_awareness == 0.85
        assert context.metrics[0].context_insight == "Good context awareness"
        assert context.metrics[0].context_thinkings == "I analyzed the context..."

    @patch("fair_forge.metrics.context.Judge")
    def test_batch_with_observation(self, mock_judge_class, mock_model):
        """Test batch method handles observation correctly."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Observation analysis...",
            {"insight": "With observation", "score": 0.9},
        )

        context = Context(retriever=MockRetriever, model=mock_model)

        batch = create_sample_batch(qa_id="qa_001", observation="The user seems confused")

        context.batch(
            session_id="test_session",
            context="Test context",
            assistant_id="test_assistant",
            batch=[batch],
        )

        # Verify judge was called with observation data
        assert mock_judge.check.called
        call_args = mock_judge.check.call_args
        assert "observation" in call_args[0][2]

    @patch("fair_forge.metrics.context.Judge")
    def test_batch_without_observation(self, mock_judge_class, mock_model):
        """Test batch method handles case without observation."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Analysis without observation...",
            {"insight": "Without observation", "score": 0.8},
        )

        context = Context(retriever=MockRetriever, model=mock_model)

        batch = create_sample_batch(qa_id="qa_001", observation=None)

        context.batch(
            session_id="test_session",
            context="Test context",
            assistant_id="test_assistant",
            batch=[batch],
        )

        # Verify judge was called with ground_truth_assistant
        call_args = mock_judge.check.call_args
        assert "ground_truth_assistant" in call_args[0][2]

    @patch("fair_forge.metrics.context.Judge")
    def test_batch_raises_on_no_result(self, mock_judge_class, mock_model):
        """Test batch raises ValueError when no result is returned."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("Some thinking", None)

        context = Context(retriever=MockRetriever, model=mock_model)

        batch = create_sample_batch(qa_id="qa_001")

        with pytest.raises(ValueError, match="No valid response"):
            context.batch(
                session_id="test_session",
                context="Test context",
                assistant_id="test_assistant",
                batch=[batch],
            )

    @patch("fair_forge.metrics.context.Judge")
    def test_run_method(self, mock_judge_class, mock_model):
        """Test the run class method."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Thinking...",
            {"insight": "Test insight", "score": 0.75},
        )

        metrics = Context.run(
            ContextDatasetRetriever,
            model=mock_model,
            verbose=False,
        )

        assert len(metrics) > 0
        assert all(isinstance(m, ContextMetric) for m in metrics)

    @patch("fair_forge.metrics.context.Judge")
    def test_judge_initialization_params(self, mock_judge_class, mock_model):
        """Test that Judge is initialized with correct parameters."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", {"insight": "test", "score": 0.5})

        context = Context(
            retriever=MockRetriever,
            model=mock_model,
            use_structured_output=False,
            bos_json_clause="[",
            eos_json_clause="]",
        )

        batch = create_sample_batch(qa_id="qa_001")
        context.batch(
            session_id="s",
            context="c",
            assistant_id="a",
            batch=[batch],
        )

        mock_judge_class.assert_called_once_with(
            model=mock_model,
            use_structured_output=False,
            bos_json_clause="[",
            eos_json_clause="]",
        )

    @patch("fair_forge.metrics.context.Judge")
    def test_verbose_mode(self, mock_judge_class, mock_model):
        """Test that verbose mode doesn't break processing."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Thinking...",
            {"insight": "Test", "score": 0.8},
        )

        context = Context(retriever=MockRetriever, model=mock_model, verbose=True)

        batch = create_sample_batch(qa_id="qa_001")
        context.batch(
            session_id="s",
            context="c",
            assistant_id="a",
            batch=[batch],
        )

        assert len(context.metrics) == 1

    @patch("fair_forge.metrics.context.Judge")
    def test_structured_output_mode(self, mock_judge_class, mock_model):
        """Test Context with structured output enabled."""
        from fair_forge.llm.schemas import ContextJudgeOutput

        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge

        expected_result = ContextJudgeOutput(score=0.85, insight="Good context")
        mock_judge.check.return_value = ("", expected_result)

        context = Context(
            retriever=MockRetriever,
            model=mock_model,
            use_structured_output=True,
        )

        batch = create_sample_batch(qa_id="qa_001")
        context.batch(
            session_id="test_session",
            context="Test context",
            assistant_id="test_assistant",
            batch=[batch],
        )

        assert len(context.metrics) == 1
        assert context.metrics[0].context_awareness == 0.85
        assert context.metrics[0].context_insight == "Good context"
