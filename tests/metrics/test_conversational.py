"""Tests for Conversational metric."""
from unittest.mock import MagicMock, patch

import pytest

from fair_forge.metrics.conversational import Conversational
from fair_forge.schemas.conversational import ConversationalMetric
from tests.fixtures.mock_data import create_sample_batch
from tests.fixtures.mock_retriever import ConversationalDatasetRetriever, MockRetriever


class TestConversationalMetric:
    """Test suite for Conversational metric."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock BaseChatModel."""
        return MagicMock()

    def test_initialization_default(self, mock_model):
        """Test Conversational initialization with default parameters."""
        conv = Conversational(retriever=MockRetriever, model=mock_model)

        assert conv.model == mock_model
        assert conv.use_structured_output is False
        assert conv.bos_json_clause == "```json"
        assert conv.eos_json_clause == "```"

    def test_initialization_custom(self, mock_model):
        """Test Conversational initialization with custom parameters."""
        conv = Conversational(
            retriever=MockRetriever,
            model=mock_model,
            use_structured_output=True,
            bos_json_clause="<json>",
            eos_json_clause="</json>",
        )

        assert conv.model == mock_model
        assert conv.use_structured_output is True
        assert conv.bos_json_clause == "<json>"
        assert conv.eos_json_clause == "</json>"

    @patch("fair_forge.metrics.conversational.Judge")
    def test_batch_processing(self, mock_judge_class, mock_model):
        """Test batch method processes interactions correctly."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "I analyzed the conversation...",
            {
                "insight": "Good conversation quality",
                "memory": 0.9,
                "language": 0.85,
                "quality_maxim": 0.8,
                "quantity_maxim": 0.75,
                "relation_maxim": 0.9,
                "manner_maxim": 0.85,
                "sensibleness": 0.88,
            },
        )

        conv = Conversational(retriever=MockRetriever, model=mock_model)

        batches = [
            create_sample_batch(qa_id="qa_001"),
            create_sample_batch(qa_id="qa_002"),
        ]

        conv.batch(
            session_id="test_session",
            context="Test context",
            assistant_id="test_assistant",
            batch=batches,
            language="english",
        )

        assert len(conv.metrics) == 2
        assert all(isinstance(m, ConversationalMetric) for m in conv.metrics)

        metric = conv.metrics[0]
        assert metric.session_id == "test_session"
        assert metric.qa_id == "qa_001"
        assert metric.conversational_memory == 0.9
        assert metric.conversational_language == 0.85
        assert metric.conversational_quality_maxim == 0.8
        assert metric.conversational_quantity_maxim == 0.75
        assert metric.conversational_relation_maxim == 0.9
        assert metric.conversational_manner_maxim == 0.85
        assert metric.conversational_sensibleness == 0.88
        assert metric.conversational_thinkings == "I analyzed the conversation..."

    @patch("fair_forge.metrics.conversational.Judge")
    def test_batch_with_observation(self, mock_judge_class, mock_model):
        """Test batch method handles observation correctly."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Observation analysis...",
            {
                "insight": "With observation",
                "memory": 0.9,
                "language": 0.85,
                "quality_maxim": 0.8,
                "quantity_maxim": 0.75,
                "relation_maxim": 0.9,
                "manner_maxim": 0.85,
                "sensibleness": 0.88,
            },
        )

        conv = Conversational(retriever=MockRetriever, model=mock_model)

        batch = create_sample_batch(qa_id="qa_001", observation="The user seems satisfied")

        conv.batch(
            session_id="test_session",
            context="Test context",
            assistant_id="test_assistant",
            batch=[batch],
        )

        # Verify judge was called with observation data
        assert mock_judge.check.called
        call_args = mock_judge.check.call_args
        assert "observation" in call_args[0][2]

    @patch("fair_forge.metrics.conversational.Judge")
    def test_batch_without_observation(self, mock_judge_class, mock_model):
        """Test batch method handles case without observation."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Analysis without observation...",
            {
                "insight": "Without observation",
                "memory": 0.9,
                "language": 0.85,
                "quality_maxim": 0.8,
                "quantity_maxim": 0.75,
                "relation_maxim": 0.9,
                "manner_maxim": 0.85,
                "sensibleness": 0.88,
            },
        )

        conv = Conversational(retriever=MockRetriever, model=mock_model)

        batch = create_sample_batch(qa_id="qa_001", observation=None)

        conv.batch(
            session_id="test_session",
            context="Test context",
            assistant_id="test_assistant",
            batch=[batch],
        )

        # Verify judge was called with ground_truth_assistant
        call_args = mock_judge.check.call_args
        assert "ground_truth_assistant" in call_args[0][2]

    @patch("fair_forge.metrics.conversational.Judge")
    def test_batch_raises_on_no_result(self, mock_judge_class, mock_model):
        """Test batch raises ValueError when no result is returned."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("Some thinking", None)

        conv = Conversational(retriever=MockRetriever, model=mock_model)

        batch = create_sample_batch(qa_id="qa_001")

        with pytest.raises(ValueError, match="No valid response"):
            conv.batch(
                session_id="test_session",
                context="Test context",
                assistant_id="test_assistant",
                batch=[batch],
            )

    @patch("fair_forge.metrics.conversational.Judge")
    def test_run_method(self, mock_judge_class, mock_model):
        """Test the run class method."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Thinking...",
            {
                "insight": "Test insight",
                "memory": 0.9,
                "language": 0.85,
                "quality_maxim": 0.8,
                "quantity_maxim": 0.75,
                "relation_maxim": 0.9,
                "manner_maxim": 0.85,
                "sensibleness": 0.88,
            },
        )

        metrics = Conversational.run(
            ConversationalDatasetRetriever,
            model=mock_model,
            verbose=False,
        )

        assert len(metrics) > 0
        assert all(isinstance(m, ConversationalMetric) for m in metrics)

    @patch("fair_forge.metrics.conversational.Judge")
    def test_judge_initialization_params(self, mock_judge_class, mock_model):
        """Test that Judge is initialized with correct parameters."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "",
            {
                "insight": "test",
                "memory": 0.5,
                "language": 0.5,
                "quality_maxim": 0.5,
                "quantity_maxim": 0.5,
                "relation_maxim": 0.5,
                "manner_maxim": 0.5,
                "sensibleness": 0.5,
            },
        )

        conv = Conversational(
            retriever=MockRetriever,
            model=mock_model,
            use_structured_output=False,
            bos_json_clause="[",
            eos_json_clause="]",
        )

        batch = create_sample_batch(qa_id="qa_001")
        conv.batch(
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

    @patch("fair_forge.metrics.conversational.Judge")
    def test_language_parameter(self, mock_judge_class, mock_model):
        """Test that language parameter is passed correctly."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "",
            {
                "insight": "test",
                "memory": 0.5,
                "language": 0.5,
                "quality_maxim": 0.5,
                "quantity_maxim": 0.5,
                "relation_maxim": 0.5,
                "manner_maxim": 0.5,
                "sensibleness": 0.5,
            },
        )

        conv = Conversational(retriever=MockRetriever, model=mock_model)

        batch = create_sample_batch(qa_id="qa_001")
        conv.batch(
            session_id="s",
            context="c",
            assistant_id="a",
            batch=[batch],
            language="spanish",
        )

        # Verify preferred_language was passed
        call_args = mock_judge.check.call_args
        assert call_args[0][2]["preferred_language"] == "spanish"

    @patch("fair_forge.metrics.conversational.Judge")
    def test_verbose_mode(self, mock_judge_class, mock_model):
        """Test that verbose mode doesn't break processing."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Thinking...",
            {
                "insight": "Test",
                "memory": 0.9,
                "language": 0.85,
                "quality_maxim": 0.8,
                "quantity_maxim": 0.75,
                "relation_maxim": 0.9,
                "manner_maxim": 0.85,
                "sensibleness": 0.88,
            },
        )

        conv = Conversational(retriever=MockRetriever, model=mock_model, verbose=True)

        batch = create_sample_batch(qa_id="qa_001")
        conv.batch(
            session_id="s",
            context="c",
            assistant_id="a",
            batch=[batch],
        )

        assert len(conv.metrics) == 1

    @patch("fair_forge.metrics.conversational.Judge")
    def test_structured_output_mode(self, mock_judge_class, mock_model):
        """Test Conversational with structured output enabled."""
        from fair_forge.llm.schemas import ConversationalJudgeOutput

        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge

        expected_result = ConversationalJudgeOutput(
            memory=8.5,
            language=9.0,
            insight="Good conversation",
            quality_maxim=8.0,
            quantity_maxim=7.5,
            relation_maxim=9.0,
            manner_maxim=8.5,
            sensibleness=9.0,
        )
        mock_judge.check.return_value = ("", expected_result)

        conv = Conversational(
            retriever=MockRetriever,
            model=mock_model,
            use_structured_output=True,
        )

        batch = create_sample_batch(qa_id="qa_001")
        conv.batch(
            session_id="test_session",
            context="Test context",
            assistant_id="test_assistant",
            batch=[batch],
        )

        assert len(conv.metrics) == 1
        assert conv.metrics[0].conversational_memory == 8.5
        assert conv.metrics[0].conversational_language == 9.0
        assert conv.metrics[0].conversational_insight == "Good conversation"
