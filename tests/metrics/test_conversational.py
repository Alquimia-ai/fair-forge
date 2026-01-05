"""Tests for Conversational metric."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from pydantic import SecretStr
from fair_forge.metrics.conversational import Conversational
from fair_forge.schemas import ConversationalMetric
from tests.fixtures.mock_retriever import MockRetriever, ConversationalDatasetRetriever
from tests.fixtures.mock_data import create_sample_batch


class TestConversationalMetric:
    """Test suite for Conversational metric."""

    def test_initialization_default(self):
        """Test Conversational initialization with default parameters."""
        conv = Conversational(retriever=MockRetriever)

        assert conv.judge_url == "https://api.groq.com/openai/v1"
        assert conv.judge_model == "deepseek-r1-distill-llama-70b"
        assert conv.judge_temperature == 0
        assert conv.judge_bos_think_token == "<think>"
        assert conv.judge_eos_think_token == "</think>"
        assert conv.judge_bos_json_clause == "```json"
        assert conv.judge_eos_json_clause == "```"

    def test_initialization_custom(self):
        """Test Conversational initialization with custom parameters."""
        conv = Conversational(
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

        assert conv.judge_url == "https://custom-api.com"
        assert conv.judge_model == "custom-model"
        assert conv.judge_temperature == 0.5
        assert conv.judge_bos_think_token == "<reasoning>"
        assert conv.judge_eos_think_token == "</reasoning>"

    @patch('fair_forge.metrics.conversational.Judge')
    def test_batch_processing(self, mock_judge_class):
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
                "sensibleness": 0.88
            }
        )

        conv = Conversational(retriever=MockRetriever)

        batches = [
            create_sample_batch(qa_id="qa_001"),
            create_sample_batch(qa_id="qa_002"),
        ]

        conv.batch(
            session_id="test_session",
            context="Test context",
            assistant_id="test_assistant",
            batch=batches,
            language="english"
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

    @patch('fair_forge.metrics.conversational.Judge')
    def test_batch_with_observation(self, mock_judge_class):
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
                "sensibleness": 0.88
            }
        )

        conv = Conversational(retriever=MockRetriever)

        batch = create_sample_batch(
            qa_id="qa_001",
            observation="The user seems satisfied"
        )

        conv.batch(
            session_id="test_session",
            context="Test context",
            assistant_id="test_assistant",
            batch=[batch]
        )

        # Verify judge was called with observation data
        assert mock_judge.check.called
        call_args = mock_judge.check.call_args
        assert "observation" in call_args[0][2]

    @patch('fair_forge.metrics.conversational.Judge')
    def test_batch_without_observation(self, mock_judge_class):
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
                "sensibleness": 0.88
            }
        )

        conv = Conversational(retriever=MockRetriever)

        batch = create_sample_batch(qa_id="qa_001", observation=None)

        conv.batch(
            session_id="test_session",
            context="Test context",
            assistant_id="test_assistant",
            batch=[batch]
        )

        # Verify judge was called with ground_truth_assistant
        call_args = mock_judge.check.call_args
        assert "ground_truth_assistant" in call_args[0][2]

    @patch('fair_forge.metrics.conversational.Judge')
    def test_batch_raises_on_no_json(self, mock_judge_class):
        """Test batch raises ValueError when no JSON is returned."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("Some thinking", None)

        conv = Conversational(retriever=MockRetriever)

        batch = create_sample_batch(qa_id="qa_001")

        with pytest.raises(ValueError, match="No JSON found"):
            conv.batch(
                session_id="test_session",
                context="Test context",
                assistant_id="test_assistant",
                batch=[batch]
            )

    @patch('fair_forge.metrics.conversational.Judge')
    def test_run_method(self, mock_judge_class):
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
                "sensibleness": 0.88
            }
        )

        metrics = Conversational.run(
            ConversationalDatasetRetriever,
            judge_api_key=SecretStr("test-key"),
            verbose=False
        )

        assert len(metrics) > 0
        assert all(isinstance(m, ConversationalMetric) for m in metrics)

    @patch('fair_forge.metrics.conversational.Judge')
    def test_judge_initialization_params(self, mock_judge_class):
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
                "sensibleness": 0.5
            }
        )

        conv = Conversational(
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
        conv.batch(
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

    @patch('fair_forge.metrics.conversational.Judge')
    def test_language_parameter(self, mock_judge_class):
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
                "sensibleness": 0.5
            }
        )

        conv = Conversational(retriever=MockRetriever)

        batch = create_sample_batch(qa_id="qa_001")
        conv.batch(
            session_id="s",
            context="c",
            assistant_id="a",
            batch=[batch],
            language="spanish"
        )

        # Verify preferred_language was passed
        call_args = mock_judge.check.call_args
        assert call_args[0][2]["preferred_language"] == "spanish"

    @patch('fair_forge.metrics.conversational.Judge')
    def test_verbose_mode(self, mock_judge_class):
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
                "sensibleness": 0.88
            }
        )

        conv = Conversational(retriever=MockRetriever, verbose=True)

        batch = create_sample_batch(qa_id="qa_001")
        conv.batch(
            session_id="s",
            context="c",
            assistant_id="a",
            batch=[batch]
        )

        assert len(conv.metrics) == 1
