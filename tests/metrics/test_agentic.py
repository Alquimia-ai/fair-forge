"""Tests for Agentic metric."""

from unittest.mock import MagicMock, patch

import pytest

from fair_forge.metrics.agentic import Agentic
from fair_forge.schemas.agentic import AgenticMetric, ToolCorrectnessScore
from tests.fixtures.mock_retriever import AgenticDatasetRetriever, MockRetriever


class TestAgenticMetric:
    """Test suite for Agentic metric."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock BaseChatModel."""
        return MagicMock()

    def test_initialization_default(self, mock_model):
        """Test Agentic initialization with default parameters."""
        agentic = Agentic(retriever=MockRetriever, model=mock_model)

        assert agentic.model == mock_model
        assert agentic.use_structured_output is True
        assert agentic.bos_json_clause == "```json"
        assert agentic.eos_json_clause == "```"
        assert agentic.threshold == 0.7
        assert agentic.tool_threshold == 1.0
        assert agentic.tool_weights == {
            "selection": 0.25,
            "parameters": 0.25,
            "sequence": 0.25,
            "utilization": 0.25,
        }

    def test_initialization_custom(self, mock_model):
        """Test Agentic initialization with custom parameters."""
        custom_weights = {
            "selection": 0.3,
            "parameters": 0.3,
            "sequence": 0.2,
            "utilization": 0.2,
        }

        agentic = Agentic(
            retriever=MockRetriever,
            model=mock_model,
            use_structured_output=True,
            bos_json_clause="<json>",
            eos_json_clause="</json>",
            threshold=0.8,
            tool_threshold=0.85,
            tool_weights=custom_weights,
        )

        assert agentic.model == mock_model
        assert agentic.use_structured_output is True
        assert agentic.bos_json_clause == "<json>"
        assert agentic.eos_json_clause == "</json>"
        assert agentic.threshold == 0.8
        assert agentic.tool_threshold == 0.85
        assert agentic.tool_weights == custom_weights

    @patch("fair_forge.metrics.agentic.Judge")
    def test_evaluate_answer_correctness(self, mock_judge_class, mock_model):
        """Test answer correctness evaluation using LLM judge."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Reasoning...",
            {"correctness_score": 0.95, "reasoning": "Perfect match"},
        )

        agentic = Agentic(retriever=MockRetriever, model=mock_model)
        judge = mock_judge_class.return_value

        score = agentic._evaluate_answer_correctness(
            judge=judge,
            query="What is 2+2?",
            answer="4",
            ground_truth="4",
        )

        assert score == 0.95
        mock_judge.check.assert_called_once()

    def test_evaluate_tool_correctness_perfect_match(self, mock_model):
        """Test tool correctness evaluation with perfect match."""
        agentic = Agentic(retriever=MockRetriever, model=mock_model)

        agentic_data = {
            "tools_used": [{"tool_name": "calculator", "parameters": {"a": 5, "b": 7}, "result": 12, "step": 1}],
            "final_answer_uses_tools": True,
        }

        ground_truth = {
            "expected_tools": [{"tool_name": "calculator", "parameters": {"a": 5, "b": 7}, "step": 1}],
            "tool_sequence_matters": True,
        }

        result = agentic._evaluate_tool_correctness(agentic_data, ground_truth)

        assert isinstance(result, ToolCorrectnessScore)
        assert result.tool_selection_correct == 1.0
        assert result.parameter_accuracy == 1.0
        assert result.sequence_correct == 1.0
        assert result.result_utilization == 1.0
        assert result.overall_correctness == 1.0
        assert result.is_correct is True

    def test_evaluate_tool_correctness_partial_match(self, mock_model):
        """Test tool correctness evaluation with partial match."""
        agentic = Agentic(retriever=MockRetriever, model=mock_model)

        agentic_data = {
            "tools_used": [{"tool_name": "calculator", "parameters": {"a": 5, "b": 8}, "result": 13, "step": 1}],
            "final_answer_uses_tools": True,
        }

        ground_truth = {
            "expected_tools": [{"tool_name": "calculator", "parameters": {"a": 5, "b": 7}, "step": 1}],
            "tool_sequence_matters": True,
        }

        result = agentic._evaluate_tool_correctness(agentic_data, ground_truth)

        assert isinstance(result, ToolCorrectnessScore)
        assert result.tool_selection_correct == 1.0
        assert result.parameter_accuracy < 1.0
        assert result.sequence_correct == 1.0
        assert result.result_utilization == 1.0
        assert result.overall_correctness < 1.0

    def test_evaluate_tool_correctness_no_utilization(self, mock_model):
        """Test tool correctness when tools not used in final answer."""
        agentic = Agentic(retriever=MockRetriever, model=mock_model)

        agentic_data = {
            "tools_used": [{"tool_name": "calculator", "parameters": {"a": 5, "b": 7}, "result": 12, "step": 1}],
            "final_answer_uses_tools": False,
        }

        ground_truth = {
            "expected_tools": [{"tool_name": "calculator", "parameters": {"a": 5, "b": 7}, "step": 1}],
            "tool_sequence_matters": True,
        }

        result = agentic._evaluate_tool_correctness(agentic_data, ground_truth)

        assert result.result_utilization == 0.0
        assert result.overall_correctness < 1.0

    def test_evaluate_tool_correctness_wrong_tool(self, mock_model):
        """Test tool correctness with wrong tool selected."""
        agentic = Agentic(retriever=MockRetriever, model=mock_model)

        agentic_data = {
            "tools_used": [{"tool_name": "search", "parameters": {"query": "5+7"}, "result": "12", "step": 1}],
            "final_answer_uses_tools": True,
        }

        ground_truth = {
            "expected_tools": [{"tool_name": "calculator", "parameters": {"a": 5, "b": 7}, "step": 1}],
            "tool_sequence_matters": True,
        }

        result = agentic._evaluate_tool_correctness(agentic_data, ground_truth)

        assert result.tool_selection_correct < 1.0
        assert result.overall_correctness < 1.0

    @patch("fair_forge.metrics.agentic.Judge")
    def test_process_conversations(self, mock_judge_class, mock_model):
        """Test processing complete conversations."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge

        # Mock scores for all interactions across 4 conversations
        # Conversation 1: 3 interactions, all correct (scores: 0.9, 0.85, 0.95)
        # Conversation 2: 2 interactions, all correct (scores: 0.9, 0.85)
        # Conversation 3: 3 interactions, only 1 correct (scores: 0.9, 0.3, 0.2)
        # Conversation 4: 1 interaction, correct (score: 0.9)
        mock_judge.check.side_effect = [
            ("", {"correctness_score": 0.9, "reasoning": "Correct"}),
            ("", {"correctness_score": 0.85, "reasoning": "Correct"}),
            ("", {"correctness_score": 0.95, "reasoning": "Correct"}),
            ("", {"correctness_score": 0.9, "reasoning": "Correct"}),
            ("", {"correctness_score": 0.85, "reasoning": "Correct"}),
            ("", {"correctness_score": 0.9, "reasoning": "Correct"}),
            ("", {"correctness_score": 0.3, "reasoning": "Incorrect"}),
            ("", {"correctness_score": 0.2, "reasoning": "Incorrect"}),
            ("", {"correctness_score": 0.9, "reasoning": "Correct"}),
        ]

        agentic = Agentic(retriever=AgenticDatasetRetriever, model=mock_model, threshold=0.7)
        metrics = agentic._process()

        assert len(metrics) == 4  # 4 conversations

        # Conversation 1: fully correct (3/3)
        assert metrics[0].session_id == "conversation_001"
        assert metrics[0].total_interactions == 3
        assert metrics[0].correct_interactions == 3
        assert metrics[0].is_fully_correct is True

        # Conversation 2: fully correct (2/2)
        assert metrics[1].session_id == "conversation_002"
        assert metrics[1].total_interactions == 2
        assert metrics[1].correct_interactions == 2
        assert metrics[1].is_fully_correct is True

        # Conversation 3: partially correct (1/3) - NOT fully correct
        assert metrics[2].session_id == "conversation_003"
        assert metrics[2].total_interactions == 3
        assert metrics[2].correct_interactions == 1
        assert metrics[2].is_fully_correct is False

        # Conversation 4: fully correct (1/1)
        assert metrics[3].session_id == "conversation_004"
        assert metrics[3].total_interactions == 1
        assert metrics[3].correct_interactions == 1
        assert metrics[3].is_fully_correct is True

    @patch("fair_forge.metrics.agentic.Judge")
    def test_process_all_correct(self, mock_judge_class, mock_model):
        """Test when all conversations are fully correct."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge

        # All interactions correct (9 total)
        mock_judge.check.return_value = ("", {"correctness_score": 0.95, "reasoning": "Perfect"})

        agentic = Agentic(retriever=AgenticDatasetRetriever, model=mock_model, threshold=0.7)
        metrics = agentic._process()

        assert len(metrics) == 4
        assert all(m.is_fully_correct for m in metrics)

    @patch("fair_forge.metrics.agentic.Judge")
    def test_process_none_correct(self, mock_judge_class, mock_model):
        """Test when no conversations are fully correct."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge

        # All interactions incorrect
        mock_judge.check.return_value = ("", {"correctness_score": 0.2, "reasoning": "Wrong"})

        agentic = Agentic(retriever=AgenticDatasetRetriever, model=mock_model, threshold=0.7)
        metrics = agentic._process()

        assert len(metrics) == 4
        assert all(not m.is_fully_correct for m in metrics)

    @patch("fair_forge.metrics.agentic.Judge")
    def test_run_method(self, mock_judge_class, mock_model):
        """Test the run class method."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge

        mock_judge.check.side_effect = [
            ("", {"correctness_score": 0.9, "reasoning": "Correct"}),
            ("", {"correctness_score": 0.8, "reasoning": "Correct"}),
            ("", {"correctness_score": 0.5, "reasoning": "Incorrect"}),
        ]

        metrics = Agentic.run(
            AgenticDatasetRetriever,
            model=mock_model,
            threshold=0.7,
            verbose=False,
        )

        assert len(metrics) > 0
        assert isinstance(metrics[0], AgenticMetric)

    @patch("fair_forge.metrics.agentic.Judge")
    def test_judge_initialization_params(self, mock_judge_class, mock_model):
        """Test that Judge is initialized with correct parameters."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", {"correctness_score": 0.8, "reasoning": "OK"})

        agentic = Agentic(
            retriever=AgenticDatasetRetriever,
            model=mock_model,
            use_structured_output=True,
            bos_json_clause="[",
            eos_json_clause="]",
            verbose=True,
        )

        agentic._process()

        mock_judge_class.assert_called_once_with(
            model=mock_model,
            use_structured_output=True,
            bos_json_clause="[",
            eos_json_clause="]",
            verbose=True,
        )

    @patch("fair_forge.metrics.agentic.Judge")
    def test_verbose_mode(self, mock_judge_class, mock_model):
        """Test that verbose mode doesn't break processing."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        # 9 interactions: 3 + 2 + 3 + 1
        mock_judge.check.side_effect = [
            ("", {"correctness_score": 0.9, "reasoning": "Good"}),
            ("", {"correctness_score": 0.85, "reasoning": "Good"}),
            ("", {"correctness_score": 0.4, "reasoning": "Bad"}),
            ("", {"correctness_score": 0.9, "reasoning": "Good"}),
            ("", {"correctness_score": 0.85, "reasoning": "Good"}),
            ("", {"correctness_score": 0.9, "reasoning": "Good"}),
            ("", {"correctness_score": 0.3, "reasoning": "Bad"}),
            ("", {"correctness_score": 0.2, "reasoning": "Bad"}),
            ("", {"correctness_score": 0.9, "reasoning": "Good"}),
        ]

        agentic = Agentic(retriever=AgenticDatasetRetriever, model=mock_model, verbose=True)
        metrics = agentic._process()

        assert len(metrics) == 4  # 4 conversations

    @patch("fair_forge.metrics.agentic.Judge")
    def test_structured_output_mode(self, mock_judge_class, mock_model):
        """Test Agentic with structured output enabled."""
        from fair_forge.metrics.agentic import AnswerCorrectnessOutput

        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge

        expected_result = AnswerCorrectnessOutput(
            correctness_score=0.95,
            reasoning="Perfect match",
        )

        # 9 interactions total
        mock_judge.check.side_effect = [("", expected_result)] * 9

        agentic = Agentic(
            retriever=AgenticDatasetRetriever,
            model=mock_model,
            use_structured_output=True,
        )

        metrics = agentic._process()

        assert len(metrics) == 4  # 4 conversations
        # All conversations fully correct (all interactions scored 0.95)
        assert all(m.is_fully_correct for m in metrics)

    @patch("fair_forge.metrics.agentic.Judge")
    def test_tool_correctness_integration(self, mock_judge_class, mock_model):
        """Test that tool correctness is evaluated alongside answer correctness."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge

        mock_judge.check.side_effect = [
            ("", {"correctness_score": 0.9, "reasoning": "Correct"}),
            ("", {"correctness_score": 0.85, "reasoning": "Correct"}),
            ("", {"correctness_score": 0.3, "reasoning": "Wrong"}),
        ]

        agentic = Agentic(retriever=AgenticDatasetRetriever, model=mock_model, threshold=0.7)
        metrics = agentic._process()

        metric = metrics[0]

        assert metric.tool_correctness_scores is not None
        assert len(metric.tool_correctness_scores) > 0
        assert isinstance(metric.tool_correctness_scores[0], ToolCorrectnessScore)
        assert metric.tool_correctness_scores[0].overall_correctness >= 0.0
        assert metric.tool_correctness_scores[0].overall_correctness <= 1.0

    @patch("fair_forge.metrics.agentic.Judge")
    def test_answer_judge_returns_none(self, mock_judge_class, mock_model):
        """Test handling when judge returns None for answer correctness."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", None)

        agentic = Agentic(retriever=AgenticDatasetRetriever, model=mock_model)

        score = agentic._evaluate_answer_correctness(
            judge=mock_judge,
            query="Test",
            answer="Answer",
            ground_truth="Truth",
        )

        assert score == 0.0

    @patch("fair_forge.metrics.agentic.Judge")
    def test_custom_tool_weights(self, mock_judge_class, mock_model):
        """Test tool correctness with custom weights."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        # 9 interactions total
        mock_judge.check.side_effect = [
            ("", {"correctness_score": 0.8, "reasoning": "OK"}),
        ] * 9

        custom_weights = {
            "selection": 0.4,
            "parameters": 0.3,
            "sequence": 0.2,
            "utilization": 0.1,
        }

        agentic = Agentic(
            retriever=AgenticDatasetRetriever,
            model=mock_model,
            tool_weights=custom_weights,
        )

        assert agentic.tool_weights == custom_weights

        metrics = agentic._process()
        assert len(metrics) == 4  # 4 conversations

    @patch("fair_forge.metrics.agentic.Judge")
    def test_threshold_boundary(self, mock_judge_class, mock_model):
        """Test threshold boundary conditions."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge

        mock_judge.check.side_effect = [
            ("", {"correctness_score": 0.7, "reasoning": "Exact threshold"}),
            ("", {"correctness_score": 0.699, "reasoning": "Just below"}),
            ("", {"correctness_score": 0.701, "reasoning": "Just above"}),
        ]

        agentic = Agentic(retriever=AgenticDatasetRetriever, model=mock_model, threshold=0.7)
        metrics = agentic._process()

        assert len(metrics[0].correct_indices) == 2

    def test_batch_method(self, mock_model):
        """Test that batch method processes interactions."""
        from fair_forge.schemas.common import Batch

        agentic = Agentic(retriever=MockRetriever, model=mock_model)

        batch_data = [
            Batch(
                qa_id="qa_001",
                query="Test query",
                assistant="Test answer",
                ground_truth_assistant="Truth",
            )
        ]

        agentic.batch(
            session_id="session_1",
            context="Test context",
            assistant_id="assistant_1",
            batch=batch_data,
            language="english",
        )

    def test_pass_at_k_formula(self):
        """Test pass@k Bernoulli formula: 1 - (1 - c/n)^k."""
        from fair_forge.metrics.agentic import pass_at_k

        # 2/3 correct, k=3: 1 - (1/3)^3 = 1 - 1/27 ≈ 0.963
        result = pass_at_k(n=3, c=2, k=3)
        assert 0.96 < result < 0.97

        # 0/3 correct: always 0.0
        result = pass_at_k(n=3, c=0, k=3)
        assert result == 0.0

        # 3/3 correct: always 1.0
        result = pass_at_k(n=3, c=3, k=3)
        assert result == 1.0

        # 9 evaluated, 3 correct, k=3: 1 - (2/3)^3 = 1 - 8/27 ≈ 0.704
        result = pass_at_k(n=9, c=3, k=3)
        assert 0.70 < result < 0.71

    def test_pass_pow_k_formula(self):
        """Test pass^k formula: (c/n)^k."""
        from fair_forge.metrics.agentic import pass_pow_k

        # 2/3 correct, k=3: (2/3)^3 ≈ 0.296
        result = pass_pow_k(n=3, c=2, k=3)
        assert 0.29 < result < 0.30

        # 0/3 correct: always 0.0
        result = pass_pow_k(n=3, c=0, k=3)
        assert result == 0.0

        # 3/3 correct: always 1.0
        result = pass_pow_k(n=3, c=3, k=3)
        assert result == 1.0

        # 9 evaluated, 3 correct, k=3: (1/3)^3 ≈ 0.037
        result = pass_pow_k(n=9, c=3, k=3)
        assert 0.037 < result < 0.038

    def test_pass_at_k_k_exceeds_n(self):
        """Test pass@k when k > n — valid with the Bernoulli model."""
        from fair_forge.metrics.agentic import pass_at_k

        # k=5 > n=3: valid, p=2/3, 1-(1/3)^5 ≈ 0.9959
        result = pass_at_k(n=3, c=2, k=5)
        assert 0.99 < result < 1.0

    def test_pass_pow_k_k_exceeds_n(self):
        """Test pass^k when k > n — valid with the Bernoulli model."""
        from fair_forge.metrics.agentic import pass_pow_k

        # k=5 > n=3: valid, (2/3)^5 ≈ 0.132
        result = pass_pow_k(n=3, c=2, k=5)
        assert 0.13 < result < 0.14

