"""Tests for Agentic metric."""

from unittest.mock import MagicMock, patch

from fair_forge.metrics.agentic import Agentic
from fair_forge.schemas.agentic import AgenticMetric, ToolCorrectnessScore
from tests.fixtures.mock_retriever import AgenticDatasetRetriever, MockRetriever


class TestAgenticMetric:
    """Test suite for Agentic metric."""

    def mock_model(self):
        return MagicMock()

    def test_initialization_stores_k(self):
        """k is stored and accessible after initialization."""
        agentic = Agentic(retriever=MockRetriever, model=MagicMock(), k=5)
        assert agentic.k == 5
        assert agentic.threshold == 0.7
        assert agentic.tool_threshold == 1.0

    def test_evaluate_tool_correctness_perfect_match(self):
        """Perfect tool match scores 1.0 on all aspects."""
        agentic = Agentic(retriever=MockRetriever, model=MagicMock(), k=3)
        result = agentic._evaluate_tool_correctness(
            {
                "tools_used": [{"tool_name": "calc", "parameters": {"a": 5, "b": 7}, "result": 12, "step": 1}],
                "final_answer_uses_tools": True,
            },
            {
                "expected_tools": [{"tool_name": "calc", "parameters": {"a": 5, "b": 7}, "step": 1}],
                "tool_sequence_matters": True,
            },
        )
        assert isinstance(result, ToolCorrectnessScore)
        assert result.overall_correctness == 1.0
        assert result.is_correct is True

    def test_evaluate_tool_correctness_wrong_params(self):
        """Wrong parameters lower the overall score below 1.0."""
        agentic = Agentic(retriever=MockRetriever, model=MagicMock(), k=3)
        result = agentic._evaluate_tool_correctness(
            {
                "tools_used": [{"tool_name": "calc", "parameters": {"a": 5, "b": 8}, "step": 1}],
                "final_answer_uses_tools": True,
            },
            {
                "expected_tools": [{"tool_name": "calc", "parameters": {"a": 5, "b": 7}, "step": 1}],
                "tool_sequence_matters": True,
            },
        )
        assert result.parameter_accuracy < 1.0
        assert result.overall_correctness < 1.0

    @patch("fair_forge.metrics.agentic.Judge")
    def test_process_global_pass_at_k(self, mock_judge_class):
        """pass_at_k and pass_pow_k are computed globally from c/n across all conversations."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.side_effect = [
            # Conversation 1: 3/3 correct → fully correct
            ("", {"correctness_score": 0.9, "reasoning": ""}),
            ("", {"correctness_score": 0.85, "reasoning": ""}),
            ("", {"correctness_score": 0.95, "reasoning": ""}),
            # Conversation 2: 2/2 correct → fully correct
            ("", {"correctness_score": 0.9, "reasoning": ""}),
            ("", {"correctness_score": 0.85, "reasoning": ""}),
            # Conversation 3: 1/3 correct → NOT fully correct
            ("", {"correctness_score": 0.9, "reasoning": ""}),
            ("", {"correctness_score": 0.3, "reasoning": ""}),
            ("", {"correctness_score": 0.2, "reasoning": ""}),
            # Conversation 4: 1/1 correct → fully correct
            ("", {"correctness_score": 0.9, "reasoning": ""}),
        ]

        agentic = Agentic(retriever=AgenticDatasetRetriever, model=MagicMock(), k=3, threshold=0.7)
        result = agentic._process()

        assert isinstance(result, AgenticMetric)
        assert result.n == 4
        assert result.c == 3  # conv1, conv2, conv4 fully correct
        assert result.p == 0.75
        assert len(result.conversations) == 4

        # Global pass@3 = 1 - (1 - 0.75)^3 = 1 - 0.015625 = 0.984375
        assert abs(result.pass_at_k - 0.984375) < 0.001
        # Global pass^3 = 0.75^3 = 0.421875
        assert abs(result.pass_pow_k - 0.421875) < 0.001

    @patch("fair_forge.metrics.agentic.Judge")
    def test_process_all_correct(self, mock_judge_class):
        """All correct conversations → pass_at_k = pass_pow_k = 1.0, c == n."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", {"correctness_score": 0.95, "reasoning": ""})

        agentic = Agentic(retriever=AgenticDatasetRetriever, model=MagicMock(), k=3, threshold=0.7)
        result = agentic._process()

        assert result.c == result.n
        assert result.p == 1.0
        assert result.pass_at_k == 1.0
        assert result.pass_pow_k == 1.0

    @patch("fair_forge.metrics.agentic.Judge")
    def test_process_none_correct(self, mock_judge_class):
        """All incorrect interactions → c=0, pass_at_k = pass_pow_k = 0.0."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", {"correctness_score": 0.2, "reasoning": ""})

        agentic = Agentic(retriever=AgenticDatasetRetriever, model=MagicMock(), k=3, threshold=0.7)
        result = agentic._process()

        assert result.c == 0
        assert result.p == 0.0
        assert result.pass_at_k == 0.0
        assert result.pass_pow_k == 0.0

    @patch("fair_forge.metrics.agentic.Judge")
    def test_run_returns_single_agentic_metric(self, mock_judge_class):
        """run() returns a single AgenticMetric with global k, pass_at_k, pass_pow_k, and conversations."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", {"correctness_score": 0.9, "reasoning": ""})

        result = Agentic.run(AgenticDatasetRetriever, k=3, model=MagicMock(), threshold=0.7)

        assert isinstance(result, AgenticMetric)
        assert result.k == 3
        assert result.n > 0
        assert len(result.conversations) == result.n
        assert 0.0 <= result.pass_at_k <= 1.0
        assert 0.0 <= result.pass_pow_k <= 1.0

    @patch("fair_forge.metrics.agentic.Judge")
    def test_threshold_boundary(self, mock_judge_class):
        """Score at threshold counts as correct; just below does not."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.side_effect = [
            ("", {"correctness_score": 0.7, "reasoning": ""}),  # exactly at threshold → correct
            ("", {"correctness_score": 0.699, "reasoning": ""}),  # just below → incorrect
            ("", {"correctness_score": 0.701, "reasoning": ""}),  # just above → correct
        ]

        agentic = Agentic(retriever=AgenticDatasetRetriever, model=MagicMock(), k=3, threshold=0.7)
        result = agentic._process()

        assert len(result.conversations[0].correct_indices) == 2

    def test_pass_at_k_formula(self):
        """Bernoulli model: 1 - (1 - c/n)^k."""
        from fair_forge.metrics.agentic import pass_at_k

        assert 0.96 < pass_at_k(n=3, c=2, k=3) < 0.97  # 1 - (1/3)^3 ≈ 0.963
        assert pass_at_k(n=3, c=0, k=3) == 0.0
        assert pass_at_k(n=3, c=3, k=3) == 1.0
        assert 0.70 < pass_at_k(n=9, c=3, k=3) < 0.71  # 1 - (2/3)^3 ≈ 0.704

    def test_pass_pow_k_formula(self):
        """Formula: (c/n)^k."""
        from fair_forge.metrics.agentic import pass_pow_k

        assert 0.29 < pass_pow_k(n=3, c=2, k=3) < 0.30  # (2/3)^3 ≈ 0.296
        assert pass_pow_k(n=3, c=0, k=3) == 0.0
        assert pass_pow_k(n=3, c=3, k=3) == 1.0

    def test_pass_at_k_k_exceeds_n(self):
        """k > n is valid with the Bernoulli model."""
        from fair_forge.metrics.agentic import pass_at_k, pass_pow_k

        assert 0.99 < pass_at_k(n=3, c=2, k=5) < 1.0  # 1-(1/3)^5 ≈ 0.9959
        assert 0.13 < pass_pow_k(n=3, c=2, k=5) < 0.14  # (2/3)^5 ≈ 0.132
