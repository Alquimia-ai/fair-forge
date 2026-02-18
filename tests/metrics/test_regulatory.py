"""Tests for Regulatory metric."""

from unittest.mock import MagicMock, patch

import pytest

from fair_forge.metrics.regulatory import Regulatory
from fair_forge.schemas.regulatory import RegulatoryMetric
from tests.fixtures.mock_data import create_sample_batch
from tests.fixtures.mock_retriever import MockRetriever, RegulatoryDatasetRetriever


SAMPLE_REGULATIONS = [
    "The assistant must verify user identity before processing financial transactions",
    "The assistant must not share personal financial data without explicit user consent",
    "The assistant must provide accurate and up-to-date account information",
    "The assistant must maintain a professional and courteous tone",
]


class TestRegulatoryMetric:
    """Test suite for Regulatory metric."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock BaseChatModel."""
        return MagicMock()

    def test_initialization_default(self, mock_model):
        """Test Regulatory initialization with default parameters."""
        regulatory = Regulatory(
            retriever=MockRetriever,
            model=mock_model,
            regulations=SAMPLE_REGULATIONS,
        )

        assert regulatory.model == mock_model
        assert regulatory.regulations == SAMPLE_REGULATIONS
        assert regulatory.use_structured_output is False
        assert regulatory.bos_json_clause == "```json"
        assert regulatory.eos_json_clause == "```"

    def test_initialization_custom(self, mock_model):
        """Test Regulatory initialization with custom parameters."""
        regulatory = Regulatory(
            retriever=MockRetriever,
            model=mock_model,
            regulations=SAMPLE_REGULATIONS,
            use_structured_output=True,
            bos_json_clause="<json>",
            eos_json_clause="</json>",
        )

        assert regulatory.model == mock_model
        assert regulatory.regulations == SAMPLE_REGULATIONS
        assert regulatory.use_structured_output is True
        assert regulatory.bos_json_clause == "<json>"
        assert regulatory.eos_json_clause == "</json>"

    @patch("fair_forge.metrics.regulatory.Judge")
    def test_batch_processing(self, mock_judge_class, mock_model):
        """Test batch method processes interactions correctly."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "I analyzed the compliance...",
            {
                "compliance_score": 0.95,
                "insight": "The response is largely compliant with regulations",
                "violated_rules": [],
                "rule_assessments": {
                    "1": {"compliant": True, "reason": "Identity verification mentioned"},
                    "2": {"compliant": True, "reason": "No unauthorized data sharing"},
                },
            },
        )

        regulatory = Regulatory(
            retriever=MockRetriever,
            model=mock_model,
            regulations=SAMPLE_REGULATIONS,
        )

        batches = [
            create_sample_batch(qa_id="qa_001"),
            create_sample_batch(qa_id="qa_002"),
        ]

        regulatory.batch(
            session_id="test_session",
            context="Banking assistant context",
            assistant_id="test_assistant",
            batch=batches,
            language="english",
        )

        assert len(regulatory.metrics) == 2
        assert all(isinstance(m, RegulatoryMetric) for m in regulatory.metrics)
        assert regulatory.metrics[0].session_id == "test_session"
        assert regulatory.metrics[0].qa_id == "qa_001"
        assert regulatory.metrics[0].compliance_score == 0.95
        assert regulatory.metrics[0].compliance_insight == "The response is largely compliant with regulations"
        assert regulatory.metrics[0].compliance_thinkings == "I analyzed the compliance..."
        assert regulatory.metrics[0].violated_rules == []

    @patch("fair_forge.metrics.regulatory.Judge")
    def test_batch_with_violations(self, mock_judge_class, mock_model):
        """Test batch method handles compliance violations correctly."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Analyzing violations...",
            {
                "compliance_score": 0.5,
                "insight": "Some regulations were violated",
                "violated_rules": ["1", "2"],
                "rule_assessments": {
                    "1": {"compliant": False, "reason": "No identity verification"},
                    "2": {"compliant": False, "reason": "Data shared without consent"},
                },
            },
        )

        regulatory = Regulatory(
            retriever=MockRetriever,
            model=mock_model,
            regulations=SAMPLE_REGULATIONS,
        )

        batch = create_sample_batch(qa_id="qa_001")

        regulatory.batch(
            session_id="test_session",
            context="Test context",
            assistant_id="test_assistant",
            batch=[batch],
        )

        assert len(regulatory.metrics) == 1
        assert regulatory.metrics[0].compliance_score == 0.5
        assert regulatory.metrics[0].violated_rules == ["1", "2"]
        assert "1" in regulatory.metrics[0].rule_assessments
        assert regulatory.metrics[0].rule_assessments["1"]["compliant"] is False

    @patch("fair_forge.metrics.regulatory.Judge")
    def test_batch_raises_on_no_result(self, mock_judge_class, mock_model):
        """Test batch raises ValueError when no result is returned."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("Some thinking", None)

        regulatory = Regulatory(
            retriever=MockRetriever,
            model=mock_model,
            regulations=SAMPLE_REGULATIONS,
        )

        batch = create_sample_batch(qa_id="qa_001")

        with pytest.raises(ValueError, match="No valid response"):
            regulatory.batch(
                session_id="test_session",
                context="Test context",
                assistant_id="test_assistant",
                batch=[batch],
            )

    @patch("fair_forge.metrics.regulatory.Judge")
    def test_run_method(self, mock_judge_class, mock_model):
        """Test the run class method."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Thinking...",
            {
                "compliance_score": 0.85,
                "insight": "Good compliance",
                "violated_rules": [],
                "rule_assessments": {},
            },
        )

        metrics = Regulatory.run(
            RegulatoryDatasetRetriever,
            model=mock_model,
            regulations=SAMPLE_REGULATIONS,
            verbose=False,
        )

        assert len(metrics) > 0
        assert all(isinstance(m, RegulatoryMetric) for m in metrics)

    @patch("fair_forge.metrics.regulatory.Judge")
    def test_judge_initialization_params(self, mock_judge_class, mock_model):
        """Test that Judge is initialized with correct parameters."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "",
            {
                "compliance_score": 0.5,
                "insight": "test",
                "violated_rules": [],
                "rule_assessments": {},
            },
        )

        regulatory = Regulatory(
            retriever=MockRetriever,
            model=mock_model,
            regulations=SAMPLE_REGULATIONS,
            use_structured_output=False,
            bos_json_clause="[",
            eos_json_clause="]",
        )

        batch = create_sample_batch(qa_id="qa_001")
        regulatory.batch(
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
            verbose=False,
        )

    @patch("fair_forge.metrics.regulatory.Judge")
    def test_verbose_mode(self, mock_judge_class, mock_model):
        """Test that verbose mode doesn't break processing."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Thinking...",
            {
                "compliance_score": 0.8,
                "insight": "Test",
                "violated_rules": [],
                "rule_assessments": {},
            },
        )

        regulatory = Regulatory(
            retriever=MockRetriever,
            model=mock_model,
            regulations=SAMPLE_REGULATIONS,
            verbose=True,
        )

        batch = create_sample_batch(qa_id="qa_001")
        regulatory.batch(
            session_id="s",
            context="c",
            assistant_id="a",
            batch=[batch],
        )

        assert len(regulatory.metrics) == 1

    @patch("fair_forge.metrics.regulatory.Judge")
    def test_structured_output_mode(self, mock_judge_class, mock_model):
        """Test Regulatory with structured output enabled."""
        from fair_forge.llm.schemas import RegulatoryJudgeOutput

        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge

        expected_result = RegulatoryJudgeOutput(
            compliance_score=0.9,
            insight="Fully compliant response",
            violated_rules=[],
            rule_assessments={"1": {"compliant": True, "reason": "OK"}},
        )
        mock_judge.check.return_value = ("", expected_result)

        regulatory = Regulatory(
            retriever=MockRetriever,
            model=mock_model,
            regulations=SAMPLE_REGULATIONS,
            use_structured_output=True,
        )

        batch = create_sample_batch(qa_id="qa_001")
        regulatory.batch(
            session_id="test_session",
            context="Test context",
            assistant_id="test_assistant",
            batch=[batch],
        )

        assert len(regulatory.metrics) == 1
        assert regulatory.metrics[0].compliance_score == 0.9
        assert regulatory.metrics[0].compliance_insight == "Fully compliant response"
        assert regulatory.metrics[0].violated_rules == []

    @patch("fair_forge.metrics.regulatory.Judge")
    def test_regulations_formatted_in_prompt(self, mock_judge_class, mock_model):
        """Test that regulations are properly formatted when passed to judge."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "",
            {
                "compliance_score": 1.0,
                "insight": "OK",
                "violated_rules": [],
                "rule_assessments": {},
            },
        )

        regulatory = Regulatory(
            retriever=MockRetriever,
            model=mock_model,
            regulations=["Rule A", "Rule B"],
        )

        batch = create_sample_batch(qa_id="qa_001")
        regulatory.batch(
            session_id="s",
            context="c",
            assistant_id="a",
            batch=[batch],
        )

        call_args = mock_judge.check.call_args
        data = call_args[0][2]
        assert "regulations" in data
        assert "1. Rule A" in data["regulations"]
        assert "2. Rule B" in data["regulations"]

    def test_metric_attributes(self, mock_model):
        """Test that all expected attributes exist in RegulatoryMetric."""
        with patch("fair_forge.metrics.regulatory.Judge") as mock_judge_class:
            mock_judge = MagicMock()
            mock_judge_class.return_value = mock_judge
            mock_judge.check.return_value = (
                "thinking",
                {
                    "compliance_score": 0.75,
                    "insight": "Partial compliance",
                    "violated_rules": ["1"],
                    "rule_assessments": {"1": {"compliant": False, "reason": "Failed"}},
                },
            )

            metrics = Regulatory.run(
                RegulatoryDatasetRetriever,
                model=mock_model,
                regulations=SAMPLE_REGULATIONS,
                verbose=False,
            )

            assert len(metrics) > 0
            m = metrics[0]

            required_attributes = [
                "session_id",
                "assistant_id",
                "qa_id",
                "compliance_score",
                "compliance_insight",
                "compliance_thinkings",
                "violated_rules",
                "rule_assessments",
            ]

            for attr in required_attributes:
                assert hasattr(m, attr), f"Missing attribute: {attr}"
