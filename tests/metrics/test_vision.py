"""Unit tests for vision hallucination metrics: FalsePositiveRate, Precision, ConfidenceScoreAnalysis."""

from unittest.mock import MagicMock, patch

import pytest

from fair_forge.metrics.vision import ConfidenceScoreAnalysis, FalsePositiveRate, Precision
from fair_forge.schemas.vision import (
    ConfidenceScoreMetric,
    FalsePositiveRateMetric,
    PrecisionMetric,
    VisionInteraction,
)
from tests.fixtures.mock_data import create_sample_batch, create_sample_dataset
from tests.fixtures.mock_retriever import MockRetriever, VisionDatasetRetriever


@pytest.fixture
def mock_model():
    return MagicMock()


def _make_judge_result(classification: str, reasoning: str = "test reasoning") -> dict:
    return {"classification": classification, "reasoning": reasoning}


def _vision_retriever(batches):
    dataset = create_sample_dataset(
        session_id="vision_session",
        assistant_id="argos_vlm",
        context="Argos security camera",
        conversation=batches,
    )
    return type("VisionRetriever", (MockRetriever,), {"load_dataset": lambda self: [dataset]})


class TestFalsePositiveRate:
    """Test suite for FalsePositiveRate metric."""

    @patch("fair_forge.metrics.vision.Judge")
    def test_initialization(self, mock_judge_class, mock_model, vision_dataset_retriever):
        metric = FalsePositiveRate(vision_dataset_retriever, model=mock_model)
        assert metric is not None
        assert metric._session_data == {}

    @patch("fair_forge.metrics.vision.Judge")
    def test_single_false_positive(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", _make_judge_result("false_positive"))

        batches = [create_sample_batch(qa_id="qa_001", assistant="event detected", ground_truth_assistant="no event")]
        retriever = _vision_retriever(batches)
        metric = FalsePositiveRate(retriever, model=mock_model)
        metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)
        metric.on_process_complete()

        assert len(metric.metrics) == 1
        result = metric.metrics[0]
        assert isinstance(result, FalsePositiveRateMetric)
        assert result.n_false_positives == 1
        assert result.n_negatives == 1
        assert result.false_positive_rate == pytest.approx(1.0)

    @patch("fair_forge.metrics.vision.Judge")
    def test_no_false_positives(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.side_effect = [
            ("", _make_judge_result("true_positive")),
            ("", _make_judge_result("true_negative")),
        ]

        batches = [
            create_sample_batch(qa_id="qa_001"),
            create_sample_batch(qa_id="qa_002"),
        ]
        retriever = _vision_retriever(batches)
        metric = FalsePositiveRate(retriever, model=mock_model)
        metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)
        metric.on_process_complete()

        result = metric.metrics[0]
        assert result.n_false_positives == 0
        assert result.false_positive_rate == pytest.approx(0.0)

    @patch("fair_forge.metrics.vision.Judge")
    def test_fpr_calculation(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.side_effect = [
            ("", _make_judge_result("true_positive")),
            ("", _make_judge_result("true_negative")),
            ("", _make_judge_result("false_positive")),
            ("", _make_judge_result("false_negative")),
        ]

        batches = [create_sample_batch(qa_id=f"qa_00{i}") for i in range(1, 5)]
        retriever = _vision_retriever(batches)
        metric = FalsePositiveRate(retriever, model=mock_model)
        metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)
        metric.on_process_complete()

        result = metric.metrics[0]
        assert result.n_predictions == 4
        assert result.n_negatives == 2  # TN + FP
        assert result.n_false_positives == 1
        assert result.false_positive_rate == pytest.approx(0.5)

    @patch("fair_forge.metrics.vision.Judge")
    def test_no_actual_negatives_returns_none(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.side_effect = [
            ("", _make_judge_result("true_positive")),
            ("", _make_judge_result("false_negative")),
        ]

        batches = [create_sample_batch(qa_id=f"qa_00{i}") for i in range(1, 3)]
        retriever = _vision_retriever(batches)
        metric = FalsePositiveRate(retriever, model=mock_model)
        metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)
        metric.on_process_complete()

        assert metric.metrics[0].false_positive_rate is None

    @patch("fair_forge.metrics.vision.Judge")
    def test_multiple_sessions(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", _make_judge_result("false_positive"))

        batches = [create_sample_batch(qa_id="qa_001")]
        dataset_a = create_sample_dataset(session_id="session_a", conversation=batches)
        dataset_b = create_sample_dataset(session_id="session_b", conversation=batches)
        retriever = type("R", (MockRetriever,), {"load_dataset": lambda self: [dataset_a, dataset_b]})

        metric = FalsePositiveRate(retriever, model=mock_model)
        metric.batch(session_id="session_a", context="cam", assistant_id="vlm", batch=batches)
        metric.batch(session_id="session_b", context="cam", assistant_id="vlm", batch=batches)
        metric.on_process_complete()

        assert len(metric.metrics) == 2

    @patch("fair_forge.metrics.vision.Judge")
    def test_run_method(self, mock_judge_class, mock_model, vision_dataset_retriever):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", _make_judge_result("true_negative"))

        results = FalsePositiveRate.run(vision_dataset_retriever, model=mock_model, verbose=False)

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], FalsePositiveRateMetric)

    @patch("fair_forge.metrics.vision.Judge")
    def test_interactions_stored(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", _make_judge_result("false_positive", "hallucinated event"))

        batches = [create_sample_batch(qa_id="qa_001")]
        retriever = _vision_retriever(batches)
        metric = FalsePositiveRate(retriever, model=mock_model)
        metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)
        metric.on_process_complete()

        interaction = metric.metrics[0].interactions[0]
        assert isinstance(interaction, VisionInteraction)
        assert interaction.qa_id == "qa_001"
        assert interaction.classification == "false_positive"
        assert interaction.reasoning == "hallucinated event"

    @patch("fair_forge.metrics.vision.Judge")
    def test_judge_none_raises(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", None)

        batches = [create_sample_batch(qa_id="qa_001")]
        retriever = _vision_retriever(batches)
        metric = FalsePositiveRate(retriever, model=mock_model)

        with pytest.raises(ValueError, match="No valid response from judge"):
            metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)


class TestPrecision:
    """Test suite for Precision metric."""

    @patch("fair_forge.metrics.vision.Judge")
    def test_initialization(self, mock_judge_class, mock_model, vision_dataset_retriever):
        metric = Precision(vision_dataset_retriever, model=mock_model)
        assert metric is not None
        assert metric._session_data == {}

    @patch("fair_forge.metrics.vision.Judge")
    def test_perfect_precision(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.side_effect = [
            ("", _make_judge_result("true_positive")),
            ("", _make_judge_result("true_positive")),
        ]

        batches = [create_sample_batch(qa_id=f"qa_00{i}") for i in range(1, 3)]
        retriever = _vision_retriever(batches)
        metric = Precision(retriever, model=mock_model)
        metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)
        metric.on_process_complete()

        result = metric.metrics[0]
        assert isinstance(result, PrecisionMetric)
        assert result.precision == pytest.approx(1.0)

    @patch("fair_forge.metrics.vision.Judge")
    def test_precision_calculation(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.side_effect = [
            ("", _make_judge_result("true_positive")),
            ("", _make_judge_result("true_negative")),
            ("", _make_judge_result("false_positive")),
            ("", _make_judge_result("false_negative")),
        ]

        batches = [create_sample_batch(qa_id=f"qa_00{i}") for i in range(1, 5)]
        retriever = _vision_retriever(batches)
        metric = Precision(retriever, model=mock_model)
        metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)
        metric.on_process_complete()

        result = metric.metrics[0]
        assert result.n_positive_predictions == 2  # TP + FP
        assert result.n_true_positives == 1
        assert result.precision == pytest.approx(0.5)

    @patch("fair_forge.metrics.vision.Judge")
    def test_no_positive_predictions_returns_none(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.side_effect = [
            ("", _make_judge_result("true_negative")),
            ("", _make_judge_result("false_negative")),
        ]

        batches = [create_sample_batch(qa_id=f"qa_00{i}") for i in range(1, 3)]
        retriever = _vision_retriever(batches)
        metric = Precision(retriever, model=mock_model)
        metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)
        metric.on_process_complete()

        assert metric.metrics[0].precision is None

    @patch("fair_forge.metrics.vision.Judge")
    def test_run_method(self, mock_judge_class, mock_model, vision_dataset_retriever):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", _make_judge_result("true_positive"))

        results = Precision.run(vision_dataset_retriever, model=mock_model, verbose=False)

        assert isinstance(results, list)
        assert isinstance(results[0], PrecisionMetric)


class TestConfidenceScoreAnalysis:
    """Test suite for ConfidenceScoreAnalysis metric."""

    @patch("fair_forge.metrics.vision.Judge")
    def test_initialization(self, mock_judge_class, mock_model, vision_dataset_retriever):
        metric = ConfidenceScoreAnalysis(vision_dataset_retriever, model=mock_model)
        assert metric is not None

    @patch("fair_forge.metrics.vision.Judge")
    def test_confidence_stats(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.side_effect = [
            ("", _make_judge_result("true_positive")),
            ("", _make_judge_result("true_negative")),
            ("", _make_judge_result("false_positive")),
            ("", _make_judge_result("false_negative")),
        ]

        batches = [
            create_sample_batch(qa_id="qa_001", agentic={"confidence": 0.9}),
            create_sample_batch(qa_id="qa_002", agentic={"confidence": 0.1}),
            create_sample_batch(qa_id="qa_003", agentic={"confidence": 0.8}),
            create_sample_batch(qa_id="qa_004", agentic={"confidence": 0.4}),
        ]
        retriever = _vision_retriever(batches)
        metric = ConfidenceScoreAnalysis(retriever, model=mock_model)
        metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)
        metric.on_process_complete()

        result = metric.metrics[0]
        assert isinstance(result, ConfidenceScoreMetric)
        assert result.n_with_confidence == 4
        assert result.confidence_mean == pytest.approx(0.55, abs=0.01)
        assert result.confidence_min == pytest.approx(0.1)
        assert result.confidence_max == pytest.approx(0.9)
        assert result.confidence_std is not None

    @patch("fair_forge.metrics.vision.Judge")
    def test_no_confidence_scores(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", _make_judge_result("true_negative"))

        batches = [create_sample_batch(qa_id="qa_001", agentic={})]
        retriever = _vision_retriever(batches)
        metric = ConfidenceScoreAnalysis(retriever, model=mock_model)
        metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)
        metric.on_process_complete()

        result = metric.metrics[0]
        assert result.n_with_confidence == 0
        assert result.confidence_mean is None
        assert result.expected_calibration_error is None
        assert result.buckets == []

    @patch("fair_forge.metrics.vision.Judge")
    def test_ece_perfect_calibration(self, mock_judge_class, mock_model):
        """A model that is always correct should have ECE close to 0."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", _make_judge_result("true_positive"))

        batches = [
            create_sample_batch(qa_id=f"qa_00{i}", agentic={"confidence": 0.95})
            for i in range(1, 5)
        ]
        retriever = _vision_retriever(batches)
        metric = ConfidenceScoreAnalysis(retriever, model=mock_model)
        metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)
        metric.on_process_complete()

        result = metric.metrics[0]
        assert result.expected_calibration_error is not None
        assert result.expected_calibration_error < 0.1

    @patch("fair_forge.metrics.vision.Judge")
    def test_ece_poor_calibration(self, mock_judge_class, mock_model):
        """A model that is always wrong despite high confidence should have high ECE."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", _make_judge_result("false_positive"))

        batches = [
            create_sample_batch(qa_id=f"qa_00{i}", agentic={"confidence": 0.95})
            for i in range(1, 5)
        ]
        retriever = _vision_retriever(batches)
        metric = ConfidenceScoreAnalysis(retriever, model=mock_model)
        metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)
        metric.on_process_complete()

        result = metric.metrics[0]
        assert result.expected_calibration_error is not None
        assert result.expected_calibration_error > 0.5

    @patch("fair_forge.metrics.vision.Judge")
    def test_buckets_populated(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.side_effect = [
            ("", _make_judge_result("true_positive")),
            ("", _make_judge_result("true_negative")),
        ]

        batches = [
            create_sample_batch(qa_id="qa_001", agentic={"confidence": 0.9}),
            create_sample_batch(qa_id="qa_002", agentic={"confidence": 0.1}),
        ]
        retriever = _vision_retriever(batches)
        metric = ConfidenceScoreAnalysis(retriever, model=mock_model)
        metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)
        metric.on_process_complete()

        result = metric.metrics[0]
        non_empty_buckets = [b for b in result.buckets if b.count > 0]
        assert len(non_empty_buckets) == 2

    @patch("fair_forge.metrics.vision.Judge")
    def test_run_method(self, mock_judge_class, mock_model, vision_dataset_retriever):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", _make_judge_result("true_negative"))

        results = ConfidenceScoreAnalysis.run(vision_dataset_retriever, model=mock_model, verbose=False)

        assert isinstance(results, list)
        assert isinstance(results[0], ConfidenceScoreMetric)

    @patch("fair_forge.metrics.vision.Judge")
    def test_compute_stats(self, mock_judge_class, mock_model, vision_dataset_retriever):
        metric = ConfidenceScoreAnalysis(vision_dataset_retriever, model=mock_model)
        mean, std, min_v, max_v = metric._compute_stats([0.2, 0.4, 0.6, 0.8])
        assert mean == pytest.approx(0.5)
        assert min_v == pytest.approx(0.2)
        assert max_v == pytest.approx(0.8)
        assert std > 0
