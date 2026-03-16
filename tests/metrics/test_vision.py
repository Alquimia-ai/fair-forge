"""Unit tests for vision hallucination metrics: FalsePositiveRate, Precision, ConfidenceScoreAnalysis."""

from unittest.mock import MagicMock, patch

import numpy as np
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

_HIGH = np.array([1.0, 0.0])  # similarity = 1.0 (correct)
_LOW = np.array([0.0, 1.0])   # similarity = 0.0 (incorrect)


def _mock_encoder(encode_sequence: list[np.ndarray]):
    """Returns a mock encoder whose encode() calls return vectors from encode_sequence in order."""
    encoder = MagicMock()
    encoder.encode.side_effect = encode_sequence
    return encoder


def _vision_retriever(batches):
    dataset = create_sample_dataset(
        session_id="vision_session",
        assistant_id="argos_vlm",
        context="Argos security camera",
        conversation=batches,
    )
    return type("VisionRetriever", (MockRetriever,), {"load_dataset": lambda self: [dataset]})


def _batch(qa_id: str, gt_detected: bool, confidence: float | None = None):
    return create_sample_batch(
        qa_id=qa_id,
        assistant="VLM description",
        ground_truth_assistant="Ground truth description",
        agentic={"confidence": confidence} if confidence is not None else {},
        ground_truth_agentic={"detected": gt_detected},
    )


def _run_metric(metric_class, batches, encoder, threshold=0.75, **kwargs):
    retriever = _vision_retriever(batches)
    metric = metric_class(retriever, threshold=threshold, **kwargs)
    metric._encoder = encoder
    metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)
    metric.on_process_complete()
    return metric


class TestFalsePositiveRate:
    def test_initialization(self, vision_dataset_retriever):
        metric = FalsePositiveRate(vision_dataset_retriever)
        assert metric._session_data == {}
        assert metric._threshold == 0.75

    def test_single_false_positive(self):
        # LOW similarity + gt_detected=False → FP
        batches = [_batch("qa_001", gt_detected=False)]
        encoder = _mock_encoder([np.array([_HIGH]), np.array([_LOW])])
        metric = _run_metric(FalsePositiveRate, batches, encoder)

        result = metric.metrics[0]
        assert isinstance(result, FalsePositiveRateMetric)
        assert result.n_false_positives == 1
        assert result.n_negatives == 1
        assert result.false_positive_rate == pytest.approx(1.0)

    def test_no_false_positives(self):
        # HIGH similarity + gt_detected=True → TP
        # HIGH similarity + gt_detected=False → TN
        batches = [_batch("qa_001", gt_detected=True), _batch("qa_002", gt_detected=False)]
        encoder = _mock_encoder([
            np.array([_HIGH, _HIGH]),
            np.array([_HIGH, _HIGH]),
        ])
        metric = _run_metric(FalsePositiveRate, batches, encoder)

        result = metric.metrics[0]
        assert result.n_false_positives == 0
        assert result.false_positive_rate == pytest.approx(0.0)

    def test_fpr_calculation(self):
        # TP, TN, FP, FN
        batches = [
            _batch("qa_001", gt_detected=True),   # HIGH → TP
            _batch("qa_002", gt_detected=False),  # HIGH → TN
            _batch("qa_003", gt_detected=False),  # LOW  → FP
            _batch("qa_004", gt_detected=True),   # LOW  → FN
        ]
        encoder = _mock_encoder([
            np.array([_HIGH, _HIGH, _LOW, _LOW]),
            np.array([_HIGH, _HIGH, _HIGH, _HIGH]),
        ])
        metric = _run_metric(FalsePositiveRate, batches, encoder)

        result = metric.metrics[0]
        assert result.n_predictions == 4
        assert result.n_negatives == 2   # TN + FP
        assert result.n_false_positives == 1
        assert result.false_positive_rate == pytest.approx(0.5)

    def test_no_actual_negatives_returns_none(self):
        batches = [_batch("qa_001", gt_detected=True), _batch("qa_002", gt_detected=True)]
        encoder = _mock_encoder([np.array([_HIGH, _LOW]), np.array([_HIGH, _HIGH])])
        metric = _run_metric(FalsePositiveRate, batches, encoder)

        assert metric.metrics[0].false_positive_rate is None

    def test_multiple_sessions(self):
        batches = [_batch("qa_001", gt_detected=False)]
        dataset_a = create_sample_dataset(session_id="session_a", conversation=batches)
        dataset_b = create_sample_dataset(session_id="session_b", conversation=batches)
        retriever = type("R", (MockRetriever,), {"load_dataset": lambda self: [dataset_a, dataset_b]})

        encoder = _mock_encoder([
            np.array([_HIGH]), np.array([_LOW]),
            np.array([_HIGH]), np.array([_LOW]),
        ])
        metric = FalsePositiveRate(retriever)
        metric._encoder = encoder
        metric.batch(session_id="session_a", context="cam", assistant_id="vlm", batch=batches)
        metric.batch(session_id="session_b", context="cam", assistant_id="vlm", batch=batches)
        metric.on_process_complete()

        assert len(metric.metrics) == 2

    def test_interactions_stored(self):
        batches = [_batch("qa_001", gt_detected=False)]
        encoder = _mock_encoder([np.array([_HIGH]), np.array([_LOW])])
        metric = _run_metric(FalsePositiveRate, batches, encoder)

        interaction = metric.metrics[0].interactions[0]
        assert isinstance(interaction, VisionInteraction)
        assert interaction.qa_id == "qa_001"
        assert interaction.classification == "false_positive"
        assert interaction.similarity_score is not None

    def test_missing_ground_truth_detected_raises(self):
        batches = [create_sample_batch(qa_id="qa_001", ground_truth_agentic={})]
        retriever = _vision_retriever(batches)
        encoder = _mock_encoder([np.array([_HIGH]), np.array([_HIGH])])
        metric = FalsePositiveRate(retriever)
        metric._encoder = encoder

        with pytest.raises(ValueError, match="Missing 'detected' in ground_truth_agentic"):
            metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)

    def test_configurable_threshold(self):
        # With threshold=0.5, a similarity of 0.0 is still below → FP
        batches = [_batch("qa_001", gt_detected=False)]
        encoder = _mock_encoder([np.array([_HIGH]), np.array([_LOW])])
        metric = _run_metric(FalsePositiveRate, batches, encoder, threshold=0.5)

        assert metric.metrics[0].interactions[0].classification == "false_positive"

    def test_run_method(self, vision_dataset_retriever):
        mock_encoder = MagicMock()
        mock_encoder.encode.side_effect = [
            np.array([_HIGH, _HIGH, _LOW, _LOW]),
            np.array([_HIGH, _HIGH, _HIGH, _HIGH]),
        ]
        with patch.object(FalsePositiveRate, "_get_encoder", return_value=mock_encoder):
            results = FalsePositiveRate.run(vision_dataset_retriever, verbose=False)

        assert isinstance(results, list)
        assert isinstance(results[0], FalsePositiveRateMetric)


class TestPrecision:
    def test_initialization(self, vision_dataset_retriever):
        metric = Precision(vision_dataset_retriever)
        assert metric._session_data == {}

    def test_perfect_precision(self):
        batches = [_batch("qa_001", gt_detected=True), _batch("qa_002", gt_detected=True)]
        encoder = _mock_encoder([np.array([_HIGH, _HIGH]), np.array([_HIGH, _HIGH])])
        metric = _run_metric(Precision, batches, encoder)

        assert isinstance(metric.metrics[0], PrecisionMetric)
        assert metric.metrics[0].precision == pytest.approx(1.0)

    def test_precision_calculation(self):
        batches = [
            _batch("qa_001", gt_detected=True),   # HIGH → TP
            _batch("qa_002", gt_detected=False),  # HIGH → TN
            _batch("qa_003", gt_detected=False),  # LOW  → FP
            _batch("qa_004", gt_detected=True),   # LOW  → FN
        ]
        encoder = _mock_encoder([
            np.array([_HIGH, _HIGH, _LOW, _LOW]),
            np.array([_HIGH, _HIGH, _HIGH, _HIGH]),
        ])
        metric = _run_metric(Precision, batches, encoder)

        result = metric.metrics[0]
        assert result.n_positive_predictions == 2  # TP + FP
        assert result.n_true_positives == 1
        assert result.precision == pytest.approx(0.5)

    def test_no_positive_predictions_returns_none(self):
        batches = [_batch("qa_001", gt_detected=False), _batch("qa_002", gt_detected=True)]
        encoder = _mock_encoder([np.array([_HIGH, _LOW]), np.array([_HIGH, _HIGH])])
        metric = _run_metric(Precision, batches, encoder)

        assert metric.metrics[0].precision is None

    def test_run_method(self, vision_dataset_retriever):
        mock_encoder = MagicMock()
        mock_encoder.encode.side_effect = [
            np.array([_HIGH, _HIGH, _LOW, _LOW]),
            np.array([_HIGH, _HIGH, _HIGH, _HIGH]),
        ]
        with patch.object(Precision, "_get_encoder", return_value=mock_encoder):
            results = Precision.run(vision_dataset_retriever, verbose=False)

        assert isinstance(results, list)
        assert isinstance(results[0], PrecisionMetric)


class TestConfidenceScoreAnalysis:
    def test_initialization(self, vision_dataset_retriever):
        metric = ConfidenceScoreAnalysis(vision_dataset_retriever)
        assert metric is not None

    def test_confidence_stats(self):
        batches = [
            _batch("qa_001", gt_detected=True, confidence=0.9),
            _batch("qa_002", gt_detected=False, confidence=0.1),
            _batch("qa_003", gt_detected=False, confidence=0.8),
            _batch("qa_004", gt_detected=True, confidence=0.4),
        ]
        encoder = _mock_encoder([
            np.array([_HIGH, _HIGH, _LOW, _LOW]),
            np.array([_HIGH, _HIGH, _HIGH, _HIGH]),
        ])
        metric = _run_metric(ConfidenceScoreAnalysis, batches, encoder)

        result = metric.metrics[0]
        assert isinstance(result, ConfidenceScoreMetric)
        assert result.n_with_confidence == 4
        assert result.confidence_mean == pytest.approx(0.55, abs=0.01)
        assert result.confidence_min == pytest.approx(0.1)
        assert result.confidence_max == pytest.approx(0.9)

    def test_no_confidence_scores(self):
        batches = [_batch("qa_001", gt_detected=False)]
        encoder = _mock_encoder([np.array([_HIGH]), np.array([_HIGH])])
        metric = _run_metric(ConfidenceScoreAnalysis, batches, encoder)

        result = metric.metrics[0]
        assert result.n_with_confidence == 0
        assert result.confidence_mean is None
        assert result.expected_calibration_error is None

    def test_ece_perfect_calibration(self):
        batches = [_batch(f"qa_00{i}", gt_detected=True, confidence=0.95) for i in range(1, 5)]
        encoder = _mock_encoder([
            np.array([_HIGH, _HIGH, _HIGH, _HIGH]),
            np.array([_HIGH, _HIGH, _HIGH, _HIGH]),
        ])
        metric = _run_metric(ConfidenceScoreAnalysis, batches, encoder)

        result = metric.metrics[0]
        assert result.expected_calibration_error is not None
        assert result.expected_calibration_error < 0.1

    def test_ece_poor_calibration(self):
        batches = [_batch(f"qa_00{i}", gt_detected=False, confidence=0.95) for i in range(1, 5)]
        encoder = _mock_encoder([
            np.array([_HIGH, _HIGH, _HIGH, _HIGH]),
            np.array([_LOW, _LOW, _LOW, _LOW]),
        ])
        metric = _run_metric(ConfidenceScoreAnalysis, batches, encoder)

        result = metric.metrics[0]
        assert result.expected_calibration_error is not None
        assert result.expected_calibration_error > 0.5

    def test_run_method(self, vision_dataset_retriever):
        mock_encoder = MagicMock()
        mock_encoder.encode.side_effect = [
            np.array([_HIGH, _HIGH, _LOW, _LOW]),
            np.array([_HIGH, _HIGH, _HIGH, _HIGH]),
        ]
        with patch.object(ConfidenceScoreAnalysis, "_get_encoder", return_value=mock_encoder):
            results = ConfidenceScoreAnalysis.run(vision_dataset_retriever, verbose=False)

        assert isinstance(results, list)
        assert isinstance(results[0], ConfidenceScoreMetric)

    def test_compute_stats(self, vision_dataset_retriever):
        metric = ConfidenceScoreAnalysis(vision_dataset_retriever)
        mean, std, min_v, max_v = metric._compute_stats([0.2, 0.4, 0.6, 0.8])
        assert mean == pytest.approx(0.5)
        assert min_v == pytest.approx(0.2)
        assert max_v == pytest.approx(0.8)
        assert std > 0
