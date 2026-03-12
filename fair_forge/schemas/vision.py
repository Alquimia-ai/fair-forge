"""Pydantic schemas for vision hallucination metrics."""

from typing import Literal

from pydantic import BaseModel

from fair_forge.schemas.metrics import BaseMetric


class VisionInteraction(BaseModel):
    qa_id: str
    classification: Literal["true_positive", "false_positive", "true_negative", "false_negative"]
    confidence: float | None = None
    reasoning: str


class FalsePositiveRateMetric(BaseMetric):
    n_predictions: int
    n_negatives: int
    n_false_positives: int
    false_positive_rate: float | None
    interactions: list[VisionInteraction]


class PrecisionMetric(BaseMetric):
    n_predictions: int
    n_positive_predictions: int
    n_true_positives: int
    precision: float | None
    interactions: list[VisionInteraction]


class ConfidenceBucket(BaseModel):
    range_min: float
    range_max: float
    count: int
    mean_confidence: float | None = None
    accuracy: float | None = None


class ConfidenceScoreMetric(BaseMetric):
    n_predictions: int
    n_with_confidence: int
    confidence_mean: float | None = None
    confidence_std: float | None = None
    confidence_min: float | None = None
    confidence_max: float | None = None
    expected_calibration_error: float | None = None
    buckets: list[ConfidenceBucket]
    interactions: list[VisionInteraction]
