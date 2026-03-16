"""Vision hallucination metrics: False Positive Rate, Precision, Confidence Score Analysis."""

import math
from abc import abstractmethod

import numpy as np
from tqdm.auto import tqdm

from fair_forge.core import FairForge, Retriever
from fair_forge.schemas import Batch
from fair_forge.schemas.vision import (
    ConfidenceBucket,
    ConfidenceScoreMetric,
    FalsePositiveRateMetric,
    PrecisionMetric,
    VisionInteraction,
)

_DEFAULT_MODEL = "all-mpnet-base-v2"
_DEFAULT_THRESHOLD = 0.75
_N_CONFIDENCE_BUCKETS = 10


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _derive_classification(similarity: float, threshold: float, gt_detected: bool) -> str:
    correct = similarity >= threshold
    if correct and gt_detected:
        return "true_positive"
    if correct and not gt_detected:
        return "true_negative"
    if not correct and gt_detected:
        return "false_negative"
    return "false_positive"


class _VisionBase(FairForge):
    """Shared base for vision hallucination metrics.

    Compares the VLM's free-text description (assistant) against the human ground
    truth (ground_truth_assistant) using cosine similarity between sentence embeddings.
    Classification into TP/FP/TN/FN uses the similarity score against a configurable
    threshold combined with the actual event label from ground_truth_agentic["detected"].

    Expected Batch fields:
        assistant                        — VLM free-text description of the scene
        ground_truth_assistant           — human description of what actually happened
        ground_truth_agentic["detected"] (bool, required) — whether an event actually occurred
        agentic["confidence"]            (float, optional) — model confidence score
    """

    def __init__(
        self,
        retriever: type[Retriever],
        model_name: str = _DEFAULT_MODEL,
        threshold: float = _DEFAULT_THRESHOLD,
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)
        self._model_name = model_name
        self._threshold = threshold
        self._encoder = None
        self._session_data: dict[str, dict] = {}

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self._model_name)
        return self._encoder

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: str | None = "english",
    ):
        if session_id not in self._session_data:
            self._session_data[session_id] = {"assistant_id": assistant_id, "interactions": []}

        encoder = self._get_encoder()
        assistants = [i.assistant for i in batch]
        ground_truths = [i.ground_truth_assistant for i in batch]
        emb_a = encoder.encode(assistants, show_progress_bar=False)
        emb_b = encoder.encode(ground_truths, show_progress_bar=False)

        for interaction, a, b in tqdm(zip(batch, emb_a, emb_b), desc=session_id, unit="event", total=len(batch)):
            self.logger.debug(f"QA ID: {interaction.qa_id}")

            gt_agentic = interaction.ground_truth_agentic or {}
            if "detected" not in gt_agentic:
                raise ValueError(f"Missing 'detected' in ground_truth_agentic for qa_id: {interaction.qa_id}")

            similarity = _cosine_similarity(a, b)
            classification = _derive_classification(similarity, self._threshold, gt_agentic["detected"])

            self._session_data[session_id]["interactions"].append(
                VisionInteraction(
                    qa_id=interaction.qa_id,
                    classification=classification,
                    similarity_score=round(similarity, 4),
                    confidence=(interaction.agentic or {}).get("confidence"),
                )
            )

    @abstractmethod
    def on_process_complete(self):
        raise NotImplementedError


class FalsePositiveRate(_VisionBase):
    """Measures the rate at which the VLM invents events that did not occur.

    FPR = False Positives / (False Positives + True Negatives)

    A high FPR means the model frequently describes events in scenes where nothing
    happened — the critical failure mode for vision-based alerting systems like Argos.
    """

    def on_process_complete(self):
        for session_id, data in self._session_data.items():
            interactions = data["interactions"]
            n_negatives = sum(
                1 for vi in interactions if vi.classification in ("true_negative", "false_positive")
            )
            n_false_positives = sum(1 for vi in interactions if vi.classification == "false_positive")
            fpr = n_false_positives / n_negatives if n_negatives > 0 else None
            self.metrics.append(
                FalsePositiveRateMetric(
                    session_id=session_id,
                    assistant_id=data["assistant_id"],
                    n_predictions=len(interactions),
                    n_negatives=n_negatives,
                    n_false_positives=n_false_positives,
                    false_positive_rate=fpr,
                    interactions=interactions,
                )
            )


class Precision(_VisionBase):
    """Measures the accuracy of the VLM's event descriptions.

    Precision = True Positives / (True Positives + False Positives)

    A low precision means the VLM frequently describes events that did not occur,
    causing the downstream LLM to raise false alarms in Argos.
    """

    def on_process_complete(self):
        for session_id, data in self._session_data.items():
            interactions = data["interactions"]
            n_positive_predictions = sum(
                1 for vi in interactions if vi.classification in ("true_positive", "false_positive")
            )
            n_true_positives = sum(1 for vi in interactions if vi.classification == "true_positive")
            precision = n_true_positives / n_positive_predictions if n_positive_predictions > 0 else None
            self.metrics.append(
                PrecisionMetric(
                    session_id=session_id,
                    assistant_id=data["assistant_id"],
                    n_predictions=len(interactions),
                    n_positive_predictions=n_positive_predictions,
                    n_true_positives=n_true_positives,
                    precision=precision,
                    interactions=interactions,
                )
            )


class ConfidenceScoreAnalysis(_VisionBase):
    """Analyzes the distribution and calibration of VLM confidence scores.

    Reads confidence from agentic["confidence"] and combines it with the derived
    correctness label to compute:
    - Descriptive statistics (mean, std, min, max)
    - Expected Calibration Error (ECE): how well confidence predicts correctness

    A high ECE means the model is overconfident or underconfident relative to its
    actual accuracy — useful for setting reliable alert thresholds in Argos.
    """

    def _compute_stats(self, values: list[float]) -> tuple[float, float, float, float]:
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        return mean, math.sqrt(variance), min(values), max(values)

    def _compute_ece(
        self, interactions: list[VisionInteraction]
    ) -> tuple[float | None, list[ConfidenceBucket]]:
        confident = [vi for vi in interactions if vi.confidence is not None]
        if not confident:
            return None, []

        bucket_width = 1.0 / _N_CONFIDENCE_BUCKETS
        buckets_data: list[dict] = [{"confidences": [], "correct": []} for _ in range(_N_CONFIDENCE_BUCKETS)]

        for vi in confident:
            idx = min(int(vi.confidence / bucket_width), _N_CONFIDENCE_BUCKETS - 1)
            buckets_data[idx]["confidences"].append(vi.confidence)
            buckets_data[idx]["correct"].append(1 if vi.classification in ("true_positive", "true_negative") else 0)

        total = len(confident)
        ece = 0.0
        buckets: list[ConfidenceBucket] = []

        for i, bd in enumerate(buckets_data):
            count = len(bd["confidences"])
            range_min = round(i * bucket_width, 2)
            range_max = round((i + 1) * bucket_width, 2)
            if count == 0:
                buckets.append(ConfidenceBucket(range_min=range_min, range_max=range_max, count=0))
                continue
            mean_conf = sum(bd["confidences"]) / count
            accuracy = sum(bd["correct"]) / count
            ece += (count / total) * abs(mean_conf - accuracy)
            buckets.append(
                ConfidenceBucket(
                    range_min=range_min,
                    range_max=range_max,
                    count=count,
                    mean_confidence=round(mean_conf, 4),
                    accuracy=round(accuracy, 4),
                )
            )

        return round(ece, 4), buckets

    def on_process_complete(self):
        for session_id, data in self._session_data.items():
            interactions = data["interactions"]
            confident = [vi for vi in interactions if vi.confidence is not None]

            mean = std = min_c = max_c = None
            if confident:
                mean, std, min_c, max_c = self._compute_stats([vi.confidence for vi in confident])
                mean, std, min_c, max_c = round(mean, 4), round(std, 4), round(min_c, 4), round(max_c, 4)

            ece, buckets = self._compute_ece(interactions)

            self.metrics.append(
                ConfidenceScoreMetric(
                    session_id=session_id,
                    assistant_id=data["assistant_id"],
                    n_predictions=len(interactions),
                    n_with_confidence=len(confident),
                    confidence_mean=mean,
                    confidence_std=std,
                    confidence_min=min_c,
                    confidence_max=max_c,
                    expected_calibration_error=ece,
                    buckets=buckets,
                    interactions=interactions,
                )
            )
