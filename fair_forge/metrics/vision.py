"""Vision hallucination metrics: False Positive Rate, Precision, Confidence Score Analysis."""

import math
from abc import abstractmethod
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from tqdm.auto import tqdm

from fair_forge.core import FairForge, Retriever
from fair_forge.llm import Judge, VisionJudgeOutput
from fair_forge.llm.prompts import vision_judge_system_prompt
from fair_forge.schemas import Batch
from fair_forge.schemas.vision import (
    ConfidenceBucket,
    ConfidenceScoreMetric,
    FalsePositiveRateMetric,
    PrecisionMetric,
    VisionInteraction,
)

if TYPE_CHECKING:
    pass

_N_CONFIDENCE_BUCKETS = 10


class _VisionBase(FairForge):
    """Shared base for vision hallucination metrics.

    Handles the LLM judge call per interaction and accumulates VisionInteraction
    objects per session. Subclasses only implement on_process_complete() to
    derive their specific metric from the accumulated classifications.
    """

    def __init__(
        self,
        retriever: type[Retriever],
        model: BaseChatModel,
        use_structured_output: bool = False,
        strict: bool = True,
        bos_json_clause: str = "```json",
        eos_json_clause: str = "```",
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)
        self._model = model
        self._use_structured_output = use_structured_output
        self._strict = strict
        self._bos_json_clause = bos_json_clause
        self._eos_json_clause = eos_json_clause
        self._session_data: dict[str, dict] = {}

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

        judge = Judge(
            model=self._model,
            use_structured_output=self._use_structured_output,
            strict=self._strict,
            bos_json_clause=self._bos_json_clause,
            eos_json_clause=self._eos_json_clause,
        )

        for interaction in tqdm(batch, desc=session_id, unit="frame"):
            self.logger.debug(f"QA ID: {interaction.qa_id}")
            _, result = judge.check(
                system_prompt=vision_judge_system_prompt,
                query="Classify the VLM prediction against the ground truth and provide your JSON response.",
                data={
                    "context": context,
                    "assistant": interaction.assistant,
                    "ground_truth_assistant": interaction.ground_truth_assistant,
                },
                output_schema=VisionJudgeOutput,
            )

            if result is None:
                raise ValueError(f"No valid response from judge for QA ID: {interaction.qa_id}")

            classification = result["classification"] if isinstance(result, dict) else result.classification
            reasoning = result["reasoning"] if isinstance(result, dict) else result.reasoning
            confidence = (interaction.agentic or {}).get("confidence")

            self._session_data[session_id]["interactions"].append(
                VisionInteraction(
                    qa_id=interaction.qa_id,
                    classification=classification,
                    confidence=confidence,
                    reasoning=reasoning,
                )
            )

    @abstractmethod
    def on_process_complete(self):
        raise NotImplementedError


class FalsePositiveRate(_VisionBase):
    """Measures the rate at which the VLM invents events that did not occur.

    FPR = False Positives / (False Positives + True Negatives)

    A high FPR indicates the model frequently generates hallucinated detections,
    which is the critical failure mode for vision-based alerting systems like Argos.
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
    """Measures the accuracy of the VLM's positive detections.

    Precision = True Positives / (True Positives + False Positives)

    A low precision means the model raises many false alarms relative to
    correct detections — directly impacting operational trust in Argos alerts.
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

    Reads confidence from interaction.agentic["confidence"] and combines it
    with judge-determined correctness to compute:
    - Descriptive statistics (mean, std, min, max)
    - Expected Calibration Error (ECE): how well confidence predicts correctness

    A high ECE means the model is overconfident or underconfident relative to
    its actual accuracy — useful for setting reliable alert thresholds in Argos.
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
            is_correct = vi.classification in ("true_positive", "true_negative")
            buckets_data[idx]["correct"].append(1 if is_correct else 0)

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
