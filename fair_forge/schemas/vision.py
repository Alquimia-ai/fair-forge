"""Pydantic schemas for vision metrics."""

from pydantic import BaseModel

from fair_forge.schemas.metrics import BaseMetric


class VisionSimilarityInteraction(BaseModel):
    qa_id: str
    similarity_score: float


class VisionSimilarityMetric(BaseMetric):
    mean_similarity: float
    min_similarity: float
    max_similarity: float
    summary: str
    interactions: list[VisionSimilarityInteraction]


class VisionHallucinationInteraction(BaseModel):
    qa_id: str
    similarity_score: float
    is_hallucination: bool


class VisionHallucinationMetric(BaseMetric):
    hallucination_rate: float
    n_hallucinations: int
    n_frames: int
    threshold: float
    summary: str
    interactions: list[VisionHallucinationInteraction]
