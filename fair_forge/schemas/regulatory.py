"""Regulatory compliance metric schemas."""

from typing import Literal

from pydantic import BaseModel, Field

from .metrics import BaseMetric


class RegulatoryChunk(BaseModel):
    """
    A chunk of regulatory text with retrieval and reranking scores.

    Attributes:
        text: The chunk text content.
        source: Source document filename.
        chunk_index: Index of this chunk within the source document.
        similarity: Cosine similarity score from embedding retrieval.
        reranker_score: Score from reranker model (higher = supports, lower = contradicts).
        verdict: Whether the chunk SUPPORTS or CONTRADICTS the agent response.
    """

    text: str
    source: str
    chunk_index: int
    similarity: float = Field(ge=0, le=1)
    reranker_score: float = Field(ge=0, le=1)
    verdict: Literal["SUPPORTS", "CONTRADICTS"]


class RegulatoryMetric(BaseMetric):
    """
    Regulatory compliance metric for evaluating assistant responses against regulatory corpus.

    Attributes:
        qa_id: Unique identifier for the Q&A interaction.
        query: The user query.
        assistant: The assistant response.
        compliance_score: Overall compliance score (0.0-1.0).
            1.0 = fully supported, 0.0 = fully contradicted.
        verdict: Overall verdict (COMPLIANT, NON_COMPLIANT, or IRRELEVANT).
        supporting_chunks: Number of chunks that support the response.
        contradicting_chunks: Number of chunks that contradict the response.
        retrieved_chunks: List of retrieved and evaluated chunks.
        insight: Explanation of the compliance evaluation.
    """

    qa_id: str
    query: str
    assistant: str
    compliance_score: float = Field(ge=0, le=1)
    verdict: Literal["COMPLIANT", "NON_COMPLIANT", "IRRELEVANT"]
    supporting_chunks: int = Field(ge=0)
    contradicting_chunks: int = Field(ge=0)
    retrieved_chunks: list[RegulatoryChunk]
    insight: str


__all__ = ["RegulatoryChunk", "RegulatoryMetric"]
