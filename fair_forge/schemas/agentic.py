"""Agentic metric schemas."""

from pydantic import BaseModel, Field

from .metrics import BaseMetric


class ToolCorrectnessScore(BaseModel):
    """
    Evaluation scores for tool usage correctness.

    Attributes:
        tool_selection_correct: Score (0.0-1.0) for selecting correct tools
        parameter_accuracy: Score (0.0-1.0) for parameter correctness
        sequence_correct: Score (0.0-1.0) for tool call order
        result_utilization: Score (0.0-1.0) for using tool results
        overall_correctness: Weighted average of all scores
        is_correct: Whether overall score meets threshold
        reasoning: Optional explanation from the judge
    """

    tool_selection_correct: float = Field(ge=0.0, le=1.0)
    parameter_accuracy: float = Field(ge=0.0, le=1.0)
    sequence_correct: float = Field(ge=0.0, le=1.0)
    result_utilization: float = Field(ge=0.0, le=1.0)
    overall_correctness: float = Field(ge=0.0, le=1.0)
    is_correct: bool
    reasoning: str | None = None


class AgenticMetric(BaseMetric):
    """
    Metric for evaluating agentic responses with pass@K and tool correctness.

    Attributes:
        qa_id: Unique identifier for the question
        k: Number of responses evaluated
        threshold: Similarity threshold for answer correctness
        correctness_scores: List of correctness scores for each response
        pass_at_k: True if at least one response is correct
        pass_pow_k: True if all K responses are correct
        correct_indices: Indices of correct responses
        tool_correctness: Optional tool usage evaluation
    """

    qa_id: str
    k: int
    threshold: float
    correctness_scores: list[float]
    pass_at_k: bool
    pass_pow_k: bool
    correct_indices: list[int]
    tool_correctness: ToolCorrectnessScore | None = None
