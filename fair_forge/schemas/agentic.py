"""Agentic metric schemas."""

from pydantic import BaseModel, Field

from .metrics import BaseMetric


class ToolCorrectnessScore(BaseModel):
    """
    Evaluation scores for tool usage correctness.

    Evaluates four aspects: tool selection (correct tools chosen), parameter accuracy
    (correct parameters passed), sequence (correct order if required), and utilization
    (tool results used in final answer). Overall score is weighted average.
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

    pass@K: At least one of K responses is correct (similarity >= threshold).
    pass^K: All K responses are correct.
    tool_correctness: Optional evaluation of tool usage quality.
    """

    qa_id: str
    k: int
    threshold: float
    correctness_scores: list[float]
    pass_at_k: bool
    pass_pow_k: bool
    correct_indices: list[int]
    tool_correctness: ToolCorrectnessScore | None = None
