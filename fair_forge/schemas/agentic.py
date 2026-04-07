"""Agentic metric schemas."""

from pydantic import BaseModel, Field


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


class AgenticConversation(BaseModel):
    """
    Per-conversation evaluation detail.

    Stores the raw correctness data for a single evaluated conversation —
    interaction-level scores, which indices passed, and optional tool scores.
    Does not contain pass@K/pass^K (those are global metrics computed across
    all conversations in AgenticMetric).
    """

    session_id: str
    assistant_id: str
    total_interactions: int
    correct_interactions: int
    is_fully_correct: bool
    threshold: float
    correctness_scores: list[float]
    correct_indices: list[int]
    tool_correctness_scores: list[ToolCorrectnessScore | None] = []


class AgenticMetric(BaseModel):
    """
    Global agentic evaluation result across all evaluated conversations.

    pass@K and pass^K are computed using the global success rate p = c/n,
    where n is the total number of conversations evaluated and c is the number
    of fully correct conversations (all interactions passed the threshold).

    pass@K = 1 - (1 - p)^k  — probability ≥1 of k attempts is fully correct.
    pass^K = p^k             — probability all k attempts are fully correct.
    """

    k: int
    n: int
    c: int
    p: float
    pass_at_k: float
    pass_at_k_ci_low: float | None = None
    pass_at_k_ci_high: float | None = None
    pass_pow_k: float
    pass_pow_k_ci_low: float | None = None
    pass_pow_k_ci_high: float | None = None
    conversations: list[AgenticConversation]
