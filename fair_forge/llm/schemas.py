"""Pydantic schemas for LLM structured outputs."""

from pydantic import BaseModel, Field


class ContextJudgeOutput(BaseModel):
    """Structured output for context evaluation."""

    score: float = Field(ge=0, le=1, description="Context alignment score (0-1)")
    insight: str = Field(description="Insight about context compliance")


class ConversationalJudgeOutput(BaseModel):
    """Structured output for conversational evaluation."""

    memory: float = Field(ge=0, le=10, description="Memory recall score (0-10)")
    language: float = Field(ge=0, le=10, description="Language appropriateness score (0-10)")
    insight: str = Field(description="Overall insight about conversation quality")
    quality_maxim: float = Field(ge=0, le=10, description="Grice quality maxim score (0-10)")
    quantity_maxim: float = Field(ge=0, le=10, description="Grice quantity maxim score (0-10)")
    relation_maxim: float = Field(ge=0, le=10, description="Grice relation maxim score (0-10)")
    manner_maxim: float = Field(ge=0, le=10, description="Grice manner maxim score (0-10)")
    sensibleness: float = Field(ge=0, le=10, description="Sensibleness score (0-10)")


class BestOfJudgeOutput(BaseModel):
    """Structured output for best-of evaluation."""

    winner: str = Field(description="Winner identifier or 'tie'")
    verdict: str = Field(description="Explanation of why this contestant won")
    confidence: float = Field(ge=0, le=1, description="Confidence in the decision (0-1)")
    reasoning: dict = Field(description="Strengths and weaknesses for each contestant")


class RegulatoryJudgeOutput(BaseModel):
    """Structured output for regulatory compliance evaluation."""

    compliance_score: float = Field(
        ge=0, le=1, description="Overall compliance score (0-1), 1 means fully compliant"
    )
    insight: str = Field(description="Summary insight about regulatory compliance")
    violated_rules: list[str] = Field(
        description="List of rule identifiers that were violated (empty if compliant)"
    )
    rule_assessments: dict = Field(
        description="Assessment for each rule: {rule_id: {compliant: bool, reason: str}}"
    )
