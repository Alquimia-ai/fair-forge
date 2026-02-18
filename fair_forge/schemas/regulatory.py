"""Regulatory metric schemas."""

from .metrics import BaseMetric


class RegulatoryMetric(BaseMetric):
    """
    Regulatory compliance metric for evaluating adherence to regulations.

    Evaluates whether an AI assistant's responses comply with a given set
    of regulations, policies, or guidelines.

    Attributes:
        qa_id: Unique identifier for the Q&A interaction
        compliance_score: Overall compliance score (0.0-1.0)
        compliance_insight: Explanation of the compliance evaluation
        compliance_thinkings: Detailed reasoning from the judge
        violated_rules: List of specific rules that were violated (if any)
        rule_assessments: Individual assessment for each rule
    """

    qa_id: str
    compliance_score: float
    compliance_insight: str
    compliance_thinkings: str
    violated_rules: list[str]
    rule_assessments: dict[str, dict]
