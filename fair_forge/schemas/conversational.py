"""Conversational metric schemas."""

from .metrics import BaseMetric


class ConversationalMetric(BaseMetric):
    """
    Conversational metric for evaluating the assistant's conversational abilities.
    """

    conversational_memory: float
    conversational_insight: str
    conversational_language: float
    conversational_quality_maxim: float
    conversational_quantity_maxim: float
    conversational_relation_maxim: float
    conversational_manner_maxim: float
    conversational_sensibleness: float
    conversational_thinkings: str
    qa_id: str
