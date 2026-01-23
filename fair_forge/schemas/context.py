"""Context metric schemas."""

from .metrics import BaseMetric


class ContextMetric(BaseMetric):
    """
    Context metric for evaluating context awareness.
    """

    context_insight: str
    context_awareness: float
    context_thinkings: str
    qa_id: str
