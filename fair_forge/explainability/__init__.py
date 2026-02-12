"""
Explainability module for Fair Forge.

This module provides token attribution analysis for language models using the
interpreto library. It helps understand which parts of the input contribute
most to the model's output.

Example:
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> from fair_forge.explainability import AttributionExplainer, AttributionMethod
    >>>
    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    >>>
    >>> explainer = AttributionExplainer(model, tokenizer)
    >>> result = explainer.explain(
    ...     messages=[{"role": "user", "content": "What is gravity?"}],
    ...     target="Gravity is the force of attraction between objects.",
    ...     method=AttributionMethod.LIME
    ... )
    >>> print(result.get_top_k(5))
"""

from fair_forge.explainability.attributions import (
    AttributionExplainer,
    compute_attributions,
)
from fair_forge.schemas.explainability import (
    AttributionBatchResult,
    AttributionMethod,
    AttributionResult,
    Granularity,
    TokenAttribution,
)

__all__ = [
    # Main class
    "AttributionExplainer",
    # Convenience function
    "compute_attributions",
    # Schemas
    "AttributionResult",
    "AttributionBatchResult",
    "TokenAttribution",
    # Enums
    "AttributionMethod",
    "Granularity",
]
