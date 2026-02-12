"""Attribution-based explainability for language models using interpreto."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from fair_forge.schemas.explainability import (
    AttributionBatchResult,
    AttributionMethod,
    AttributionResult,
    Granularity,
    TokenAttribution,
)
from fair_forge.utils.logging import VerboseLogger

if TYPE_CHECKING:
    from interpreto.attributions.base import AttributionExplainer as InterpretoExplainer
    from transformers import PreTrainedModel, PreTrainedTokenizer


# Mapping from our enum to interpreto classes
_METHOD_MAP: dict[AttributionMethod, str] = {
    AttributionMethod.SALIENCY: "Saliency",
    AttributionMethod.INTEGRATED_GRADIENTS: "IntegratedGradients",
    AttributionMethod.GRADIENT_SHAP: "GradientShap",
    AttributionMethod.SMOOTH_GRAD: "SmoothGrad",
    AttributionMethod.SQUARE_GRAD: "SquareGrad",
    AttributionMethod.VAR_GRAD: "VarGrad",
    AttributionMethod.INPUT_X_GRADIENT: "InputxGradient",
    AttributionMethod.LIME: "Lime",
    AttributionMethod.KERNEL_SHAP: "KernelShap",
    AttributionMethod.OCCLUSION: "Occlusion",
    AttributionMethod.SOBOL: "Sobol",
}


def _get_interpreto_class(method: AttributionMethod) -> type:
    """Dynamically import and return the interpreto explainer class."""
    import interpreto

    class_name = _METHOD_MAP[method]
    if not hasattr(interpreto, class_name):
        msg = f"Interpreto does not have method '{class_name}'. Available: {dir(interpreto)}"
        raise ImportError(msg)
    return getattr(interpreto, class_name)  # type: ignore[no-any-return]


def _get_interpreto_granularity(granularity: Granularity):
    """Convert our Granularity enum to interpreto's Granularity."""
    from interpreto import Granularity as InterpretoGranularity

    mapping = {
        Granularity.TOKEN: InterpretoGranularity.TOKEN,
        Granularity.WORD: InterpretoGranularity.WORD,
        Granularity.SENTENCE: InterpretoGranularity.SENTENCE,
    }
    return mapping[granularity]


class AttributionExplainer:
    """
    Compute token/word/sentence attributions for language model responses.

    This class wraps the interpreto library to provide explainability for
    transformer-based language models. It computes how much each input token
    contributes to the model's output.

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from fair_forge.explainability import AttributionExplainer, AttributionMethod
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        >>>
        >>> explainer = AttributionExplainer(model, tokenizer)
        >>> result = explainer.explain(
        ...     messages=[
        ...         {"role": "system", "content": "Answer concisely."},
        ...         {"role": "user", "content": "What is gravity?"}
        ...     ],
        ...     target="Gravity is the force that attracts objects toward each other.",
        ...     method=AttributionMethod.LIME
        ... )
        >>> print(result.top_attributions[:5])
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        *,
        default_method: AttributionMethod = AttributionMethod.LIME,
        default_granularity: Granularity = Granularity.WORD,
        verbose: bool = False,
    ):
        """
        Initialize the attribution explainer.

        Args:
            model: A HuggingFace PreTrainedModel (e.g., AutoModelForCausalLM)
            tokenizer: The corresponding tokenizer for the model
            default_method: Default attribution method to use
            default_granularity: Default granularity level (token, word, or sentence)
            verbose: Enable verbose logging
        """
        self.model = model
        self.tokenizer = tokenizer
        self.default_method = default_method
        self.default_granularity = default_granularity
        self.logger = VerboseLogger(verbose)

        # Cache for interpreto explainer instances
        self._explainer_cache: dict[tuple[AttributionMethod, Granularity], InterpretoExplainer] = {}

    def _get_explainer(
        self,
        method: AttributionMethod,
        granularity: Granularity,
        **kwargs: Any,
    ) -> InterpretoExplainer:
        """Get or create an interpreto explainer instance."""
        cache_key = (method, granularity)

        # Only use cache if no extra kwargs are provided
        if not kwargs and cache_key in self._explainer_cache:
            return self._explainer_cache[cache_key]

        explainer_class = _get_interpreto_class(method)
        interpreto_granularity = _get_interpreto_granularity(granularity)

        self.logger.info(f"Creating {method.value} explainer with {granularity.value} granularity")

        explainer = explainer_class(
            self.model,
            self.tokenizer,
            granularity=interpreto_granularity,
            **kwargs,
        )

        if not kwargs:
            self._explainer_cache[cache_key] = explainer

        return explainer

    def _format_prompt(
        self,
        messages: list[dict[str, str]],
        *,
        enable_thinking: bool = False,
    ) -> str:
        """
        Format messages into a prompt string using the tokenizer's chat template.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            enable_thinking: Enable thinking mode if supported by tokenizer

        Returns:
            Formatted prompt string
        """
        # Check if tokenizer supports chat template
        if not hasattr(self.tokenizer, "apply_chat_template"):
            # Fallback: simple concatenation
            self.logger.warning("Tokenizer does not support chat template, using simple concatenation")
            return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)

        # Try to apply chat template with optional parameters
        try:
            result = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                enable_thinking=enable_thinking,
                add_generation_prompt=True,
            )
        except TypeError:
            # Some tokenizers don't support enable_thinking
            result = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # apply_chat_template with tokenize=False returns str
        return str(result)

    def _parse_attributions(
        self,
        interpreto_result: Any,
        prompt: str,
        target: str,
        method: AttributionMethod,
        granularity: Granularity,
    ) -> AttributionResult:
        """Parse interpreto attribution result into our schema."""
        # interpreto returns a list, we take the first element
        attr_data = interpreto_result[0] if isinstance(interpreto_result, list) else interpreto_result

        # Extract tokens and scores from interpreto result
        # The exact structure depends on interpreto's output format
        tokens = []
        scores = []

        # Try different attribute names that interpreto might use
        if hasattr(attr_data, "tokens") and hasattr(attr_data, "attributions"):
            tokens = attr_data.tokens
            scores = attr_data.attributions
        elif hasattr(attr_data, "words") and hasattr(attr_data, "scores"):
            tokens = attr_data.words
            scores = attr_data.scores
        elif hasattr(attr_data, "input_tokens") and hasattr(attr_data, "attribution_scores"):
            tokens = attr_data.input_tokens
            scores = attr_data.attribution_scores
        elif isinstance(attr_data, dict):
            tokens = attr_data.get("tokens") or attr_data.get("words") or []
            scores = attr_data.get("attributions") or attr_data.get("scores") or []
        else:
            # Try to access as tuple or list
            try:
                if len(attr_data) >= 2:
                    tokens, scores = attr_data[0], attr_data[1]
            except (TypeError, IndexError):
                self.logger.warning(f"Could not parse interpreto result: {type(attr_data)}")
                tokens = []
                scores = []

        # Ensure scores is a list of floats
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        scores = [float(s) for s in scores]

        # Normalize scores to [0, 1] range
        if scores:
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            if score_range > 0:
                normalized = [(s - min_score) / score_range for s in scores]
            else:
                normalized = [0.5] * len(scores)
        else:
            normalized = []

        # Create TokenAttribution objects
        attributions = [
            TokenAttribution(
                text=str(token),
                score=score,
                position=i,
                normalized_score=norm_score,
            )
            for i, (token, score, norm_score) in enumerate(zip(tokens, scores, normalized, strict=False))
        ]

        return AttributionResult(
            prompt=prompt,
            target=target,
            method=method,
            granularity=granularity,
            attributions=attributions,
        )

    def explain(
        self,
        messages: list[dict[str, str]],
        target: str,
        *,
        method: AttributionMethod | None = None,
        granularity: Granularity | None = None,
        max_length: int = 512,
        enable_thinking: bool = False,
        **explainer_kwargs: Any,
    ) -> AttributionResult:
        """
        Compute attributions for a model response given input messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                      Example: [{"role": "user", "content": "Hello"}]
            target: The model's response/output to explain
            method: Attribution method to use (defaults to instance default)
            granularity: Granularity level (defaults to instance default)
            max_length: Maximum sequence length for attribution computation
            enable_thinking: Enable thinking mode in chat template (if supported)
            **explainer_kwargs: Additional kwargs passed to the interpreto explainer

        Returns:
            AttributionResult containing token attributions and metadata
        """
        method = method or self.default_method
        granularity = granularity or self.default_granularity

        self.logger.info(f"Computing {method.value} attributions at {granularity.value} level")

        # Format prompt using chat template
        prompt = self._format_prompt(messages, enable_thinking=enable_thinking)
        self.logger.debug(f"Formatted prompt: {prompt[:200]}...")

        # Get or create explainer
        explainer = self._get_explainer(method, granularity, **explainer_kwargs)

        # Compute attributions
        start_time = time.time()
        result = explainer(prompt, target, max_length=max_length)
        compute_time = time.time() - start_time

        self.logger.info(f"Attribution computed in {compute_time:.2f}s")

        # Parse and return result
        attribution_result = self._parse_attributions(result, prompt, target, method, granularity)
        attribution_result.metadata["compute_time_seconds"] = compute_time
        attribution_result.metadata["max_length"] = max_length

        return attribution_result

    def explain_batch(
        self,
        items: list[tuple[list[dict[str, str]], str]],
        *,
        method: AttributionMethod | None = None,
        granularity: Granularity | None = None,
        max_length: int = 512,
        enable_thinking: bool = False,
        **explainer_kwargs: Any,
    ) -> AttributionBatchResult:
        """
        Compute attributions for multiple message/target pairs.

        Args:
            items: List of (messages, target) tuples
            method: Attribution method to use
            granularity: Granularity level
            max_length: Maximum sequence length
            enable_thinking: Enable thinking mode in chat template
            **explainer_kwargs: Additional kwargs for the explainer

        Returns:
            AttributionBatchResult containing all results
        """
        method = method or self.default_method
        granularity = granularity or self.default_granularity

        self.logger.info(f"Processing batch of {len(items)} items")
        start_time = time.time()

        results = []
        for messages, target in items:
            result = self.explain(
                messages=messages,
                target=target,
                method=method,
                granularity=granularity,
                max_length=max_length,
                enable_thinking=enable_thinking,
                **explainer_kwargs,
            )
            results.append(result)

        total_time = time.time() - start_time

        # Get model name
        model_name = getattr(self.model.config, "_name_or_path", "unknown")

        return AttributionBatchResult(
            results=results,
            model_name=model_name,
            total_compute_time_seconds=total_time,
        )

    def visualize(
        self,
        result: AttributionResult,
        *,
        return_html: bool = False,
    ) -> str | None:
        """
        Display or return HTML visualization of attributions.

        Args:
            result: AttributionResult to visualize
            return_html: If True, return HTML string instead of displaying

        Returns:
            HTML string if return_html=True, otherwise None (displays in notebook)
        """
        from interpreto import AttributionVisualization

        # Create a mock attribution object that interpreto can visualize
        class _MockAttribution:
            def __init__(self, tokens: list[str], scores: list[float]):
                self.tokens = tokens
                self.attributions = scores

        mock_attr = _MockAttribution(
            tokens=[attr.text for attr in result.attributions],
            scores=[attr.score for attr in result.attributions],
        )

        viz = AttributionVisualization(mock_attr)

        if return_html:
            html_result: str = viz.to_html()
            return html_result

        viz.display()
        return None


def compute_attributions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    target: str,
    *,
    method: AttributionMethod = AttributionMethod.LIME,
    granularity: Granularity = Granularity.WORD,
    **kwargs: Any,
) -> AttributionResult:
    """
    Convenience function to compute attributions in one call.

    This is a simpler interface for one-off attribution computations.
    For repeated use, instantiate AttributionExplainer directly.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        messages: Chat messages
        target: Model's response to explain
        method: Attribution method
        granularity: Granularity level
        **kwargs: Additional kwargs for explain()

    Returns:
        AttributionResult
    """
    explainer = AttributionExplainer(
        model=model,
        tokenizer=tokenizer,
        default_method=method,
        default_granularity=granularity,
    )
    return explainer.explain(messages=messages, target=target, **kwargs)
