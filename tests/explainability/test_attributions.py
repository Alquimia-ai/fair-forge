"""Tests for Attribution Explainer."""

from unittest.mock import MagicMock, patch

import pytest

from fair_forge.schemas.explainability import (
    AttributionBatchResult,
    AttributionMethod,
    AttributionResult,
    Granularity,
    TokenAttribution,
)


class TestAttributionSchemas:
    """Test suite for attribution schemas."""

    def test_token_attribution_creation(self):
        """Test TokenAttribution creation."""
        attr = TokenAttribution(
            text="hello",
            score=0.8,
            position=0,
            normalized_score=0.9,
        )
        assert attr.text == "hello"
        assert attr.score == 0.8
        assert attr.position == 0
        assert attr.normalized_score == 0.9

    def test_token_attribution_without_normalized_score(self):
        """Test TokenAttribution without normalized_score."""
        attr = TokenAttribution(
            text="world",
            score=-0.5,
            position=1,
        )
        assert attr.text == "world"
        assert attr.score == -0.5
        assert attr.normalized_score is None

    def test_attribution_result_creation(self):
        """Test AttributionResult creation."""
        attributions = [
            TokenAttribution(text="hello", score=0.8, position=0),
            TokenAttribution(text="world", score=0.2, position=1),
        ]
        result = AttributionResult(
            prompt="Hello world",
            target="Greeting response",
            method=AttributionMethod.LIME,
            granularity=Granularity.WORD,
            attributions=attributions,
        )
        assert result.prompt == "Hello world"
        assert result.target == "Greeting response"
        assert result.method == AttributionMethod.LIME
        assert result.granularity == Granularity.WORD
        assert len(result.attributions) == 2

    def test_attribution_result_top_attributions(self):
        """Test top_attributions property."""
        attributions = [
            TokenAttribution(text="low", score=0.1, position=0),
            TokenAttribution(text="high", score=0.9, position=1),
            TokenAttribution(text="medium", score=0.5, position=2),
        ]
        result = AttributionResult(
            prompt="test",
            target="test",
            method=AttributionMethod.SALIENCY,
            granularity=Granularity.TOKEN,
            attributions=attributions,
        )
        top = result.top_attributions
        assert top[0].text == "high"
        assert top[1].text == "medium"
        assert top[2].text == "low"

    def test_attribution_result_get_top_k(self):
        """Test get_top_k method."""
        attributions = [TokenAttribution(text=f"word_{i}", score=i * 0.1, position=i) for i in range(10)]
        result = AttributionResult(
            prompt="test",
            target="test",
            method=AttributionMethod.LIME,
            granularity=Granularity.WORD,
            attributions=attributions,
        )
        top_3 = result.get_top_k(3)
        assert len(top_3) == 3
        assert top_3[0].text == "word_9"

    def test_attribution_result_to_dict_for_visualization(self):
        """Test to_dict_for_visualization method."""
        attributions = [
            TokenAttribution(text="a", score=0.5, position=0, normalized_score=0.7),
            TokenAttribution(text="b", score=0.3, position=1, normalized_score=0.4),
        ]
        result = AttributionResult(
            prompt="test",
            target="test",
            method=AttributionMethod.LIME,
            granularity=Granularity.WORD,
            attributions=attributions,
        )
        viz_dict = result.to_dict_for_visualization()
        assert viz_dict["tokens"] == ["a", "b"]
        assert viz_dict["scores"] == [0.5, 0.3]
        assert viz_dict["normalized_scores"] == [0.7, 0.4]

    def test_attribution_batch_result_iteration(self):
        """Test AttributionBatchResult iteration and indexing."""
        results = [
            AttributionResult(
                prompt=f"prompt_{i}",
                target=f"target_{i}",
                method=AttributionMethod.LIME,
                granularity=Granularity.WORD,
                attributions=[],
            )
            for i in range(3)
        ]
        batch = AttributionBatchResult(
            results=results,
            model_name="test-model",
            total_compute_time_seconds=1.5,
        )
        assert len(batch) == 3
        assert batch[0].prompt == "prompt_0"
        assert batch[1].prompt == "prompt_1"
        assert list(batch)[2].prompt == "prompt_2"


class TestAttributionMethod:
    """Test suite for AttributionMethod enum."""

    def test_gradient_methods(self):
        """Test gradient-based methods are available."""
        gradient_methods = [
            AttributionMethod.SALIENCY,
            AttributionMethod.INTEGRATED_GRADIENTS,
            AttributionMethod.GRADIENT_SHAP,
            AttributionMethod.SMOOTH_GRAD,
            AttributionMethod.SQUARE_GRAD,
            AttributionMethod.VAR_GRAD,
            AttributionMethod.INPUT_X_GRADIENT,
        ]
        for method in gradient_methods:
            assert isinstance(method.value, str)

    def test_perturbation_methods(self):
        """Test perturbation-based methods are available."""
        perturbation_methods = [
            AttributionMethod.LIME,
            AttributionMethod.KERNEL_SHAP,
            AttributionMethod.OCCLUSION,
            AttributionMethod.SOBOL,
        ]
        for method in perturbation_methods:
            assert isinstance(method.value, str)


class TestGranularity:
    """Test suite for Granularity enum."""

    def test_granularity_values(self):
        """Test granularity enum values."""
        assert Granularity.TOKEN.value == "token"
        assert Granularity.WORD.value == "word"
        assert Granularity.SENTENCE.value == "sentence"


class TestAttributionExplainer:
    """Test suite for AttributionExplainer class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock HuggingFace model."""
        model = MagicMock()
        model.config._name_or_path = "test-model"
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted prompt"
        return tokenizer

    def test_explainer_initialization(self, mock_model, mock_tokenizer):
        """Test AttributionExplainer initialization."""
        from fair_forge.explainability import AttributionExplainer

        explainer = AttributionExplainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            default_method=AttributionMethod.LIME,
            default_granularity=Granularity.WORD,
        )
        assert explainer.model == mock_model
        assert explainer.tokenizer == mock_tokenizer
        assert explainer.default_method == AttributionMethod.LIME
        assert explainer.default_granularity == Granularity.WORD

    def test_explainer_initialization_defaults(self, mock_model, mock_tokenizer):
        """Test AttributionExplainer default parameters."""
        from fair_forge.explainability import AttributionExplainer

        explainer = AttributionExplainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        assert explainer.default_method == AttributionMethod.LIME
        assert explainer.default_granularity == Granularity.WORD

    @patch("fair_forge.explainability.attributions._get_interpreto_class")
    @patch("fair_forge.explainability.attributions._get_interpreto_granularity")
    def test_explain_method(
        self,
        mock_granularity_fn,
        mock_class_fn,
        mock_model,
        mock_tokenizer,
    ):
        """Test explain method creates correct result."""
        from fair_forge.explainability import AttributionExplainer

        # Mock interpreto explainer
        mock_interpreto_explainer = MagicMock()
        mock_interpreto_result = MagicMock()
        mock_interpreto_result.tokens = ["hello", "world"]
        mock_interpreto_result.attributions = [0.8, 0.2]
        mock_interpreto_explainer.return_value = [mock_interpreto_result]
        mock_class_fn.return_value = MagicMock(return_value=mock_interpreto_explainer)

        mock_granularity_fn.return_value = "WORD"

        explainer = AttributionExplainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        result = explainer.explain(
            messages=[{"role": "user", "content": "Hello"}],
            target="World",
        )

        assert isinstance(result, AttributionResult)
        assert result.method == AttributionMethod.LIME
        assert result.granularity == Granularity.WORD

    def test_format_prompt_with_chat_template(self, mock_model, mock_tokenizer):
        """Test _format_prompt uses chat template."""
        from fair_forge.explainability import AttributionExplainer

        mock_tokenizer.apply_chat_template.return_value = "formatted: Hello"

        explainer = AttributionExplainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        messages = [{"role": "user", "content": "Hello"}]
        result = explainer._format_prompt(messages)

        assert result == "formatted: Hello"
        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_format_prompt_fallback(self, mock_model, mock_tokenizer):
        """Test _format_prompt fallback when chat template not available."""
        from fair_forge.explainability import AttributionExplainer

        # Remove chat template method
        del mock_tokenizer.apply_chat_template

        explainer = AttributionExplainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = explainer._format_prompt(messages)

        assert "system: Be helpful" in result
        assert "user: Hello" in result

    @patch("fair_forge.explainability.attributions._get_interpreto_class")
    @patch("fair_forge.explainability.attributions._get_interpreto_granularity")
    def test_explain_batch(
        self,
        mock_granularity_fn,
        mock_class_fn,
        mock_model,
        mock_tokenizer,
    ):
        """Test explain_batch processes multiple items."""
        from fair_forge.explainability import AttributionExplainer

        # Mock interpreto
        mock_interpreto_explainer = MagicMock()
        mock_result = MagicMock()
        mock_result.tokens = ["a", "b"]
        mock_result.attributions = [0.5, 0.5]
        mock_interpreto_explainer.return_value = [mock_result]
        mock_class_fn.return_value = MagicMock(return_value=mock_interpreto_explainer)
        mock_granularity_fn.return_value = "WORD"

        explainer = AttributionExplainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        items = [
            ([{"role": "user", "content": "Q1"}], "A1"),
            ([{"role": "user", "content": "Q2"}], "A2"),
        ]

        batch_result = explainer.explain_batch(items)

        assert isinstance(batch_result, AttributionBatchResult)
        assert len(batch_result) == 2
        assert batch_result.model_name == "test-model"

    def test_parse_attributions_dict_format(self, mock_model, mock_tokenizer):
        """Test _parse_attributions handles dict format."""
        from fair_forge.explainability import AttributionExplainer

        explainer = AttributionExplainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        mock_result = [{"tokens": ["a", "b", "c"], "attributions": [0.1, 0.5, 0.4]}]

        result = explainer._parse_attributions(
            mock_result,
            prompt="test prompt",
            target="test target",
            method=AttributionMethod.LIME,
            granularity=Granularity.WORD,
        )

        assert len(result.attributions) == 3
        assert result.attributions[0].text == "a"
        assert result.attributions[1].score == 0.5


class TestConvenienceFunction:
    """Test suite for compute_attributions convenience function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.config._name_or_path = "test-model"
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted"
        return tokenizer

    @patch("fair_forge.explainability.attributions._get_interpreto_class")
    @patch("fair_forge.explainability.attributions._get_interpreto_granularity")
    def test_compute_attributions(
        self,
        mock_granularity_fn,
        mock_class_fn,
        mock_model,
        mock_tokenizer,
    ):
        """Test compute_attributions convenience function."""
        from fair_forge.explainability import compute_attributions

        # Mock interpreto
        mock_explainer = MagicMock()
        mock_result = MagicMock()
        mock_result.tokens = ["test"]
        mock_result.attributions = [0.5]
        mock_explainer.return_value = [mock_result]
        mock_class_fn.return_value = MagicMock(return_value=mock_explainer)
        mock_granularity_fn.return_value = "WORD"

        result = compute_attributions(
            model=mock_model,
            tokenizer=mock_tokenizer,
            messages=[{"role": "user", "content": "Hello"}],
            target="World",
            method=AttributionMethod.LIME,
        )

        assert isinstance(result, AttributionResult)


class TestMethodMapping:
    """Test attribution method to interpreto class mapping."""

    def test_get_interpreto_class_lime(self):
        """Test getting LIME class from interpreto."""
        import sys

        # Create a mock interpreto module
        mock_interpreto = MagicMock()
        mock_interpreto.Lime = MagicMock()
        sys.modules["interpreto"] = mock_interpreto

        try:
            from fair_forge.explainability.attributions import _get_interpreto_class

            result = _get_interpreto_class(AttributionMethod.LIME)
            assert result == mock_interpreto.Lime
        finally:
            del sys.modules["interpreto"]

    def test_get_interpreto_class_not_found(self):
        """Test error when method not found in interpreto."""
        import sys

        # Create a mock interpreto module without Lime
        mock_interpreto = MagicMock(spec=[])  # Empty spec means no attributes
        sys.modules["interpreto"] = mock_interpreto

        try:
            from fair_forge.explainability.attributions import _get_interpreto_class

            with pytest.raises(ImportError, match="Interpreto does not have method"):
                _get_interpreto_class(AttributionMethod.LIME)
        finally:
            del sys.modules["interpreto"]
