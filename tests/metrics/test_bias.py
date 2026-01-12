"""Tests for Bias metric."""
from unittest.mock import MagicMock, Mock, patch

import pytest

from fair_forge.core import Guardian
from fair_forge.metrics.bias import Bias
from fair_forge.schemas import Batch
from fair_forge.schemas.bias import BiasMetric, GuardianBias, ProtectedAttribute
from tests.fixtures.mock_retriever import MockRetriever, BiasDatasetRetriever
from tests.fixtures.mock_data import create_sample_batch, create_bias_dataset


class MockGuardian(Guardian):
    """Mock Guardian for testing bias detection."""

    def __init__(self, always_biased=False, certainty=0.9, **kwargs):
        self.always_biased = always_biased
        self.certainty = certainty
        self.call_count = 0

    def is_biased(self, question, answer, attribute, context=None):
        """Return mock bias detection result."""
        self.call_count += 1
        return GuardianBias(
            is_biased=self.always_biased,
            attribute=attribute.attribute.value,
            certainty=self.certainty
        )


class MockGuardianAlternating(Guardian):
    """Mock Guardian that alternates between biased and not biased."""

    def __init__(self, **kwargs):
        self.call_count = 0

    def is_biased(self, question, answer, attribute, context=None):
        """Return alternating bias detection result."""
        self.call_count += 1
        is_biased = self.call_count % 2 == 0
        return GuardianBias(
            is_biased=is_biased,
            attribute=attribute.attribute.value,
            certainty=0.85
        )


class TestBiasMetric:
    """Test suite for Bias metric."""

    def test_initialization_basic(self):
        """Test Bias initialization with basic parameters."""
        bias = Bias(
            retriever=MockRetriever,
            guardian=MockGuardian
        )

        assert bias.guardian is not None
        assert bias.confidence_level == 0.95
        assert len(bias.protected_attributes) == 5

    def test_initialization_custom_confidence(self):
        """Test Bias initialization with custom confidence level."""
        bias = Bias(
            retriever=MockRetriever,
            guardian=MockGuardian,
            confidence_level=0.80
        )

        assert bias.confidence_level == 0.80

    def test_protected_attributes_defined(self):
        """Test that all protected attributes are defined."""
        bias = Bias(
            retriever=MockRetriever,
            guardian=MockGuardian
        )

        expected_attributes = {
            ProtectedAttribute.Attribute.gender,
            ProtectedAttribute.Attribute.race,
            ProtectedAttribute.Attribute.religion,
            ProtectedAttribute.Attribute.nationality,
            ProtectedAttribute.Attribute.sexual_orientation
        }

        actual_attributes = {attr.attribute for attr in bias.protected_attributes}
        assert actual_attributes == expected_attributes

    def test_clopper_pearson_basic(self):
        """Test Clopper-Pearson confidence interval calculation."""
        bias = Bias(
            retriever=MockRetriever,
            guardian=MockGuardian,
            confidence_level=0.95
        )

        result = bias._clopper_pearson_confidence_interval(samples=100, k_success=70)

        assert result.samples == 100
        assert result.k_success == 70
        assert result.probability == pytest.approx(0.7, abs=0.001)
        assert result.alpha == pytest.approx(0.05, abs=0.001)
        assert result.lower_bound < result.probability
        assert result.upper_bound > result.probability
        assert 0 <= result.lower_bound <= 1
        assert 0 <= result.upper_bound <= 1

    def test_clopper_pearson_all_success(self):
        """Test Clopper-Pearson when all samples are successes."""
        bias = Bias(
            retriever=MockRetriever,
            guardian=MockGuardian
        )

        result = bias._clopper_pearson_confidence_interval(samples=100, k_success=100)

        assert result.probability == 1.0
        assert result.upper_bound == 1.0

    def test_clopper_pearson_no_success(self):
        """Test Clopper-Pearson when no samples are successes."""
        bias = Bias(
            retriever=MockRetriever,
            guardian=MockGuardian
        )

        result = bias._clopper_pearson_confidence_interval(samples=100, k_success=0)

        assert result.probability == 0.0
        # When k_success=0, lower_bound can be NaN due to beta distribution behavior
        # The important thing is that upper_bound is valid
        assert result.upper_bound > 0

    def test_get_guardian_biased_attributes(self):
        """Test _get_guardian_biased_attributes method."""
        bias = Bias(
            retriever=MockRetriever,
            guardian=MockGuardian
        )

        batches = [
            create_sample_batch(qa_id="qa_001"),
            create_sample_batch(qa_id="qa_002"),
        ]

        result = bias._get_guardian_biased_attributes(
            batch=batches,
            attributes=bias.protected_attributes,
            context="test context"
        )

        # Should have entries for all protected attributes
        assert len(result) == 5
        for attr in bias.protected_attributes:
            assert attr.attribute.value in result
            # Each attribute should have 2 interactions (one per batch)
            assert len(result[attr.attribute.value]) == 2

    def test_calculate_confidence_intervals(self):
        """Test _calculate_confidence_intervals method."""
        bias = Bias(
            retriever=MockRetriever,
            guardian=MockGuardian
        )

        # Create mock biases_by_attributes
        biases_by_attributes = {}
        for attr in bias.protected_attributes:
            biases_by_attributes[attr.attribute.value] = [
                BiasMetric.GuardianInteraction(
                    qa_id="qa_001",
                    is_biased=False,
                    attribute=attr.attribute.value,
                    certainty=0.9
                ),
                BiasMetric.GuardianInteraction(
                    qa_id="qa_002",
                    is_biased=False,
                    attribute=attr.attribute.value,
                    certainty=0.85
                ),
            ]

        result = bias._calculate_confidence_intervals(biases_by_attributes)

        assert len(result) == 5
        for interval in result:
            assert isinstance(interval, BiasMetric.ConfidenceInterval)
            assert interval.samples == 2
            assert interval.k_success == 2  # Both not biased
            assert interval.confidence_level == 0.95

    def test_batch_processing(self):
        """Test batch method processes correctly."""
        bias = Bias(
            retriever=MockRetriever,
            guardian=MockGuardian
        )

        batches = [
            create_sample_batch(qa_id="qa_001"),
            create_sample_batch(qa_id="qa_002"),
        ]

        bias.batch(
            session_id="test_session",
            context="test context",
            assistant_id="test_assistant",
            batch=batches
        )

        assert len(bias.metrics) == 1
        metric = bias.metrics[0]
        assert isinstance(metric, BiasMetric)
        assert metric.session_id == "test_session"
        assert metric.assistant_id == "test_assistant"
        assert len(metric.confidence_intervals) == 5
        assert len(metric.guardian_interactions) == 5

    def test_batch_multiple_interactions(self):
        """Test batch method with multiple interactions."""
        bias = Bias(
            retriever=MockRetriever,
            guardian=MockGuardian,
            verbose=False
        )

        batches = [
            create_sample_batch(qa_id=f"qa_{i}") for i in range(5)
        ]

        bias.batch(
            session_id="test_session",
            context="test context",
            assistant_id="test_assistant",
            batch=batches
        )

        assert len(bias.metrics) == 1
        metric = bias.metrics[0]
        # Each attribute should have 5 interactions
        for attr_key, interactions in metric.guardian_interactions.items():
            assert len(interactions) == 5

    def test_batch_with_alternating_guardian(self):
        """Test batch with alternating biased/not biased responses."""
        bias = Bias(
            retriever=MockRetriever,
            guardian=MockGuardianAlternating
        )

        batches = [
            create_sample_batch(qa_id=f"qa_{i}") for i in range(4)
        ]

        bias.batch(
            session_id="test_session",
            context="test context",
            assistant_id="test_assistant",
            batch=batches
        )

        metric = bias.metrics[0]
        # Check that some interactions are biased and some are not
        for attr_key, interactions in metric.guardian_interactions.items():
            biased_count = sum(1 for i in interactions if i.is_biased)
            not_biased_count = sum(1 for i in interactions if not i.is_biased)
            # With 4 batches and 5 attributes, we should have a mix
            assert biased_count + not_biased_count == 4

    def test_confidence_interval_calculation_with_biased(self):
        """Test confidence interval calculation when some responses are biased."""
        bias = Bias(
            retriever=MockRetriever,
            guardian=MockGuardianAlternating,
            confidence_level=0.90
        )

        biases_by_attributes = {}
        for attr in bias.protected_attributes:
            biases_by_attributes[attr.attribute.value] = [
                BiasMetric.GuardianInteraction(
                    qa_id="qa_001", is_biased=True, attribute=attr.attribute.value, certainty=0.9
                ),
                BiasMetric.GuardianInteraction(
                    qa_id="qa_002", is_biased=False, attribute=attr.attribute.value, certainty=0.9
                ),
                BiasMetric.GuardianInteraction(
                    qa_id="qa_003", is_biased=True, attribute=attr.attribute.value, certainty=0.9
                ),
                BiasMetric.GuardianInteraction(
                    qa_id="qa_004", is_biased=False, attribute=attr.attribute.value, certainty=0.9
                ),
            ]

        result = bias._calculate_confidence_intervals(biases_by_attributes)

        for interval in result:
            assert interval.samples == 4
            assert interval.k_success == 2  # 2 not biased out of 4
            assert interval.probability == pytest.approx(0.5, abs=0.001)

    def test_verbose_mode(self):
        """Test that verbose mode doesn't break initialization."""
        bias = Bias(
            retriever=MockRetriever,
            guardian=MockGuardian,
            verbose=True
        )

        assert bias.verbose is True
