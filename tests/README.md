# Fair-Forge Tests

This directory contains the test suite for Fair-Forge metrics.

## Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and shared fixtures
├── README.md                # This file
│
├── fixtures/                # Test fixtures and mock data
│   ├── __init__.py
│   ├── mock_data.py         # Mock datasets for testing
│   └── mock_retriever.py    # Mock retriever implementations
│
└── metrics/                 # Metric tests
    ├── __init__.py
    ├── test_humanity.py     # Humanity metric tests
    └── test_toxicity.py     # Toxicity metric tests
```

## Running Tests

### Run all tests
```bash
# With uv
uv run pytest

# With pytest directly
pytest
```

### Run specific test file
```bash
# Humanity tests
uv run pytest tests/metrics/test_humanity.py

# Toxicity tests
uv run pytest tests/metrics/test_toxicity.py
```

### Run specific test class or function
```bash
# Run specific test class
uv run pytest tests/metrics/test_humanity.py::TestHumanityMetric

# Run specific test function
uv run pytest tests/metrics/test_humanity.py::TestHumanityMetric::test_humanity_initialization
```

### Run with coverage
```bash
# Generate coverage report
uv run pytest --cov=fair_forge --cov-report=html

# View HTML coverage report
open htmlcov/index.html
```

### Run with verbose output
```bash
uv run pytest -v
```

### Run with markers
```bash
# Run only unit tests
uv run pytest -m unit

# Skip slow tests
uv run pytest -m "not slow"
```

## Test Categories

### Humanity Metric Tests (`test_humanity.py`)
Tests for emotional analysis using the NRC lexicon:
- Initialization and configuration
- Text tokenization
- Emotion lexicon loading
- Emotion distribution calculation
- Emotional entropy computation
- Spearman correlation with ground truth
- Batch processing
- Metric attributes validation

### Toxicity Metric Tests (`test_toxicity.py`)
Tests for toxic language detection with clustering:
- Initialization with statistical modes (Frequentist/Bayesian)
- Custom configuration
- Text tokenization (including Unicode)
- Toxic word set building
- Toxic word counting
- Toxicity score calculation
- Binary toxicity classification
- Weight normalization
- DR (Demographic Representation) computation
- DTO (Directed Toxicity) computation
- ASB (Associated Sentiment Bias) computation
- Batch processing with mocked dependencies
- Metric attributes validation

## Fixtures

### Shared Fixtures (conftest.py)
- `sample_batch`: Single Q&A interaction
- `sample_dataset`: Complete conversation dataset
- `emotional_dataset`: Emotionally rich dataset for Humanity tests
- `toxicity_dataset`: Dataset for Toxicity tests
- Various retriever fixtures for different test scenarios
- Mock API keys and configurations

### Mock Data (fixtures/mock_data.py)
Factory functions for creating test datasets:
- `create_sample_batch()`: Create individual Q&A batches
- `create_sample_dataset()`: Create complete conversation datasets
- `create_emotional_dataset()`: Emotionally rich content
- `create_toxicity_dataset()`: Toxicity testing content
- And more specialized datasets...

### Mock Retrievers (fixtures/mock_retriever.py)
Mock retriever implementations:
- `MockRetriever`: Customizable retriever with predefined datasets
- `EmptyRetriever`: Returns empty dataset list
- `SingleDatasetRetriever`: Returns single dataset
- `MultipleDatasetRetriever`: Returns multiple datasets
- Specialized retrievers for each metric type

## Writing New Tests

When adding tests for new metrics:

1. Create a new test file in `tests/metrics/` (e.g., `test_new_metric.py`)
2. Add mock data factories in `tests/fixtures/mock_data.py` if needed
3. Add specialized retrievers in `tests/fixtures/mock_retriever.py` if needed
4. Add fixtures to `conftest.py` if they're shared across multiple test files
5. Follow the existing test structure and naming conventions

### Test Structure Template

```python
"""Unit tests for NewMetric metric."""
import pytest
from unittest.mock import Mock, patch

from fair_forge.metrics import NewMetric
from fair_forge.schemas import NewMetricResult


class TestNewMetric:
    """Test suite for NewMetric."""

    def test_metric_initialization(self, mock_retriever):
        """Test that NewMetric initializes correctly."""
        metric = NewMetric(mock_retriever)
        assert metric is not None

    def test_batch_processing(self, mock_retriever, sample_dataset):
        """Test batch processing."""
        metric = NewMetric(mock_retriever)
        # Add test logic...

    # Add more tests...
```

## Best Practices

1. **Use mocks for external dependencies**: Mock LLM APIs, embedding models, etc.
2. **Test edge cases**: Empty inputs, invalid data, boundary conditions
3. **Keep tests isolated**: Each test should be independent
4. **Use descriptive names**: Test names should clearly describe what they test
5. **Add docstrings**: Explain what each test is testing
6. **Use fixtures**: Reuse common test data via fixtures
7. **Test both success and failure**: Test expected behavior and error handling

## Dependencies

Tests require the following packages (already in dev-dependencies):
- `pytest>=7.0.0`: Test framework
- `pytest-cov>=4.0.0`: Coverage reporting

## Continuous Integration

Tests are run automatically on:
- Pull requests
- Commits to main branch
- Release tags

Coverage reports are generated and can be viewed in the CI pipeline.

## Troubleshooting

### Tests failing due to missing dependencies
```bash
uv sync
```

### Tests failing due to imports
Make sure you're running tests from the project root:
```bash
cd /path/to/fair-forge
uv run pytest
```

### Coverage not working
Ensure pytest-cov is installed:
```bash
uv add --dev pytest-cov
```
