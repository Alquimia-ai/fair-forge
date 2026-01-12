# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Alquimia AI Fair Forge is a performance-measurement library for evaluating AI models and assistants. It provides metrics for fairness, toxicity, bias, conversational quality, and more.

## Development Commands

```bash
# Install dependencies
uv sync

# Run scripts in development
uv run python your_script.py

# Run all tests with coverage
uv run pytest

# Run a single test file
uv run pytest tests/metrics/test_toxicity.py

# Run a specific test
uv run pytest tests/metrics/test_toxicity.py::test_function_name

# Run tests in parallel
uv run pytest -n auto

# Skip slow tests
uv run pytest -m "not slow"

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy fair_forge

# Build package
uv build
```

## Architecture

### Core Pattern: FairForge Base Class
All metrics inherit from `FairForge` (in `fair_forge/core/base.py`). The pattern:
1. Subclass `FairForge`
2. Implement `batch()` method to process conversation batches
3. Append results to `self.metrics`
4. Use via `MyMetric.run(RetrieverClass, **kwargs)` which handles instantiation and processing

### Data Flow
```
Retriever.load_dataset() -> list[Dataset]
    ↓
FairForge._process() iterates datasets
    ↓
Metric.batch() processes each conversation
    ↓
Results in self.metrics
```

### Key Modules

- **`fair_forge/metrics/`**: Metric implementations (Toxicity, Bias, Context, Conversational, Humanity, BestOf, Agentic). Uses lazy imports.
- **`fair_forge/core/`**: Base classes - `FairForge`, `Retriever`, `Guardian`, `ToxicityLoader`, `SentimentAnalyzer`
- **`fair_forge/schemas/`**: Pydantic models for data validation (`Dataset`, `Batch`, metric-specific schemas)
- **`fair_forge/runners/`**: Test execution against AI systems (`BaseRunner`, `AlquimiaRunner`)
- **`fair_forge/storage/`**: Storage backends for test datasets (`LocalStorage`, `LakeFSStorage`)
- **`fair_forge/statistical/`**: Statistical modes (`FrequentistMode`, `BayesianMode`) for metrics like Toxicity
- **`fair_forge/guardians/`**: Bias detection implementations (IBMGranite, LlamaGuard)
- **`fair_forge/loaders/`**: Dataset loaders (e.g., `HurtlexLoader` for toxicity lexicons)
- **`fair_forge/llm/`**: LLM integration (`Judge`, prompts, schemas for structured outputs)
- **`fair_forge/extractors/`**: Group extraction implementations (`EmbeddingGroupExtractor`)
- **`fair_forge/utils/`**: Utilities (logging configuration)

### Custom Retriever Pattern
Users must implement a `Retriever` subclass to load their data:
```python
from fair_forge.core.retriever import Retriever
from fair_forge.schemas.common import Dataset

class MyRetriever(Retriever):
    def load_dataset(self) -> list[Dataset]:
        # Load and return datasets
        pass
```

### Test Fixtures
Shared fixtures in `tests/conftest.py` provide mock retrievers and datasets for each metric type (e.g., `toxicity_dataset_retriever`, `bias_dataset_retriever`).

## Key Data Structures

- **`Dataset`**: A conversation session with `session_id`, `assistant_id`, `language`, `context`, and `conversation` (list of Batch)
- **`Batch`**: Single Q&A interaction with `query`, `assistant`, `ground_truth_assistant`, `qa_id`
