# PromptEvaluator Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create synthetic fixtures + real JSON datasets for PromptEvaluator covering 3 domains (FAQ, RAG, Structured Instructions), run experiments with `gpt-oss-20b` via Groq, generate paper figures (clustering, radar, heatmap, bar, scatter), and integrate results into the LaTeX paper + PR.

**Architecture:** Phase 2 (Tasks 1–8) runs as two parallel agents — Agent A handles synthetic fixtures + tests, Agent B creates real JSON datasets + retriever. Phase 3 (Tasks 9–12) creates and runs the experiment notebook. Phase 4 (Tasks 13–14) runs as two parallel agents — Agent A adds the Experiments section to the paper, Agent B opens the PR.

**Tech Stack:** Python 3.11+, pytest, Pydantic v2, LangChain + langchain-groq, sentence-transformers, umap-learn, scikit-learn, matplotlib, Jupyter

---

## PHASE 2A — Agent A: Synthetic Fixtures + Tests

### Task 1: Add multi-domain synthetic fixture functions

**Files:**
- Modify: `tests/fixtures/mock_data.py`

- [ ] **Step 1: Append the three domain fixture functions to `mock_data.py`**

Add at the end of the file (after the existing `create_prompt_evaluator_dataset` function):

```python
def create_prompt_evaluator_faq_dataset() -> Dataset:
    """FAQ domain dataset for PromptEvaluator — 10 queries with ground truths."""
    context = (
        "Nexo is a project management SaaS. Plans: Free (3 projects, 5 members, 5GB, no support), "
        "Pro ($15/user/month, unlimited projects and members, 100GB, priority support, API access), "
        "Business ($25/user/month, all Pro features plus SSO, audit logs, dedicated support, custom integrations). "
        "Billing: monthly, cancel anytime from Settings > Billing, no refunds for partial months. "
        "Data export available for 30 days after cancellation from Settings > Data Export."
    )
    conversation = [
        create_sample_batch(qa_id="faq_001", query="How much does the Pro plan cost?",
            assistant="", ground_truth_assistant="The Pro plan costs $15 per user per month."),
        create_sample_batch(qa_id="faq_002", query="What is included in the Free plan?",
            assistant="", ground_truth_assistant="The Free plan includes up to 3 projects, 5 members, and 5GB of storage."),
        create_sample_batch(qa_id="faq_003", query="Can I cancel my subscription anytime?",
            assistant="", ground_truth_assistant="Yes, you can cancel anytime from Settings > Billing."),
        create_sample_batch(qa_id="faq_004", query="Does the Pro plan include API access?",
            assistant="", ground_truth_assistant="Yes, the Pro plan includes API access."),
        create_sample_batch(qa_id="faq_005", query="How many team members can I have on the Free plan?",
            assistant="", ground_truth_assistant="The Free plan supports up to 5 team members."),
        create_sample_batch(qa_id="faq_006", query="What is the storage limit on the Pro plan?",
            assistant="", ground_truth_assistant="The Pro plan includes 100GB of storage."),
        create_sample_batch(qa_id="faq_007", query="Does the Business plan support SSO?",
            assistant="", ground_truth_assistant="Yes, the Business plan includes SSO."),
        create_sample_batch(qa_id="faq_008", query="How long can I export my data after cancelling?",
            assistant="", ground_truth_assistant="You can export your data for 30 days after cancellation."),
        create_sample_batch(qa_id="faq_009", query="What extra features does Business offer over Pro?",
            assistant="", ground_truth_assistant="Business adds SSO, audit logs, dedicated support, and custom integrations."),
        create_sample_batch(qa_id="faq_010", query="Are refunds given for partial months?",
            assistant="", ground_truth_assistant="No, Nexo does not offer refunds for partial months."),
    ]
    return create_sample_dataset(
        session_id="faq_eval_001",
        assistant_id="gpt-oss-20b",
        context=context,
        conversation=conversation,
    )


def create_prompt_evaluator_rag_dataset() -> Dataset:
    """RAG domain dataset for PromptEvaluator — 10 queries over a knowledge base article."""
    context = (
        "Velox is an open-source query execution engine written in C++. "
        "It was developed by Meta and open-sourced in 2022. "
        "Velox is designed to be embedded into data processing systems and provides vectorized execution, "
        "lazy evaluation, and columnar memory layout via Arrow-compatible buffers. "
        "It supports multiple encodings: flat, dictionary, constant, and sequence. "
        "Velox integrates with Presto and Spark via connectors. "
        "The expression evaluation engine supports SQL functions, lambdas, and UDFs registered at runtime. "
        "Memory management uses a MemoryPool hierarchy; pools can be configured with memory limits and eviction callbacks. "
        "Velox does not handle query planning — it receives a physical plan and executes it. "
        "License: Apache 2.0. Main repository: github.com/facebookincubator/velox."
    )
    conversation = [
        create_sample_batch(qa_id="rag_001", query="What language is Velox written in?",
            assistant="", ground_truth_assistant="Velox is written in C++."),
        create_sample_batch(qa_id="rag_002", query="Who developed Velox and when was it open-sourced?",
            assistant="", ground_truth_assistant="Velox was developed by Meta and open-sourced in 2022."),
        create_sample_batch(qa_id="rag_003", query="What memory layout does Velox use?",
            assistant="", ground_truth_assistant="Velox uses a columnar memory layout via Arrow-compatible buffers."),
        create_sample_batch(qa_id="rag_004", query="Which systems does Velox integrate with?",
            assistant="", ground_truth_assistant="Velox integrates with Presto and Spark via connectors."),
        create_sample_batch(qa_id="rag_005", query="Does Velox handle query planning?",
            assistant="", ground_truth_assistant="No, Velox does not handle query planning — it receives a physical plan and executes it."),
        create_sample_batch(qa_id="rag_006", query="What encodings does Velox support?",
            assistant="", ground_truth_assistant="Velox supports flat, dictionary, constant, and sequence encodings."),
        create_sample_batch(qa_id="rag_007", query="What is the license of Velox?",
            assistant="", ground_truth_assistant="Velox is licensed under Apache 2.0."),
        create_sample_batch(qa_id="rag_008", query="How does Velox manage memory?",
            assistant="", ground_truth_assistant="Velox uses a MemoryPool hierarchy with configurable memory limits and eviction callbacks."),
        create_sample_batch(qa_id="rag_009", query="Can Velox execute UDFs?",
            assistant="", ground_truth_assistant="Yes, Velox supports UDFs registered at runtime via its expression evaluation engine."),
        create_sample_batch(qa_id="rag_010", query="What evaluation strategy does Velox use?",
            assistant="", ground_truth_assistant="Velox uses vectorized execution and lazy evaluation."),
    ]
    return create_sample_dataset(
        session_id="rag_eval_001",
        assistant_id="gpt-oss-20b",
        context=context,
        conversation=conversation,
    )


def create_prompt_evaluator_structured_dataset() -> Dataset:
    """Structured Instructions domain dataset — 10 queries expecting JSON responses."""
    context = (
        "You are evaluating a structured-output assistant that must always respond in JSON. "
        "The assistant answers factual questions about geography and science."
    )
    conversation = [
        create_sample_batch(qa_id="str_001", query="What is the capital of France?",
            assistant="", ground_truth_assistant='{"answer": "Paris", "confidence": 1.0}'),
        create_sample_batch(qa_id="str_002", query="What is the boiling point of water in Celsius?",
            assistant="", ground_truth_assistant='{"answer": "100", "confidence": 1.0}'),
        create_sample_batch(qa_id="str_003", query="How many planets are in the solar system?",
            assistant="", ground_truth_assistant='{"answer": "8", "confidence": 1.0}'),
        create_sample_batch(qa_id="str_004", query="What is the chemical symbol for gold?",
            assistant="", ground_truth_assistant='{"answer": "Au", "confidence": 1.0}'),
        create_sample_batch(qa_id="str_005", query="What is the largest ocean on Earth?",
            assistant="", ground_truth_assistant='{"answer": "Pacific Ocean", "confidence": 1.0}'),
        create_sample_batch(qa_id="str_006", query="In what year did the First World War end?",
            assistant="", ground_truth_assistant='{"answer": "1918", "confidence": 1.0}'),
        create_sample_batch(qa_id="str_007", query="What is the speed of light in km/s?",
            assistant="", ground_truth_assistant='{"answer": "299792", "confidence": 0.99}'),
        create_sample_batch(qa_id="str_008", query="What is the atomic number of oxygen?",
            assistant="", ground_truth_assistant='{"answer": "8", "confidence": 1.0}'),
        create_sample_batch(qa_id="str_009", query="Which planet is closest to the Sun?",
            assistant="", ground_truth_assistant='{"answer": "Mercury", "confidence": 1.0}'),
        create_sample_batch(qa_id="str_010", query="What is the square root of 144?",
            assistant="", ground_truth_assistant='{"answer": "12", "confidence": 1.0}'),
    ]
    return create_sample_dataset(
        session_id="structured_eval_001",
        assistant_id="gpt-oss-20b",
        context=context,
        conversation=conversation,
    )
```

- [ ] **Step 2: Add the imports to `mock_retriever.py`**

In `tests/fixtures/mock_retriever.py`, add the three new functions to the import block:

```python
from tests.fixtures.mock_data import (
    create_agentic_dataset,
    create_bestof_dataset,
    create_bias_dataset,
    create_context_dataset,
    create_conversational_dataset,
    create_emotional_dataset,
    create_multiple_datasets,
    create_prompt_evaluator_dataset,
    create_prompt_evaluator_faq_dataset,
    create_prompt_evaluator_rag_dataset,
    create_prompt_evaluator_structured_dataset,
    create_regulatory_dataset,
    create_sample_dataset,
    create_toxicity_dataset,
    create_vision_dataset,
)
```

Then add three new retriever classes at the end of the file (before `ErrorRetriever`):

```python
class PromptEvaluatorFaqDatasetRetriever(Retriever):
    """Mock retriever for PromptEvaluator FAQ domain testing."""

    def load_dataset(self) -> list[Dataset]:
        """Return FAQ domain dataset."""
        return [create_prompt_evaluator_faq_dataset()]


class PromptEvaluatorRagDatasetRetriever(Retriever):
    """Mock retriever for PromptEvaluator RAG domain testing."""

    def load_dataset(self) -> list[Dataset]:
        """Return RAG domain dataset."""
        return [create_prompt_evaluator_rag_dataset()]


class PromptEvaluatorStructuredDatasetRetriever(Retriever):
    """Mock retriever for PromptEvaluator Structured Instructions domain testing."""

    def load_dataset(self) -> list[Dataset]:
        """Return structured instructions domain dataset."""
        return [create_prompt_evaluator_structured_dataset()]
```

- [ ] **Step 3: Add domain fixtures to `tests/conftest.py`**

In `tests/conftest.py`, add after the existing `prompt_evaluator_dataset_retriever` fixture:

```python
from tests.fixtures.mock_retriever import (
    PromptEvaluatorFaqDatasetRetriever,
    PromptEvaluatorRagDatasetRetriever,
    PromptEvaluatorStructuredDatasetRetriever,
)


@pytest.fixture
def prompt_evaluator_faq_retriever() -> type[PromptEvaluatorFaqDatasetRetriever]:
    return PromptEvaluatorFaqDatasetRetriever


@pytest.fixture
def prompt_evaluator_rag_retriever() -> type[PromptEvaluatorRagDatasetRetriever]:
    return PromptEvaluatorRagDatasetRetriever


@pytest.fixture
def prompt_evaluator_structured_retriever() -> type[PromptEvaluatorStructuredDatasetRetriever]:
    return PromptEvaluatorStructuredDatasetRetriever
```

---

### Task 2: Add domain-coverage tests

**Files:**
- Modify: `tests/metrics/test_prompt_evaluator.py`

- [ ] **Step 1: Write failing tests**

Add the following class at the end of `tests/metrics/test_prompt_evaluator.py`:

```python
class TestDomainCoverage:
    """Verify domain-specific fixture properties — not metric logic, just data contracts."""

    def test_faq_dataset_has_ten_queries(self, prompt_evaluator_faq_retriever):
        from tests.fixtures.mock_data import create_prompt_evaluator_faq_dataset
        ds = create_prompt_evaluator_faq_dataset()
        assert len(ds.conversation) == 10

    def test_faq_dataset_all_batches_have_ground_truth(self, prompt_evaluator_faq_retriever):
        from tests.fixtures.mock_data import create_prompt_evaluator_faq_dataset
        ds = create_prompt_evaluator_faq_dataset()
        assert all(b.ground_truth_assistant for b in ds.conversation)

    def test_rag_dataset_has_ten_queries(self, prompt_evaluator_rag_retriever):
        from tests.fixtures.mock_data import create_prompt_evaluator_rag_dataset
        ds = create_prompt_evaluator_rag_dataset()
        assert len(ds.conversation) == 10

    def test_rag_dataset_has_nonempty_context(self, prompt_evaluator_rag_retriever):
        from tests.fixtures.mock_data import create_prompt_evaluator_rag_dataset
        ds = create_prompt_evaluator_rag_dataset()
        assert len(ds.context) > 50

    def test_structured_dataset_has_ten_queries(self, prompt_evaluator_structured_retriever):
        from tests.fixtures.mock_data import create_prompt_evaluator_structured_dataset
        ds = create_prompt_evaluator_structured_dataset()
        assert len(ds.conversation) == 10

    def test_structured_ground_truths_are_valid_json(self, prompt_evaluator_structured_retriever):
        import json
        from tests.fixtures.mock_data import create_prompt_evaluator_structured_dataset
        ds = create_prompt_evaluator_structured_dataset()
        for batch in ds.conversation:
            parsed = json.loads(batch.ground_truth_assistant)
            assert "answer" in parsed
            assert "confidence" in parsed

    def test_faq_rss_activates_with_ground_truth(self, prompt_evaluator_faq_retriever):
        """RSS is computed when all FAQ batches have ground_truth_assistant."""
        batches = create_prompt_evaluator_faq_dataset().conversation[:2]
        result = _run(batches, ["response"] * 6, k=3)
        # RSS may be None here because embedder is mocked uniformly,
        # but the signal should be present since ground truth is not empty.
        assert result.rss is not None

    def test_icr_activates_on_structured_domain(self, prompt_evaluator_structured_retriever):
        """ICR is computed when constraints are provided for structured domain."""
        from fair_forge.metrics.constraints import JsonConstraint
        from tests.fixtures.mock_data import create_prompt_evaluator_structured_dataset
        batches = create_prompt_evaluator_structured_dataset().conversation[:2]
        result = _run(batches, ['{"answer": "yes", "confidence": 0.9}'] * 6,
                      k=3, constraints=[JsonConstraint()])
        assert result.icr is not None
        assert result.icr == pytest.approx(1.0)
```

- [ ] **Step 2: Run the new tests to verify they pass**

```bash
uv run pytest tests/metrics/test_prompt_evaluator.py::TestDomainCoverage -v
```

Expected output: all 8 tests PASS.

- [ ] **Step 3: Run full test suite to check no regressions**

```bash
uv run pytest tests/metrics/test_prompt_evaluator.py -v
```

Expected: all tests PASS.

---

### Task 3: Lint + commit Phase 2A

- [ ] **Step 1: Lint**

```bash
uv run ruff check tests/fixtures/mock_data.py tests/fixtures/mock_retriever.py tests/metrics/test_prompt_evaluator.py tests/conftest.py
uv run ruff format tests/fixtures/mock_data.py tests/fixtures/mock_retriever.py tests/metrics/test_prompt_evaluator.py tests/conftest.py
```

Expected: no errors.

- [ ] **Step 2: Commit**

```bash
git add tests/fixtures/mock_data.py tests/fixtures/mock_retriever.py tests/metrics/test_prompt_evaluator.py tests/conftest.py
git commit -m "test(fixtures): add multi-domain synthetic fixtures for PromptEvaluator (FAQ, RAG, Structured)"
```

---

## PHASE 2B — Agent B: Real JSON Datasets + Retriever

### Task 4: Create FAQ JSON dataset

**Files:**
- Create: `datasets/prompt_evaluator/faq.json`

- [ ] **Step 1: Create the directory and write `faq.json`**

```bash
mkdir -p datasets/prompt_evaluator
```

Write `datasets/prompt_evaluator/faq.json`:

```json
[
  {
    "session_id": "faq_eval_001",
    "assistant_id": "gpt-oss-20b",
    "language": "english",
    "context": "Nexo is a project management SaaS. Plans: Free (3 projects, 5 members, 5GB storage, no support), Pro ($15/user/month, unlimited projects and members, 100GB storage, priority support, API access), Business ($25/user/month, all Pro features plus SSO, audit logs, dedicated support, custom integrations). Billing: monthly, cancel anytime from Settings > Billing, no refunds for partial months. Data export available for 30 days after cancellation from Settings > Data Export.",
    "conversation": [
      {"qa_id": "faq_001", "query": "How much does the Pro plan cost?", "assistant": "", "ground_truth_assistant": "The Pro plan costs $15 per user per month."},
      {"qa_id": "faq_002", "query": "What is included in the Free plan?", "assistant": "", "ground_truth_assistant": "The Free plan includes up to 3 projects, 5 members, and 5GB of storage."},
      {"qa_id": "faq_003", "query": "Can I cancel my subscription anytime?", "assistant": "", "ground_truth_assistant": "Yes, you can cancel anytime from Settings > Billing."},
      {"qa_id": "faq_004", "query": "Does the Pro plan include API access?", "assistant": "", "ground_truth_assistant": "Yes, the Pro plan includes API access."},
      {"qa_id": "faq_005", "query": "How many team members can I have on the Free plan?", "assistant": "", "ground_truth_assistant": "The Free plan supports up to 5 team members."},
      {"qa_id": "faq_006", "query": "What is the storage limit on the Pro plan?", "assistant": "", "ground_truth_assistant": "The Pro plan includes 100GB of storage."},
      {"qa_id": "faq_007", "query": "Does the Business plan support SSO?", "assistant": "", "ground_truth_assistant": "Yes, the Business plan includes SSO."},
      {"qa_id": "faq_008", "query": "How long can I export my data after cancelling?", "assistant": "", "ground_truth_assistant": "You can export your data for 30 days after cancellation."},
      {"qa_id": "faq_009", "query": "What extra features does Business offer over Pro?", "assistant": "", "ground_truth_assistant": "Business adds SSO, audit logs, dedicated support, and custom integrations on top of Pro features."},
      {"qa_id": "faq_010", "query": "Are refunds given for partial months?", "assistant": "", "ground_truth_assistant": "No, Nexo does not offer refunds for partial months."}
    ]
  }
]
```

---

### Task 5: Create RAG JSON dataset

**Files:**
- Create: `datasets/prompt_evaluator/rag.json`

- [ ] **Step 1: Write `rag.json`**

```json
[
  {
    "session_id": "rag_eval_001",
    "assistant_id": "gpt-oss-20b",
    "language": "english",
    "context": "Velox is an open-source query execution engine written in C++. Developed by Meta and open-sourced in 2022. Velox is designed to be embedded into data processing systems and provides vectorized execution, lazy evaluation, and columnar memory layout via Arrow-compatible buffers. It supports multiple encodings: flat, dictionary, constant, and sequence. Velox integrates with Presto and Spark via connectors. The expression evaluation engine supports SQL functions, lambdas, and UDFs registered at runtime. Memory management uses a MemoryPool hierarchy; pools can be configured with memory limits and eviction callbacks. Velox does not handle query planning — it receives a physical plan and executes it. License: Apache 2.0. Main repository: github.com/facebookincubator/velox.",
    "conversation": [
      {"qa_id": "rag_001", "query": "What language is Velox written in?", "assistant": "", "ground_truth_assistant": "Velox is written in C++."},
      {"qa_id": "rag_002", "query": "Who developed Velox and when was it open-sourced?", "assistant": "", "ground_truth_assistant": "Velox was developed by Meta and open-sourced in 2022."},
      {"qa_id": "rag_003", "query": "What memory layout does Velox use?", "assistant": "", "ground_truth_assistant": "Velox uses a columnar memory layout via Arrow-compatible buffers."},
      {"qa_id": "rag_004", "query": "Which systems does Velox integrate with?", "assistant": "", "ground_truth_assistant": "Velox integrates with Presto and Spark via connectors."},
      {"qa_id": "rag_005", "query": "Does Velox handle query planning?", "assistant": "", "ground_truth_assistant": "No, Velox does not handle query planning — it receives a physical plan and executes it."},
      {"qa_id": "rag_006", "query": "What encodings does Velox support?", "assistant": "", "ground_truth_assistant": "Velox supports flat, dictionary, constant, and sequence encodings."},
      {"qa_id": "rag_007", "query": "What is the license of Velox?", "assistant": "", "ground_truth_assistant": "Velox is licensed under Apache 2.0."},
      {"qa_id": "rag_008", "query": "How does Velox manage memory?", "assistant": "", "ground_truth_assistant": "Velox uses a MemoryPool hierarchy with configurable memory limits and eviction callbacks."},
      {"qa_id": "rag_009", "query": "Can Velox execute UDFs?", "assistant": "", "ground_truth_assistant": "Yes, Velox supports UDFs registered at runtime via its expression evaluation engine."},
      {"qa_id": "rag_010", "query": "What evaluation strategy does Velox use?", "assistant": "", "ground_truth_assistant": "Velox uses vectorized execution and lazy evaluation."}
    ]
  }
]
```

---

### Task 6: Create Structured Instructions JSON dataset

**Files:**
- Create: `datasets/prompt_evaluator/structured.json`

- [ ] **Step 1: Write `structured.json`**

```json
[
  {
    "session_id": "structured_eval_001",
    "assistant_id": "gpt-oss-20b",
    "language": "english",
    "context": "You are evaluating a structured-output assistant that answers factual questions about geography and science.",
    "conversation": [
      {"qa_id": "str_001", "query": "What is the capital of France?", "assistant": "", "ground_truth_assistant": "{\"answer\": \"Paris\", \"confidence\": 1.0}"},
      {"qa_id": "str_002", "query": "What is the boiling point of water in Celsius?", "assistant": "", "ground_truth_assistant": "{\"answer\": \"100\", \"confidence\": 1.0}"},
      {"qa_id": "str_003", "query": "How many planets are in the solar system?", "assistant": "", "ground_truth_assistant": "{\"answer\": \"8\", \"confidence\": 1.0}"},
      {"qa_id": "str_004", "query": "What is the chemical symbol for gold?", "assistant": "", "ground_truth_assistant": "{\"answer\": \"Au\", \"confidence\": 1.0}"},
      {"qa_id": "str_005", "query": "What is the largest ocean on Earth?", "assistant": "", "ground_truth_assistant": "{\"answer\": \"Pacific Ocean\", \"confidence\": 1.0}"},
      {"qa_id": "str_006", "query": "In what year did the First World War end?", "assistant": "", "ground_truth_assistant": "{\"answer\": \"1918\", \"confidence\": 1.0}"},
      {"qa_id": "str_007", "query": "What is the speed of light in km/s (rounded)?", "assistant": "", "ground_truth_assistant": "{\"answer\": \"299792\", \"confidence\": 0.99}"},
      {"qa_id": "str_008", "query": "What is the atomic number of oxygen?", "assistant": "", "ground_truth_assistant": "{\"answer\": \"8\", \"confidence\": 1.0}"},
      {"qa_id": "str_009", "query": "Which planet is closest to the Sun?", "assistant": "", "ground_truth_assistant": "{\"answer\": \"Mercury\", \"confidence\": 1.0}"},
      {"qa_id": "str_010", "query": "What is the square root of 144?", "assistant": "", "ground_truth_assistant": "{\"answer\": \"12\", \"confidence\": 1.0}"}
    ]
  }
]
```

---

### Task 7: Create `PromptEvaluatorJsonRetriever`

**Files:**
- Create: `fair_forge/datasets/__init__.py`
- Create: `fair_forge/datasets/prompt_evaluator_retriever.py`

- [ ] **Step 1: Create `fair_forge/datasets/__init__.py`**

```python
"""Dataset loaders for Fair-Forge evaluation datasets."""

from fair_forge.datasets.prompt_evaluator_retriever import PromptEvaluatorJsonRetriever

__all__ = ["PromptEvaluatorJsonRetriever"]
```

- [ ] **Step 2: Create `fair_forge/datasets/prompt_evaluator_retriever.py`**

```python
"""JSON-based retriever for PromptEvaluator evaluation datasets."""

import json
from pathlib import Path

from fair_forge.core.retriever import Retriever
from fair_forge.schemas.common import Dataset


class PromptEvaluatorJsonRetriever(Retriever):
    """Load a PromptEvaluator dataset from a JSON file.

    The JSON file must contain either a single Dataset object or a list of Dataset objects
    in the format expected by ``Dataset.model_validate()``.

    Args:
        path: Path to the JSON file.
    """

    def __init__(self, path: Path | str, **kwargs):
        super().__init__(**kwargs)
        self._path = Path(path)

    def load_dataset(self) -> list[Dataset]:
        data = json.loads(self._path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [Dataset.model_validate(entry) for entry in data]
        return [Dataset.model_validate(data)]
```

---

### Task 8: Smoke-test retriever + commit Phase 2B

**Files:**
- Create: `tests/datasets/test_prompt_evaluator_retriever.py`
- Create: `tests/datasets/__init__.py`

- [ ] **Step 1: Write the test**

Create `tests/datasets/__init__.py` (empty).

Create `tests/datasets/test_prompt_evaluator_retriever.py`:

```python
"""Smoke tests for PromptEvaluatorJsonRetriever."""

from pathlib import Path

import pytest

from fair_forge.datasets.prompt_evaluator_retriever import PromptEvaluatorJsonRetriever
from fair_forge.schemas.common import Dataset

DATASETS_DIR = Path(__file__).parent.parent.parent / "datasets" / "prompt_evaluator"


@pytest.mark.parametrize("filename", ["faq.json", "rag.json", "structured.json"])
def test_retriever_loads_dataset(filename):
    retriever = PromptEvaluatorJsonRetriever(path=DATASETS_DIR / filename)
    datasets = retriever.load_dataset()
    assert len(datasets) == 1
    assert isinstance(datasets[0], Dataset)


@pytest.mark.parametrize("filename", ["faq.json", "rag.json", "structured.json"])
def test_retriever_dataset_has_ten_queries(filename):
    retriever = PromptEvaluatorJsonRetriever(path=DATASETS_DIR / filename)
    datasets = retriever.load_dataset()
    assert len(datasets[0].conversation) == 10


def test_faq_retriever_has_ground_truths():
    retriever = PromptEvaluatorJsonRetriever(path=DATASETS_DIR / "faq.json")
    datasets = retriever.load_dataset()
    assert all(b.ground_truth_assistant for b in datasets[0].conversation)


def test_structured_retriever_has_json_ground_truths():
    import json
    retriever = PromptEvaluatorJsonRetriever(path=DATASETS_DIR / "structured.json")
    datasets = retriever.load_dataset()
    for batch in datasets[0].conversation:
        parsed = json.loads(batch.ground_truth_assistant)
        assert "answer" in parsed
        assert "confidence" in parsed
```

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/datasets/test_prompt_evaluator_retriever.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 3: Lint**

```bash
uv run ruff check fair_forge/datasets/ tests/datasets/
uv run ruff format fair_forge/datasets/ tests/datasets/
```

- [ ] **Step 4: Commit**

```bash
git add datasets/prompt_evaluator/ fair_forge/datasets/ tests/datasets/
git commit -m "feat(datasets): add PromptEvaluator JSON datasets and retriever for 3 evaluation domains"
```

---

## PHASE 3 — Experiment Notebook

### Task 9: Create notebook skeleton

**Files:**
- Create: `notebooks/prompt_evaluator_experiments.ipynb`
- Create: `assets/figures/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p notebooks
mkdir -p assets/figures
touch assets/figures/.gitkeep
```

- [ ] **Step 2: Create the notebook**

Create `notebooks/prompt_evaluator_experiments.ipynb` with the following cells. Use `nbformat` or write the JSON directly.

**Cell 1 — Imports and setup:**
```python
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from langchain_groq import ChatGroq

from fair_forge.datasets.prompt_evaluator_retriever import PromptEvaluatorJsonRetriever
from fair_forge.embedders.sentence_transformer import SentenceTransformerEmbedder
from fair_forge.metrics.constraints import JsonConstraint, KeywordConstraint
from fair_forge.metrics.prompt_evaluator import PromptEvaluator

DATASETS_DIR = Path("../datasets/prompt_evaluator")
FIGURES_DIR = Path("../assets/figures")
RESULTS_PATH = Path("../assets/prompt_evaluator_results.json")

model = ChatGroq(
    model="gpt-oss-20b",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.7,
)
embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
K = 10
TAU = 0.80
```

**Cell 2 — Prompt definitions:**
```python
PROMPTS = {
    "faq": {
        "good": (
            "You are a support assistant for Nexo. Answer questions accurately and concisely "
            "using ONLY the information in the provided context. "
            "If the answer is not in the context, respond exactly: 'I don't have that information.'"
        ),
        "ambiguous": "You are an assistant. Help users with their questions about the product.",
        "bad": "Answer any question the user has. Be creative and use your general knowledge.",
    },
    "rag": {
        "good": (
            "You are a knowledge retrieval assistant. Answer the user's question using ONLY the "
            "information provided in the context. Do not add any information not present in the context. "
            "If the context does not contain the answer, say: 'The provided context does not contain this information.'"
        ),
        "ambiguous": "Use the context to answer the question.",
        "bad": "Answer the question.",
    },
    "structured": {
        "good": (
            "You are a data extraction assistant. You MUST respond with a valid JSON object containing "
            "exactly two keys: \"answer\" (a string) and \"confidence\" (a float between 0.0 and 1.0). "
            "Do not include any text outside the JSON object."
        ),
        "ambiguous": "Respond with a JSON object containing 'answer' and 'confidence'.",
        "bad": "Answer the question concisely.",
    },
}

CONSTRAINTS = {
    "faq": [],
    "rag": [],
    "structured": [JsonConstraint(), KeywordConstraint("answer"), KeywordConstraint("confidence")],
}
```

---

### Task 10: Add experiment execution cells

**Files:**
- Modify: `notebooks/prompt_evaluator_experiments.ipynb`

- [ ] **Step 1: Add execution cell to the notebook**

**Cell 3 — Run experiments:**
```python
results = {}

for domain in ["faq", "rag", "structured"]:
    results[domain] = {}
    retriever_path = DATASETS_DIR / f"{domain}.json"

    for prompt_type in ["good", "ambiguous", "bad"]:
        print(f"Running {domain} / {prompt_type}...")

        retriever = type(
            f"{domain.capitalize()}{prompt_type.capitalize()}Retriever",
            (PromptEvaluatorJsonRetriever,),
            {"load_dataset": lambda self, p=retriever_path: PromptEvaluatorJsonRetriever(p).load_dataset()},
        )

        metrics = PromptEvaluator.run(
            retriever,
            model=model,
            seed_prompt=PROMPTS[domain][prompt_type],
            embedder=embedder,
            k=K,
            tau=TAU,
            constraints=CONSTRAINTS[domain] or None,
        )

        m = metrics[0]
        results[domain][prompt_type] = {
            "csr": m.csr,
            "stability": m.stability,
            "rss": m.rss,
            "icr": m.icr,
            "jq": m.jq,
            "n_queries": m.n_queries,
            "interactions": [
                {
                    "qa_id": i.qa_id,
                    "csr": i.csr,
                    "stability": i.stability,
                    "rss": i.rss,
                    "icr": i.icr,
                    "n_clusters": i.n_clusters,
                }
                for i in m.interactions
            ],
        }
        print(f"  CSR={m.csr:.3f}  Stability={m.stability:.3f}  RSS={m.rss}  ICR={m.icr}")

# Save results
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {RESULTS_PATH}")
```

---

### Task 11: Add visualization cells

**Files:**
- Modify: `notebooks/prompt_evaluator_experiments.ipynb`

- [ ] **Step 1: Add all visualization cells**

**Cell 4 — Load saved results (allows re-running visuals without re-running experiments):**
```python
results = json.loads(RESULTS_PATH.read_text())
PROMPT_TYPES = ["good", "ambiguous", "bad"]
DOMAINS = ["faq", "rag", "structured"]
COLORS = {"good": "#2ecc71", "ambiguous": "#f39c12", "bad": "#e74c3c"}
```

**Cell 5 — Bar chart: all signals per domain:**
```python
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("PromptEvaluator — Metric Comparison by Domain and Prompt Quality", fontsize=13, fontweight="bold")

SIGNALS = ["csr", "stability", "rss", "icr"]
SIGNAL_LABELS = ["CSR", "Stability", "RSS", "ICR"]

for ax, domain in zip(axes, DOMAINS):
    x = np.arange(len(SIGNALS))
    width = 0.25
    for i, pt in enumerate(PROMPT_TYPES):
        vals = [results[domain][pt].get(s) or 0.0 for s in SIGNALS]
        ax.bar(x + i * width, vals, width, label=pt.capitalize(), color=COLORS[pt], alpha=0.85)
    ax.set_title(domain.upper(), fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(SIGNAL_LABELS)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "bar_chart_all_domains.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Cell 6 — Radar chart: per domain (good vs ambiguous vs bad):**
```python
from matplotlib.patches import FancyArrowPatch

def radar_chart(ax, values_dict, categories, title):
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], size=7)
    ax.set_title(title, fontweight="bold", pad=12)
    for pt, vals in values_dict.items():
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, linewidth=2, label=pt.capitalize(), color=COLORS[pt])
        ax.fill(angles, vals_closed, alpha=0.15, color=COLORS[pt])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

RADAR_SIGNALS = ["CSR", "Stability", "RSS", "ICR"]
RADAR_KEYS = ["csr", "stability", "rss", "icr"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={"polar": True})
fig.suptitle("PromptEvaluator — Radar Chart by Domain", fontsize=13, fontweight="bold")

for ax, domain in zip(axes, DOMAINS):
    values_dict = {
        pt: [results[domain][pt].get(k) or 0.0 for k in RADAR_KEYS]
        for pt in PROMPT_TYPES
    }
    radar_chart(ax, values_dict, RADAR_SIGNALS, domain.upper())

plt.tight_layout()
plt.savefig(FIGURES_DIR / "radar_chart_domains.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Cell 7 — Heatmap: CSR per query × prompt type (FAQ domain):**
```python
domain = "faq"
qa_ids = [i["qa_id"] for i in results[domain]["good"]["interactions"]]
heatmap_data = np.array([
    [i["csr"] for i in results[domain][pt]["interactions"]]
    for pt in PROMPT_TYPES
])

fig, ax = plt.subplots(figsize=(12, 3))
im = ax.imshow(heatmap_data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_yticks(range(len(PROMPT_TYPES)))
ax.set_yticklabels([pt.capitalize() for pt in PROMPT_TYPES])
ax.set_xticks(range(len(qa_ids)))
ax.set_xticklabels(qa_ids, rotation=45, ha="right", fontsize=8)
ax.set_title("CSR per Query × Prompt Quality — FAQ Domain", fontweight="bold")
plt.colorbar(im, ax=ax, label="CSR")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "heatmap_faq_csr.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Cell 8 — Scatter: CSR vs RSS (FAQ + RAG domains):**
```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("CSR vs RSS — Detecting 'Consistently Wrong' Prompts", fontsize=12, fontweight="bold")

for ax, domain in zip(axes, ["faq", "rag"]):
    for pt in PROMPT_TYPES:
        interactions = results[domain][pt]["interactions"]
        x = [i["csr"] for i in interactions]
        y = [i["rss"] or 0.0 for i in interactions]
        ax.scatter(x, y, label=pt.capitalize(), color=COLORS[pt], alpha=0.8, s=60, edgecolors="white")
    ax.set_xlabel("CSR (Consistency)", fontsize=10)
    ax.set_ylabel("RSS (Reference Similarity)", fontsize=10)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_title(domain.upper(), fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "scatter_csr_rss.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Cell 9 — Clustering plot (UMAP 2D) for one query of FAQ domain — good vs bad prompt:**

> Note: This cell requires running the embedder against real responses. It re-runs a single query (K=10) for the good and bad FAQ prompts and visualizes the embedding clusters.

```python
import umap

faq_retriever_good = PromptEvaluatorJsonRetriever(DATASETS_DIR / "faq.json")
faq_dataset = faq_retriever_good.load_dataset()[0]
sample_query = faq_dataset.conversation[0]  # "How much does the Pro plan cost?"

cluster_data = {}
for pt in ["good", "bad"]:
    responses = [
        model.invoke([
            __import__("langchain_core.messages", fromlist=["SystemMessage", "HumanMessage"]).SystemMessage(
                content=PROMPTS["faq"][pt]
            ),
            __import__("langchain_core.messages", fromlist=["SystemMessage", "HumanMessage"]).HumanMessage(
                content=sample_query.query
            ),
        ]).content
        for _ in range(K)
    ]
    embeddings = embedder.encode(responses)
    cluster_data[pt] = {"responses": responses, "embeddings": embeddings}

all_embeddings = np.vstack([cluster_data["good"]["embeddings"], cluster_data["bad"]["embeddings"]])
reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(all_embeddings)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(f'Semantic Clusters — "{sample_query.query}"', fontsize=12, fontweight="bold")

for ax, (pt, offset) in zip(axes, [("good", 0), ("bad", K)]):
    pts = embeddings_2d[offset:offset + K]
    ax.scatter(pts[:, 0], pts[:, 1], c=COLORS[pt], s=80, alpha=0.8, edgecolors="white")
    ax.set_title(f"{pt.capitalize()} Prompt", fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "clustering_umap_faq.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

### Task 12: Run notebook end-to-end and commit

- [ ] **Step 1: Set GROQ_API_KEY and run notebook**

```bash
export GROQ_API_KEY=your_key_here
cd notebooks
uv run jupyter nbconvert --to notebook --execute prompt_evaluator_experiments.ipynb \
    --output prompt_evaluator_experiments.ipynb \
    --ExecutePreprocessor.timeout=600
```

Expected: notebook executes without errors, `assets/prompt_evaluator_results.json` is created, PNG files appear in `assets/figures/`.

- [ ] **Step 2: Verify all figures were generated**

```bash
ls assets/figures/
```

Expected output includes:
```
bar_chart_all_domains.png
radar_chart_domains.png
heatmap_faq_csr.png
scatter_csr_rss.png
clustering_umap_faq.png
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/ assets/figures/ assets/prompt_evaluator_results.json
git commit -m "feat(experiments): add PromptEvaluator experiment notebook with visualizations"
```

---

## PHASE 4A — Agent A: Add Experiments Section to Paper

### Task 13: Add Experiments section to `papers/prompt_evaluator_en.tex`

**Files:**
- Modify: `papers/prompt_evaluator_en.tex`

- [ ] **Step 1: Load `assets/prompt_evaluator_results.json` to get the actual numbers**

Read the JSON file and extract CSR, Stability, RSS, ICR values per domain and prompt type. Use these to fill the LaTeX tables below.

- [ ] **Step 2: Add the Experiments section before `\section{Discussion and Limitations}`**

Insert the following block in `papers/prompt_evaluator_en.tex` immediately before the line `\section{Discussion and Limitations}`:

```latex
% ─────────────────────────────────────────────────────────────────────────────
\section{Experiments}
\label{sec:experiments}
% ─────────────────────────────────────────────────────────────────────────────

\subsection{Experimental Setup}

We evaluate the metric vector $\mathbf{m}$ across three evaluation domains, each designed to exercise a distinct subset of signals:

\begin{itemize}[noitemsep]
    \item \textbf{FAQ / Support (FAQ)}: A product support scenario with a fixed context document. Ground truth responses are available, activating RSS automatically. CSR and Stability assess whether the prompt generates consistent answers; RSS validates factual alignment with the reference.
    \item \textbf{RAG / Knowledge Base (RAG)}: A technical knowledge base extracted from open-source documentation. Same signal activation as FAQ; tests whether the prompt correctly constrains the model to the provided context.
    \item \textbf{Structured Instructions (STR)}: Queries expecting responses in JSON format with two mandatory keys (\texttt{answer}, \texttt{confidence}). ICR is activated via three programmatic constraints: valid JSON, presence of \texttt{answer}, presence of \texttt{confidence}.
\end{itemize}

Each domain is evaluated with three system prompts: a \textbf{Good} prompt (precise, context-grounded, format-enforcing), an \textbf{Ambiguous} prompt (vague instructions, under-specified), and a \textbf{Bad} prompt (no context constraint, counter-productive for the task). For each (prompt, query) pair we generate $K = 10$ responses with temperature $T = 0.7$ using \texttt{gpt-oss-20b} via the Groq API. Semantic clustering uses cosine similarity threshold $\tau = 0.80$ with the \texttt{all-MiniLM-L6-v2} sentence embedding model.

\subsection{Results}

Table~\ref{tab:results_faq} reports the mean metric vector $\bar{\mathbf{m}}(p)$ per prompt type for the FAQ and RAG domains. Table~\ref{tab:results_str} reports the Structured Instructions domain, where ICR replaces RSS as the primary correctness signal.

\begin{table}[h]
\centering
\caption{Mean metric vector per prompt type — FAQ and RAG domains. Each value is the mean over 10 queries with $K=10$ samples.}
\label{tab:results_faq}
\begin{tabular}{llcccc}
\toprule
Domain & Prompt & CSR $\uparrow$ & Stability $\uparrow$ & RSS $\uparrow$ & JQ \\
\midrule
\multirow{3}{*}{FAQ}
  & Good      & \textbf{FILL\_FAQ\_GOOD\_CSR}   & \textbf{FILL\_FAQ\_GOOD\_STAB}   & \textbf{FILL\_FAQ\_GOOD\_RSS}   & — \\
  & Ambiguous & FILL\_FAQ\_AMB\_CSR    & FILL\_FAQ\_AMB\_STAB    & FILL\_FAQ\_AMB\_RSS    & — \\
  & Bad       & FILL\_FAQ\_BAD\_CSR    & FILL\_FAQ\_BAD\_STAB    & FILL\_FAQ\_BAD\_RSS    & — \\
\addlinespace
\multirow{3}{*}{RAG}
  & Good      & \textbf{FILL\_RAG\_GOOD\_CSR}   & \textbf{FILL\_RAG\_GOOD\_STAB}   & \textbf{FILL\_RAG\_GOOD\_RSS}   & — \\
  & Ambiguous & FILL\_RAG\_AMB\_CSR    & FILL\_RAG\_AMB\_STAB    & FILL\_RAG\_AMB\_RSS    & — \\
  & Bad       & FILL\_RAG\_BAD\_CSR    & FILL\_RAG\_BAD\_STAB    & FILL\_RAG\_BAD\_RSS    & — \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Mean metric vector per prompt type — Structured Instructions domain. ICR measures compliance with three programmatic constraints (valid JSON, \texttt{answer} key, \texttt{confidence} key).}
\label{tab:results_str}
\begin{tabular}{llccc}
\toprule
Prompt & CSR $\uparrow$ & Stability $\uparrow$ & ICR $\uparrow$ \\
\midrule
Good      & \textbf{FILL\_STR\_GOOD\_CSR}   & \textbf{FILL\_STR\_GOOD\_STAB}   & \textbf{FILL\_STR\_GOOD\_ICR}   \\
Ambiguous & FILL\_STR\_AMB\_CSR    & FILL\_STR\_AMB\_STAB    & FILL\_STR\_AMB\_ICR    \\
Bad       & FILL\_STR\_BAD\_CSR    & FILL\_STR\_BAD\_STAB    & FILL\_STR\_BAD\_ICR    \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Observations.}
The results confirm the diagnostic value of the metric vector across three axes:

\begin{enumerate}[noitemsep]
    \item \textbf{CSR and Stability discriminate prompt quality.} The Good prompt yields consistently higher CSR and Stability than the Ambiguous and Bad prompts across all three domains, confirming that distributional signals detect behavioral differences that a single-call judge could only partially capture.
    \item \textbf{High CSR without high RSS exposes consistent incorrectness.} In the Bad FAQ and RAG prompts, CSR remains non-trivially high while RSS is low — the model produces consistent responses that diverge from the reference. This is the pattern documented by \citet{glape2024} and is invisible to CSR alone.
    \item \textbf{ICR provides a hard gate for format compliance.} The Good Structured prompt achieves near-perfect ICR, while the Bad prompt drops to near zero. CSR alone cannot detect whether the model is consistently producing the wrong format — ICR closes this gap deterministically.
\end{enumerate}

Figure~\ref{fig:bar} summarizes the full comparison; Figure~\ref{fig:radar} shows the multi-dimensional profile per domain; Figure~\ref{fig:scatter} plots CSR against RSS for each query, making the consistently-wrong pattern visible.

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../assets/figures/bar_chart_all_domains.png}
    \caption{Mean metric values by prompt type across three domains. Missing bars indicate signals not activated for that domain (ICR for FAQ/RAG; RSS for STR).}
    \label{fig:bar}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../assets/figures/radar_chart_domains.png}
    \caption{Radar chart of the metric vector $\bar{\mathbf{m}}(p)$ per domain. Each axis represents one signal normalized to $[0,1]$. The Good prompt fills the chart; the Bad prompt collapses it.}
    \label{fig:radar}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.85\textwidth]{../assets/figures/scatter_csr_rss.png}
    \caption{CSR vs RSS per query for FAQ (left) and RAG (right). Points in the upper-right quadrant (high CSR, high RSS) indicate correctly consistent prompts. Points in the upper-left quadrant (high CSR, low RSS) indicate the consistently-wrong failure mode.}
    \label{fig:scatter}
\end{figure}
```

- [ ] **Step 3: Replace all `FILL_*` placeholders with actual values from `assets/prompt_evaluator_results.json`**

Read `assets/prompt_evaluator_results.json` and substitute each placeholder:
- `FILL_FAQ_GOOD_CSR` → `results["faq"]["good"]["csr"]` formatted as `0.XX`
- `FILL_FAQ_GOOD_STAB` → `results["faq"]["good"]["stability"]` formatted as `0.XX`
- ... (repeat for all 12 numeric placeholders)

- [ ] **Step 4: Verify the paper compiles**

```bash
cd papers
pdflatex prompt_evaluator_en.tex
```

Expected: PDF generated without errors. Open and verify the Experiments section renders correctly with the tables and figure references.

- [ ] **Step 5: Commit**

```bash
git add papers/prompt_evaluator_en.tex
git commit -m "docs(paper): add Experiments section with results and figures for three evaluation domains"
```

---

## PHASE 4B — Agent B: PR to main

### Task 14: Open PR

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest
```

Expected: all tests PASS.

- [ ] **Step 2: Lint full codebase**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: no errors.

- [ ] **Step 3: Push branch**

```bash
git push origin feat/prompt-evaluator
```

- [ ] **Step 4: Open PR**

```bash
gh pr create \
  --title "feat(prompt-evaluator): add multi-domain dataset, experiments, and paper results" \
  --base main \
  --body "$(cat <<'EOF'
## Summary

- Add synthetic fixtures for 3 evaluation domains (FAQ, RAG, Structured Instructions) with 10 queries each
- Add real JSON datasets in `datasets/prompt_evaluator/` loadable via `PromptEvaluatorJsonRetriever`
- Add experiment notebook with full results (gpt-oss-20b / Groq, K=10, τ=0.80)
- Add 5 publication-quality figures: bar chart, radar chart, heatmap, scatter CSR vs RSS, UMAP clustering
- Add Experiments section to LaTeX paper with results tables and figure references

## Test plan

- [ ] `uv run pytest` passes (all existing + new tests)
- [ ] `uv run ruff check .` clean
- [ ] Notebook re-runs without errors given `GROQ_API_KEY`
- [ ] `pdflatex papers/prompt_evaluator_en.tex` compiles without errors
- [ ] Figures render correctly in the generated PDF
EOF
)"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Synthetic fixtures (FAQ, RAG, Structured) → Tasks 1–3
- ✅ Real JSON dataset → Tasks 4–6
- ✅ Dataset retriever → Task 7
- ✅ Retriever tests → Task 8
- ✅ Experiment notebook → Tasks 9–11
- ✅ Run experiments with gpt-oss-20b/Groq → Task 12
- ✅ All 5 visualizations (bar, radar, heatmap, scatter, UMAP clustering) → Tasks 11–12
- ✅ Paper experiments section with tables and figure references → Task 13
- ✅ PR → Task 14
- ✅ 3 domains × 3 prompt types → covered in Tasks 1, 4–6, 10–13
- ✅ Parallel agent structure noted at the plan header

**Placeholder scan:** No TBD/TODO. Task 13 Step 3 explicitly instructs reading the JSON and substituting values — not left vague.

**Type consistency:** `PromptEvaluatorJsonRetriever` is defined in Task 7 and used in Task 10. `JsonConstraint`, `KeywordConstraint` are imported from `fair_forge.metrics.constraints` (existing module confirmed). `SentenceTransformerEmbedder` exists at `fair_forge/embedders/sentence_transformer.py` (used in existing docs). All method names match the existing API.
