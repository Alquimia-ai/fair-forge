# Metrics Module Pattern

Fair-Forge metrics evaluate AI assistant responses. Each metric inherits from `FairForge` base class and uses a `Retriever` to load datasets.

## Available Metrics

| Metric | Extra | Description |
|--------|-------|-------------|
| BestOf | `bestof` | Tournament-style comparison of multiple assistants |
| Toxicity | `toxicity` | Detect toxic language in responses |
| Bias | `bias` | Detect bias using guardian models |
| Context | `context` | Evaluate context relevance |
| Conversational | `conversational` | Evaluate conversation quality |
| Humanity | `humanity` | Measure human-likeness of responses |

## Lambda Implementation

```python
"""run.py - Metrics pattern"""
import os
from typing import Any

from fair_forge import Retriever
from fair_forge.schemas import Dataset, Batch

# Import your metric
from fair_forge.metrics.best_of import BestOf
# Or: from fair_forge.metrics.toxicity import Toxicity
# Or: from fair_forge.metrics.bias import Bias
# etc.

# Import LLM provider (for LLM-based metrics)
from langchain_groq import ChatGroq


class PayloadRetriever(Retriever):
    """Load datasets from Lambda payload."""

    def __init__(self, payload: dict):
        self.payload = payload

    def load_dataset(self) -> list[Dataset]:
        datasets = []
        for data in self.payload.get("datasets", []):
            datasets.append(Dataset.model_validate(data))
        return datasets


def run(payload: dict) -> dict[str, Any]:
    """Run metric on payload datasets."""
    config = payload.get("config", {})
    api_key = config.get("api_key") or os.environ.get("LLM_API_KEY")

    # Initialize LLM (for LLM-based metrics like BestOf, Context, Conversational)
    model = ChatGroq(
        model=config.get("model", "qwen/qwen3-32b"),
        api_key=api_key,
        temperature=0.0,
    )

    # Create retriever
    retriever = PayloadRetriever(payload)

    # Run metric
    metrics = BestOf.run(
        lambda: retriever,
        model=model,
        use_structured_output=True,
        criteria=config.get("criteria", "Overall quality"),
        verbose=config.get("verbose", False),
    )

    return {
        "success": True,
        "metrics": [m.model_dump() for m in metrics],
        "count": len(metrics),
    }
```

## Request Payload Format

```json
{
  "datasets": [
    {
      "session_id": "session-1",
      "assistant_id": "assistant-a",
      "language": "english",
      "context": "Optional context string",
      "conversation": [
        {
          "qa_id": "qa-1",
          "query": "User question",
          "assistant": "Assistant response",
          "ground_truth_assistant": "Expected response (optional)"
        }
      ]
    }
  ],
  "config": {
    "api_key": "your-llm-api-key",
    "model": "qwen/qwen3-32b",
    "criteria": "Evaluation criteria description",
    "verbose": false
  }
}
```

## Metric-Specific Notes

### BestOf
- Requires multiple datasets with different `assistant_id` values
- Runs tournament-style elimination to find best assistant
- Returns winner and contest details

### Toxicity
- Uses statistical analysis (Frequentist or Bayesian modes)
- Does not require LLM - uses built-in toxicity detection
- Install with `toxicity` extra for full dependencies

### Bias
- Uses guardian models (IBM Granite, Llama Guard)
- Requires `bias` extra and torch

### Context / Conversational
- LLM-based evaluation
- Uses Judge class internally
