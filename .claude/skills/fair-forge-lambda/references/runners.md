# Runners Module Pattern

Fair-Forge runners execute test batches against AI systems (agents, models, APIs) and collect responses for evaluation.

## Available Runners

| Runner | Extra | Description |
|--------|-------|-------------|
| AlquimiaRunner | `runners` | Execute tests against Alquimia AI agents |
| BaseRunner | (base) | Abstract base class for custom runners |

## Lambda Implementation

Runners are async, so Lambda needs an event loop wrapper.

```python
"""run.py - Runners pattern"""
import asyncio
import os
from typing import Any

from fair_forge.runners import AlquimiaRunner
from fair_forge.schemas import Dataset, Batch


def run(payload: dict) -> dict[str, Any]:
    """Run tests against an AI system."""
    return asyncio.get_event_loop().run_until_complete(_async_run(payload))


async def _async_run(payload: dict) -> dict[str, Any]:
    """Async runner implementation."""
    config = payload.get("config", {})

    # Initialize runner
    runner = AlquimiaRunner(
        base_url=config.get("base_url") or os.environ.get("ALQUIMIA_BASE_URL"),
        api_key=config.get("api_key") or os.environ.get("ALQUIMIA_API_KEY"),
        agent_id=config.get("agent_id"),
        channel_id=config.get("channel_id"),
        api_version=config.get("api_version", ""),
    )

    # Load datasets from payload
    datasets = []
    for data in payload.get("datasets", []):
        datasets.append(Dataset.model_validate(data))

    # Run all datasets
    results = []
    summaries = []

    for dataset in datasets:
        updated_dataset, summary = await runner.run_dataset(dataset)
        results.append(updated_dataset.model_dump())
        summaries.append(summary)

    return {
        "success": True,
        "datasets": results,
        "summaries": summaries,
        "total_datasets": len(results),
    }
```

## Request Payload Format

```json
{
  "datasets": [
    {
      "session_id": "test-session-1",
      "assistant_id": "target-assistant",
      "language": "english",
      "context": "",
      "conversation": [
        {
          "qa_id": "test-1",
          "query": "What is the capital of France?",
          "assistant": "",
          "ground_truth_assistant": "Paris is the capital of France."
        }
      ]
    }
  ],
  "config": {
    "base_url": "https://api.alquimia.ai",
    "api_key": "your-alquimia-api-key",
    "agent_id": "your-agent-id",
    "channel_id": "your-channel-id"
  }
}
```

## Response Format

```json
{
  "success": true,
  "datasets": [
    {
      "session_id": "test-session-1",
      "assistant_id": "target-assistant",
      "conversation": [
        {
          "qa_id": "test-1",
          "query": "What is the capital of France?",
          "assistant": "The capital of France is Paris.",
          "ground_truth_assistant": "Paris is the capital of France."
        }
      ]
    }
  ],
  "summaries": [
    {
      "session_id": "test-session-1",
      "total_batches": 1,
      "successes": 1,
      "failures": 0,
      "total_execution_time_ms": 1234.5,
      "avg_batch_time_ms": 1234.5
    }
  ],
  "total_datasets": 1
}
```

## Custom Runner Implementation

Extend `BaseRunner` for custom AI systems:

```python
from fair_forge.schemas.runner import BaseRunner
from fair_forge.schemas import Batch, Dataset

class MyCustomRunner(BaseRunner):
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint

    async def run_batch(self, batch: Batch, session_id: str, **kwargs) -> tuple[Batch, bool, float]:
        """Execute single test case."""
        import time
        start = time.time()

        try:
            # Call your AI system here
            response = await self._call_my_api(batch.query, session_id)
            execution_time = (time.time() - start) * 1000

            updated_batch = batch.model_copy(update={"assistant": response})
            return updated_batch, True, execution_time

        except Exception as e:
            execution_time = (time.time() - start) * 1000
            updated_batch = batch.model_copy(update={"assistant": f"[ERROR] {e}"})
            return updated_batch, False, execution_time

    async def run_dataset(self, dataset: Dataset, **kwargs) -> tuple[Dataset, dict]:
        """Execute all batches in dataset."""
        updated_batches = []
        successes = failures = 0
        total_time = 0.0

        for batch in dataset.conversation:
            updated, success, time_ms = await self.run_batch(batch, dataset.session_id)
            updated_batches.append(updated)
            total_time += time_ms
            if success:
                successes += 1
            else:
                failures += 1

        updated_dataset = dataset.model_copy(update={"conversation": updated_batches})

        summary = {
            "session_id": dataset.session_id,
            "total_batches": len(dataset.conversation),
            "successes": successes,
            "failures": failures,
            "total_execution_time_ms": total_time,
            "avg_batch_time_ms": total_time / len(dataset.conversation) if dataset.conversation else 0,
        }

        return updated_dataset, summary
```
