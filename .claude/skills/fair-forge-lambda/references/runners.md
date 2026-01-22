# Runners Module Pattern

Fair-Forge runners execute test batches against AI systems (agents, models, APIs) and collect responses for evaluation.

## Available Runners

| Runner | Extra | Description |
|--------|-------|-------------|
| AlquimiaRunner | `runners` | Execute tests against Alquimia AI agents |
| LLMRunner | (built-in) | Execute tests directly against LangChain LLMs |
| BaseRunner | (base) | Abstract base class for custom runners |

## Lambda Implementation

The Lambda supports two modes: Alquimia mode and LLM mode with dynamic connectors.

```python
"""run.py - Runners pattern with dual mode support"""
import asyncio
import importlib
import os
import time
from typing import Any

from fair_forge.runners import AlquimiaRunner
from fair_forge.schemas import Dataset, Batch


def create_llm_connector(connector_config: dict) -> Any:
    """Factory method to create LLM connector from dynamic class path."""
    class_path = connector_config.get("class_path")
    params = connector_config.get("params", {})

    if not class_path:
        raise ValueError("connector.class_path is required")

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if "api_key" not in params or not params["api_key"]:
        env_key = os.environ.get("LLM_API_KEY")
        if env_key:
            params["api_key"] = env_key

    return cls(**params)


class LLMRunner:
    """Runner that executes tests directly against LangChain LLMs."""

    def __init__(self, llm: Any):
        self.llm = llm

    async def run_batch(self, batch: Batch, session_id: str, context: str = "") -> tuple[Batch, bool, float]:
        start = time.time()
        try:
            prompt = f"Context:\n{context}\n\nQuestion: {batch.query}" if context else batch.query
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            execution_time = (time.time() - start) * 1000
            return batch.model_copy(update={"assistant": content}), True, execution_time
        except Exception as e:
            execution_time = (time.time() - start) * 1000
            return batch.model_copy(update={"assistant": f"[ERROR] {e}"}), False, execution_time

    async def run_dataset(self, dataset: Dataset) -> tuple[Dataset, dict]:
        updated_batches = []
        successes = failures = 0
        total_time = 0.0

        for batch in dataset.conversation:
            updated, success, time_ms = await self.run_batch(batch, dataset.session_id, dataset.context)
            updated_batches.append(updated)
            total_time += time_ms
            successes += 1 if success else 0
            failures += 0 if success else 1

        return dataset.model_copy(update={"conversation": updated_batches}), {
            "session_id": dataset.session_id,
            "total_batches": len(dataset.conversation),
            "successes": successes,
            "failures": failures,
            "total_execution_time_ms": total_time,
            "avg_batch_time_ms": total_time / len(dataset.conversation) if dataset.conversation else 0,
        }


def run(payload: dict) -> dict[str, Any]:
    """Run tests - supports Alquimia mode (config) or LLM mode (connector)."""
    return asyncio.get_event_loop().run_until_complete(_async_run(payload))


async def _async_run(payload: dict) -> dict[str, Any]:
    config = payload.get("config", {})
    connector_config = payload.get("connector", {})

    raw_datasets = payload.get("datasets", [])
    if not raw_datasets:
        return {"success": False, "error": "No datasets provided"}

    datasets = [Dataset.model_validate(data) for data in raw_datasets]

    # Choose runner based on payload
    if connector_config:
        llm = create_llm_connector(connector_config)
        runner = LLMRunner(llm)
    else:
        runner = AlquimiaRunner(
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
            agent_id=config.get("agent_id"),
            channel_id=config.get("channel_id"),
        )

    results, summaries = [], []
    for dataset in datasets:
        updated_dataset, summary = await runner.run_dataset(dataset)
        results.append(updated_dataset.model_dump())
        summaries.append(summary)

    return {"success": True, "datasets": results, "summaries": summaries, "total_datasets": len(results)}
```

## Request Payload Format

### LLM Mode (Direct LLM Testing)

```json
{
  "connector": {
    "class_path": "langchain_groq.chat_models.ChatGroq",
    "params": {
      "model": "qwen/qwen3-32b",
      "api_key": "your-api-key"
    }
  },
  "datasets": [
    {
      "session_id": "test-session-1",
      "assistant_id": "target-assistant",
      "language": "english",
      "context": "Optional context for all queries",
      "conversation": [
        {
          "qa_id": "test-1",
          "query": "What is the capital of France?",
          "assistant": "",
          "ground_truth_assistant": "Paris is the capital of France."
        }
      ]
    }
  ]
}
```

### Alquimia Mode

```json
{
  "datasets": [...],
  "config": {
    "base_url": "https://api.alquimia.ai",
    "api_key": "your-alquimia-api-key",
    "agent_id": "your-agent-id",
    "channel_id": "your-channel-id"
  }
}
```

## Supported Connectors (LLM Mode)

| Provider | class_path |
|----------|-----------|
| Groq | `langchain_groq.chat_models.ChatGroq` |
| OpenAI | `langchain_openai.chat_models.ChatOpenAI` |
| Google Gemini | `langchain_google_genai.chat_models.ChatGoogleGenerativeAI` |
| Ollama | `langchain_ollama.chat_models.ChatOllama` |

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
