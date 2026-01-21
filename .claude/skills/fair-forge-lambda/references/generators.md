# Generators Module Pattern

Fair-Forge generators create synthetic test datasets from context documents using LLMs.

## Available Generators

| Generator | Extra | Description |
|-----------|-------|-------------|
| BaseGenerator | `generators` | Uses any LangChain-compatible model |
| AlquimiaGenerator | `generators-alquimia` | Uses Alquimia agent API |

## Lambda Implementation

Generators are async and require context to be passed in the payload.

```python
"""run.py - Generators pattern"""
import asyncio
import os
from typing import Any

from fair_forge.generators import BaseGenerator, LocalMarkdownLoader
from langchain_groq import ChatGroq


def run(payload: dict) -> dict[str, Any]:
    """Generate synthetic test datasets."""
    return asyncio.get_event_loop().run_until_complete(_async_run(payload))


async def _async_run(payload: dict) -> dict[str, Any]:
    """Async generator implementation."""
    config = payload.get("config", {})
    api_key = config.get("api_key") or os.environ.get("LLM_API_KEY")

    # Initialize LLM
    model = ChatGroq(
        model=config.get("model", "qwen/qwen3-32b"),
        api_key=api_key,
        temperature=0.7,  # Higher for creative generation
    )

    # Initialize generator
    generator = BaseGenerator(model=model, use_structured_output=True)

    # Create context loader
    loader = LocalMarkdownLoader(
        max_chunk_size=config.get("max_chunk_size", 2000),
        min_chunk_size=config.get("min_chunk_size", 200),
    )

    # Get context from payload or file
    context_content = payload.get("context", "")
    assistant_id = config.get("assistant_id", "test-assistant")

    # For Lambda, we need to handle context differently
    # Option 1: Context passed directly in payload
    # Option 2: Context file path (if mounted/available)

    if context_content:
        # Write context to temp file for loader
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(context_content)
            temp_path = f.name

        datasets = await generator.generate_dataset(
            context_loader=loader,
            source=temp_path,
            assistant_id=assistant_id,
            num_queries_per_chunk=config.get("num_queries", 3),
            language=config.get("language", "english"),
            seed_examples=config.get("seed_examples"),
            conversation_mode=config.get("conversation_mode", False),
        )

        # Cleanup temp file
        import os as os_module
        os_module.unlink(temp_path)
    else:
        return {"success": False, "error": "No context provided"}

    return {
        "success": True,
        "datasets": [d.model_dump() for d in datasets],
        "total_datasets": len(datasets),
        "total_batches": sum(len(d.conversation) for d in datasets),
    }
```

## Request Payload Format

```json
{
  "context": "# Knowledge Base\n\nYour markdown content here...\n\n## Section 1\n\nContent for section 1...",
  "config": {
    "api_key": "your-llm-api-key",
    "model": "qwen/qwen3-32b",
    "assistant_id": "my-assistant",
    "num_queries": 3,
    "language": "english",
    "conversation_mode": false,
    "max_chunk_size": 2000,
    "min_chunk_size": 200,
    "seed_examples": [
      "What are the main features of X?",
      "How does Y compare to Z?"
    ]
  }
}
```

## Response Format

```json
{
  "success": true,
  "datasets": [
    {
      "session_id": "uuid-generated",
      "assistant_id": "my-assistant",
      "language": "english",
      "context": "Combined chunk content...",
      "conversation": [
        {
          "qa_id": "chunk-1_q1",
          "query": "Generated question about the content?",
          "assistant": "",
          "ground_truth_assistant": ""
        }
      ]
    }
  ],
  "total_datasets": 1,
  "total_batches": 3
}
```

## Generation Modes

### Independent Queries (default)
Each generated query is independent - unrelated questions per chunk.

```python
datasets = await generator.generate_dataset(
    ...,
    conversation_mode=False,  # Default
)
```

### Conversation Mode
Generates coherent multi-turn conversations where each turn builds on the previous.

```python
datasets = await generator.generate_dataset(
    ...,
    conversation_mode=True,
)
```

## Chunk Selection Strategies

### Sequential (default)
Process all chunks in order as a single dataset.

```python
from fair_forge.generators import SequentialStrategy

datasets = await generator.generate_dataset(
    ...,
    selection_strategy=SequentialStrategy(),
)
```

### Random Sampling
Randomly sample chunks multiple times for varied datasets.

```python
from fair_forge.generators import RandomSamplingStrategy

datasets = await generator.generate_dataset(
    ...,
    selection_strategy=RandomSamplingStrategy(
        num_samples=5,      # Generate 5 datasets
        chunks_per_sample=3 # 3 chunks per dataset
    ),
)
```

## Using Alquimia Generator

For Alquimia-specific generation:

```python
from fair_forge.generators import create_alquimia_generator

generator = create_alquimia_generator(
    base_url=config.get("base_url"),
    api_key=config.get("api_key"),
    agent_id=config.get("agent_id"),
    channel_id=config.get("channel_id"),
)

datasets = await generator.generate_dataset(...)
```
