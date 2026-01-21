"""Fair-Forge Generators Lambda business logic.

Generates synthetic test datasets from context documents using LLMs.
"""
import asyncio
import os
import tempfile
from typing import Any

from fair_forge.generators import BaseGenerator, LocalMarkdownLoader
from langchain_groq import ChatGroq


def run(payload: dict) -> dict[str, Any]:
    """Generate synthetic test datasets.

    Args:
        payload: Request JSON body with context and config

    Returns:
        dict: Generated datasets

    Example payload:
        {
            "context": "# Knowledge Base\n\nYour markdown content...",
            "config": {
                "api_key": "your-llm-api-key",
                "model": "qwen/qwen3-32b",
                "assistant_id": "my-assistant",
                "num_queries": 3,
                "language": "english",
                "conversation_mode": false,
                "max_chunk_size": 2000,
                "min_chunk_size": 200,
                "seed_examples": ["Example question 1?", "Example question 2?"]
            }
        }
    """
    return asyncio.get_event_loop().run_until_complete(_async_run(payload))


async def _async_run(payload: dict) -> dict[str, Any]:
    """Async generator implementation."""
    config = payload.get("config", {})
    api_key = config.get("api_key") or os.environ.get("LLM_API_KEY")

    if not api_key:
        return {"success": False, "error": "No API key provided"}

    # Initialize LLM
    model = ChatGroq(
        model=config.get("model", "qwen/qwen3-32b"),
        api_key=api_key,
        temperature=config.get("temperature", 0.7),
    )

    # Initialize generator
    generator = BaseGenerator(model=model, use_structured_output=True)

    # Create context loader
    loader = LocalMarkdownLoader(
        max_chunk_size=config.get("max_chunk_size", 2000),
        min_chunk_size=config.get("min_chunk_size", 200),
    )

    # Get context from payload
    context_content = payload.get("context", "")
    assistant_id = config.get("assistant_id", "test-assistant")

    if not context_content:
        return {"success": False, "error": "No context provided"}

    # Write context to temp file for loader
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(context_content)
        temp_path = f.name

    try:
        datasets = await generator.generate_dataset(
            context_loader=loader,
            source=temp_path,
            assistant_id=assistant_id,
            num_queries_per_chunk=config.get("num_queries", 3),
            language=config.get("language", "english"),
            seed_examples=config.get("seed_examples"),
            conversation_mode=config.get("conversation_mode", False),
        )
    finally:
        # Cleanup temp file
        os.unlink(temp_path)

    return {
        "success": True,
        "datasets": [d.model_dump() for d in datasets],
        "total_datasets": len(datasets),
        "total_batches": sum(len(d.conversation) for d in datasets),
    }
