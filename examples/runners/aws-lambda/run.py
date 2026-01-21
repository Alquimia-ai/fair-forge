"""Fair-Forge Runners Lambda business logic.

Executes test batches against AI systems and collects responses for evaluation.
"""
import asyncio
import os
from typing import Any

from fair_forge.runners import AlquimiaRunner
from fair_forge.schemas import Dataset


def run(payload: dict) -> dict[str, Any]:
    """Run tests against an AI system.

    Args:
        payload: Request JSON body with datasets and config

    Returns:
        dict: Test results with responses from the AI system

    Example payload:
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
                            "ground_truth_assistant": "Paris is the capital."
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
    """
    return asyncio.get_event_loop().run_until_complete(_async_run(payload))


async def _async_run(payload: dict) -> dict[str, Any]:
    """Async runner implementation."""
    config = payload.get("config", {})

    base_url = config.get("base_url") or os.environ.get("ALQUIMIA_BASE_URL")
    api_key = config.get("api_key") or os.environ.get("ALQUIMIA_API_KEY")
    agent_id = config.get("agent_id")
    channel_id = config.get("channel_id")

    if not base_url:
        return {"success": False, "error": "No base_url provided"}
    if not api_key:
        return {"success": False, "error": "No api_key provided"}
    if not agent_id:
        return {"success": False, "error": "No agent_id provided"}

    # Initialize runner
    runner = AlquimiaRunner(
        base_url=base_url,
        api_key=api_key,
        agent_id=agent_id,
        channel_id=channel_id,
        api_version=config.get("api_version", ""),
    )

    # Load datasets from payload
    raw_datasets = payload.get("datasets", [])
    if not raw_datasets:
        return {"success": False, "error": "No datasets provided"}

    datasets = []
    for data in raw_datasets:
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
