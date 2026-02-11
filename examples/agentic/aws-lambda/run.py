"""Fair-Forge Agentic Lambda business logic.

Evaluates agent responses using pass@K and tool correctness metrics.
"""

import importlib
import os
from typing import Any

from fair_forge.core import Retriever
from fair_forge.metrics.agentic import Agentic
from fair_forge.schemas import Dataset


def create_llm_connector(connector_config: dict) -> Any:
    """Factory method to create LLM connector from dynamic class path.

    Args:
        connector_config: Configuration dict with:
            - class_path: Full class path (e.g., "langchain_groq.chat_models.ChatGroq")
            - params: Dict of parameters to pass to the class constructor

    Returns:
        Instantiated LLM connector

    Supported connectors:
        - langchain_groq.chat_models.ChatGroq
        - langchain_openai.chat_models.ChatOpenAI
        - langchain_google_genai.chat_models.ChatGoogleGenerativeAI
        - langchain_ollama.chat_models.ChatOllama
    """
    class_path = connector_config.get("class_path")
    params = connector_config.get("params", {})

    if not class_path:
        raise ValueError("connector.class_path is required")

    # Dynamic import
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Support environment variable fallback for api_key
    if "api_key" not in params or not params["api_key"]:
        env_key = os.environ.get("LLM_API_KEY")
        if env_key:
            params["api_key"] = env_key

    return cls(**params)


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
    """Run Agentic metric on payload datasets.

    Args:
        payload: Request JSON body with connector, datasets and config

    Returns:
        dict: Agentic evaluation results with pass@K, pass^K, and tool correctness

    Example payload:
        {
            "connector": {
                "class_path": "langchain_groq.chat_models.ChatGroq",
                "params": {
                    "model": "llama-3.3-70b-versatile",
                    "api_key": "your-api-key",
                    "temperature": 0.0
                }
            },
            "datasets": [
                {
                    "session_id": "eval_session",
                    "assistant_id": "agent_response_1",
                    "language": "english",
                    "context": "System context...",
                    "conversation": [
                        {
                            "qa_id": "q1",
                            "query": "What is 5 + 3?",
                            "assistant": "The result is 8.",
                            "ground_truth_assistant": "5 + 3 equals 8",
                            "agentic": {
                                "tools_used": [
                                    {
                                        "tool_name": "calculator",
                                        "parameters": {"operation": "add", "a": 5, "b": 3},
                                        "result": 8,
                                        "step": 1
                                    }
                                ],
                                "final_answer_uses_tools": true
                            },
                            "ground_truth_agentic": {
                                "expected_tools": [
                                    {
                                        "tool_name": "calculator",
                                        "parameters": {"operation": "add", "a": 5, "b": 3},
                                        "step": 1
                                    }
                                ],
                                "tool_sequence_matters": false
                            }
                        }
                    ]
                },
                {
                    "session_id": "eval_session",
                    "assistant_id": "agent_response_2",
                    "language": "english",
                    "context": "System context...",
                    "conversation": [
                        {
                            "qa_id": "q1",
                            "query": "What is 5 + 3?",
                            "assistant": "5 + 3 is 8.",
                            "ground_truth_assistant": "5 + 3 equals 8",
                            "agentic": {
                                "tools_used": [
                                    {
                                        "tool_name": "calculator",
                                        "parameters": {"operation": "add", "a": 5, "b": 3},
                                        "result": 8,
                                        "step": 1
                                    }
                                ],
                                "final_answer_uses_tools": true
                            },
                            "ground_truth_agentic": {
                                "expected_tools": [
                                    {
                                        "tool_name": "calculator",
                                        "parameters": {"operation": "add", "a": 5, "b": 3},
                                        "step": 1
                                    }
                                ],
                                "tool_sequence_matters": false
                            }
                        }
                    ]
                }
            ],
            "config": {
                "threshold": 0.7,
                "tool_threshold": 0.75,
                "tool_weights": {
                    "selection": 0.25,
                    "parameters": 0.25,
                    "sequence": 0.25,
                    "utilization": 0.25
                },
                "use_structured_output": false,
                "verbose": false
            }
        }
    """
    # Get connector config
    connector_config = payload.get("connector", {})
    if not connector_config:
        return {"success": False, "error": "No connector configuration provided"}

    try:
        model = create_llm_connector(connector_config)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Failed to create LLM connector: {e}"}

    # Validate datasets
    datasets = payload.get("datasets", [])
    if not datasets:
        return {"success": False, "error": "No datasets provided"}

    # For Agentic metric, we need multiple responses for the same qa_id (K responses)
    # Each dataset should have the same qa_id but different assistant_id
    qa_ids = set()
    for d in datasets:
        for conv in d.get("conversation", []):
            qa_ids.add(conv.get("qa_id"))

    if not qa_ids:
        return {"success": False, "error": "No qa_ids found in datasets"}

    config = payload.get("config", {})

    # Run Agentic metric
    try:
        metrics = Agentic.run(
            lambda: PayloadRetriever(payload),
            model=model,
            threshold=config.get("threshold", 0.7),
            tool_threshold=config.get("tool_threshold", 0.75),
            tool_weights=config.get("tool_weights"),
            use_structured_output=config.get("use_structured_output", False),
            verbose=config.get("verbose", False),
        )
    except Exception as e:
        return {"success": False, "error": f"Agentic evaluation failed: {e}"}

    if not metrics:
        return {"success": False, "error": "No metrics produced"}

    # Extract results
    results = []
    for metric in metrics:
        result = {
            "qa_id": metric.qa_id,
            "k": metric.k,
            "threshold": metric.threshold,
            "pass_at_k": metric.pass_at_k,
            "pass_pow_k": metric.pass_pow_k,
            "correctness_scores": metric.correctness_scores,
            "correct_indices": metric.correct_indices,
        }

        if metric.tool_correctness:
            result["tool_correctness"] = {
                "tool_selection_correct": metric.tool_correctness.tool_selection_correct,
                "parameter_accuracy": metric.tool_correctness.parameter_accuracy,
                "sequence_correct": metric.tool_correctness.sequence_correct,
                "result_utilization": metric.tool_correctness.result_utilization,
                "overall_correctness": metric.tool_correctness.overall_correctness,
                "is_correct": metric.tool_correctness.is_correct,
                "reasoning": metric.tool_correctness.reasoning,
            }

        results.append(result)

    return {
        "success": True,
        "metrics": results,
        "count": len(results),
        "summary": {
            "total_qa_ids": len(results),
            "pass_at_k_count": sum(1 for m in metrics if m.pass_at_k),
            "pass_pow_k_count": sum(1 for m in metrics if m.pass_pow_k),
        },
    }
