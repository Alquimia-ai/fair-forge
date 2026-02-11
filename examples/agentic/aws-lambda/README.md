# Fair-Forge Agentic Lambda

AWS Lambda function for evaluating AI agent responses with pass@K and tool correctness metrics.

## Description

This Lambda function uses the Fair-Forge `Agentic` metric to evaluate multiple agent responses (K responses) for the same question. It provides three types of evaluations:

1. **pass@K**: At least one of the K responses is correct (similarity score >= threshold)
2. **pass^K**: All K responses are correct
3. **Tool Correctness**: Evaluates proper tool usage across four dimensions:
   - **Selection** (25%): Were the correct tools chosen?
   - **Parameters** (25%): Were the correct parameters passed?
   - **Sequence** (25%): Were tools used in the correct order? (if required)
   - **Utilization** (25%): Were tool results properly used in the final answer?

The metric uses an LLM judge to evaluate answer correctness and direct dictionary comparison for tool correctness.

## Invoke URL

```
<DEPLOY_TO_GET_URL>
```

## Supported LLM Providers

| Provider | class_path |
|----------|-----------|
| Groq | `langchain_groq.chat_models.ChatGroq` |
| OpenAI | `langchain_openai.chat_models.ChatOpenAI` |
| Google Gemini | `langchain_google_genai.chat_models.ChatGoogleGenerativeAI` |
| Ollama | `langchain_ollama.chat_models.ChatOllama` |

## Test Example

### Using Groq

```bash
curl -s -X POST "<INVOKE_URL>/run" \
  -H "Content-Type: application/json" \
  -d '{
    "connector": {
      "class_path": "langchain_groq.chat_models.ChatGroq",
      "params": {
        "model": "llama-3.3-70b-versatile",
        "api_key": "your-groq-api-key",
        "temperature": 0.0
      }
    },
    "datasets": [
      {
        "session_id": "eval_session",
        "assistant_id": "agent_response_1",
        "language": "english",
        "context": "You are a helpful calculator agent.",
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
        "context": "You are a helpful calculator agent.",
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
      "verbose": false
    }
  }' | jq
```

### Using OpenAI

```bash
curl -s -X POST "<INVOKE_URL>/run" \
  -H "Content-Type: application/json" \
  -d '{
    "connector": {
      "class_path": "langchain_openai.chat_models.ChatOpenAI",
      "params": {
        "model": "gpt-4o-mini",
        "api_key": "your-openai-api-key"
      }
    },
    "datasets": [...],
    "config": {
      "threshold": 0.7
    }
  }' | jq
```

## Request Format

```json
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
      "context": "System context",
      "conversation": [
        {
          "qa_id": "q1",
          "query": "User question",
          "assistant": "Agent response",
          "ground_truth_assistant": "Expected response",
          "agentic": {
            "tools_used": [
              {
                "tool_name": "tool_name",
                "parameters": {},
                "result": "tool_result",
                "step": 1
              }
            ],
            "final_answer_uses_tools": true
          },
          "ground_truth_agentic": {
            "expected_tools": [
              {
                "tool_name": "tool_name",
                "parameters": {},
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
```

### Connector Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `connector.class_path` | string | Yes | Full class path of the LangChain chat model |
| `connector.params` | object | Yes | Parameters passed to the chat model constructor |
| `connector.params.model` | string | Yes | Model name/identifier |
| `connector.params.api_key` | string | Yes | API key for the LLM provider |
| `connector.params.temperature` | float | No | Sampling temperature (default varies by provider) |

### Agentic-Specific Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `datasets` | array | Yes | List of datasets with K responses for the same qa_id |
| `datasets[].session_id` | string | Yes | Session identifier (same across K responses) |
| `datasets[].assistant_id` | string | Yes | Unique identifier for each response (K different IDs) |
| `datasets[].conversation[].qa_id` | string | Yes | Question identifier (same across K responses) |
| `datasets[].conversation[].agentic` | object | No | Agent's actual tool usage |
| `datasets[].conversation[].agentic.tools_used` | array | No | List of tools used by the agent |
| `datasets[].conversation[].agentic.final_answer_uses_tools` | boolean | No | Whether tool results are used in final answer |
| `datasets[].conversation[].ground_truth_agentic` | object | No | Expected tool usage |
| `datasets[].conversation[].ground_truth_agentic.expected_tools` | array | No | List of expected tools |
| `datasets[].conversation[].ground_truth_agentic.tool_sequence_matters` | boolean | No | Whether tool order matters (default: false) |
| `config.threshold` | float | No | Answer correctness threshold (0.0-1.0, default: 0.7) |
| `config.tool_threshold` | float | No | Tool correctness threshold (0.0-1.0, default: 0.75) |
| `config.tool_weights` | object | No | Weights for tool correctness components (default: 0.25 each) |
| `config.use_structured_output` | boolean | No | Use LangChain structured output (default: false) |
| `config.verbose` | boolean | No | Enable verbose logging (default: false) |

## Response Format

```json
{
  "success": true,
  "metrics": [
    {
      "qa_id": "q1",
      "k": 3,
      "threshold": 0.7,
      "pass_at_k": true,
      "pass_pow_k": false,
      "correctness_scores": [0.85, 0.92, 0.65],
      "correct_indices": [0, 1],
      "tool_correctness": {
        "tool_selection_correct": 1.0,
        "parameter_accuracy": 1.0,
        "sequence_correct": 1.0,
        "result_utilization": 1.0,
        "overall_correctness": 1.0,
        "is_correct": true,
        "reasoning": "All tools used correctly"
      }
    }
  ],
  "count": 1,
  "summary": {
    "total_qa_ids": 1,
    "pass_at_k_count": 1,
    "pass_pow_k_count": 0
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether evaluation succeeded |
| `metrics` | array | List of evaluation results (one per qa_id) |
| `metrics[].qa_id` | string | Question identifier |
| `metrics[].k` | integer | Number of responses evaluated |
| `metrics[].threshold` | float | Answer correctness threshold used |
| `metrics[].pass_at_k` | boolean | At least one response is correct |
| `metrics[].pass_pow_k` | boolean | All responses are correct |
| `metrics[].correctness_scores` | array | Correctness scores for each response (0.0-1.0) |
| `metrics[].correct_indices` | array | Indices of correct responses (score >= threshold) |
| `metrics[].tool_correctness` | object | Tool usage evaluation (if ground_truth_agentic provided) |
| `metrics[].tool_correctness.tool_selection_correct` | float | Tool selection score (0.0-1.0) |
| `metrics[].tool_correctness.parameter_accuracy` | float | Parameter accuracy score (0.0-1.0) |
| `metrics[].tool_correctness.sequence_correct` | float | Sequence correctness score (0.0-1.0) |
| `metrics[].tool_correctness.result_utilization` | float | Result utilization score (0.0-1.0) |
| `metrics[].tool_correctness.overall_correctness` | float | Weighted average of all components |
| `metrics[].tool_correctness.is_correct` | boolean | Overall correctness >= tool_threshold |
| `metrics[].tool_correctness.reasoning` | string | Optional explanation |
| `count` | integer | Total number of qa_ids evaluated |
| `summary.total_qa_ids` | integer | Total questions evaluated |
| `summary.pass_at_k_count` | integer | Number of questions with pass@K=true |
| `summary.pass_pow_k_count` | integer | Number of questions with pass^K=true |

## Error Responses

```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `No connector configuration provided` | Missing `connector` field | Add connector with class_path and params |
| `connector.class_path is required` | Missing class_path in connector | Specify the LLM class path |
| `Failed to create LLM connector` | Invalid class_path or params | Check class_path spelling and required params |
| `No datasets provided` | Missing or empty `datasets` | Add at least one dataset |
| `No qa_ids found in datasets` | Datasets have no conversations | Add conversations with qa_id |
| `Agentic evaluation failed` | Evaluation error | Check logs for details |

## View Logs

```bash
aws logs tail "/aws/lambda/fair-forge-agentic" --follow --region us-east-1
```

## Deployment Commands

```bash
# Deploy (from examples/agentic/aws-lambda/)
./scripts/deploy.sh agentic us-east-1

# Update (rebuild and redeploy)
./scripts/update.sh agentic us-east-1

# Cleanup (remove all AWS resources)
./scripts/cleanup.sh agentic us-east-1
```

## Environment Variables

The Lambda function supports the following environment variables:

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | Fallback API key if not provided in request |

## Understanding pass@K vs pass^K

- **pass@K**: Measures if *at least one* of K responses is correct. This is useful for evaluating whether an agent can produce a correct answer with multiple attempts.

- **pass^K**: Measures if *all* K responses are correct. This is a stricter metric that evaluates consistency and reliability.

## Tool Correctness Evaluation

Tool correctness is evaluated across four dimensions:

1. **Selection (25%)**: Did the agent choose the right tools?
2. **Parameters (25%)**: Did the agent pass correct parameters to the tools?
3. **Sequence (25%)**: Did the agent use tools in the correct order? (only matters if `tool_sequence_matters=true`)
4. **Utilization (25%)**: Did the agent use tool results in the final answer?

The overall score is a weighted average. You can customize weights in `config.tool_weights`.

## Installation

```bash
# Install Fair-Forge with agentic support
pip install alquimia-fair-forge[agentic]

# Install LLM provider (choose one or more)
pip install langchain-groq        # For Groq
pip install langchain-openai      # For OpenAI
pip install langchain-google-genai # For Google Gemini
pip install langchain-ollama      # For Ollama
```

---

Built with [Fair-Forge](https://github.com/Alquimia-ai/fair-forge) - AI Evaluation Framework
