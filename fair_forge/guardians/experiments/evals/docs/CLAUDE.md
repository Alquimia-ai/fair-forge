# CLAUDE.md - Guardians Evaluations

This file provides guidance for working with the evaluation notebooks in `/fair_forge/guardians/experiments/evals/`.

## Overview

This directory contains comprehensive evaluations of multiple AI safety guardrail models and architectures. The evaluations test models against a curated dataset of safe/unsafe content across various risk categories.

## Test Data Structure

Located in `data/test_cases.json`:
- Each test case has: `text`, `category`, `expected` (SAFE/UNSAFE/INJECTION)
- Categories include: benign, jailbreak, profanity, abuse, violence, sexual, social_bias, pii, copyright, self_harm, illegal, system_info, mixed

## Notebooks

### Tests.ipynb

Primary evaluation notebook testing 5 different guardrail approaches.

#### Models Evaluated

1. **Qwen3Guard-Gen-0.6B** (CPU-compatible)
   - Multi-purpose model
   - Returns safety label (Safe/Unsafe/Controversial) + category classification
   - Function: `evaluar_qwen(prompt)`
   - Output: safety label, category, confidence scores, latency

2. **Pipeline Multicapa** (~1B params, 5 specialized SLMs)
   - Layer 1: Llama-Prompt-Guard-2-86M (jailbreak/injection)
   - Layer 2: Granite-Guardian-HAP-125M (hate/abuse/profanity)
   - Layer 3: Detoxify Multilingual ~560M (7 toxicity categories)
   - Layer 4: Suicide-BERT ~110M (self-harm/suicide)
   - Layer 5: DeBERTa-PII 184M (54 PII categories)
   - Function: `cascade_classify(text)` - stops at first unsafe layer
   - Individual layer functions: `classify_llama()`, `classify_granite()`, `classify_detoxify()`, `classify_suicide()`, `classify_pii()`

3. **GPT-OSS-SAFEGUARD-20B** (requires GPU)
   - OpenAI's open-source safeguard model
   - Function: `classify_gpt_oss(text)`
   - Returns binary safe/unsafe classification

4. **Llama-Guard-3-8B** (via HuggingFace API)
   - Meta's guardrail model
   - Uses HF Endpoint: `xtq3z5luegoyrx63.us-east-1.aws.endpoints.huggingface.cloud`
   - Function: `classify_llama_guard(text)`
   - Returns safe/unsafe with category codes (S1, S2, etc.)
   - Requires `HF_TOKEN` environment variable

5. **Granite-Guardian-3.1-2B** (via HuggingFace API)
   - IBM's multi-category risk detection model
   - Uses HF Endpoint: `elhdnq2rylqmcfrc.us-east-1.aws.endpoints.huggingface.cloud`
   - Function: `classify_granite_guardian(text, risk_name)`
   - Risk categories: harm, jailbreak, profanity, violence, sexual_content, social_bias, unethical_behavior
   - Category mapping in `CATEGORY_TO_RISK` dict
   - Requires `HF_TOKEN` environment variable

#### Key Functions

```python
# Qwen3Guard
evaluar_qwen(prompt) -> dict
  Returns: safety, safety_prob, category, category_prob, time_ms

# Pipeline Multicapa
cascade_classify(text) -> dict
  Returns: final_label, reason, stopped_at, details, latency_ms

# Individual pipeline layers
classify_llama(text) -> dict  # Layer 1: Jailbreak/Injection
classify_granite(text) -> dict  # Layer 2: Hate/Abuse
classify_detoxify(text, threshold=0.5) -> dict  # Layer 3: Toxicity
classify_suicide(text) -> dict  # Layer 4: Self-harm
classify_pii(text) -> dict  # Layer 5: PII detection

# GPT-OSS
classify_gpt_oss(text) -> dict
  Returns: label, response, latency_ms

# Llama-Guard (API)
classify_llama_guard(text) -> dict
  Returns: label, categories, response, latency_ms

# Granite Guardian (API)
classify_granite_guardian(text, risk_name) -> dict
  Returns: label, is_risky, risk_name, response, latency_ms
```

#### Evaluation Metrics

All models are evaluated on:
- **Accuracy**: % of correct predictions (SAFE vs UNSAFE)
- **Latency**: Response time in milliseconds
- **Per-category breakdown**: Accuracy split by test case category
- **Layer analysis** (Pipeline only): Which layer detected unsafe content

#### Configuration Requirements

- **GPU Models**: Qwen3Guard, GPT-OSS-SAFEGUARD, Pipeline models (can run on CPU but slower)
- **API Models**: Llama-Guard-3, Granite-Guardian-3.1
  - Require `HF_TOKEN` environment variable for HuggingFace endpoints
  - Use OpenAI-compatible client interface

#### Results Format

All tests generate pandas DataFrames with columns:
- `id`: Test case number
- `text`: Truncated input text
- `category`: Test case category
- `expected`: Ground truth label
- `predicted`: Model prediction
- `latency_ms` or `ms`: Response time
- `correct` or `ok`: ✅ if correct, ❌ if wrong
- Model-specific columns (e.g., `stopped_at` for pipeline, `risk_eval` for Granite)

## Usage Patterns

### Running Individual Model Tests

```python
# Load test cases
with open('data/test_cases.json') as f:
    test_cases = json.load(f)

# Test single model
for case in test_cases:
    result = evaluar_qwen(case["text"])  # or other classify function
    # Process result...
```

### Comparing Models

Run all test cells sequentially to generate comparison tables. Each model section produces:
1. Detailed results table
2. Overall accuracy
3. Average latency
4. Per-category breakdown (where applicable)

### Adding New Test Cases

Add to `data/test_cases.json`:
```json
{
  "text": "Your test prompt here",
  "category": "category_name",
  "expected": "SAFE" or "UNSAFE" or "INJECTION"
}
```

### Environment Setup

```bash
# For API-based models
export HF_TOKEN="your_huggingface_token"

# Install dependencies
pip install transformers torch detoxify pandas
```

## Notes

- INJECTION cases are treated as UNSAFE for evaluation purposes
- Pipeline cascade stops at first unsafe detection for efficiency
- API-based models require active HuggingFace endpoints
- GPU memory requirements vary by model size (0.6B to 20B params)
- Detoxify returns 7 categories: toxicity, severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit

---

### studio.ipynb

Integration testing notebook for deploying guardrail models in Alquimia runtime.

#### Purpose

Test different LangChain connector configurations for HuggingFace endpoints before deploying to production.

#### Llama-Prompt-Guard-2-86M Configuration Options

The notebook provides 3 different configuration approaches for `meta-llama/Llama-Prompt-Guard-2-86M`:

**OPTION 1: ChatOpenAI (for TGI-compatible endpoints)**
```json
{
  "class_path": "langchain_openai.ChatOpenAI",
  "params": {
    "base_url": "https://your-endpoint.huggingface.cloud/v1",
    "model": "tgi",
    "temperature": 0,
    "max_tokens": 10,
    "api_key": "dummy"
  }
}
```
- Use when endpoint exposes OpenAI-compatible API
- Requires `/v1` suffix in URL
- Works with Text Generation Inference (TGI) deployments

**OPTION 2: HuggingFaceEndpoint (langchain_huggingface)**
```json
{
  "class_path": "langchain_huggingface.llms.HuggingFaceEndpoint",
  "params": {
    "endpoint_url": "https://your-endpoint.huggingface.cloud",
    "task": "text-generation",
    "max_new_tokens": 5,
    "temperature": 0.01,
    "top_k": 1
  }
}
```
- Standard approach for HuggingFace Inference Endpoints
- Most commonly used configuration

**OPTION 3: HuggingFaceEndpoint (langchain_community)**
```json
{
  "class_path": "langchain_community.llms.HuggingFaceEndpoint",
  "params": {
    "endpoint_url": "https://your-endpoint.huggingface.cloud",
    "max_new_tokens": 5,
    "temperature": 0.0,
    "return_full_text": false
  }
}
```
- Legacy implementation with broader compatibility
- Useful when newer package versions have issues

#### Testing Workflow

1. **Run direct endpoint test** - Execute the HTTP testing cell to verify endpoint accessibility and response format
2. **Identify working payload format** - Note which request format (simple inputs vs. parameterized) succeeds
3. **Select matching LangChain connector** - Choose Option 1/2/3 based on endpoint behavior
4. **Copy configuration to Alquimia** - Deploy the working config JSON to production

#### Key Implementation Notes

- **Model Output**: Llama-Prompt-Guard-2-86M returns "BENIGN" or "MALICIOUS" as plain text
- **System Prompt**: Use minimal prompt `{{query}}` - model is pre-trained for classification
- **Token Limit**: Set `max_new_tokens: 5` since output is single-word
- **No API Key**: Public test endpoints use `api_key: "dummy"` or empty string
- **Memory Strategy**: Use `"dory"` (no memory) for stateless classification tasks

#### Agent Structure for Alquimia

```json
{
  "assistant_id": "prompt-guard",
  "dante": {
    "profile": {
      "short_term_memory_strategy": {"memory_strategy_id": "dory"},
      "system_prompt": "{{query}}"
    },
    "config": {
      "type": "generative-dynamic",
      "connector": {
        "class_path": "...",
        "params": {...}
      }
    }
  }
}
```

#### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing langchain package | Install `langchain-huggingface` or `langchain-community` |
| `401 Unauthorized` | API key required | Add `huggingfacehub_api_token` param or use public endpoint |
| `422 Validation Error` | Wrong parameter names | Check class_path matches param schema (ChatOpenAI vs HuggingFaceEndpoint) |
| Response parsing error | Endpoint returns classification dict | Use ChatOpenAI option for structured outputs |
| Timeout errors | Cold start on endpoint | Retry or increase timeout in connector config |
