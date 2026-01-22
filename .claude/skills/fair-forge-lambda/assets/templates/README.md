# Fair-Forge {MODULE_NAME} Lambda

AWS Lambda function for {MODULE_DESCRIPTION}.

## Description

{DETAILED_DESCRIPTION}

## Invoke URL

```
{INVOKE_URL}
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
curl -s -X POST "{INVOKE_URL}" \
  -H "Content-Type: application/json" \
  -d '{
    "connector": {
      "class_path": "langchain_groq.chat_models.ChatGroq",
      "params": {
        "model": "qwen/qwen3-32b",
        "api_key": "your-groq-api-key",
        "temperature": 0.7
      }
    },
    {EXAMPLE_PAYLOAD_FIELDS}
  }'
```

### Using OpenAI

```bash
curl -s -X POST "{INVOKE_URL}" \
  -H "Content-Type: application/json" \
  -d '{
    "connector": {
      "class_path": "langchain_openai.chat_models.ChatOpenAI",
      "params": {
        "model": "gpt-4o-mini",
        "api_key": "your-openai-api-key"
      }
    },
    {EXAMPLE_PAYLOAD_FIELDS}
  }'
```

## Request Format

```json
{
  "connector": {
    "class_path": "langchain_groq.chat_models.ChatGroq",
    "params": {
      "model": "qwen/qwen3-32b",
      "api_key": "your-api-key",
      "temperature": 0.7
    }
  },
  {REQUEST_SCHEMA}
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

### Module-Specific Fields

{REQUEST_FIELDS_TABLE}

## Response Format

```json
{RESPONSE_SCHEMA}
```

### Response Fields

{RESPONSE_FIELDS_TABLE}

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
| `No API key provided` | Missing api_key in params | Add api_key to connector.params |

## View Logs

```bash
aws logs tail "/aws/lambda/fair-forge-{MODULE_EXTRA}" --follow --region {AWS_REGION}
```

## Deployment Commands

```bash
# Deploy
./scripts/deploy.sh {MODULE_EXTRA} {AWS_REGION}

# Update (rebuild and redeploy)
./scripts/update.sh {MODULE_EXTRA} {AWS_REGION}

# Cleanup (remove all AWS resources)
./scripts/cleanup.sh {MODULE_EXTRA} {AWS_REGION}
```

## Environment Variables

The Lambda function supports the following environment variables:

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | Fallback API key if not provided in request |

---

## Template Placeholders

When creating a new README from this template, replace these placeholders:

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `{MODULE_NAME}` | Human-readable module name | `Generators`, `Runners`, `BestOf` |
| `{MODULE_DESCRIPTION}` | Brief description | `generating synthetic test datasets` |
| `{DETAILED_DESCRIPTION}` | Full description of functionality | See existing READMEs |
| `{INVOKE_URL}` | API Gateway endpoint URL | `https://xxx.execute-api.us-east-2.amazonaws.com/run` |
| `{MODULE_EXTRA}` | pip extras name | `generators`, `runners`, `bestof` |
| `{AWS_REGION}` | AWS region | `us-east-2` |
| `{EXAMPLE_PAYLOAD_FIELDS}` | Module-specific request fields | `"context": "...", "config": {...}` |
| `{REQUEST_SCHEMA}` | Full request JSON schema | See module reference docs |
| `{REQUEST_FIELDS_TABLE}` | Markdown table of request fields | Field, Type, Required, Description |
| `{RESPONSE_SCHEMA}` | Full response JSON schema | See module reference docs |
| `{RESPONSE_FIELDS_TABLE}` | Markdown table of response fields | Field, Type, Description |
