---
name: fair-forge-lambda
description: Generate AWS Lambda deployment boilerplate for Fair-Forge modules. Use when deploying Fair-Forge metrics (BestOf, Toxicity, Bias), runners (test execution against AI systems), or generators (synthetic dataset creation) to AWS Lambda. Handles local wheel installation since fair-forge is not on PyPI.
---

# Fair-Forge Lambda Deployment

Generate AWS Lambda container deployment for Fair-Forge modules. Builds wheel from local repository since fair-forge is not available on PyPI.

## Supported Modules

| Module Type | Extras | Use Case |
|-------------|--------|----------|
| Metrics | `bestof`, `toxicity`, `bias`, `context`, `conversational`, `humanity` | Evaluate AI responses |
| Runners | `runners` | Execute tests against AI systems |
| Generators | `generators`, `generators-alquimia` | Create synthetic test datasets |

## Workflow

### 1. Generate Lambda Project

Create the deployment directory:

```
examples/{module}/aws-lambda/
├── Dockerfile       # Multi-stage build: wheel + Lambda
├── handler.py       # Lambda entrypoint
├── run.py           # Business logic (customize this)
├── requirements.txt # Additional runtime dependencies
└── scripts/
    ├── deploy.sh
    ├── update.sh
    └── cleanup.sh
```

Copy templates from `assets/templates/` and scripts from `scripts/`.

### 2. Implement run.py

Edit `run.py` with your module logic. See reference docs for patterns:

- `references/metrics.md` - Metrics implementation patterns
- `references/runners.md` - Runners implementation patterns
- `references/generators.md` - Generators implementation patterns

### 3. Update requirements.txt

Add your LLM provider or other dependencies:

```
langchain-groq      # Groq
langchain-openai    # OpenAI
langchain-anthropic # Anthropic
```

### 4. Deploy

From `examples/{module}/aws-lambda/`:

```bash
./scripts/deploy.sh {extra-name} us-east-2
```

Examples:
```bash
./scripts/deploy.sh bestof us-east-2      # BestOf metric
./scripts/deploy.sh runners us-east-2     # Runners module
./scripts/deploy.sh generators us-east-2  # Generators module
```

### 5. Update / Cleanup

```bash
./scripts/update.sh {extra-name} us-east-2   # Rebuild and update
./scripts/cleanup.sh {extra-name} us-east-2  # Remove all resources
```

## Dockerfile Build Process

Multi-stage build:
1. **Builder stage**: Builds fair-forge wheel from local source
2. **Final stage**: Installs wheel with module extras, copies handler

```dockerfile
ARG MODULE_EXTRA=bestof
RUN pip install "/tmp/$(ls /tmp/*.whl)[${MODULE_EXTRA}]"
```

## Bundled Resources

### scripts/
- `deploy.sh` - Full deployment (ECR + Lambda + API Gateway)
- `update.sh` - Rebuild and update Lambda
- `cleanup.sh` - Remove all AWS resources

### assets/templates/
- `Dockerfile` - Multi-stage Lambda container build
- `handler.py` - Lambda entrypoint wrapper
- `run.py` - Business logic template (customize this)
- `requirements.txt` - Runtime dependencies

### references/
- `metrics.md` - Metrics module implementation patterns
- `runners.md` - Runners module implementation patterns
- `generators.md` - Generators module implementation patterns

## Lambda Configuration

Default settings (adjust in deploy.sh):
- **Timeout**: 300 seconds (5 minutes)
- **Memory**: 2048 MB
- **Python**: 3.11
