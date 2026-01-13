# Alquimia AI Fair Forge

<p align="center">
  <img src='https://www.alquimia.ai/logo-alquimia.svg' width=800>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#metrics">Metrics</a> •
  <a href="#usage">Usage</a> •
  <a href="#examples">Examples</a> •
  <a href="#contributing">Create your own metric</a>
</p>

## Overview

**Alquimia AI Fair Forge** is a powerful performance-measurement component designed specifically for evaluating AI models and assistants. It provides clarity and insightful metrics, helping you understand and improve your AI applications through comprehensive analysis and evaluation.

## Installation

### Requirements

- **Python**: >= 3.11 (recommended: 3.11, 3.12, or 3.13)
- **uv**: We recommend using [uv](https://docs.astral.sh/uv/) for fast and reliable Python package management

### Quick Start

1. Install uv if you haven't already:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:

```shell
git clone https://github.com/Alquimia-ai/fair-forge.git
cd fair-forge
```

3. Create and activate a virtual environment:

```shell
uv venv
source .venv/bin/activate
```

4. Build the package:

```shell
make build
```

5. Install the package with the modules you need:

```shell
# Install with a specific module
uv pip install "dist/alquimia_fair_forge-0.1.1-py3-none-any.whl[toxicity]"

# Install with multiple modules
uv pip install "dist/alquimia_fair_forge-0.1.1-py3-none-any.whl[toxicity,bias]"

# Install everything
uv pip install "dist/alquimia_fair_forge-0.1.1-py3-none-any.whl[all]"
```

### Available Modules

Fair Forge is modular - install only what you need:

| Module | Description | Dependencies |
|--------|-------------|--------------|
| `context` | Context awareness metric (LLM-based) | Core only |
| `conversational` | Grice's maxims evaluation (LLM-based) | Core only |
| `bestof` | Tournament-style response comparison (LLM-based) | Core only |
| `humanity` | Emotional analysis metric | numpy, pandas |
| `toxicity` | Toxic language detection with DIDT framework | nltk, numpy, pandas, scikit-learn, umap-learn, hdbscan, sentence-transformers, torch |
| `bias` | Bias detection across protected attributes | numpy, pandas, scikit-learn, umap-learn, hdbscan, sentence-transformers, torch |
| `metrics` | All metrics above | All metric dependencies |
| `runners` | Test execution against AI systems | alquimia-client, python-dotenv, httpx, aiosseclient |
| `generators` | Synthetic test dataset generation (base) | httpx |
| `generators-openai` | Generators with OpenAI support | httpx, langchain-openai |
| `generators-groq` | Generators with Groq support | httpx, langchain-groq |
| `generators-all` | Generators with all LLM providers | httpx, langchain-openai, langchain-groq |
| `cloud` | Cloud storage backends (S3, LakeFS) | boto3, lakefs |
| `all` | Everything | All dependencies |
| `dev` | Development tools | pytest, ruff, mypy, sphinx, jupyter |

### Installation Examples

```shell
# For toxicity analysis only
uv pip install "dist/alquimia_fair_forge-0.1.1-py3-none-any.whl[toxicity]"

# For bias detection only
uv pip install "dist/alquimia_fair_forge-0.1.1-py3-none-any.whl[bias]"

# For LLM-based metrics (context, conversational, bestof)
# Note: You also need to install your LLM provider (e.g., langchain-openai, langchain-groq)
uv pip install "dist/alquimia_fair_forge-0.1.1-py3-none-any.whl[context]"
uv pip install langchain-openai  # or your preferred provider

# For running tests against AI systems
uv pip install "dist/alquimia_fair_forge-0.1.1-py3-none-any.whl[runners]"

# For cloud storage support
uv pip install "dist/alquimia_fair_forge-0.1.1-py3-none-any.whl[cloud]"

# For all metrics
uv pip install "dist/alquimia_fair_forge-0.1.1-py3-none-any.whl[metrics]"

# For everything (all modules)
uv pip install "dist/alquimia_fair_forge-0.1.1-py3-none-any.whl[all]"
```

### Installing in Jupyter Notebooks

When working in Jupyter notebooks, use this pattern to install from a local build:

```python
import sys
!uv pip install --python {sys.executable} --force-reinstall "../../dist/alquimia_fair_forge-0.1.1-py3-none-any.whl[toxicity]"
```

Replace `toxicity` with the module(s) you need, or use `all` for everything.

### For Development

If you're contributing to this project:

```shell
# Clone and setup
git clone https://github.com/Alquimia-ai/fair-forge.git
cd fair-forge
uv venv
source .venv/bin/activate

# Install all dependencies including dev tools
uv sync

# Run scripts in development
uv run python your_script.py

# Install in editable mode
uv pip install -e ".[dev]"
```

## How It Works

Alquimia AI Fair Forge evaluates AI models and assistants through six comprehensive metrics:

- **Conversational**: Measures the quality and effectiveness of conversations using Grice's maxims
- **Humanity**: Evaluates how human-like and natural the responses are through emotional analysis
- **Bias**: Analyzes potential biases across protected attributes (gender, race, religion, etc.)
- **Toxicity**: Detects and quantifies toxic language patterns using clustering and lexicon analysis
- **Context**: Assesses how well the assistant maintains and uses context
- **BestOf**: Tournament-style evaluation to determine the best response among multiple candidates

For detailed information about how each metric is computed, please refer to our [technical documentation](docs/journal.pdf) or [LaTeX source](docs/journal.tex).

### Dataset Structure

To use Alquimia AI Fair Forge, you need to provide a dataset in the following JSON format:

```json
[
  {
    "session_id": "123",
    "assistant_id": "456",
    "language": "english",
    "context": "You are a helpful assistant.",
    "conversation": [
      {
        "qa_id": "123",
        "query": "What is Alquimia AI?",
        "ground_truth_assistant": "Is an startup that its aim is to construct assistants",
        "assistant": "I'm so happy to answer your question. Alquimia AI Is an startup dedicated to construct assistants."
      }
    ]
  }
]
```

#### Dataset Fields Explained:

- `session_id`: Unique identifier for the conversation session
- `assistant_id`: Identifier for the specific assistant being evaluated
- `language`: The language in which the assistant should respond (used by the humanity metric)
- `context`: The development environment or context of the assistant
- `conversation`: Array of interactions containing:
  - `qa_id`: Unique identifier for the question-answer pair
  - `query`: The user's question
  - `ground_truth_assistant`: The expected or ideal response
  - `assistant`: The actual response from the assistant being evaluated

## Usage

### Setting Up Your Custom Retriever

To use your dataset with Fair Forge metrics, you need to create a custom retriever:

```python
import json
from fair_forge import Retriever, Dataset

class CustomRetriever(Retriever):
    def load_dataset(self) -> list[Dataset]:
        datasets = []
        with open("dataset.json") as infile:
            for dataset in json.load(infile):
                datasets.append(Dataset.model_validate(dataset))
        return datasets
```

### Using the Metrics

#### Context Metric

The Context metric evaluates how well AI responses align with the provided context. It uses an LLM as a judge to score responses.

```python
from fair_forge.metrics.context import Context
from langchain_openai import ChatOpenAI

# Initialize your LLM judge (any LangChain-compatible chat model)
judge_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
)

metrics = Context.run(
    CustomRetriever,
    model=judge_model,
    use_structured_output=True,  # Use LangChain's structured output
    verbose=True,
)

# Access results
for metric in metrics:
    print(f"QA ID: {metric.qa_id}")
    print(f"Context Awareness Score: {metric.context_awareness}")
    print(f"Insight: {metric.context_insight}")
```

#### Humanity Metric

The Humanity metric evaluates how human-like and natural the responses are through emotional analysis using the NRC Emotion Lexicon.

```python
from fair_forge.metrics.humanity import Humanity

metrics = Humanity.run(
    CustomRetriever,
    verbose=True,
)

# Access results
for metric in metrics:
    print(f"Session: {metric.session_id}")
    print(f"Emotion Correlation: {metric.emotion_correlation}")
```

#### Conversational Metric

The Conversational metric evaluates dialogue quality using Grice's maxims (quality, quantity, relation, manner) and sensibleness.

```python
from fair_forge.metrics.conversational import Conversational
from langchain_openai import ChatOpenAI

judge_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
)

metrics = Conversational.run(
    CustomRetriever,
    model=judge_model,
    use_structured_output=True,
    verbose=True,
)

# Access results
for metric in metrics:
    print(f"QA ID: {metric.qa_id}")
    print(f"Quality Maxim: {metric.conversational_quality_maxim}")
    print(f"Sensibleness: {metric.conversational_sensibleness}")
```

#### Bias Metric

The Bias metric provides comprehensive bias analysis across multiple protected attributes including gender, race, religion, nationality, and sexual orientation. It uses guardian-based bias detection and statistical confidence intervals to provide detailed insights into potential biases in AI assistant responses.

The metric is highly flexible and allows you to implement your own bias detection mechanisms through the Guardian interface. Fair Forge provides two built-in guardians out of the box:

1. **IBMGranite**: Uses IBM's Granite model for bias detection
2. **LLamaGuard**: Uses Meta's LLamaGuard model for bias detection

You can also create your own guardian by implementing the Guardian interface:

```python
from typing import Optional
from fair_forge.core.guardian import Guardian
from fair_forge.schemas.bias import ProtectedAttribute, GuardianBias

class MyCustomGuardian(Guardian):
    def is_biased(self, question: str, answer: str, attribute: ProtectedAttribute, context: Optional[str] = None) -> GuardianBias:
        # Implement your bias detection logic here
        # Return a GuardianBias object with:
        # - is_biased: Whether bias was detected
        # - attribute: The specific attribute that showed bias
        # - certainty: Confidence score for the detection
        pass
```

Here's an example using the built-in IBMGranite guardian:

```python
import os
from getpass import getpass
from pydantic import SecretStr
from fair_forge.metrics.bias import Bias
from fair_forge.guardians import IBMGranite
from fair_forge.guardians.llms.providers import OpenAIGuardianProvider
from fair_forge.schemas.bias import GuardianLLMConfig

# Set up your environment variables
GUARDIAN_URL = os.environ.get("GUARDIAN_URL")
GUARDIAN_MODEL_NAME = os.environ.get("GUARDIAN_MODEL_NAME")
GUARDIAN_API_KEY = SecretStr(getpass("Please enter your Guardian API key: "))

# Configure and run the Bias metric
metrics = Bias.run(
    CustomRetriever,  # Use your custom retriever
    guardian=IBMGranite,
    confidence_level=0.80,
    config=GuardianLLMConfig(
        model=GUARDIAN_MODEL_NAME,
        api_key=GUARDIAN_API_KEY.get_secret_value(),
        url=GUARDIAN_URL,
        temperature=0.0,
        provider=OpenAIGuardianProvider,
        logprobs=True
    ),
    verbose=True
)
```

And here's an example using your custom guardian:

```python
from fair_forge.metrics.bias import Bias

# Run the Bias metric with your custom guardian
metrics = Bias.run(
    CustomRetriever,
    guardian=MyCustomGuardian,
    confidence_level=0.80,
    verbose=True,
)
```

The Bias metric supports the following configuration options:

- `confidence_level`: Statistical confidence level for bias detection (default: 0.95)
- `guardian`: The Guardian class to use for bias detection

#### Toxicity Metric

The Toxicity metric detects and quantifies toxic language patterns in AI responses with a focus on **fairness across demographic groups**. It combines clustering analysis, lexicon-based detection, and the DIDT fairness framework (Demographic Representation, Directed Toxicity, Associated Sentiment Bias) with pluggable statistical computation modes.

**Key Features:**
- **Statistical Modes**: Choose between Frequentist (point estimates) or Bayesian (full posterior distributions)
- **DIDT Framework**: Comprehensive fairness analysis across demographic groups
- **Group Detection**: Automatic detection of demographic group mentions via embedding similarity
- **Clustering**: HDBSCAN+UMAP clustering to identify toxic patterns

**Example with Frequentist Mode (Default):**

```python
from fair_forge.metrics.toxicity import Toxicity
from fair_forge.loaders import HurtlexLoader
from fair_forge import FrequentistMode

# Run Toxicity with frequentist statistics (returns point estimates)
metrics = Toxicity.run(
    CustomRetriever,  # Use your custom retriever
    embedding_model="all-MiniLM-L6-v2",
    toxicity_loader=HurtlexLoader,
    statistical_mode=FrequentistMode(),  # Optional: this is the default
    group_prototypes={
        'gender': ['man', 'woman', 'non-binary'],
        'race': ['white', 'black', 'asian', 'latino'],
        'religion': ['christian', 'muslim', 'jewish', 'hindu', 'buddhist']
    },
    # Clustering parameters
    toxicity_min_cluster_size=5,
    toxicity_cluster_selection_epsilon=0.01,
    toxicity_cluster_use_latent_space=True,
    # UMAP parameters
    umap_n_components=2,
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    umap_random_state=42,
    umap_metric="cosine",
    # DIDT weights (default: 1/3 each)
    w_DR=1.0/3.0,
    w_ASB=1.0/3.0,
    w_DTO=1.0/3.0,
    verbose=True
)
```

**Example with Bayesian Mode:**

```python
from fair_forge.metrics.toxicity import Toxicity
from fair_forge.loaders import HurtlexLoader
from fair_forge import BayesianMode

# Run Toxicity with Bayesian statistics (returns posterior distributions with credible intervals)
metrics = Toxicity.run(
    CustomRetriever,
    embedding_model="all-MiniLM-L6-v2",
    toxicity_loader=HurtlexLoader,
    statistical_mode=BayesianMode(
        mc_samples=10000,      # Number of Monte Carlo samples
        ci_level=0.95,         # Credible interval level (95%)
        dirichlet_prior=1.0,   # Dirichlet prior for distribution divergence
        beta_prior_a=1.0,      # Beta prior alpha for rate estimation
        beta_prior_b=1.0       # Beta prior beta for rate estimation
    ),
    group_prototypes={
        'gender': ['man', 'woman', 'non-binary'],
        'race': ['white', 'black', 'asian', 'latino']
    },
    verbose=True
)
```

**Configuration Parameters:**

Core Parameters:
- `group_prototypes` (required if no `group_extractor`): Dictionary mapping group categories to prototype words for detection
- `statistical_mode`: FrequentistMode() or BayesianMode() (default: FrequentistMode())
- `embedding_model`: Model for text embeddings (default: "all-MiniLM-L6-v2")
- `toxicity_loader`: Loader for toxicity lexicon (default: HurtlexLoader)
- `sentiment_analyzer`: Optional SentimentAnalyzer instance for ASB calculation. If not provided, ASB will be 0.

DIDT Framework Weights:
- `w_DR`: Weight for Demographic Representation (default: 1/3)
- `w_DTO`: Weight for Directed Toxicity per Group (default: 1/3)
- `w_ASB`: Weight for Associated Sentiment Bias (default: 1/3)

Group Extractor Parameters:
- `group_extractor`: Optional custom BaseGroupExtractor instance. If not provided, uses EmbeddingGroupExtractor with the parameters below.
- `group_prototypes`: Dictionary mapping group categories to prototype words (required if `group_extractor` is None)
- `group_thresholds`: Custom similarity thresholds per group (default: 0.5 for all)
- `group_default_threshold`: Default threshold for group detection (default: 0.5)
- `group_extractor_batch_size`: Batch size for group extraction (default: 64)
- `group_extractor_normalize_embeddings`: Whether to normalize embeddings for group extraction (default: True)

Group Profiling Parameters:
- `group_toxicity_threshold`: Minimum toxicity score to count as toxic (default: 0.0)
- `group_reference_q`: Optional reference distribution for DR calculation. If not provided, uses uniform distribution across groups.

Clustering Parameters:
- `toxicity_min_cluster_size`: Minimum cluster size for HDBSCAN (default: 5)
- `toxicity_cluster_selection_epsilon`: Epsilon for cluster selection (default: 0.01)
- `toxicity_cluster_selection_method`: Cluster distance metric (default: "euclidean")
- `toxicity_cluster_use_latent_space`: Use latent space for clustering (default: True)

UMAP Parameters:
- `umap_n_components`: Dimensionality of latent space (default: 2)
- `umap_n_neighbors`: Number of neighbors for local structure (default: 15)
- `umap_min_dist`: Minimum distance between points (default: 0.1)
- `umap_random_state`: Random seed for reproducibility (default: 42)
- `umap_metric`: Distance metric (default: "cosine")

**Output Structure:**

The Toxicity metric returns results containing:

For **Frequentist Mode**:
- `cluster_profiling`: Toxicity scores per cluster
- `group_profiling.frequentist`: Point estimates for DR, DTO, ASB, DIDT
- `group_profiling.N_i`: Count of mentions per group
- `group_profiling.K_i`: Count of toxic mentions per group
- `group_profiling.p_i`: Observed proportions per group
- `group_profiling.T_i`: Toxicity rates per group

For **Bayesian Mode**:
- `cluster_profiling`: Toxicity scores per cluster
- `group_profiling.bayesian.summary`: Posterior means and 95% credible intervals for DR, DTO, ASB, DIDT
- `group_profiling.bayesian.priors`: Prior parameters used
- `group_profiling.bayesian.mc_samples`: Number of Monte Carlo samples

**Custom Toxicity Loaders**

You can create your own toxicity dataset loader by implementing the ToxicityLoader interface:

```python
from fair_forge import ToxicityLoader
from fair_forge.schemas.toxicity import ToxicityDataset

class MyCustomToxicityLoader(ToxicityLoader):
    def load(self, language: str) -> list[ToxicityDataset]:
        # Load your custom toxicity dataset
        # Return a list of ToxicityDataset objects containing:
        # - word: The potentially offensive word
        # - category: The category of toxicity
        return [
            ToxicityDataset(word="offensive_word", category="category_name")
            # ... more entries
        ]
```

Then use your custom loader with the Toxicity metric:

```python
from fair_forge.metrics.toxicity import Toxicity

metrics = Toxicity.run(
    CustomRetriever,
    toxicity_loader=MyCustomToxicityLoader,
    group_prototypes={'gender': ['man', 'woman']},
    verbose=True,
)
```

#### BestOf Metric

The BestOf metric implements a tournament-style evaluation to determine the best response among multiple candidates. It uses a judge LLM to compare responses in pairs and determines the winner through elimination rounds.

```python
from fair_forge.metrics.best_of import BestOf
from langchain_openai import ChatOpenAI

judge_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
)

metrics = BestOf.run(
    CustomRetriever,
    model=judge_model,
    use_structured_output=True,
    criteria="BestOf",  # Label for the evaluation criteria
    verbose=True,
)
```

The BestOf metric:
- Creates tournament brackets from multiple response candidates
- Uses LLM judges to evaluate pairs of responses
- Tracks the winner through multiple rounds
- Provides detailed reasoning for each decision
- Returns the final winner with confidence scores

When `verbose=True` is set, the following information will be logged:

- Dataset loading progress
- Metric calculation steps
- API calls and responses
- Processing status for each conversation
- Detailed error messages (if any)
- Performance metrics and timing information

The IBM Granite Guardian model is specifically trained to detect various risks including:

- Social Bias
- Harmful Content
- Unethical Behavior
- Jailbreaking Attempts
- Violence
- Profanity
- Sexual Content

## Examples

For practical examples of how to use Alquimia AI Fair Forge, please refer to our example implementations in the repository:

- [Context Metric Notebook](examples/context/context.ipynb) - Interactive notebook demonstrating the Context metric
- [Bias Metric Notebook](examples/bias/bias.ipynb) - Interactive notebook demonstrating the Bias metric
- [Custom Retriever Example](examples/helpers/retriever.py) - Implementation of a custom dataset retriever
- [Sample Dataset](examples/data/dataset.json) - Example dataset format

Each example includes detailed comments and explanations to help you understand the implementation details. Feel free to use these examples as a starting point for your own implementations.

## Runners Module

Fair Forge includes a modular **runners** system for executing test suites against AI systems. The runners module provides a flexible architecture for testing agents, models, and APIs with comprehensive storage backends.

### Architecture

The runners module consists of 2 main components:

1. **Runners** (`fair_forge.runners`): Execute test batches against AI systems
   - `BaseRunner`: Abstract interface for custom runner implementations
   - `AlquimiaRunner`: Built-in implementation for Alquimia AI agents

2. **Storage** (`fair_forge.storage`): Load test datasets and save results
   - `LocalStorage`: Local filesystem storage
   - `LakeFSStorage`: Cloud-based version control with LakeFS

### Quick Start with AlquimiaRunner

```python
import asyncio
from pathlib import Path
from datetime import datetime
from fair_forge.runners import AlquimiaRunner
from fair_forge.storage import create_local_storage

# 1. Configure storage
storage = create_local_storage(
    tests_dir=Path("./test_datasets"),
    results_dir=Path("./test_results"),
)

# 2. Configure runner
runner = AlquimiaRunner(
    base_url="https://api.alquimia.ai",
    api_key="your-api-key",
    agent_id="your-agent-id",
    channel_id="your-channel-id",
)

# 3. Load and execute tests
async def run_tests():
    datasets = storage.load_datasets()

    for dataset in datasets:
        updated_dataset, summary = await runner.run_dataset(dataset)
        print(f"Completed: {summary['successes']}/{summary['total_batches']} passed")

        # Save results
        storage.save_results([updated_dataset], "run_001", datetime.now())

asyncio.run(run_tests())
```

### Storage Backends

#### Local Storage

Store test datasets and results on local filesystem:

```python
from fair_forge.storage import create_local_storage

storage = create_local_storage(
    tests_dir="./test_datasets",
    results_dir="./test_results",
    enabled_suites=None,  # None = load all suites
)
```

**Directory Structure**:
```
test_datasets/
├── prompt_injection.json
├── toxicity.json
└── general_qa.json

test_results/
└── test_run_20260108_143022_abc123.json
```

#### LakeFS Storage

Store test datasets and results in LakeFS for version control:

```python
from fair_forge.storage import create_lakefs_storage

storage = create_lakefs_storage(
    host="https://lakefs.example.com",
    username="admin",
    password="your-password",
    repo_id="fair-forge-tests",
    enabled_suites=None,
    tests_prefix="tests/",
    results_prefix="results/",
    branch_name="main",
)
```

#### Storage Factory

Use the factory function for dynamic backend selection:

```python
from fair_forge.storage import create_storage

# Local storage
storage = create_storage(
    backend="local",
    tests_dir="./tests",
    results_dir="./results",
)

# LakeFS storage
storage = create_storage(
    backend="lakefs",
    host="https://lakefs.example.com",
    username="admin",
    password="password",
    repo_id="my-repo",
)
```

### Tests Format

Test suites use Fair Forge's standard Dataset schema:

```json
{
  "session_id": "test_suite_001",
  "assistant_id": "my_agent",
  "language": "english",
  "context": "Test suite description",
  "conversation": [
    {
      "qa_id": "test_001",
      "query": "What is AI?",
      "assistant": "",
      "ground_truth_assistant": "AI is artificial intelligence",
      "observation": "Basic knowledge test",
      "agentic": {},
      "ground_truth_agentic": {}
    }
  ]
}
```

### Test Suite Filtering

Filter which test suites to load using the `enabled_suites` parameter:

```python
# Load only specific test suites
storage = create_local_storage(
    tests_dir="./test_datasets",
    results_dir="./test_results",
    enabled_suites=["prompt_injection", "toxicity"],
)

# Load all test suites
storage = create_local_storage(
    tests_dir="./test_datasets",
    results_dir="./test_results",
    enabled_suites=None,
)
```

The `enabled_suites` filter matches against JSON filenames without extension.

### Creating Custom Runners

Implement custom runners by extending the `BaseRunner` interface:

```python
from fair_forge.schemas.runner import BaseRunner
from fair_forge.schemas.common import Batch, Dataset
from typing import Any

class CustomRunner(BaseRunner):
    """Custom runner for your AI system."""

    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint

    async def run_batch(
        self,
        batch: Batch,
        session_id: str,
        **kwargs: Any
    ) -> tuple[Batch, bool, float]:
        """Execute a single test case."""
        import time
        start = time.time()

        try:
            # Call your AI system API
            response = await self._call_api(batch.query, session_id)

            # Update batch with response
            updated_batch = batch.model_copy(
                update={"assistant": response}
            )

            exec_time = (time.time() - start) * 1000
            return updated_batch, True, exec_time

        except Exception as e:
            exec_time = (time.time() - start) * 1000
            error_batch = batch.model_copy(
                update={"assistant": f"[ERROR] {str(e)}"}
            )
            return error_batch, False, exec_time

    async def run_dataset(
        self,
        dataset: Dataset,
        **kwargs: Any
    ) -> tuple[Dataset, dict[str, Any]]:
        """Execute all test cases in a dataset."""
        # Implement dataset execution logic
        # See AlquimiaRunner for reference implementation
        pass
```

### Integration with Fair Forge Metrics

Test results are fully compatible with Fair Forge metrics for comprehensive analysis:

```python
import json
from fair_forge import Retriever, Dataset
from fair_forge.metrics.toxicity import Toxicity
from fair_forge.metrics.bias import Bias
from fair_forge.metrics.context import Context

# Create a custom retriever for your test results
class TestResultsRetriever(Retriever):
    def load_dataset(self) -> list[Dataset]:
        with open("test_results/test_run_latest.json") as f:
            data = json.load(f)
            return [Dataset.model_validate(d) for d in data]

# Analyze test results with metrics
toxicity_metrics = Toxicity.run(TestResultsRetriever, ...)
bias_metrics = Bias.run(TestResultsRetriever, ...)
context_metrics = Context.run(TestResultsRetriever, ...)
```

This enables comprehensive evaluation of agent behavior using toxicity, bias, conversational quality, and other metrics on your test execution results.

## Generators Module

Fair Forge includes a **generators** module for creating synthetic test datasets from context documents. This enables automated test generation for evaluating AI assistants against your documentation or knowledge base.

### Architecture

The generators module consists of 2 main components:

1. **Generators** (`fair_forge.generators`): Generate test queries from context
   - `BaseGenerator`: Abstract interface for custom generator implementations
   - `OpenAIGenerator`: Uses OpenAI models via LangChain (GPT-4, GPT-3.5)
   - `GroqGenerator`: Uses Groq's ultra-fast inference API (Llama, Mixtral)
   - `AlquimiaGenerator`: Uses Alquimia's internal LLM endpoint
   - `LangChainGenerator`: Base class for any LangChain-compatible model

2. **Context Loaders** (`fair_forge.generators.context_loaders`): Load and chunk context documents
   - `BaseContextLoader`: Abstract interface for custom loaders
   - `LocalMarkdownLoader`: Load and chunk local markdown files

### Quick Start with OpenAI

```python
import asyncio
import os
from fair_forge.generators import create_openai_generator, create_markdown_loader

# Set API key via environment variable
os.environ["OPENAI_API_KEY"] = "your-api-key"

# 1. Create context loader
loader = create_markdown_loader(max_chunk_size=2000)

# 2. Create OpenAI generator
generator = create_openai_generator(
    model_name="gpt-4o-mini",  # or "gpt-4o", "gpt-3.5-turbo"
    temperature=0.7,
)

# 3. Generate dataset
async def generate():
    dataset = await generator.generate_dataset(
        context_loader=loader,
        source="./docs/knowledge_base.md",
        assistant_id="my-assistant",
        num_queries_per_chunk=3,
    )
    return dataset

dataset = asyncio.run(generate())
print(f"Generated {len(dataset.conversation)} test queries")
```

### Quick Start with Groq

Groq provides ultra-fast inference for open-source models:

```python
import asyncio
import os
from fair_forge.generators import create_groq_generator, create_markdown_loader

# Set API key via environment variable
os.environ["GROQ_API_KEY"] = "your-api-key"

# Create Groq generator with Llama 3.1
generator = create_groq_generator(
    model_name="llama-3.1-70b-versatile",  # or "llama-3.1-8b-instant", "mixtral-8x7b-32768"
    temperature=0.7,
)

# Generate dataset
loader = create_markdown_loader()
dataset = await generator.generate_dataset(
    context_loader=loader,
    source="./docs/knowledge_base.md",
    assistant_id="my-assistant",
)
```

### Quick Start with Alquimia

The Alquimia generator uses an agent configured in your Alquimia workspace. The `context`, `seed_examples`, and `num_queries` are passed as extra data that gets injected into the agent's system prompt.

```python
from fair_forge.generators import create_alquimia_generator, create_markdown_loader

generator = create_alquimia_generator(
    base_url="https://api.alquimia.ai",
    api_key="your-api-key",
    agent_id="your-agent-id",      # Agent configured in Alquimia workspace
    channel_id="your-channel-id",  # Channel identifier
)

loader = create_markdown_loader()
dataset = await generator.generate_dataset(
    context_loader=loader,
    source="./docs/knowledge_base.md",
    assistant_id="my-assistant",
)
```

**Note:** AlquimiaGenerator does not support custom system prompts. Configure the agent's prompt in your Alquimia workspace to accept `context`, `num_queries`, and `seed_examples` as template variables.

### Available Generator Models

| Provider | Models | Configuration |
|----------|--------|---------------------|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo` | `OPENAI_API_KEY` env var |
| Groq | `llama-3.1-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768` | `GROQ_API_KEY` env var |
| Alquimia | Agent-based (workspace configured) | `agent_id`, `channel_id` params |

### Context Loaders

#### LocalMarkdownLoader

Load and chunk local markdown files using a hybrid strategy:

```python
from fair_forge.generators import create_markdown_loader

loader = create_markdown_loader(
    max_chunk_size=2000,   # Maximum characters per chunk
    min_chunk_size=200,    # Minimum characters per chunk
    overlap=100,           # Overlap between size-based chunks
    header_levels=[1, 2, 3],  # Split on H1, H2, H3 headers
)

# Load and chunk a markdown file
chunks = loader.load("./docs/knowledge_base.md")
for chunk in chunks:
    print(f"Chunk: {chunk.chunk_id}")
    print(f"Content: {chunk.content[:100]}...")
```

**Chunking Strategy:**
1. **Primary**: Split by markdown headers (H1, H2, H3, etc.)
2. **Fallback**: Split by character count for long sections without headers

### Generator Configuration

#### Seed Examples

Guide the query generation style using seed examples:

```python
dataset = await generator.generate_dataset(
    context_loader=loader,
    source="./docs/knowledge_base.md",
    assistant_id="my-assistant",
    num_queries_per_chunk=3,
    seed_examples=[
        "What are the main features of the platform?",
        "How do I configure the authentication system?",
    ],
)
```

#### Custom System Prompt (OpenAI/Groq only)

Use a custom system prompt for domain-specific generation with OpenAI or Groq generators:

```python
custom_prompt = """You are a QA specialist creating test questions.

Context:
{context}

{seed_examples_section}

Generate exactly {num_queries} questions based on this context.

You MUST respond with valid JSON:
{{"queries": [{{"query": "...", "difficulty": "easy|medium|hard", "query_type": "factual|inferential"}}], "chunk_summary": "..."}}"""

dataset = await generator.generate_dataset(
    context_loader=loader,
    source="./docs/knowledge_base.md",
    assistant_id="my-assistant",
    custom_system_prompt=custom_prompt,
)
```

### Creating Custom Context Loaders

Implement custom loaders by extending the `BaseContextLoader` interface:

```python
from fair_forge.schemas.generators import BaseContextLoader, Chunk

class JsonContextLoader(BaseContextLoader):
    """Custom loader for JSON documents."""

    def load(self, source: str) -> list[Chunk]:
        import json
        from pathlib import Path

        with open(source) as f:
            data = json.load(f)

        chunks = []
        for key, value in data.items():
            chunks.append(Chunk(
                content=f"{key}: {json.dumps(value)}",
                chunk_id=f"json_{key}",
                metadata={"key": key},
            ))
        return chunks
```

### Integration with Runners

Generated datasets are fully compatible with Fair Forge runners:

```python
from fair_forge.runners import AlquimiaRunner
from fair_forge.storage import create_local_storage

# Generate test dataset
dataset = await generator.generate_dataset(...)

# Run tests against AI system
runner = AlquimiaRunner(...)
updated_dataset, summary = await runner.run_dataset(dataset)

# Analyze results with metrics
from fair_forge.metrics.context import Context
metrics = Context.run(TestResultsRetriever, ...)
```

## Contributing

To contribute a new metric to Fair Forge, follow these steps:

1. Create a new Python file in the `fair_forge/metrics` directory with your metric name (e.g., `my_metric.py`)

2. Follow the basic structure:

```python
from typing import Type, Optional
from fair_forge import FairForge, Retriever, Batch

class MyMetric(FairForge):
    def __init__(self, retriever: Type[Retriever], **kwargs):
        super().__init__(retriever, **kwargs)

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: Optional[str] = "english",
    ):
        # Implement your metric logic here
        for interaction in batch:
            # Process each interaction
            # Append results to self.metrics
            pass
```

3. Implement your metric logic in the `batch` method, which receives:

   - `session_id`: Unique identifier for the conversation session
   - `context`: The development environment or context of the assistant
   - `assistant_id`: Identifier for the specific assistant being evaluated
   - `batch`: List of interactions to evaluate
   - `language`: Optional language parameter (defaults to "english")


4. In file `fair_forge/metrics/__init__.py` add your metric to the `__all__` list:

```python
__all__ = [
    # ... existing metrics
    "MyMetric",
]
```

   Users will import your metric directly from its module:
   ```python
   from fair_forge.metrics.my_metric import MyMetric
   ```

5. Document your metric in the `docs/journal.tex` file, including:
   - Mathematical formulation
   - Implementation details
   - Usage examples
   - References to relevant research

6. Submit a pull request with your implementation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
