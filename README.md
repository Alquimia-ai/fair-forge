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

To get started with Alquimia AI Fair Forge, follow these simple steps:

1. First. Activate your venv

```shell
source venv/bin/activate
```

1. Then, build the package:

```shell
make package
```

2. Then install it using pip:

```shell
pip install dist/alquimia_fair_forge-0.0.1.tar.gz -q
```

## How It Works

Alquimia AI Fair Forge evaluates AI models and assistants through four main metrics:

- **Conversational**: Measures the quality and effectiveness of conversations
- **Humanity**: Evaluates how human-like and natural the responses are
- **Bias**: Analyzes potential biases in the responses
- **Context**: Assesses how well the assistant maintains and uses context

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
from fair_forge.schemas import Dataset
from fair_forge import Retriever

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

```python
from getpass import getpass
from fair_forge.metrics import Context

judge_api_key = SecretStr(getpass("Please enter your Judge API key: "))

metrics = Context.run(
    CustomRetriever,
    judge_api_key=judge_api_key,
    verbose=True  # Enable detailed logging
)
```

#### Humanity Metric

```python
from fair_forge.metrics import Humanity

metrics = Humanity.run(
    CustomRetriever,
    verbose=True  # Enable detailed logging
)
```

#### Conversational Metric

```python
from getpass import getpass
from fair_forge.metrics import Conversational

judge_api_key = SecretStr(getpass("Please enter your Judge API key: "))

metrics = Conversational.run(
    CustomRetriever,
    judge_api_key=judge_api_key,
    verbose=True  # Enable detailed logging
)
```

#### Bias Metric

The Bias metric provides comprehensive bias analysis across multiple protected attributes including gender, race, religion, nationality, and sexual orientation. It uses a combination of clustering techniques, confidence intervals, and guardian-based bias detection to provide detailed insights into potential biases in AI assistant responses.

The metric is highly flexible and allows you to implement your own bias detection mechanisms through the Guardian interface. Fair Forge provides two built-in guardians out of the box:

1. **IBMGranite**: Uses IBM's Granite model for bias detection
2. **LLamaGuard**: Uses Meta's LLamaGuard model for bias detection

You can also create your own guardian by implementing the Guardian interface:

```python
from fair_forge import Guardian
from fair_forge.schemas import ProtectedAttribute, GuardianBias
from typing import Optional

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
from getpass import getpass
from fair_forge.metrics import Bias
from fair_forge.guardians import IBMGranite, GuardianLLMConfig, OpenAIGuardianProvider
from fair_forge.retrievers import LocalRetriever

# Set up your environment variables
GUARDIAN_URL = os.environ.get("GUARDIAN_URL")
GUARDIAN_MODEL_NAME = os.environ.get("GUARDIAN_MODEL_NAME")
GUARDIAN_API_KEY = SecretStr(getpass("Please enter your Guardian API key: "))

# Configure and run the Bias metric
metrics = Bias.run(
    LocalRetriever,
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
    toxicity_min_cluster_size=2,
    toxicity_cluster_use_latent_space=True,
    umap_n_neighbors=30,
    verbose=True
)
```

And here's an example using your custom guardian:

```python
from fair_forge.metrics import Bias
from fair_forge.retrievers import LocalRetriever

# Create an instance of your custom guardian
my_guardian = MyCustomGuardian()

# Run the Bias metric with your custom guardian
metrics = Bias.run(
    LocalRetriever,
    guardian=my_guardian,
    confidence_level=0.80,
    verbose=True
)
```

The Bias metric supports various configuration options for clustering and analysis:

- `confidence_level`: Statistical confidence level for bias detection (default: 0.95)
- `embedding_model`: Model for generating text embeddings (default: "all-MiniLM-L6-v2")
- `toxicity_min_cluster_size`: Minimum size for clusters in HDBSCAN (default: 5)
- `toxicity_cluster_selection_epsilon`: Epsilon value for cluster selection (default: 0.01)
- `toxicity_cluster_selection_method`: Method used for cluster selection (default: "euclidean")
- `toxicity_cluster_use_latent_space`: Whether to use latent space for clustering (default: True)
- `umap_n_components`: Number of components for UMAP dimensionality reduction (default: 2)
- `umap_n_neighbors`: Number of neighbors for UMAP (default: 15)
- `umap_min_dist`: Minimum distance parameter for UMAP (default: 0.1)
- `umap_random_state`: Random state for UMAP reproducibility (default: 42)
- `umap_metric`: Metric used for UMAP (default: "cosine")

### Custom Toxicity Loaders

The Bias metric uses toxicity analysis to identify potentially harmful content in clusters of responses. You can create your own toxicity dataset loader by implementing the ToxicityLoader interface:

```python
from fair_forge import ToxicityLoader
from fair_forge.schemas import ToxicityDataset

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

Then use your custom loader with the Bias metric:

```python
metrics = Bias.run(
    LocalRetriever,
    guardian=my_guardian,
    toxicity_loader=MyCustomToxicityLoader,
    verbose=True
)
```

The toxicity loader is used during cluster analysis to calculate the percentage of potentially offensive words in each cluster of responses. This helps identify patterns of harmful content in the assistant's responses.

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

- [Openshift AI Pipeline](examples/fair-forge.pipeline) - A simple implementation showing how to set up and use the basic metrics using Elyra pipeline to be executed.
- [Custom Retriever Example](examples/helpers/retriever.py) - Implementation of a custom dataset retriever

Each example includes detailed comments and explanations to help you understand the implementation details. Feel free to use these examples as a starting point for your own implementations.

## Contributing

To contribute a new metric to Fair Forge, follow these steps:

1. Create a new Python file in the `fair_forge/metrics` directory with your metric name (e.g., `my_metric.py`)

2. Follow the basic structure:

```python
from fair_forge import FairForge, Retriever
from typing import Type, Optional
from fair_forge.schemas import Batch

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
            pass
```

3. Implement your metric logic in the `batch` method, which receives:

   - `session_id`: Unique identifier for the conversation session
   - `context`: The development environment or context of the assistant
   - `assistant_id`: Identifier for the specific assistant being evaluated
   - `batch`: List of interactions to evaluate
   - `language`: Optional language parameter (defaults to "english")


4. In file `metrics/__init__.py` add your metric file export:

```python
from .my_metric import MyMetric

__all__ = [
    'MyMetric'
]
```

5. Document your metric in the `docs/journal.tex` file, including:
   - Mathematical formulation
   - Implementation details
   - Usage examples
   - References to relevant research

6. Submit a pull request with your implementation

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
