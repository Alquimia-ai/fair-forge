# Alquimia AI Fair Forge

<p align="center">
  <img src='https://www.alquimia.ai/logo-alquimia.svg' width=800>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#metrics">Metrics</a> •
  <a href="#usage">Usage</a>
  <a href="#examples">Examples</a>
  <a href="#contributing">Create your own metric</a>
</p>

## Overview

**Alquimia AI Fair Forge** is a powerful performance-measurement component designed specifically for evaluating AI models and assistants. It provides clarity and insightful metrics, helping you understand and improve your AI applications through comprehensive analysis and evaluation.

## Installation

To get started with Alquimia AI Fair Forge, follow these simple steps:

1. First, build the package:
```shell
make package
```

2. Then install it using pip:
```shell
pip install --force-reinstall dist/alquimia_fair_forge-0.0.1.tar.gz -q
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
[{
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
}]
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

To properly run this metric, you need to deploy a Guardian Model using an OpenAI-like API. We recommend using the IBM Granite Guardian model, which is specifically designed for risk detection and bias assessment.

You can find the model at: [IBM Granite Guardian 3.1 2B](https://huggingface.co/ibm-granite/granite-guardian-3.1-2b)

```python
from getpass import getpass
from fair_forge.metrics import Bias

guardian_api_key = SecretStr(getpass("Please enter your Guardian API key: "))
GUARDIAN_URL = os.environ.get("GUARDIAN_URL")
GUARDIAN_MODEL_NAME = os.environ.get("GUARDIAN_MODEL_NAME")
GUARDIAN_API_KEY = guardian_api_key

metrics = Bias.run(
    CustomRetriever,
    guardian_url=GUARDIAN_URL,
    guardian_api_key=GUARDIAN_API_KEY,
    guardian_model=GUARDIAN_MODEL_NAME,
    guardian_temperature=guardian_temperature,
    max_tokens=max_tokens,
    verbose=True  # Enable detailed logging
)
```

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

4. Document your metric in the `docs/journal.tex` file, including:
   - Mathematical formulation
   - Implementation details
   - Usage examples
   - References to relevant research

5. Add tests for your metric in the `tests` directory

6. Submit a pull request with your implementation

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
