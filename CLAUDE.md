# Fair-Forge - Claude Reference Guide

## Project Overview

**Fair-Forge** is an AI evaluation framework by Alquimia AI for measuring fairness, quality, and safety of AI assistant responses.

- **Version**: 0.0.1 (Alpha)
- **Python**: 3.10, 3.11, 3.12, 3.13
- **Package Manager**: uv (recommended)
- **License**: MIT
- **Repository**: https://github.com/Alquimia-ai/fair-forge

## Architecture at a Glance (NEW MODULAR STRUCTURE)

```
fair_forge/
├── __init__.py                   # Public API exports
│
├── core/                         # Core abstractions & base classes
│   ├── __init__.py
│   ├── base.py                   # FairForge base class
│   ├── retriever.py              # Retriever ABC
│   ├── guardian.py               # Guardian ABC
│   ├── loader.py                 # ToxicityLoader ABC
│   ├── extractor.py              # BaseGroupExtractor ABC
│   └── exceptions.py             # Custom exceptions
│
├── schemas/                      # Pydantic models (organized by domain)
│   ├── __init__.py
│   ├── common.py                 # Dataset, Batch, Logprobs
│   ├── metrics.py                # BaseMetric
│   ├── bias.py                   # BiasMetric, GuardianBias, ProtectedAttribute
│   ├── toxicity.py               # ToxicityMetric, GroupProfiling, etc.
│   ├── conversational.py         # ConversationalMetric
│   ├── humanity.py               # HumanityMetric
│   ├── context.py                # ContextMetric
│   ├── best_of.py                # BestOfMetric
│   └── agentic.py                # AgenticMetric
│
├── metrics/                      # Metric implementations
│   ├── __init__.py
│   ├── humanity.py
│   ├── conversational.py
│   ├── context.py
│   ├── bias.py
│   ├── toxicity.py               # ✨ NOW USES STATISTICAL MODES!
│   ├── best_of.py                # Renamed from bestOf.py
│   └── agentic.py
│
├── statistical/                  # ✨ NEW! Statistical computation modes
│   ├── __init__.py
│   ├── base.py                   # StatisticalMode ABC
│   ├── frequentist.py            # FrequentistMode
│   └── bayesian.py               # BayesianMode
│
├── guardians/                    # Bias detection guardians
│   ├── __init__.py               # IBMGranite, LLamaGuard
│   └── providers/
│       ├── __init__.py
│       ├── base.py               # Provider ABC
│       ├── huggingface.py
│       └── openai.py
│
├── loaders/                      # ✨ NEW! Data loaders
│   ├── __init__.py
│   └── hurtlex.py                # HurtlexLoader
│
├── extractors/                   # ✨ NEW! Feature/group extractors
│   ├── __init__.py
│   └── embedding.py              # EmbeddingGroupExtractor
│
├── llm/                          # ✨ LLM utilities (MIGRATED!)
│   ├── __init__.py
│   ├── judge.py                  # LLM judge interface
│   ├── cot.py                    # Chain-of-Thought reasoning
│   └── prompts.py                # LLM system prompts
│
├── utils/                        # ✨ General utilities
│   ├── __init__.py
│   └── logging.py                # VerboseLogger
│
└── artifacts/                    # Static data files
    ├── lexicons/
    │   └── nrc_emotion.csv       # NRC emotion lexicon
    ├── risk_atlases/
    │   └── ibm_ai_risk_atlas.yml
    └── toxicity/
        └── hurtlex_*.tsv
```

## NEW: Statistical Modes (Strategy Pattern)

Fair-Forge now implements pluggable statistical computation modes using the **Strategy Pattern**. This allows metrics to use either frequentist or Bayesian statistics without coupling to specific implementations.

### StatisticalMode Interface

```python
from fair_forge.statistical import StatisticalMode, FrequentistMode, BayesianMode

# All modes provide these generic primitives:
mode.distribution_divergence(observed, reference)  # Compute divergence between distributions
mode.rate_estimation(successes, trials)            # Estimate rates with uncertainty
mode.aggregate_metrics(metrics, weights)           # Aggregate multiple metrics
mode.dispersion_metric(values, center="mean")      # Compute dispersion/spread
mode.get_result_type()                             # Returns "point_estimate" or "distribution"
```

### FrequentistMode

Returns point estimates (floats):

```python
from fair_forge.statistical import FrequentistMode

mode = FrequentistMode()
divergence = mode.distribution_divergence(
    observed={'A': 0.3, 'B': 0.7},
    reference={'A': 0.5, 'B': 0.5}
)
# Returns: 0.2 (float)
```

### BayesianMode

Returns full posterior distributions with credible intervals:

```python
from fair_forge.statistical import BayesianMode

mode = BayesianMode(
    mc_samples=5000,
    ci_level=0.95,
    dirichlet_prior=1.0,
    beta_prior_a=1.0,
    beta_prior_b=1.0
)

divergence = mode.distribution_divergence(
    observed_counts={'A': 30, 'B': 70},  # Note: counts, not proportions
    reference={'A': 0.5, 'B': 0.5}
)
# Returns: {'mean': 0.21, 'ci_low': 0.15, 'ci_high': 0.27, 'samples': array([...])}
```

### Using Statistical Modes in Toxicity Metric

```python
from fair_forge.metrics import Toxicity
from fair_forge.statistical import BayesianMode
from fair_forge.retrievers import LocalRetriever

# Use Bayesian mode
toxicity = Toxicity.run(
    LocalRetriever,
    statistical_mode=BayesianMode(mc_samples=10000, ci_level=0.95),
    group_prototypes={
        'gender': ['man', 'woman', 'non-binary'],
        'race': ['white', 'black', 'asian', 'latino'],
    },
    verbose=True
)

# Or use Frequentist mode (default)
toxicity = Toxicity.run(
    LocalRetriever,
    # statistical_mode defaults to FrequentistMode()
    group_prototypes={...},
    verbose=True
)
```

### Benefits of Strategy Pattern

1. **Decoupling**: Toxicity metric doesn't know about frequentist/Bayesian details
2. **Extensibility**: Easy to add new modes (Bootstrap, Permutation, etc.) without modifying Toxicity
3. **Reusability**: Statistical modes can be used in ANY metric, not just Toxicity
4. **Testability**: Each mode is independently testable
5. **Consistency**: All metrics can use the same statistical framework

## Core Abstractions

### 1. FairForge (Base Metric Class)
All metrics inherit from this abstract class:

```python
from fair_forge import FairForge, Retriever
from typing import Type

class MyMetric(FairForge):
    def __init__(self, retriever: Type[Retriever], **kwargs):
        super().__init__(retriever, **kwargs)

    def batch(self, session_id: str, context: str,
              assistant_id: str, batch: list[Batch],
              language: str = "english"):
        # Process each batch and append results to self.metrics
        pass
```

**Key Methods**:
- `batch()`: Process a single conversation batch (must implement)
- `run()`: Class method to execute metric (inherited)
- `_process()`: Internal orchestration (inherited)

**Key Attributes**:
- `retriever`: Data source
- `metrics`: Collected results (list)
- `dataset`: Loaded data
- `verbose`: Enable detailed logging

### 2. Retriever (Data Loading)
Abstract class for loading datasets from various sources:

```python
from fair_forge import Retriever
from fair_forge.schemas import Dataset

class CustomRetriever(Retriever):
    def load_dataset(self) -> list[Dataset]:
        # Load and return datasets
        pass
```

**Built-in Implementations**:
- `LocalRetriever`: Load from local JSON files
- `LakeFSRetriever`: Load from LakeFS distributed storage

### 3. Guardian (Bias Detection)
Abstract class for bias detection:

```python
from fair_forge import Guardian
from fair_forge.schemas import ProtectedAttribute, GuardianBias

class MyGuardian(Guardian):
    def is_biased(self, question: str, answer: str,
                  attribute: ProtectedAttribute,
                  context: Optional[str] = None) -> GuardianBias:
        # Return GuardianBias(is_biased, attribute, certainty)
        pass
```

**Built-in Guardians**:
- `IBMGranite`: IBM's Granite model
- `LLamaGuard`: Meta's LLamaGuard model

**Protected Attributes**: gender, race, religion, nationality, sexual_orientation

### 4. ToxicityLoader
Abstract class for loading toxicity datasets:

```python
from fair_forge import ToxicityLoader
from fair_forge.schemas import ToxicityDataset

class MyToxicityLoader(ToxicityLoader):
    def load(self, language: str) -> list[ToxicityDataset]:
        # Return list of ToxicityDataset(word, category)
        pass
```

**Built-in**: `HurtlexLoader` (multilingual HurtLex lexicon)

## Data Schema

### Input Format (Dataset)
```json
[{
  "session_id": "123",
  "assistant_id": "456",
  "language": "english",
  "context": "System context...",
  "conversation": [{
    "qa_id": "123",
    "query": "User question",
    "assistant": "Assistant response",
    "ground_truth_assistant": "Expected response",
    "observation": "Optional observation",
    "agentic": {},  // Optional agentic data
    "ground_truth_agentic": {},
    "logprobs": {}  // Optional log probabilities
  }]
}]
```

### Core Pydantic Models
- `Dataset`: Top-level conversation session
- `Batch`: Single Q&A interaction
- `BaseMetric`: Base for all metric outputs
- Metric-specific: `HumanityMetric`, `ConversationalMetric`, `BiasMetric`, `ToxicityMetric`, `BestOfMetric`, `ContextMetric`, `AgenticMetric`

## Metrics Deep Dive

### 1. Humanity (humanity.py)
**Purpose**: Emotional analysis using NRC lexicon

**How it works**:
- Tokenizes text with NLTK
- Matches against NRC emotion lexicon (8 emotions)
- Calculates emotional entropy
- Compares Spearman correlation with ground truth

**Usage**:
```python
from fair_forge.metrics import Humanity

metrics = Humanity.run(CustomRetriever, verbose=True)
```

**Output**: `HumanityMetric`
- `humanity_assistant_emotional_entropy`: float
- `humanity_ground_truth_spearman`: float
- `humanity_assistant_{emotion}`: float (anger, anticipation, disgust, fear, joy, sadness, surprise, trust)

### 2. Conversational (conversational.py)
**Purpose**: Evaluate using Grice's maxims

**Analyzes**: memory, language, quantity/quality/relation/manner maxims, sensibleness

**Usage**:
```python
from fair_forge.metrics import Conversational
from pydantic import SecretStr

metrics = Conversational.run(
    CustomRetriever,
    judge_api_key=SecretStr("your-api-key"),
    verbose=True
)
```

**Output**: `ConversationalMetric`
- `conversational_memory`: float (0-1)
- `conversational_language`: float
- `conversational_{maxim}_maxim`: float (quality, quantity, relation, manner)
- `conversational_sensibleness`: float
- `conversational_thinkings`: str (Chain-of-Thought reasoning)

### 3. Context (context.py)
**Purpose**: Context awareness and adherence

**Usage**:
```python
from fair_forge.metrics import Context

metrics = Context.run(
    CustomRetriever,
    judge_api_key=SecretStr("your-api-key"),
    verbose=True
)
```

**Output**: `ContextMetric`
- Awareness score and insights

### 4. Bias (bias.py) - Complex
**Purpose**: Protected attribute bias detection with statistical confidence

**Features**:
- Analyzes 5 protected attributes
- Pluggable guardians
- Clopper-Pearson confidence intervals
- Statistical confidence (default: 0.95)

**Usage**:
```python
from fair_forge.metrics import Bias
from fair_forge.guardians import IBMGranite, GuardianLLMConfig, OpenAIGuardianProvider

metrics = Bias.run(
    CustomRetriever,
    guardian=IBMGranite,
    confidence_level=0.80,
    config=GuardianLLMConfig(
        model="model-name",
        api_key="your-api-key",
        url="https://api-url",
        temperature=0.0,
        provider=OpenAIGuardianProvider,
        logprobs=True
    ),
    verbose=True
)
```

**Output**: `BiasMetric`
- `confidence_intervals`: list[ConfidenceInterval]
- `guardian_interactions`: dict[str, list[GuardianInteraction]]

### 5. Toxicity (toxicity.py) - Most Complex (579 lines)
**Purpose**: Toxic language detection with clustering

**Features**:
- **Clustering**: HDBSCAN + UMAP
- **Detection**: HurtLex lexicon
- **Profiling**: DIDT framework
  - DR: Demographic Representation
  - ASB: Associated Sentiment Bias
  - DTO: Directed Toxicity per Group
  - DIDT: Combined index
- **Group Extraction**: Embedding similarity
- **Statistical Modes**: Frequentist & Bayesian

**Usage**:
```python
from fair_forge.metrics import Toxicity, HurtlexLoader

metrics = Toxicity.run(
    CustomRetriever,
    embedding_model="all-MiniLM-L6-v2",
    toxicity_loader=HurtlexLoader,
    toxicity_min_cluster_size=5,
    toxicity_cluster_selection_epsilon=0.01,
    toxicity_cluster_selection_method="euclidean",
    toxicity_cluster_use_latent_space=True,
    umap_n_components=2,
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    umap_random_state=42,
    umap_metric="cosine",
    verbose=True
)
```

**Output**: `ToxicityMetric`
- `cluster_profiling`: dict[float, float]
- `group_profiling`: Optional[GroupProfiling]
- `assistant_space`: AssistantSpace (embeddings, latent_space, cluster_labels)

### 6. BestOf (bestOf.py)
**Purpose**: Tournament-style response evaluation

**Features**:
- Bracket-based tournament
- LLM judge for pairwise comparisons
- Confidence scores
- Chain-of-Thought reasoning

**Usage**:
```python
from fair_forge.metrics import BestOf

metrics = BestOf.run(
    CustomRetriever,
    judge_api_key=SecretStr("your-api-key"),
    verbose=True
)
```

**Output**: `BestOfMetric`
- `bestof_winner_id`: str
- `bestof_contests`: list[BestOfContest]

### 7. Agentic (agentic.py)
**Status**: Placeholder/stub (19 lines)

## Helper Systems

### Judge (judge.py)
LLM judge interface for reasoning and evaluation:

```python
from fair_forge.helpers import Judge

judge = Judge(
    api_key="your-key",
    model="deepseek-r1-distill-llama-70b",
    temperature=0.0,
    chain_of_thought=True  # Enable CoT
)

result = judge.infer(messages=[...])
```

**Features**:
- Chain-of-Thought support
- JSON extraction
- Configurable model/temperature
- Default: Groq API with Deepseek R1 Distill

### Chain-of-Thought (cot.py)
Extract and manage reasoning from models with thinking tokens:

```python
from fair_forge.helpers import CoT

cot = CoT(api_key="your-key", model="model-name")
result = cot.infer(messages=[...])
# Splits output into thinking and answer sections
```

## Key Dependencies

**Core**:
- `pydantic>=2.0.0`: Data validation
- `scipy>=1.14.1`: Statistical calculations
- `nltk`: NLP
- `transformers`, `torch`: Models
- `sentence-transformers`: Embeddings

**Clustering/ML**:
- `hdbscan`: Clustering
- `umap-learn`: Dimensionality reduction
- `scikit-learn`: ML utilities

**LLM Integration**:
- `langchain-openai`, `langchain-groq`

**Data/Storage**:
- `pandas`: Data manipulation
- `boto3`: AWS integration
- `lakefs`: Distributed storage

**Utilities**:
- `jinja2`: Templating
- `tqdm`: Progress bars

## Installation & Setup

### Development with uv (Recommended)
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync
git clone https://github.com/Alquimia-ai/fair-forge.git
cd fair-forge
uv sync

# Run scripts
uv run python your_script.py
```

### Add to Your Project
```bash
# With uv
uv add alquimia-fair-forge

# With pip
pip install alquimia-fair-forge
```

### Local Development (Editable)
```bash
# With uv
uv pip install -e .

# With pip
pip install -e .
```

## Common Patterns

### Workflow
```
1. Load Data → Retriever.load_dataset() → list[Dataset]
2. Process → FairForge.run(retriever, **config)
   a. Initialize metric with retriever
   b. For each Dataset:
      - For each Batch in conversation:
        * Apply metric logic (batch method)
        * Append result to self.metrics
3. Return → list[MetricResult]
```

### Design Patterns
- **Abstract Factory**: Base classes enable plugin architecture
- **Strategy**: Different metric implementations
- **Template Method**: FairForge._process() orchestrates pipeline
- **Provider**: Guardian providers abstract LLM backends

## File Locations

### Metrics
- `fair_forge/metrics/humanity.py`: Emotional analysis
- `fair_forge/metrics/conversational.py`: Grice's maxims
- `fair_forge/metrics/context.py`: Context awareness
- `fair_forge/metrics/bias.py`: Protected attributes
- `fair_forge/metrics/toxicity.py`: Toxic language (complex)
- `fair_forge/metrics/bestOf.py`: Tournament evaluation
- `fair_forge/metrics/agentic.py`: Placeholder

### Core Files
- `fair_forge/__init__.py`: Core abstractions (line ~200)
- `fair_forge/schemas.py`: All Pydantic models
- `fair_forge/prompts.py`: LLM system prompts

### Guardians
- `fair_forge/guardians/IBMGranite.py`: IBM guardian
- `fair_forge/guardians/LLamaGuard.py`: Meta guardian
- `fair_forge/guardians/llms/providers.py`: Provider implementations

### Examples
- `examples/pipeline/helpers/retriever.py`: LocalRetriever, LakeFSRetriever
- `examples/dataset.json`: Example datasets
- `examples/dataset_bestOf.json`: BestOf examples


## Development Commands

```bash
# Build package
make build  # or: uv build

# Run tests
make test

# Format code
make format

# Lint
make lint

# Install from local build
uv pip install dist/alquimia_fair_forge-0.0.1-py3-none-any.whl
```

## Environment Variables

Common environment variables used:
- `GUARDIAN_URL`: Guardian API endpoint
- `GUARDIAN_MODEL_NAME`: Guardian model name
- `GUARDIAN_API_KEY`: Guardian API key
- `GROQ_API_KEY`: Groq API key (for judge)

## Key Insights

1. **Extensibility**: All core components are abstract classes - easy to extend
2. **Statistical Rigor**: Bias metric uses Clopper-Pearson intervals
3. **Complexity**: Toxicity metric is most complex (HDBSCAN + UMAP + DIDT framework)
4. **LLM Judges**: Conversational, Context, BestOf use LLM judges with CoT
5. **Multilingual**: Toxicity supports multilingual datasets via HurtLex
6. **Modular**: Each metric is independent and can be used standalone
7. **Type Safety**: Heavy use of Pydantic for validation
8. **Progress Tracking**: tqdm integration for long-running operations

## Quick Reference: Creating a Custom Metric

1. Create `fair_forge/metrics/my_metric.py`
2. Inherit from `FairForge`
3. Implement `batch()` method
4. Add to `metrics/__init__.py`
5. Document in `docs/journal.tex`
6. Submit PR

## Documentation

- Technical docs: `docs/journal.pdf` or `docs/journal.tex`
- Usage: `README.md`
- Contributing: `CONTRIBUTING.md`
