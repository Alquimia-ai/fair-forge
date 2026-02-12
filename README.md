# Fair Forge

<p align="center">
  <img src="https://cdn.prod.website-files.com/68f2b0d5efa00a8133ee0b23/698275cf13a9d1263497724f_Alquimia%20Logo.svg" width="800" alt="Alquimia AI">
</p>

<p align="center">
  <strong>Performance-measurement library for evaluating AI models and assistants</strong>
</p>

<p align="center">
  <a href="https://github.com/Alquimia-ai/fair-forge/actions/workflows/release.yml"><img src="https://github.com/Alquimia-ai/fair-forge/actions/workflows/release.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/alquimia-fair-forge/"><img src="https://img.shields.io/pypi/v/alquimia-fair-forge" alt="PyPI"></a>
  <a href="https://pypi.org/project/alquimia-fair-forge/"><img src="https://img.shields.io/pypi/pyversions/alquimia-fair-forge" alt="Python Versions"></a>
  <a href="https://github.com/Alquimia-ai/fair-forge/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Alquimia-ai/fair-forge" alt="License"></a>
</p>

<p align="center">
  <a href="https://fairforge.alquimia.ai">Documentation</a> •
  <a href="https://github.com/Alquimia-ai/fair-forge">GitHub</a>
</p>

## What is Fair Forge?

Fair Forge provides comprehensive metrics for evaluating AI systems:

- **Toxicity** - Detect toxic language patterns with DIDT fairness framework
- **Bias** - Analyze biases across protected attributes (gender, race, religion, etc.)
- **Context** - Assess how well responses align with provided context
- **Conversational** - Evaluate dialogue quality using Grice's maxims
- **Humanity** - Measure how natural responses are through emotional analysis
- **BestOf** - Tournament-style comparison to find the best response
- **Explainability** - Compute token attributions to understand model decisions

## Quick Start

```bash
# Install with pip
pip install alquimia-fair-forge

# Or install specific modules
pip install "alquimia-fair-forge[toxicity]"
pip install "alquimia-fair-forge[bias]"
pip install "alquimia-fair-forge[explainability]"
pip install "alquimia-fair-forge[all]"
```

```python
from fair_forge import Retriever, Dataset
from fair_forge.metrics.toxicity import Toxicity

class MyRetriever(Retriever):
    def load_dataset(self) -> list[Dataset]:
        # Load your dataset here
        pass

metrics = Toxicity.run(MyRetriever, verbose=True)
```

### Explainability

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from fair_forge.explainability import AttributionExplainer, Lime

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Format prompt according to your model (user responsibility)
messages = [{"role": "user", "content": "What is gravity?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False)

explainer = AttributionExplainer(model, tokenizer)
result = explainer.explain(
    prompt=prompt,
    target="Gravity is the force of attraction between objects.",
    method=Lime,  # Pass the method class directly
)

# Get most important words
for attr in result.get_top_k(5):
    print(f"'{attr.text}': {attr.score:.4f}")
```

## Documentation

For complete documentation, guides, and API reference visit:

**[https://fairforge.alquimia.ai](https://fairforge.alquimia.ai)**

## License

MIT License - see [LICENSE](LICENSE) for details.
