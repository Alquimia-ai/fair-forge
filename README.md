# Fair Forge

<p align="center">
  <img src="https://www.alquimia.ai/logo-alquimia.svg" width="800" alt="Alquimia AI">
</p>

<p align="center">
  <strong>Performance-measurement library for evaluating AI models and assistants</strong>
</p>

<p align="center">
  <a href="https://fairforge.alquimia.ai">Documentation</a> •
  <a href="https://fairforge.alquimia.ai/llms.txt">LLM-friendly docs</a> •
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

## Quick Start

```bash
# Install with pip
pip install fair-forge

# Or install specific modules
pip install "fair-forge[toxicity]"
pip install "fair-forge[bias]"
pip install "fair-forge[all]"
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

## Documentation

For complete documentation, guides, and API reference visit:

**[https://fairforge.alquimia.ai](https://fairforge.alquimia.ai)**

### For LLMs

If you want to use the documentation with an LLM, use:

**[https://fairforge.alquimia.ai/llms.txt](https://fairforge.alquimia.ai/llms.txt)**

## License

MIT License - see [LICENSE](LICENSE) for details.
