"""Groq Cloud generator implementation using LangChain."""

import os
from typing import Optional

from loguru import logger

from .langchain_generator import LangChainGenerator


class GroqGenerator(LangChainGenerator):
    """Generator implementation using Groq Cloud via LangChain.

    This generator uses Groq's ultra-fast inference API with models like
    Llama, Mixtral, and Gemma to generate synthetic test queries from
    context documents.

    The API key is read from the GROQ_API_KEY environment variable by default.

    Args:
        model_name: Groq model name (default: "llama-3.1-70b-versatile")
        temperature: Sampling temperature (0.0-2.0, default: 0.7)
        max_tokens: Maximum tokens in response (default: 2048)
        api_key: Optional API key (defaults to GROQ_API_KEY env var)
        use_structured_output: If True, use with_structured_output() for parsing

    Available Models (as of 2024):
        - llama-3.1-70b-versatile (recommended for quality)
        - llama-3.1-8b-instant (faster, good for simple tasks)
        - llama3-groq-70b-8192-tool-use-preview
        - mixtral-8x7b-32768
        - gemma2-9b-it

    Example:
        ```python
        import os
        os.environ["GROQ_API_KEY"] = "your-api-key"

        generator = GroqGenerator(model_name="llama-3.1-70b-versatile")
        dataset = await generator.generate_dataset(
            context_loader=loader,
            source="./docs/knowledge_base.md",
            assistant_id="my-assistant",
        )
        ```
    """

    def __init__(
        self,
        model_name: str = "llama-3.1-70b-versatile",
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        api_key: Optional[str] = None,
        use_structured_output: bool = True,
        **kwargs,
    ):
        try:
            from langchain_groq import ChatGroq
        except ImportError as e:
            raise ImportError(
                "langchain-groq is required for GroqGenerator. "
                "Install it with: pip install langchain-groq"
            ) from e

        # Get API key from parameter or environment
        resolved_api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter."
            )

        logger.info(f"Initializing Groq generator with model: {model_name}")

        model = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=resolved_api_key,
        )

        super().__init__(
            model=model,
            use_structured_output=use_structured_output,
            **kwargs,
        )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens


__all__ = ["GroqGenerator"]
