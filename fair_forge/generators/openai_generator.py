"""OpenAI generator implementation using LangChain."""

import os
from typing import Optional

from loguru import logger

from .langchain_generator import LangChainGenerator


class OpenAIGenerator(LangChainGenerator):
    """Generator implementation using OpenAI via LangChain.

    This generator uses OpenAI's chat models (GPT-4, GPT-3.5, etc.) to generate
    synthetic test queries from context documents.

    The API key is read from the OPENAI_API_KEY environment variable by default.

    Args:
        model_name: OpenAI model name (default: "gpt-4o-mini")
        temperature: Sampling temperature (0.0-2.0, default: 0.7)
        max_tokens: Maximum tokens in response (default: 2048)
        api_key: Optional API key (defaults to OPENAI_API_KEY env var)
        use_structured_output: If True, use with_structured_output() for parsing

    Example:
        ```python
        import os
        os.environ["OPENAI_API_KEY"] = "your-api-key"

        generator = OpenAIGenerator(model_name="gpt-4o-mini")
        dataset = await generator.generate_dataset(
            context_loader=loader,
            source="./docs/knowledge_base.md",
            assistant_id="my-assistant",
        )
        ```
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        api_key: Optional[str] = None,
        use_structured_output: bool = True,
        **kwargs,
    ):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain-openai is required for OpenAIGenerator. "
                "Install it with: pip install langchain-openai"
            ) from e

        # Get API key from parameter or environment
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        logger.info(f"Initializing OpenAI generator with model: {model_name}")

        model = ChatOpenAI(
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


__all__ = ["OpenAIGenerator"]
