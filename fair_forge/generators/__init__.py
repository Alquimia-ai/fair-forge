"""Generators module for Fair Forge.

This module provides synthetic dataset generators for creating test datasets
from context documents to evaluate AI assistants.

Available generators:
- AlquimiaGenerator: Uses Alquimia's internal LLM endpoint
- OpenAIGenerator: Uses OpenAI's models via LangChain (requires OPENAI_API_KEY)
- GroqGenerator: Uses Groq's fast inference API via LangChain (requires GROQ_API_KEY)
- LangChainGenerator: Base class for any LangChain-compatible model

Chunk Selection Strategies:
- SequentialStrategy: Process all chunks in order (default behavior)
- RandomSamplingStrategy: Randomly sample chunks multiple times

Features:
- Independent query generation: Generate unrelated questions per chunk
- Conversation mode: Generate coherent multi-turn conversations where each
  turn builds on the previous one within a chunk
"""

from typing import Literal, Optional

from loguru import logger

from fair_forge.schemas.generators import (
    BaseChunkSelectionStrategy,
    BaseContextLoader,
    BaseGenerator,
    Chunk,
    ConversationTurn,
    GeneratedConversationOutput,
    GeneratedQueriesOutput,
    GeneratedQuery,
)

from .alquimia_generator import AlquimiaGenerator
from .context_loaders import LocalMarkdownLoader
from .groq_generator import GroqGenerator
from .langchain_generator import LangChainGenerator
from .openai_generator import OpenAIGenerator
from .strategies import RandomSamplingStrategy, SequentialStrategy


def create_alquimia_generator(
    base_url: str,
    api_key: str,
    agent_id: str,
    channel_id: str,
    api_version: str = "",
) -> AlquimiaGenerator:
    """Create an Alquimia agent-based generator.

    The generator uses the Alquimia client to invoke an agent that generates
    test queries. Context, seed examples, and num_queries are passed as
    extra data kwargs that get injected into the agent's system prompt.

    Args:
        base_url: Alquimia API base URL
        api_key: Alquimia API key
        agent_id: Agent/assistant identifier for query generation
        channel_id: Channel identifier
        api_version: API version (optional)

    Returns:
        AlquimiaGenerator: Configured generator instance
    """
    logger.info("Creating Alquimia generator")
    return AlquimiaGenerator(
        base_url=base_url,
        api_key=api_key,
        agent_id=agent_id,
        channel_id=channel_id,
        api_version=api_version,
    )


def create_openai_generator(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: Optional[int] = 2048,
    api_key: Optional[str] = None,
    use_structured_output: bool = True,
) -> OpenAIGenerator:
    """Create an OpenAI-based generator.

    The API key is read from the OPENAI_API_KEY environment variable by default.

    Args:
        model_name: OpenAI model name (default: "gpt-4o-mini")
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
        api_key: Optional API key (defaults to OPENAI_API_KEY env var)
        use_structured_output: If True, use with_structured_output() for parsing

    Returns:
        OpenAIGenerator: Configured generator instance
    """
    logger.info(f"Creating OpenAI generator with model: {model_name}")
    return OpenAIGenerator(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        use_structured_output=use_structured_output,
    )


def create_groq_generator(
    model_name: str = "llama-3.1-70b-versatile",
    temperature: float = 0.7,
    max_tokens: Optional[int] = 2048,
    api_key: Optional[str] = None,
    use_structured_output: bool = True,
) -> GroqGenerator:
    """Create a Groq Cloud-based generator.

    The API key is read from the GROQ_API_KEY environment variable by default.

    Args:
        model_name: Groq model name (default: "llama-3.1-70b-versatile")
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
        api_key: Optional API key (defaults to GROQ_API_KEY env var)
        use_structured_output: If True, use with_structured_output() for parsing

    Returns:
        GroqGenerator: Configured generator instance
    """
    logger.info(f"Creating Groq generator with model: {model_name}")
    return GroqGenerator(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        use_structured_output=use_structured_output,
    )


def create_markdown_loader(
    max_chunk_size: int = 2000,
    min_chunk_size: int = 200,
    overlap: int = 100,
    header_levels: list[int] | None = None,
) -> LocalMarkdownLoader:
    """Create a local markdown context loader.

    Args:
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk
        overlap: Overlap between size-based chunks
        header_levels: Header levels to split on

    Returns:
        LocalMarkdownLoader: Configured loader instance
    """
    logger.info("Creating local markdown loader")
    return LocalMarkdownLoader(
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        overlap=overlap,
        header_levels=header_levels,
    )


def create_generator(
    backend: Literal["alquimia", "openai", "groq"],
    **kwargs,
) -> BaseGenerator:
    """Factory function to create a generator based on type.

    Args:
        backend: Generator backend type ("alquimia", "openai", or "groq")
        **kwargs: Backend-specific configuration

    Returns:
        BaseGenerator: Configured generator instance

    Raises:
        ValueError: If backend type is unknown

    Examples:
        >>> # Create OpenAI generator
        >>> generator = create_generator(
        ...     backend="openai",
        ...     model_name="gpt-4o-mini",
        ... )
        >>>
        >>> # Create Groq generator
        >>> generator = create_generator(
        ...     backend="groq",
        ...     model_name="llama-3.1-70b-versatile",
        ... )
        >>>
        >>> # Create Alquimia generator
        >>> generator = create_generator(
        ...     backend="alquimia",
        ...     base_url="https://api.alquimia.ai",
        ...     api_key="your-key",
        ...     model_id="your-model",
        ... )
    """
    if backend == "alquimia":
        return create_alquimia_generator(**kwargs)
    elif backend == "openai":
        return create_openai_generator(**kwargs)
    elif backend == "groq":
        return create_groq_generator(**kwargs)
    raise ValueError(
        f"Unknown generator backend: {backend}. "
        f"Valid options: 'alquimia', 'openai', 'groq'"
    )


def create_context_loader(
    loader_type: Literal["markdown"],
    **kwargs,
) -> BaseContextLoader:
    """Factory function to create a context loader based on type.

    Args:
        loader_type: Context loader type
        **kwargs: Loader-specific configuration

    Returns:
        BaseContextLoader: Configured loader instance

    Raises:
        ValueError: If loader type is unknown
    """
    if loader_type == "markdown":
        return create_markdown_loader(**kwargs)
    raise ValueError(f"Unknown context loader type: {loader_type}")


__all__ = [
    # Base classes and schemas
    "BaseGenerator",
    "BaseContextLoader",
    "BaseChunkSelectionStrategy",
    "Chunk",
    "GeneratedQuery",
    "GeneratedQueriesOutput",
    "ConversationTurn",
    "GeneratedConversationOutput",
    # Generator implementations
    "AlquimiaGenerator",
    "OpenAIGenerator",
    "GroqGenerator",
    "LangChainGenerator",
    "LocalMarkdownLoader",
    # Chunk selection strategies
    "SequentialStrategy",
    "RandomSamplingStrategy",
    # Factory functions
    "create_alquimia_generator",
    "create_openai_generator",
    "create_groq_generator",
    "create_markdown_loader",
    "create_generator",
    "create_context_loader",
]
