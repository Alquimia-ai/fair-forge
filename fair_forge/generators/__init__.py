"""Generators module for Fair Forge.

This module provides synthetic dataset generators for creating test datasets
from context documents to evaluate AI assistants.

The BaseGenerator class uses LangChain's BaseChatModel interface, allowing
any LangChain-compatible model to be used for generation.

Available generators:
- BaseGenerator: Base class that accepts any LangChain BaseChatModel
- AlquimiaGenerator: Implementation using Alquimia's agent API (adapter)
- AlquimiaChatModel: LangChain adapter for Alquimia agents

Chunk Selection Strategies:
- SequentialStrategy: Process all chunks in order (default behavior)
- RandomSamplingStrategy: Randomly sample chunks multiple times

Features:
- Independent query generation: Generate unrelated questions per chunk
- Conversation mode: Generate coherent multi-turn conversations where each
  turn builds on the previous one within a chunk

Usage with LangChain models:
    ```python
    from langchain_openai import ChatOpenAI
    from fair_forge.generators import BaseGenerator, create_markdown_loader

    # Create any LangChain-compatible model
    model = ChatOpenAI(model="gpt-4o-mini")

    # Create generator with the model
    generator = BaseGenerator(model=model)

    # Generate test datasets
    loader = create_markdown_loader()
    datasets = await generator.generate_dataset(
        context_loader=loader,
        source="./docs/knowledge_base.md",
        assistant_id="my-assistant",
    )
    ```

Usage with Alquimia:
    ```python
    from fair_forge.generators import create_alquimia_generator

    generator = create_alquimia_generator(
        base_url="https://api.alquimia.ai",
        api_key="your-key",
        agent_id="your-agent",
        channel_id="your-channel",
    )

    datasets = await generator.generate_dataset(...)
    ```
"""

from typing import Optional

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

from .alquimia_generator import AlquimiaChatModel, AlquimiaGenerator
from .context_loaders import LocalMarkdownLoader
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
    extra_data kwargs directly to the Alquimia agent.

    Args:
        base_url: Alquimia API base URL
        api_key: Alquimia API key
        agent_id: Agent/assistant identifier for query generation
        channel_id: Channel identifier
        api_version: API version (optional)

    Returns:
        AlquimiaGenerator: Configured generator instance

    Example:
        ```python
        generator = create_alquimia_generator(
            base_url="https://api.alquimia.ai",
            api_key="your-key",
            agent_id="your-agent",
            channel_id="your-channel",
        )
        ```
    """
    logger.info("Creating Alquimia generator")
    return AlquimiaGenerator(
        base_url=base_url,
        api_key=api_key,
        agent_id=agent_id,
        channel_id=channel_id,
        api_version=api_version,
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


def create_context_loader(
    loader_type: str,
    **kwargs,
) -> BaseContextLoader:
    """Factory function to create a context loader based on type.

    Args:
        loader_type: Context loader type ("markdown")
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
    "AlquimiaChatModel",
    "LocalMarkdownLoader",
    # Chunk selection strategies
    "SequentialStrategy",
    "RandomSamplingStrategy",
    # Factory functions
    "create_alquimia_generator",
    "create_markdown_loader",
    "create_context_loader",
]
