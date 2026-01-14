"""Generator schemas and interfaces for Fair Forge.

This module defines the abstract base classes and Pydantic models for
synthetic dataset generation from context documents.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional

from pydantic import BaseModel, Field

from .common import Dataset


class Chunk(BaseModel):
    """Represents a chunk of context for query generation.

    Attributes:
        content: The text content of the chunk
        chunk_id: Unique identifier for this chunk
        metadata: Optional metadata (e.g., header hierarchy, source file)
    """

    content: str
    chunk_id: str
    metadata: Optional[dict] = Field(default_factory=dict)


class GeneratedQuery(BaseModel):
    """A single generated query from the LLM.

    Attributes:
        query: The generated question/query text
        difficulty: Optional difficulty level (easy, medium, hard)
        query_type: Type of query (factual, inferential, application, etc.)
    """

    query: str
    difficulty: Optional[str] = None
    query_type: Optional[str] = None


class GeneratedQueriesOutput(BaseModel):
    """Structured output schema for LLM query generation.

    Attributes:
        queries: List of generated queries for a chunk
        chunk_summary: Brief summary of the chunk content
    """

    queries: list[GeneratedQuery] = Field(
        description="List of generated queries based on the chunk content"
    )
    chunk_summary: str = Field(
        description="Brief summary of the chunk content (1-2 sentences)"
    )


class ConversationTurn(BaseModel):
    """A single turn in a contextualized conversation.

    Represents one query in a multi-turn conversation where each turn
    builds on the previous one for coherent dialogue flow.

    Attributes:
        query: The question/query for this turn
        expected_context: What this turn builds upon (context from previous turns)
        difficulty: Optional difficulty level (easy, medium, hard)
        query_type: Type of query (factual, inferential, follow-up, etc.)
        turn_number: Position in the conversation (1-indexed)
    """

    query: str
    expected_context: Optional[str] = Field(
        default=None,
        description="Context this turn builds upon from previous turns",
    )
    difficulty: Optional[str] = None
    query_type: Optional[str] = None
    turn_number: int = Field(ge=1, description="Position in conversation (1-indexed)")


class GeneratedConversationOutput(BaseModel):
    """Structured output schema for conversation generation.

    Used when generating coherent multi-turn conversations from a chunk.

    Attributes:
        turns: List of conversation turns in order
        conversation_summary: Brief description of the conversation flow
        chunk_summary: Summary of the source chunk content
    """

    turns: list[ConversationTurn] = Field(
        description="Ordered list of conversation turns",
    )
    conversation_summary: str = Field(
        description="Brief summary of the conversation flow and topics covered",
    )
    chunk_summary: str = Field(
        description="Brief summary of the chunk content (1-2 sentences)",
    )


class BaseContextLoader(ABC):
    """Abstract base class for loading and chunking context documents.

    Context loaders are responsible for reading source documents and
    splitting them into chunks suitable for query generation.
    """

    def __init__(self, **kwargs):
        """Initialize the context loader with optional configuration.

        Args:
            **kwargs: Implementation-specific configuration parameters.
        """
        self.kwargs = kwargs

    @abstractmethod
    def load(self, source: str) -> list[Chunk]:
        """Load and chunk a context document.

        Args:
            source: Path to the document or document identifier.

        Returns:
            list[Chunk]: List of document chunks ready for processing.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Must be implemented by subclass")


class BaseChunkSelectionStrategy(ABC):
    """Abstract base class for chunk selection strategies.

    Strategies determine how chunks are selected and grouped for
    dataset generation. Each yielded group becomes a separate dataset.

    Example strategies:
        - Sequential: Process all chunks in order (single group)
        - Random Sampling: Randomly select k chunks, repeat n times
        - Clustering: Group related chunks together
    """

    @abstractmethod
    def select(self, chunks: list[Chunk]) -> Iterator[list[Chunk]]:
        """Select and group chunks for processing.

        Args:
            chunks: All available chunks from the context loader.

        Yields:
            list[Chunk]: Groups of chunks to process together.
                Each group becomes a separate dataset.

        Example:
            >>> strategy = SequentialStrategy()
            >>> for chunk_group in strategy.select(chunks):
            ...     # Process this group as one dataset
            ...     pass
        """
        raise NotImplementedError("Must be implemented by subclass")


class BaseGenerator(ABC):
    """Abstract base class for synthetic dataset generators.

    Generators use LLMs to create test queries from context chunks
    and produce Dataset objects compatible with Fair Forge metrics.
    """

    def __init__(self, **kwargs):
        """Initialize the generator with configuration.

        Args:
            **kwargs: Implementation-specific configuration parameters.
        """
        self.kwargs = kwargs

    @abstractmethod
    async def generate_queries(
        self,
        chunk: Chunk,
        num_queries: int = 3,
        seed_examples: Optional[list[str]] = None,
    ) -> list[GeneratedQuery]:
        """Generate independent queries for a single context chunk.

        Args:
            chunk: The context chunk to generate queries from.
            num_queries: Number of queries to generate per chunk.
            seed_examples: Optional example queries to guide generation.

        Returns:
            list[GeneratedQuery]: Generated queries for the chunk.
        """
        raise NotImplementedError("Must be implemented by subclass")

    @abstractmethod
    async def generate_conversation(
        self,
        chunk: Chunk,
        num_turns: int = 3,
        seed_examples: Optional[list[str]] = None,
    ) -> list[ConversationTurn]:
        """Generate a coherent multi-turn conversation from a chunk.

        Creates follow-up questions where each turn builds on the previous,
        maintaining a natural conversation flow exploring the chunk's content.

        Args:
            chunk: The context chunk to generate conversation from.
            num_turns: Number of conversation turns to generate.
            seed_examples: Optional example conversations to guide style.

        Returns:
            list[ConversationTurn]: Ordered conversation turns.
        """
        raise NotImplementedError("Must be implemented by subclass")

    @abstractmethod
    async def generate_dataset(
        self,
        context_loader: BaseContextLoader,
        source: str,
        assistant_id: str,
        num_queries_per_chunk: int = 3,
        language: str = "english",
        seed_examples: Optional[list[str]] = None,
        custom_system_prompt: Optional[str] = None,
        selection_strategy: Optional["BaseChunkSelectionStrategy"] = None,
        conversation_mode: bool = False,
    ) -> list[Dataset]:
        """Generate datasets from a context source.

        Args:
            context_loader: Loader to read and chunk the source document.
            source: Path or identifier for the context document.
            assistant_id: ID for the assistant being tested.
            num_queries_per_chunk: Queries/turns to generate per chunk.
            language: Language for the generated queries.
            seed_examples: Example queries to guide style/format.
            custom_system_prompt: Override the default system prompt.
            selection_strategy: Strategy for selecting/grouping chunks.
                Defaults to SequentialStrategy (process all chunks once).
            conversation_mode: If True, generate coherent multi-turn
                conversations instead of independent queries.

        Returns:
            list[Dataset]: Generated datasets (one per chunk group from strategy).
                When using default sequential strategy, returns single-element list.
        """
        raise NotImplementedError("Must be implemented by subclass")
