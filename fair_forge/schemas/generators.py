"""Generator schemas and interfaces for Fair Forge.

This module defines the abstract base classes and Pydantic models for
synthetic dataset generation from context documents.
"""

from abc import ABC, abstractmethod
from typing import Optional

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
        """Generate queries for a single context chunk.

        Args:
            chunk: The context chunk to generate queries from.
            num_queries: Number of queries to generate per chunk.
            seed_examples: Optional example queries to guide generation.

        Returns:
            list[GeneratedQuery]: Generated queries for the chunk.
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
    ) -> Dataset:
        """Generate a complete dataset from a context source.

        Args:
            context_loader: Loader to read and chunk the source document.
            source: Path or identifier for the context document.
            assistant_id: ID for the assistant being tested.
            num_queries_per_chunk: Queries to generate per chunk.
            language: Language for the generated queries.
            seed_examples: Example queries to guide style/format.
            custom_system_prompt: Override the default system prompt.

        Returns:
            Dataset: Complete dataset with generated queries.
        """
        raise NotImplementedError("Must be implemented by subclass")
