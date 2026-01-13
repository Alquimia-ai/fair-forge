"""Base LangChain generator implementation."""

import uuid
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from fair_forge.schemas.common import Batch, Dataset
from fair_forge.schemas.generators import (
    BaseContextLoader,
    BaseGenerator,
    Chunk,
    GeneratedQueriesOutput,
    GeneratedQuery,
)

DEFAULT_SYSTEM_PROMPT = """You are an expert at creating test questions for AI evaluation.
Your task is to generate diverse, high-quality questions based on the provided context.

Guidelines:
1. Generate questions that test different cognitive levels (recall, comprehension, application)
2. Ensure questions are grounded in the provided context
3. Vary question types: factual, inferential, comparative, analytical
4. Questions should be clear and unambiguous
5. Avoid yes/no questions; prefer open-ended questions that require substantive answers

Context:
{context}

{seed_examples_section}

Generate exactly {num_queries} questions based on this context."""


class LangChainGenerator(BaseGenerator):
    """Base generator implementation using LangChain.

    This generator uses any LangChain-compatible chat model to generate
    synthetic test queries from context documents.

    Args:
        model: LangChain BaseChatModel instance
        use_structured_output: If True, use with_structured_output() for parsing
    """

    def __init__(
        self,
        model: BaseChatModel,
        use_structured_output: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.use_structured_output = use_structured_output

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: str,
    ) -> GeneratedQueriesOutput:
        """Call the LLM and parse response.

        Args:
            prompt: User prompt
            system_prompt: System prompt for the LLM

        Returns:
            GeneratedQueriesOutput: Parsed structured output
        """
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", prompt),
        ])

        if self.use_structured_output:
            structured_model = self.model.with_structured_output(GeneratedQueriesOutput)
            chain = chat_prompt | structured_model
            result = chain.invoke({})
            return result
        else:
            chain = chat_prompt | self.model
            response = chain.invoke({})
            content = str(response.content)
            return self._parse_json_response(content)

    def _parse_json_response(self, content: str) -> GeneratedQueriesOutput:
        """Parse JSON from model response.

        Args:
            content: Raw model response content

        Returns:
            GeneratedQueriesOutput: Parsed output
        """
        import json
        import re

        # Try to extract JSON from markdown code blocks
        pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError(f"No JSON found in response: {content[:200]}")

        return GeneratedQueriesOutput.model_validate_json(json_str)

    async def generate_queries(
        self,
        chunk: Chunk,
        num_queries: int = 3,
        seed_examples: Optional[list[str]] = None,
        custom_system_prompt: Optional[str] = None,
    ) -> list[GeneratedQuery]:
        """Generate queries for a single chunk using LangChain.

        Args:
            chunk: Context chunk to generate queries from
            num_queries: Number of queries to generate
            seed_examples: Optional example queries for style guidance
            custom_system_prompt: Optional custom system prompt

        Returns:
            list[GeneratedQuery]: Generated queries
        """
        logger.debug(f"Generating {num_queries} queries for chunk {chunk.chunk_id}")

        seed_section = ""
        if seed_examples:
            examples_str = "\n".join(f"- {ex}" for ex in seed_examples)
            seed_section = f"\nExample questions (match this style):\n{examples_str}\n"

        system_prompt = custom_system_prompt or DEFAULT_SYSTEM_PROMPT
        formatted_prompt = system_prompt.format(
            context=chunk.content,
            seed_examples_section=seed_section,
            num_queries=num_queries,
        )

        # Add JSON format instruction if not using structured output
        if not self.use_structured_output:
            formatted_prompt += """

You MUST respond with a valid JSON object in this exact format:
{
    "queries": [
        {"query": "Your question here", "difficulty": "easy|medium|hard", "query_type": "factual|inferential|comparative|analytical"}
    ],
    "chunk_summary": "Brief 1-2 sentence summary of the context"
}"""

        try:
            output = await self._call_llm(
                prompt=f"Generate {num_queries} questions.",
                system_prompt=formatted_prompt,
            )
            logger.debug(
                f"Generated {len(output.queries)} queries for chunk {chunk.chunk_id}"
            )
            return output.queries
        except Exception as e:
            logger.error(f"Failed to generate queries for chunk {chunk.chunk_id}: {e}")
            raise

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
            context_loader: Loader to chunk the source document
            source: Path to the context document
            assistant_id: ID for the assistant being tested
            num_queries_per_chunk: Queries per chunk
            language: Language for queries
            seed_examples: Example queries for style guidance
            custom_system_prompt: Override default system prompt

        Returns:
            Dataset: Complete dataset with Batch objects
        """
        logger.info(f"Loading context from: {source}")
        chunks = context_loader.load(source)
        logger.info(f"Loaded {len(chunks)} chunks from source")

        session_id = str(uuid.uuid4())
        batches: list[Batch] = []
        full_context = "\n\n".join(chunk.content for chunk in chunks)

        for chunk in chunks:
            queries = await self.generate_queries(
                chunk=chunk,
                num_queries=num_queries_per_chunk,
                seed_examples=seed_examples,
                custom_system_prompt=custom_system_prompt,
            )

            for i, generated_query in enumerate(queries):
                qa_id = f"{chunk.chunk_id}_q{i + 1}"
                batch = Batch(
                    qa_id=qa_id,
                    query=generated_query.query,
                    assistant="",  # To be filled by runner
                    ground_truth_assistant="",  # Not applicable for generated queries
                    observation=f"Generated from chunk: {chunk.chunk_id}",
                    agentic=_build_agentic_metadata(chunk, generated_query),
                    ground_truth_agentic={},
                )
                batches.append(batch)

        logger.info(f"Generated {len(batches)} total queries across {len(chunks)} chunks")

        return Dataset(
            session_id=session_id,
            assistant_id=assistant_id,
            language=language,
            context=full_context,
            conversation=batches,
        )


def _build_agentic_metadata(
    chunk: Chunk, generated_query: GeneratedQuery
) -> dict[str, Any]:
    """Build agentic metadata for a batch.

    Args:
        chunk: The source chunk
        generated_query: The generated query

    Returns:
        dict: Agentic metadata
    """
    metadata: dict[str, Any] = {"chunk_id": chunk.chunk_id}
    if generated_query.difficulty:
        metadata["difficulty"] = generated_query.difficulty
    if generated_query.query_type:
        metadata["query_type"] = generated_query.query_type
    return metadata


__all__ = ["LangChainGenerator"]
