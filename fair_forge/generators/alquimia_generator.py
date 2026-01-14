"""Alquimia AI generator implementation."""

import json
import re
import uuid
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from fair_forge.schemas.common import Batch, Dataset
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

from .strategies import SequentialStrategy


class AlquimiaGenerator(BaseGenerator):
    """Generator implementation using Alquimia's agent API.

    This generator uses the Alquimia client to invoke an agent that generates
    synthetic test queries from context documents. The context, seed examples,
    and number of queries are passed as extra data kwargs that get injected
    into the agent's system prompt.

    Args:
        base_url: Alquimia API base URL
        api_key: Alquimia API key for authentication
        agent_id: Agent/assistant identifier for query generation
        channel_id: Channel identifier
        api_version: API version (optional)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        agent_id: str,
        channel_id: str,
        api_version: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.agent_id = agent_id
        self.channel_id = channel_id
        self.api_version = api_version

    async def _invoke_agent(
        self,
        query: str,
        session_id: str,
        extra_data: dict[str, Any],
    ) -> str:
        """Invoke the Alquimia agent and return the response.

        Args:
            query: User query to send to the agent
            session_id: Session identifier for conversation context
            extra_data: Additional kwargs to pass to the agent (injected into system prompt)

        Returns:
            str: Complete response from the agent

        Raises:
            ImportError: If alquimia_client is not installed
            ValueError: If no stream_id is returned or response is empty
        """
        try:
            from alquimia_client import AlquimiaClient
        except ImportError as e:
            raise ImportError(
                "alquimia-client is required for AlquimiaGenerator. "
                "Install it with: uv pip install alquimia-client"
            ) from e

        async with AlquimiaClient(
            base_url=self.base_url,
            api_key=self.api_key,
            api_version=self.api_version,
        ) as client:
            result = await client.infer(
                assistant_id=self.agent_id,
                session_id=session_id,
                channel=self.channel_id,
                query=query,
                date=datetime.now().strftime("%Y-%m-%d"),
                **extra_data,
            )

            stream_id = result.get("stream_id")
            if not stream_id:
                raise ValueError("No stream_id returned from agent")

            response = ""
            async for event in client.stream(stream_id):
                response = event["response"]["data"]["content"]

            if not response:
                raise ValueError("Empty response from agent")

            return response

    def _parse_response(self, response: str) -> GeneratedQueriesOutput:
        """Parse the agent response to extract generated queries.

        Args:
            response: Raw response from the agent

        Returns:
            GeneratedQueriesOutput: Parsed structured output
        """
        # Try to extract JSON from markdown code blocks
        pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # If no JSON found, try to parse as plain text queries
                return self._parse_plain_text_response(response)

        try:
            return GeneratedQueriesOutput.model_validate_json(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}, trying plain text")
            return self._parse_plain_text_response(response)

    def _parse_conversation_response(self, response: str) -> GeneratedConversationOutput:
        """Parse the agent response to extract conversation turns.

        Args:
            response: Raw response from the agent

        Returns:
            GeneratedConversationOutput: Parsed structured output
        """
        # Try to extract JSON from markdown code blocks
        pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # If no JSON found, try to parse as plain text
                return self._parse_plain_text_conversation(response)

        try:
            return GeneratedConversationOutput.model_validate_json(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse JSON conversation: {e}, trying plain text")
            return self._parse_plain_text_conversation(response)

    def _parse_plain_text_response(self, response: str) -> GeneratedQueriesOutput:
        """Parse plain text response when JSON parsing fails.

        Args:
            response: Raw response from the agent

        Returns:
            GeneratedQueriesOutput: Parsed output with queries extracted from text
        """
        queries = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Check if it looks like a question (ends with ?)
            if line.endswith("?"):
                # Remove common prefixes like "1.", "- ", "* ", etc.
                cleaned = re.sub(r"^[\d]+[\.\)]\s*", "", line)
                cleaned = re.sub(r"^[-*•]\s*", "", cleaned)
                if cleaned:
                    queries.append(GeneratedQuery(query=cleaned))

        if not queries:
            # If no questions found, treat the whole response as one query
            queries.append(GeneratedQuery(query=response.strip()))

        return GeneratedQueriesOutput(
            queries=queries,
            chunk_summary="Parsed from plain text response",
        )

    def _parse_plain_text_conversation(
        self, response: str
    ) -> GeneratedConversationOutput:
        """Parse plain text response as conversation turns.

        Args:
            response: Raw response from the agent

        Returns:
            GeneratedConversationOutput: Parsed conversation turns
        """
        turns = []
        lines = response.strip().split("\n")
        turn_number = 1

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.endswith("?"):
                cleaned = re.sub(r"^[\d]+[\.\)]\s*", "", line)
                cleaned = re.sub(r"^[-*•]\s*", "", cleaned)
                if cleaned:
                    turns.append(
                        ConversationTurn(
                            query=cleaned,
                            turn_number=turn_number,
                        )
                    )
                    turn_number += 1

        if not turns:
            turns.append(
                ConversationTurn(
                    query=response.strip(),
                    turn_number=1,
                )
            )

        return GeneratedConversationOutput(
            turns=turns,
            conversation_summary="Parsed from plain text response",
            chunk_summary="Parsed from plain text response",
        )

    async def generate_queries(
        self,
        chunk: Chunk,
        num_queries: int = 3,
        seed_examples: Optional[list[str]] = None,
        custom_system_prompt: Optional[str] = None,  # Not used with Alquimia
    ) -> list[GeneratedQuery]:
        """Generate queries for a single chunk using Alquimia agent.

        The context, seed_examples, and num_queries are passed as extra_data
        kwargs that get injected into the agent's system prompt.

        Args:
            chunk: Context chunk to generate queries from
            num_queries: Number of queries to generate
            seed_examples: Optional example queries for style guidance
            custom_system_prompt: Not used with Alquimia (ignored)

        Returns:
            list[GeneratedQuery]: Generated queries
        """
        logger.debug(f"Generating {num_queries} queries for chunk {chunk.chunk_id}")

        # Build extra data that will be injected into the system prompt
        extra_data: dict[str, Any] = {
            "context": chunk.content,
            "num_queries": num_queries,
        }

        if seed_examples:
            extra_data["seed_examples"] = seed_examples

        # Create a session ID for this generation
        session_id = f"gen_{chunk.chunk_id}_{uuid.uuid4().hex[:8]}"

        # Query to send to the agent
        query = (
            f"Generate {num_queries} diverse test questions based on the provided context. "
            f"Return the questions in JSON format with 'queries' array containing objects "
            f"with 'query', 'difficulty' (easy/medium/hard), and 'query_type' "
            f"(factual/inferential/comparative/analytical) fields."
        )

        try:
            response = await self._invoke_agent(query, session_id, extra_data)
            output = self._parse_response(response)
            logger.debug(
                f"Generated {len(output.queries)} queries for chunk {chunk.chunk_id}"
            )
            return output.queries
        except Exception as e:
            logger.error(f"Failed to generate queries for chunk {chunk.chunk_id}: {e}")
            raise

    async def generate_conversation(
        self,
        chunk: Chunk,
        num_turns: int = 3,
        seed_examples: Optional[list[str]] = None,
        custom_system_prompt: Optional[str] = None,  # Not used with Alquimia
    ) -> list[ConversationTurn]:
        """Generate a coherent multi-turn conversation from a chunk.

        Creates follow-up questions where each turn builds on the previous,
        maintaining a natural conversation flow exploring the chunk's content.

        Args:
            chunk: The context chunk to generate conversation from.
            num_turns: Number of conversation turns to generate.
            seed_examples: Optional example conversations to guide style.
            custom_system_prompt: Not used with Alquimia (ignored).

        Returns:
            list[ConversationTurn]: Ordered conversation turns.
        """
        logger.debug(
            f"Generating {num_turns}-turn conversation for chunk {chunk.chunk_id}"
        )

        # Build extra data that will be injected into the system prompt
        extra_data: dict[str, Any] = {
            "context": chunk.content,
            "num_turns": num_turns,
            "conversation_mode": True,
        }

        if seed_examples:
            extra_data["seed_examples"] = seed_examples

        # Create a session ID for this generation
        session_id = f"conv_{chunk.chunk_id}_{uuid.uuid4().hex[:8]}"

        # Query to send to the agent
        query = (
            f"Generate a coherent {num_turns}-turn conversation based on the provided context. "
            f"Each question should naturally follow from and build upon the previous one. "
            f"Start with a broad question, then drill down into specifics. "
            f"Return in JSON format with 'turns' array containing objects with 'query', "
            f"'turn_number', 'difficulty', 'query_type', and 'expected_context' fields."
        )

        try:
            response = await self._invoke_agent(query, session_id, extra_data)
            output = self._parse_conversation_response(response)
            logger.debug(
                f"Generated {len(output.turns)} turns for chunk {chunk.chunk_id}"
            )
            return output.turns
        except Exception as e:
            logger.error(
                f"Failed to generate conversation for chunk {chunk.chunk_id}: {e}"
            )
            raise

    async def generate_dataset(
        self,
        context_loader: BaseContextLoader,
        source: str,
        assistant_id: str,
        num_queries_per_chunk: int = 3,
        language: str = "english",
        seed_examples: Optional[list[str]] = None,
        custom_system_prompt: Optional[str] = None,  # Not used with Alquimia
        selection_strategy: Optional[BaseChunkSelectionStrategy] = None,
        conversation_mode: bool = False,
    ) -> list[Dataset]:
        """Generate datasets from a context source.

        Args:
            context_loader: Loader to chunk the source document
            source: Path to the context document
            assistant_id: ID for the assistant being tested
            num_queries_per_chunk: Queries/turns per chunk
            language: Language for queries
            seed_examples: Example queries for style guidance
            custom_system_prompt: Not used with Alquimia (ignored)
            selection_strategy: Strategy for selecting/grouping chunks.
                Defaults to SequentialStrategy (process all chunks once).
            conversation_mode: If True, generate coherent multi-turn
                conversations instead of independent queries.

        Returns:
            list[Dataset]: Generated datasets (one per chunk group from strategy).
        """
        logger.info(f"Loading context from: {source}")
        chunks = context_loader.load(source)
        logger.info(f"Loaded {len(chunks)} chunks from source")

        # Use default sequential strategy if none provided
        strategy = selection_strategy or SequentialStrategy()
        logger.info(f"Using chunk selection strategy: {strategy}")

        datasets: list[Dataset] = []

        # Process each chunk group from the strategy
        for chunk_group in strategy.select(chunks):
            session_id = str(uuid.uuid4())
            batches: list[Batch] = []
            full_context = "\n\n".join(chunk.content for chunk in chunk_group)

            if conversation_mode:
                # Generate coherent conversations per chunk
                for chunk in chunk_group:
                    turns = await self.generate_conversation(
                        chunk=chunk,
                        num_turns=num_queries_per_chunk,
                        seed_examples=seed_examples,
                    )

                    for turn in turns:
                        qa_id = f"{chunk.chunk_id}_t{turn.turn_number}"
                        batch = Batch(
                            qa_id=qa_id,
                            query=turn.query,
                            assistant="",
                            ground_truth_assistant="",
                            observation=f"Conversation turn {turn.turn_number} from chunk: {chunk.chunk_id}",
                            agentic=_build_conversation_metadata(chunk, turn),
                            ground_truth_agentic={},
                        )
                        batches.append(batch)
            else:
                # Generate independent queries per chunk
                for chunk in chunk_group:
                    queries = await self.generate_queries(
                        chunk=chunk,
                        num_queries=num_queries_per_chunk,
                        seed_examples=seed_examples,
                    )

                    for i, generated_query in enumerate(queries):
                        qa_id = f"{chunk.chunk_id}_q{i + 1}"
                        batch = Batch(
                            qa_id=qa_id,
                            query=generated_query.query,
                            assistant="",
                            ground_truth_assistant="",
                            observation=f"Generated from chunk: {chunk.chunk_id}",
                            agentic=_build_agentic_metadata(chunk, generated_query),
                            ground_truth_agentic={},
                        )
                        batches.append(batch)

            logger.info(
                f"Generated {len(batches)} batches for chunk group "
                f"({len(chunk_group)} chunks)"
            )

            dataset = Dataset(
                session_id=session_id,
                assistant_id=assistant_id,
                language=language,
                context=full_context,
                conversation=batches,
            )
            datasets.append(dataset)

        logger.info(
            f"Generated {len(datasets)} dataset(s) with "
            f"{sum(len(d.conversation) for d in datasets)} total batches"
        )

        return datasets


def _build_agentic_metadata(
    chunk: Chunk, generated_query: GeneratedQuery
) -> dict[str, Any]:
    """Build agentic metadata for a query batch.

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


def _build_conversation_metadata(
    chunk: Chunk, turn: ConversationTurn
) -> dict[str, Any]:
    """Build agentic metadata for a conversation turn batch.

    Args:
        chunk: The source chunk
        turn: The conversation turn

    Returns:
        dict: Agentic metadata including conversation tracking
    """
    metadata: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "turn_number": turn.turn_number,
        "conversation_mode": True,
    }
    if turn.turn_number > 1:
        metadata["builds_on"] = f"{chunk.chunk_id}_t{turn.turn_number - 1}"
    if turn.difficulty:
        metadata["difficulty"] = turn.difficulty
    if turn.query_type:
        metadata["query_type"] = turn.query_type
    if turn.expected_context:
        metadata["expected_context"] = turn.expected_context
    return metadata


__all__ = ["AlquimiaGenerator"]
