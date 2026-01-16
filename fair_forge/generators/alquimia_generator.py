"""Alquimia AI generator adapter implementation.

This module provides an adapter that wraps the Alquimia client as a LangChain
BaseChatModel, allowing it to be used with the BaseGenerator interface.
"""

import asyncio
import re
import uuid
from datetime import datetime
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from loguru import logger
from pydantic import Field

from fair_forge.schemas.generators import (
    BaseGenerator,
    Chunk,
    ConversationTurn,
    GeneratedConversationOutput,
    GeneratedQueriesOutput,
    GeneratedQuery,
)


class AlquimiaChatModel(BaseChatModel):
    """LangChain chat model adapter for Alquimia's agent API.

    This adapter wraps the Alquimia client to implement LangChain's BaseChatModel
    interface, allowing Alquimia agents to be used with any LangChain-compatible
    tooling.

    Args:
        base_url: Alquimia API base URL
        api_key: Alquimia API key for authentication
        agent_id: Agent/assistant identifier for query generation
        channel_id: Channel identifier
        api_version: API version (optional)

    Example:
        ```python
        model = AlquimiaChatModel(
            base_url="https://api.alquimia.ai",
            api_key="your-api-key",
            agent_id="your-agent-id",
            channel_id="your-channel-id",
        )

        # Use with BaseGenerator
        generator = BaseGenerator(model=model)
        ```
    """

    base_url: str = Field(description="Alquimia API base URL")
    api_key: str = Field(description="Alquimia API key")
    agent_id: str = Field(description="Agent/assistant identifier")
    channel_id: str = Field(description="Channel identifier")
    api_version: str = Field(default="", description="API version")

    # Extra data to pass to the Alquimia agent (set before invoking)
    _extra_data: dict[str, Any] = {}

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "alquimia"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return identifying parameters for this LLM."""
        return {
            "base_url": self.base_url,
            "agent_id": self.agent_id,
            "channel_id": self.channel_id,
        }

    def set_extra_data(self, extra_data: dict[str, Any]) -> None:
        """Set extra data to pass to the Alquimia agent.

        Args:
            extra_data: Dictionary with context, num_queries, seed_examples, etc.
        """
        self._extra_data = extra_data

    def clear_extra_data(self) -> None:
        """Clear the extra data."""
        self._extra_data = {}

    async def _ainvoke_agent(
        self,
        query: str,
        session_id: str,
        extra_data: dict[str, Any],
    ) -> str:
        """Invoke the Alquimia agent asynchronously and return the response.

        Args:
            query: User query to send to the agent
            session_id: Session identifier for conversation context
            extra_data: Additional kwargs to pass to the agent

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
                "alquimia-client is required for AlquimiaChatModel. "
                "Install it with: uv pip install alquimia-client"
            ) from e

        # Strip trailing slash from base_url
        clean_base_url = self.base_url.rstrip("/")

        async with AlquimiaClient(
            base_url=clean_base_url,
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

    def _invoke_agent_sync(
        self,
        query: str,
        session_id: str,
        extra_data: dict[str, Any],
    ) -> str:
        """Synchronous wrapper for _ainvoke_agent.

        Args:
            query: User query to send to the agent
            session_id: Session identifier
            extra_data: Additional kwargs

        Returns:
            str: Complete response from the agent
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new event loop for this thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._ainvoke_agent(query, session_id, extra_data),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._ainvoke_agent(query, session_id, extra_data)
                )
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(self._ainvoke_agent(query, session_id, extra_data))

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the Alquimia agent.

        Args:
            messages: List of messages in the conversation
            stop: Optional list of stop sequences (not used by Alquimia)
            run_manager: Callback manager for LLM run
            **kwargs: Additional keyword arguments (can include extra_data)

        Returns:
            ChatResult: The generated response
        """
        # Extract the user query from the last human message
        user_query = ""
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                user_query = str(message.content)
                break

        # Use extra_data from kwargs, instance attribute, or empty dict
        extra_data = kwargs.get("extra_data", self._extra_data or {})

        # Generate a session ID for this invocation
        session_id = f"gen_{uuid.uuid4().hex[:12]}"

        logger.debug(f"Invoking Alquimia agent with session {session_id}")
        logger.debug(f"Extra data: {extra_data}")

        # Call the Alquimia agent
        response = self._invoke_agent_sync(user_query, session_id, extra_data)

        # Return as ChatResult
        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])

    def with_structured_output(
        self,
        schema: type,
        **kwargs: Any,
    ) -> "AlquimiaStructuredOutputModel":
        """Return a model that outputs structured data.

        Args:
            schema: Pydantic model class for output validation
            **kwargs: Additional configuration

        Returns:
            AlquimiaStructuredOutputModel: Model that parses output to schema
        """
        return AlquimiaStructuredOutputModel(
            base_model=self,
            schema=schema,
        )


class AlquimiaStructuredOutputModel(Runnable):
    """Wrapper for AlquimiaChatModel that parses output to structured schema.

    This class handles parsing the Alquimia agent's response into the expected
    Pydantic schema (GeneratedQueriesOutput or GeneratedConversationOutput).
    Inherits from Runnable to support LangChain's pipe operator.
    """

    def __init__(
        self,
        base_model: AlquimiaChatModel,
        schema: type,
    ):
        """Initialize with base model and output schema.

        Args:
            base_model: The underlying AlquimiaChatModel
            schema: Pydantic model class for output validation
        """
        self.base_model = base_model
        self._schema = schema

    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ) -> Any:
        """Invoke the model and parse output to schema.

        Args:
            input: Input value (can be dict or BaseMessage list from prompt)
            config: Optional runnable config
            **kwargs: Additional arguments

        Returns:
            Parsed output matching the schema
        """
        # Handle input from ChatPromptTemplate (list of messages)
        if isinstance(input, list) and len(input) > 0:
            messages = input
        elif isinstance(input, dict):
            messages = input.get("messages", [])
        else:
            messages = []

        result = self.base_model._generate(messages)
        content = result.generations[0].message.content

        return self._parse_response(str(content))

    def _parse_response(self, response: str) -> Any:
        """Parse the response to the output schema.

        Args:
            response: Raw response from the agent

        Returns:
            Parsed Pydantic model instance
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
                # Fall back to plain text parsing
                return self._parse_plain_text_response(response)

        try:
            return self._schema.model_validate_json(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}, trying plain text")
            return self._parse_plain_text_response(response)

    def _parse_plain_text_response(self, response: str) -> Any:
        """Parse plain text response when JSON parsing fails.

        Args:
            response: Raw response from the agent

        Returns:
            Parsed output with data extracted from text
        """
        if self._schema == GeneratedQueriesOutput:
            queries = []
            lines = response.strip().split("\n")

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.endswith("?"):
                    cleaned = re.sub(r"^[\d]+[\.\)]\s*", "", line)
                    cleaned = re.sub(r"^[-*•]\s*", "", cleaned)
                    if cleaned:
                        queries.append(GeneratedQuery(query=cleaned))

            if not queries:
                queries.append(GeneratedQuery(query=response.strip()))

            return GeneratedQueriesOutput(
                queries=queries,
                chunk_summary="Parsed from plain text response",
            )

        elif self._schema == GeneratedConversationOutput:
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

        else:
            raise ValueError(f"Unsupported output schema: {self._schema}")


class AlquimiaGenerator(BaseGenerator):
    """Generator implementation using Alquimia's agent API.

    This generator calls the Alquimia agent directly, passing context,
    seed examples, and configuration as extra_data kwargs.

    Args:
        base_url: Alquimia API base URL
        api_key: Alquimia API key for authentication
        agent_id: Agent/assistant identifier for query generation
        channel_id: Channel identifier
        api_version: API version (optional)

    Example:
        ```python
        generator = AlquimiaGenerator(
            base_url="https://api.alquimia.ai",
            api_key="your-api-key",
            agent_id="your-agent-id",
            channel_id="your-channel-id",
        )

        datasets = await generator.generate_dataset(
            context_loader=loader,
            source="./docs/knowledge_base.md",
            assistant_id="my-assistant",
        )
        ```
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
        """Initialize the Alquimia generator.

        Args:
            base_url: Alquimia API base URL
            api_key: Alquimia API key
            agent_id: Agent/assistant identifier
            channel_id: Channel identifier
            api_version: API version
            **kwargs: Additional configuration
        """
        # Create the Alquimia chat model adapter
        model = AlquimiaChatModel(
            base_url=base_url,
            api_key=api_key,
            agent_id=agent_id,
            channel_id=channel_id,
            api_version=api_version,
        )

        # Store config for reference
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.agent_id = agent_id
        self.channel_id = channel_id
        self.api_version = api_version

        # Initialize parent - always use_structured_output=False since we handle parsing
        super().__init__(
            model=model,
            use_structured_output=False,
            **kwargs,
        )

    async def generate_queries(
        self,
        chunk: Chunk,
        num_queries: int = 3,
        seed_examples: Optional[list[str]] = None,
        custom_system_prompt: Optional[str] = None,
    ) -> list[GeneratedQuery]:
        """Generate independent queries for a single chunk using Alquimia.

        Passes context and configuration directly as extra_data to the agent.

        Args:
            chunk: Context chunk to generate queries from
            num_queries: Number of queries to generate
            seed_examples: Optional example queries for style guidance
            custom_system_prompt: Optional custom system prompt (ignored for Alquimia)

        Returns:
            list[GeneratedQuery]: Generated queries
        """
        logger.debug(f"Generating {num_queries} queries for chunk {chunk.chunk_id}")

        # Build extra_data to pass to the Alquimia agent
        extra_data: dict[str, Any] = {
            "context": chunk.content,
            "num_queries": num_queries,
        }

        if seed_examples:
            extra_data["seed_examples"] = seed_examples

        # Set extra_data on the model and invoke
        self.model.set_extra_data(extra_data)

        try:
            # Generate a simple query to trigger the agent
            session_id = f"gen_{uuid.uuid4().hex[:12]}"
            response = await self.model._ainvoke_agent(
                query=f"Generate {num_queries} questions based on the provided context.",
                session_id=session_id,
                extra_data=extra_data,
            )

            # Parse the response
            output = self._parse_queries_response(response)
            logger.debug(
                f"Generated {len(output.queries)} queries for chunk {chunk.chunk_id}"
            )
            return output.queries

        except Exception as e:
            logger.error(f"Failed to generate queries for chunk {chunk.chunk_id}: {e}")
            raise
        finally:
            self.model.clear_extra_data()

    async def generate_conversation(
        self,
        chunk: Chunk,
        num_turns: int = 3,
        seed_examples: Optional[list[str]] = None,
        custom_system_prompt: Optional[str] = None,
    ) -> list[ConversationTurn]:
        """Generate a coherent multi-turn conversation from a chunk.

        Passes context and configuration directly as extra_data to the agent.

        Args:
            chunk: Context chunk for conversation
            num_turns: Number of conversation turns
            seed_examples: Optional example questions for style
            custom_system_prompt: Optional custom system prompt (ignored for Alquimia)

        Returns:
            list[ConversationTurn]: Generated conversation turns
        """
        logger.debug(f"Generating {num_turns}-turn conversation for chunk {chunk.chunk_id}")

        # Build extra_data to pass to the Alquimia agent
        extra_data: dict[str, Any] = {
            "context": chunk.content,
            "num_turns": num_turns,
            "conversation_mode": True,
        }

        if seed_examples:
            extra_data["seed_examples"] = seed_examples

        # Set extra_data on the model and invoke
        self.model.set_extra_data(extra_data)

        try:
            # Generate a simple query to trigger the agent
            session_id = f"gen_{uuid.uuid4().hex[:12]}"
            response = await self.model._ainvoke_agent(
                query=f"Generate a {num_turns}-turn conversation based on the provided context.",
                session_id=session_id,
                extra_data=extra_data,
            )

            # Parse the response
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
        finally:
            self.model.clear_extra_data()

    def _parse_queries_response(self, response: str) -> GeneratedQueriesOutput:
        """Parse the agent response into GeneratedQueriesOutput.

        Args:
            response: Raw response from the agent

        Returns:
            GeneratedQueriesOutput: Parsed output
        """
        # Try to extract JSON from markdown code blocks
        pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return GeneratedQueriesOutput.model_validate_json(json_str)
            except Exception:
                pass

        # Try to find raw JSON
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return GeneratedQueriesOutput.model_validate_json(json_match.group(0))
            except Exception:
                pass

        # Fall back to plain text parsing
        queries = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.endswith("?"):
                cleaned = re.sub(r"^[\d]+[\.\)]\s*", "", line)
                cleaned = re.sub(r"^[-*•]\s*", "", cleaned)
                if cleaned:
                    queries.append(GeneratedQuery(query=cleaned))

        if not queries:
            queries.append(GeneratedQuery(query=response.strip()))

        return GeneratedQueriesOutput(
            queries=queries,
            chunk_summary="Parsed from plain text response",
        )

    def _parse_conversation_response(self, response: str) -> GeneratedConversationOutput:
        """Parse the agent response into GeneratedConversationOutput.

        Args:
            response: Raw response from the agent

        Returns:
            GeneratedConversationOutput: Parsed output
        """
        # Try to extract JSON from markdown code blocks
        pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return GeneratedConversationOutput.model_validate_json(json_str)
            except Exception:
                pass

        # Try to find raw JSON
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return GeneratedConversationOutput.model_validate_json(json_match.group(0))
            except Exception:
                pass

        # Fall back to plain text parsing
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


__all__ = ["AlquimiaGenerator", "AlquimiaChatModel"]
