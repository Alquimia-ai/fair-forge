"""Alquimia AI generator adapter implementation.

This module provides an adapter that wraps the Alquimia client as a LangChain
BaseChatModel, allowing it to be used with the BaseGenerator interface.
"""

import asyncio
import json
import re
import uuid
from datetime import datetime
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from loguru import logger
from pydantic import Field

from fair_forge.schemas.generators import (
    BaseGenerator,
    GeneratedConversationOutput,
    GeneratedQueriesOutput,
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

    def _extract_extra_data_from_messages(
        self, messages: List[BaseMessage]
    ) -> dict[str, Any]:
        """Extract extra data (context, num_queries, etc.) from system message.

        The system prompt contains the context and configuration that should
        be passed to the Alquimia agent as extra_data kwargs.

        Args:
            messages: List of messages to extract data from

        Returns:
            dict: Extra data to pass to the Alquimia agent
        """
        extra_data: dict[str, Any] = {}

        for message in messages:
            if isinstance(message, SystemMessage):
                content = str(message.content)

                # Extract context from the system prompt
                context_match = re.search(
                    r"Context:\s*\n(.*?)(?:\n\n|\nExample|\nGenerate)",
                    content,
                    re.DOTALL,
                )
                if context_match:
                    extra_data["context"] = context_match.group(1).strip()

                # Extract num_queries from the system prompt
                num_queries_match = re.search(
                    r"Generate exactly (\d+) questions", content
                )
                if num_queries_match:
                    extra_data["num_queries"] = int(num_queries_match.group(1))

                # Extract num_turns for conversation mode
                num_turns_match = re.search(
                    r"conversation with exactly (\d+) turns", content
                )
                if num_turns_match:
                    extra_data["num_turns"] = int(num_turns_match.group(1))
                    extra_data["conversation_mode"] = True

                # Extract seed examples if present
                seed_match = re.search(
                    r"Example questions \(match this style\):\s*\n(.*?)\n\n",
                    content,
                    re.DOTALL,
                )
                if seed_match:
                    seed_text = seed_match.group(1)
                    seeds = [
                        line.strip().lstrip("- ")
                        for line in seed_text.split("\n")
                        if line.strip()
                    ]
                    if seeds:
                        extra_data["seed_examples"] = seeds

        return extra_data

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
            **kwargs: Additional keyword arguments

        Returns:
            ChatResult: The generated response
        """
        # Extract the user query from the last human message
        user_query = ""
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                user_query = str(message.content)
                break

        # Extract extra data from system message
        extra_data = self._extract_extra_data_from_messages(messages)

        # Generate a session ID for this invocation
        session_id = f"gen_{uuid.uuid4().hex[:12]}"

        logger.debug(f"Invoking Alquimia agent with session {session_id}")

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
        from fair_forge.schemas.generators import (
            ConversationTurn,
            GeneratedConversationOutput,
            GeneratedQueriesOutput,
            GeneratedQuery,
        )

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

    This generator wraps the Alquimia client as a LangChain-compatible model,
    allowing it to be used with the BaseGenerator interface.

    The context, seed examples, and number of queries are passed as extra
    data kwargs that get injected into the agent's system prompt.

    Args:
        base_url: Alquimia API base URL
        api_key: Alquimia API key for authentication
        agent_id: Agent/assistant identifier for query generation
        channel_id: Channel identifier
        api_version: API version (optional)
        use_structured_output: If True, use structured output parsing

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
        use_structured_output: bool = True,
        **kwargs,
    ):
        """Initialize the Alquimia generator.

        Args:
            base_url: Alquimia API base URL
            api_key: Alquimia API key
            agent_id: Agent/assistant identifier
            channel_id: Channel identifier
            api_version: API version
            use_structured_output: If True, use structured output parsing
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

        # Initialize parent with the adapter
        super().__init__(
            model=model,
            use_structured_output=use_structured_output,
            **kwargs,
        )


__all__ = ["AlquimiaGenerator", "AlquimiaChatModel"]
