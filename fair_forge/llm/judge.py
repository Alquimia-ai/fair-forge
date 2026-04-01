"""Judge class for LLM-based evaluation."""

import json
import logging
import re
from typing import TypeVar

from langchain.agents import create_agent
from langchain.agents.factory import ProviderStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from fair_forge.utils.logging import VerboseLogger

T = TypeVar("T", bound=BaseModel)


class Judge:
    """LLM-based judge for evaluating AI responses.

    Supports two modes:
    - Structured output: Uses create_agent with response_format for schema validation
    - Regex extraction: Parses JSON from model response using regex patterns

    Each call to check() is atomic by default. When chat_history is enabled,
    both human queries and assistant responses are preserved across calls.

    Args:
        model: LangChain BaseChatModel instance
        use_structured_output: If True, use create_agent with response_format
        bos_json_clause: Opening marker for JSON block (default: ```json)
        eos_json_clause: Closing marker for JSON block (default: ```)
        chat_history: If True, accumulate conversation history across check() calls
    """

    def __init__(
        self,
        model: BaseChatModel,
        use_structured_output: bool = False,
        strict: bool = True,
        bos_json_clause: str = "```json",
        eos_json_clause: str = "```",
        verbose: bool = False,
        chat_history: bool = False,
    ):
        self.model = model
        self.use_structured_output = use_structured_output
        self.strict = strict
        self.bos_json_clause = bos_json_clause
        self.eos_json_clause = eos_json_clause
        self.verbose = verbose
        self.chat_history_enabled = chat_history
        self._chat_history: list[tuple[str, str]] = []
        self.logger = VerboseLogger(verbose=verbose)
        mode = "enabled" if chat_history else "disabled"
        self.logger.info(f"Judge initialized with chat_history {mode}")

    def check(
        self,
        system_prompt: str,
        query: str,
        data: dict,
        output_schema: type[T] | None = None,
    ) -> tuple[str, T | dict | None]:
        """Evaluate using the model.

        If use_structured_output=True and output_schema provided:
            - Renders the system prompt template with data variables
            - Uses create_agent with response_format for structured output
        Else:
            - Includes full JSON schema in prompt
            - Uses regex extraction from response

        Args:
            system_prompt: System prompt template for the evaluation
            query: User query to evaluate
            data: Template variables for the system prompt
            output_schema: Pydantic model for structured output validation

        Returns:
            Tuple of (reasoning_content, result) where result is either a Pydantic
            model instance (structured mode) or dict (regex mode). reasoning_content
            is extracted from LangChain's additional_kwargs when available.
        """
        if not self.chat_history_enabled and self._chat_history:
            self.logger.warning(
                f"Chat history is disabled but contains {len(self._chat_history)} stale entries. Clearing."
            )
            self._chat_history.clear()

        if self.chat_history_enabled:
            self.logger.debug(f"Chat history has {len(self._chat_history)} entries")

        if self.use_structured_output and output_schema:
            return self._check_structured(system_prompt, query, data, output_schema)
        return self._check_regex(system_prompt, query, data, output_schema)

    def _render_system_prompt(self, system_prompt: str, data: dict) -> str:
        return system_prompt.format_map(data)

    def _get_json_schema_for_prompt(self, schema: type[BaseModel]) -> str:
        schema_json = schema.model_json_schema()
        props = schema_json.get("properties", {})
        field_lines = [
            f'    "{name}": <{prop.get("type", "any")}> // {prop.get("description", "")}'
            for name, prop in props.items()
        ]
        fields_str = "\n".join(field_lines)
        return f"""
After your reasoning, provide ONLY the final answer in the following JSON format:
```json
{{
{fields_str}
}}
```

Do not include any additional text after the JSON.
"""

    def _check_structured(
        self,
        system_prompt: str,
        query: str,
        data: dict,
        output_schema: type[T],
    ) -> tuple[str, T | None]:
        rendered_system = self._render_system_prompt(system_prompt, data)
        agent = create_agent(
            model=self.model,
            response_format=ProviderStrategy(output_schema, strict=self.strict),
            system_prompt=rendered_system,
        )

        messages = [*self._chat_history, ("human", query)]

        max_retries = 5
        result = None
        for attempt in range(max_retries):
            try:
                result = agent.invoke({"messages": messages})
                parsed = result.get("structured_response")

                if parsed is None and attempt < max_retries - 1:
                    self.logger.warning(
                        f"Retry {attempt + 1}/{max_retries} - model returned invalid JSON. " f"Raw result: {result}"
                    )
                    continue

                break
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if "400" in str(e) and attempt < max_retries - 1:
                    continue
                raise

        reasoning = ""
        response_content = ""
        if result:
            for msg in reversed(result.get("messages", [])):
                reasoning_content = getattr(msg, "additional_kwargs", {}).get("reasoning_content", "")
                if reasoning_content:
                    reasoning = reasoning_content
                if not response_content and hasattr(msg, "content") and msg.content:
                    response_content = str(msg.content)
                if reasoning and response_content:
                    break

        if self.chat_history_enabled:
            self._chat_history.append(("human", query))
            parsed = result.get("structured_response") if result else None
            assistant_content = response_content or (
                json.dumps(parsed.model_dump() if isinstance(parsed, BaseModel) else parsed) if parsed else ""
            )
            if assistant_content:
                self._chat_history.append(("assistant", assistant_content))

        return reasoning, result.get("structured_response") if result else None

    def _check_regex(
        self,
        system_prompt: str,
        query: str,
        data: dict,
        output_schema: type[BaseModel] | None = None,
    ) -> tuple[str, dict | None]:
        if output_schema:
            schema_instruction = self._get_json_schema_for_prompt(output_schema)
            enhanced_prompt = system_prompt + schema_instruction
        else:
            enhanced_prompt = system_prompt

        messages = [("system", enhanced_prompt), *self._chat_history, ("human", query)]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.model
        response = chain.invoke(data)
        content = str(response.content)
        reasoning = response.additional_kwargs.get("reasoning_content", "")
        json_data = self._extract_json(content)

        if self.chat_history_enabled:
            self._chat_history.append(("human", query))
            self._chat_history.append(("assistant", content))

        return reasoning, json_data

    def _extract_json(self, text: str) -> dict | None:
        pattern = rf"{re.escape(self.bos_json_clause)}\s*(\{{.*?\}})\s*{re.escape(self.eos_json_clause)}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                logging.exception("[FAIR FORGE/JUDGE] JSON decode error")
                return None
        logging.error(f"[FAIR FORGE/JUDGE] No JSON found between {self.bos_json_clause} and {self.eos_json_clause}")
        return None
