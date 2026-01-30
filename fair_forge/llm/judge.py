"""Judge class for LLM-based evaluation."""

import json
import logging
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from fair_forge.utils.logging import VerboseLogger

T = TypeVar("T", bound=BaseModel)


class Judge:
    """LLM-based judge for evaluating AI responses.

    Supports two modes:
    - Structured output: Uses LangChain's with_structured_output() for automatic
      schema validation via API
    - Regex extraction: Parses JSON from model response using regex patterns

    Reasoning content is automatically extracted from LangChain's
    additional_kwargs.reasoning_content when available.

    Args:
        model: LangChain BaseChatModel instance
        use_structured_output: If True, use with_structured_output() for parsing
        bos_json_clause: Opening marker for JSON block (default: ```json)
        eos_json_clause: Closing marker for JSON block (default: ```)
    """

    def __init__(
        self,
        model: BaseChatModel,
        use_structured_output: bool = False,
        bos_json_clause: str = "```json",
        eos_json_clause: str = "```",
        verbose: bool = False,
    ):
        self.model = model
        self.use_structured_output = use_structured_output
        self.bos_json_clause = bos_json_clause
        self.eos_json_clause = eos_json_clause
        self.verbose = verbose
        self.chat_history: list[tuple[str, str]] = []
        self.logger = VerboseLogger(verbose=verbose)

    def check(
        self,
        system_prompt: str,
        query: str,
        data: dict,
        output_schema: type[T] | None = None,
    ) -> tuple[str, T | dict | None]:
        """Evaluate using the model.

        If use_structured_output=True and output_schema provided:
            - Adds simple "return JSON" instruction to prompt
            - Uses model.with_structured_output(output_schema) - schema passed via API
        Else:
            - Includes full JSON schema in prompt
            - Uses regex extraction from response

        Args:
            system_prompt: System prompt for the evaluation
            query: User query to evaluate
            data: Template variables for the prompt
            output_schema: Pydantic model for structured output validation

        Returns:
            Tuple of (reasoning_content, result) where result is either a Pydantic
            model instance (structured mode) or dict (regex mode). reasoning_content
            is extracted from LangChain's additional_kwargs when available.
        """
        if self.use_structured_output and output_schema:
            return self._check_structured(system_prompt, query, data, output_schema)
        return self._check_regex(system_prompt, query, data, output_schema)

    def _get_json_schema_for_prompt(self, schema: type[BaseModel]) -> str:
        """Generate full JSON schema instruction to include in prompt."""
        schema_json = schema.model_json_schema()
        props = schema_json.get("properties", {})

        field_lines = []
        for name, prop in props.items():
            field_type = prop.get("type", "any")
            desc = prop.get("description", "")
            field_lines.append(f'    "{name}": <{field_type}> // {desc}')

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
        """Use with_structured_output for schema validation via API."""
        structured_model = self.model.with_structured_output(
            output_schema,
            include_raw=True,
        )
        self.chat_history.append(("human", query))
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), *self.chat_history])
        chain = prompt | structured_model

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = chain.invoke(data)

                # Extract parsed result
                parsed = result.get("parsed") if isinstance(result, dict) else result

                # Retry if model returned invalid JSON (None result)
                if parsed is None and attempt < max_retries - 1:
                    self.logger.warning(f"Retry {attempt + 1}/{max_retries} - model returned invalid JSON")
                    continue

                break
            except Exception as e:
                if "400" in str(e) and attempt < max_retries - 1:
                    self.logger.warning(f"Retry {attempt + 1}/{max_retries} after 400 error")
                    continue
                raise

        reasoning = ""

        # Extract reasoning from content_blocks if available
        raw_response = result.get("raw") if isinstance(result, dict) else None
        if raw_response is not None:
            content_blocks = getattr(raw_response, "content_blocks", None)
            if content_blocks and isinstance(content_blocks, list):
                reasoning_steps = [
                    b for b in content_blocks if isinstance(b, dict) and b.get("type") == "reasoning"
                ]
                reasoning = " ".join(step.get("reasoning", "") for step in reasoning_steps)

        return reasoning, parsed

    def _check_regex(
        self,
        system_prompt: str,
        query: str,
        data: dict,
        output_schema: type[BaseModel] | None = None,
    ) -> tuple[str, dict | None]:
        """Use regex extraction - full schema included in prompt if provided."""
        if output_schema:
            schema_instruction = self._get_json_schema_for_prompt(output_schema)
            enhanced_prompt = system_prompt + schema_instruction
        else:
            enhanced_prompt = system_prompt

        self.chat_history.append(("human", query))
        prompt = ChatPromptTemplate.from_messages([("system", enhanced_prompt), *self.chat_history])
        chain = prompt | self.model
        response = chain.invoke(data)
        content = str(response.content)

        reasoning = response.additional_kwargs.get("reasoning_content", "")

        json_data = self._extract_json(content)
        return reasoning, json_data

    def _extract_json(self, text: str) -> dict | None:
        """Extract JSON from text using configured markers."""
        import re

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
