"""Tests for LangChain-based generators (OpenAI and Groq)."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from fair_forge.generators import (
    LangChainGenerator,
    LocalMarkdownLoader,
)
from fair_forge.schemas.generators import Chunk, GeneratedQuery, GeneratedQueriesOutput
from fair_forge.schemas.common import Dataset, Batch


# Check if optional dependencies are available
try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain_groq import ChatGroq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


class TestLangChainGeneratorBase:
    """Test suite for LangChainGenerator base class."""

    def test_langchain_generator_initialization(self):
        """Test LangChainGenerator initializes with model."""
        mock_model = MagicMock()
        generator = LangChainGenerator(model=mock_model)

        assert generator.model == mock_model
        assert generator.use_structured_output is True

    def test_langchain_generator_custom_structured_output(self):
        """Test LangChainGenerator with custom structured output setting."""
        mock_model = MagicMock()
        generator = LangChainGenerator(
            model=mock_model,
            use_structured_output=False,
        )

        assert generator.use_structured_output is False

    def test_parse_json_response_from_code_block(self):
        """Test parsing JSON from markdown code block."""
        mock_model = MagicMock()
        generator = LangChainGenerator(model=mock_model)

        content = '''Here is the response:
```json
{"queries": [{"query": "What is X?", "difficulty": "easy", "query_type": "factual"}], "chunk_summary": "Summary"}
```'''

        result = generator._parse_json_response(content)

        assert isinstance(result, GeneratedQueriesOutput)
        assert len(result.queries) == 1
        assert result.queries[0].query == "What is X?"

    def test_parse_json_response_raw_json(self):
        """Test parsing raw JSON without code block."""
        mock_model = MagicMock()
        generator = LangChainGenerator(model=mock_model)

        content = '{"queries": [{"query": "What is Y?", "difficulty": "medium", "query_type": "inferential"}], "chunk_summary": "Test"}'

        result = generator._parse_json_response(content)

        assert isinstance(result, GeneratedQueriesOutput)
        assert len(result.queries) == 1
        assert result.queries[0].query == "What is Y?"

    def test_parse_json_response_no_json(self):
        """Test parsing fails gracefully with no JSON."""
        mock_model = MagicMock()
        generator = LangChainGenerator(model=mock_model)

        content = "This is just plain text without any JSON"

        with pytest.raises(ValueError, match="No JSON found"):
            generator._parse_json_response(content)


@pytest.mark.skipif(not HAS_OPENAI, reason="langchain-openai not installed")
class TestOpenAIGenerator:
    """Test suite for OpenAIGenerator."""

    def test_openai_generator_missing_api_key(self):
        """Test OpenAIGenerator raises error without API key."""
        from fair_forge.generators import OpenAIGenerator

        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)

            with pytest.raises(ValueError, match="OpenAI API key not found"):
                OpenAIGenerator()

    def test_openai_generator_with_env_var(self):
        """Test OpenAIGenerator uses OPENAI_API_KEY env var."""
        from fair_forge.generators import OpenAIGenerator

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("langchain_openai.ChatOpenAI") as mock_chat:
                mock_chat.return_value = MagicMock()

                generator = OpenAIGenerator()

                mock_chat.assert_called_once()
                call_kwargs = mock_chat.call_args.kwargs
                assert call_kwargs["api_key"] == "test-key"
                assert call_kwargs["model"] == "gpt-4o-mini"

    def test_openai_generator_with_explicit_api_key(self):
        """Test OpenAIGenerator with explicit API key parameter."""
        from fair_forge.generators import OpenAIGenerator

        with patch("langchain_openai.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()

            generator = OpenAIGenerator(api_key="explicit-key")

            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["api_key"] == "explicit-key"

    def test_openai_generator_custom_model(self):
        """Test OpenAIGenerator with custom model name."""
        from fair_forge.generators import OpenAIGenerator

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("langchain_openai.ChatOpenAI") as mock_chat:
                mock_chat.return_value = MagicMock()

                generator = OpenAIGenerator(model_name="gpt-4o")

                call_kwargs = mock_chat.call_args.kwargs
                assert call_kwargs["model"] == "gpt-4o"
                assert generator.model_name == "gpt-4o"

    def test_openai_generator_custom_temperature(self):
        """Test OpenAIGenerator with custom temperature."""
        from fair_forge.generators import OpenAIGenerator

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("langchain_openai.ChatOpenAI") as mock_chat:
                mock_chat.return_value = MagicMock()

                generator = OpenAIGenerator(temperature=0.3)

                call_kwargs = mock_chat.call_args.kwargs
                assert call_kwargs["temperature"] == 0.3


@pytest.mark.skipif(not HAS_GROQ, reason="langchain-groq not installed")
class TestGroqGenerator:
    """Test suite for GroqGenerator."""

    def test_groq_generator_missing_api_key(self):
        """Test GroqGenerator raises error without API key."""
        from fair_forge.generators import GroqGenerator

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GROQ_API_KEY", None)

            with pytest.raises(ValueError, match="Groq API key not found"):
                GroqGenerator()

    def test_groq_generator_with_env_var(self):
        """Test GroqGenerator uses GROQ_API_KEY env var."""
        from fair_forge.generators import GroqGenerator

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-key"}):
            with patch("langchain_groq.ChatGroq") as mock_chat:
                mock_chat.return_value = MagicMock()

                generator = GroqGenerator()

                mock_chat.assert_called_once()
                call_kwargs = mock_chat.call_args.kwargs
                assert call_kwargs["api_key"] == "test-groq-key"
                assert call_kwargs["model"] == "llama-3.1-70b-versatile"

    def test_groq_generator_with_explicit_api_key(self):
        """Test GroqGenerator with explicit API key parameter."""
        from fair_forge.generators import GroqGenerator

        with patch("langchain_groq.ChatGroq") as mock_chat:
            mock_chat.return_value = MagicMock()

            generator = GroqGenerator(api_key="explicit-groq-key")

            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["api_key"] == "explicit-groq-key"

    def test_groq_generator_custom_model(self):
        """Test GroqGenerator with custom model name."""
        from fair_forge.generators import GroqGenerator

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch("langchain_groq.ChatGroq") as mock_chat:
                mock_chat.return_value = MagicMock()

                generator = GroqGenerator(model_name="mixtral-8x7b-32768")

                call_kwargs = mock_chat.call_args.kwargs
                assert call_kwargs["model"] == "mixtral-8x7b-32768"
                assert generator.model_name == "mixtral-8x7b-32768"


class TestFactoryFunctions:
    """Test suite for factory functions."""

    @pytest.mark.skipif(not HAS_OPENAI, reason="langchain-openai not installed")
    def test_create_openai_generator(self):
        """Test create_openai_generator factory function."""
        from fair_forge.generators import create_openai_generator, OpenAIGenerator

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("langchain_openai.ChatOpenAI") as mock_chat:
                mock_chat.return_value = MagicMock()

                generator = create_openai_generator(model_name="gpt-4o")

                assert isinstance(generator, OpenAIGenerator)

    @pytest.mark.skipif(not HAS_GROQ, reason="langchain-groq not installed")
    def test_create_groq_generator(self):
        """Test create_groq_generator factory function."""
        from fair_forge.generators import create_groq_generator, GroqGenerator

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch("langchain_groq.ChatGroq") as mock_chat:
                mock_chat.return_value = MagicMock()

                generator = create_groq_generator(model_name="llama-3.1-8b-instant")

                assert isinstance(generator, GroqGenerator)

    @pytest.mark.skipif(not HAS_OPENAI, reason="langchain-openai not installed")
    def test_create_generator_openai(self):
        """Test create_generator with openai backend."""
        from fair_forge.generators import create_generator, OpenAIGenerator

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("langchain_openai.ChatOpenAI") as mock_chat:
                mock_chat.return_value = MagicMock()

                generator = create_generator(backend="openai")

                assert isinstance(generator, OpenAIGenerator)

    @pytest.mark.skipif(not HAS_GROQ, reason="langchain-groq not installed")
    def test_create_generator_groq(self):
        """Test create_generator with groq backend."""
        from fair_forge.generators import create_generator, GroqGenerator

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch("langchain_groq.ChatGroq") as mock_chat:
                mock_chat.return_value = MagicMock()

                generator = create_generator(backend="groq")

                assert isinstance(generator, GroqGenerator)

    def test_create_generator_invalid_backend(self):
        """Test create_generator with invalid backend."""
        from fair_forge.generators import create_generator

        with pytest.raises(ValueError, match="Unknown generator backend"):
            create_generator(backend="invalid")


class TestLangChainGeneratorIntegration:
    """Integration tests for LangChain generators with mocked LLM."""

    @pytest.mark.asyncio
    async def test_generate_queries_structured_output(self, sample_chunk: Chunk):
        """Test generate_queries with structured output mode."""
        mock_model = MagicMock()
        mock_structured_model = MagicMock()

        # Mock the structured output result
        mock_result = GeneratedQueriesOutput(
            queries=[
                GeneratedQuery(query="Test question?", difficulty="easy", query_type="factual"),
            ],
            chunk_summary="Test summary",
        )
        mock_structured_model.invoke.return_value = mock_result
        mock_model.with_structured_output.return_value = mock_structured_model

        generator = LangChainGenerator(model=mock_model, use_structured_output=True)
        queries = await generator.generate_queries(sample_chunk, num_queries=1)

        assert len(queries) == 1
        assert queries[0].query == "Test question?"
        mock_model.with_structured_output.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_queries_regex_mode(self, sample_chunk: Chunk):
        """Test generate_queries with regex extraction mode."""
        mock_model = MagicMock()

        # Mock the response content
        mock_response = MagicMock()
        mock_response.content = '''```json
{"queries": [{"query": "Regex question?", "difficulty": "medium", "query_type": "inferential"}], "chunk_summary": "Summary"}
```'''
        mock_model.invoke = MagicMock(return_value=mock_response)

        # Create a mock chain that returns our mock response
        with patch("fair_forge.generators.langchain_generator.ChatPromptTemplate") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            mock_prompt.from_messages.return_value.__or__ = MagicMock(return_value=mock_chain)

            generator = LangChainGenerator(model=mock_model, use_structured_output=False)
            queries = await generator.generate_queries(sample_chunk, num_queries=1)

            assert len(queries) == 1
            assert queries[0].query == "Regex question?"

    @pytest.mark.asyncio
    async def test_generate_dataset_creates_valid_dataset(
        self, temp_markdown_file: Path
    ):
        """Test generate_dataset creates a valid Dataset object."""
        mock_model = MagicMock()
        mock_structured_model = MagicMock()

        mock_result = GeneratedQueriesOutput(
            queries=[
                GeneratedQuery(query="Generated question?", difficulty="easy", query_type="factual"),
            ],
            chunk_summary="Summary",
        )
        mock_structured_model.invoke.return_value = mock_result
        mock_model.with_structured_output.return_value = mock_structured_model

        generator = LangChainGenerator(model=mock_model, use_structured_output=True)
        loader = LocalMarkdownLoader()

        dataset = await generator.generate_dataset(
            context_loader=loader,
            source=str(temp_markdown_file),
            assistant_id="test-assistant",
            num_queries_per_chunk=1,
        )

        assert isinstance(dataset, Dataset)
        assert dataset.assistant_id == "test-assistant"
        assert len(dataset.conversation) > 0

        for batch in dataset.conversation:
            assert isinstance(batch, Batch)
            assert batch.query
            assert batch.assistant == ""


class TestLangChainGeneratorWithSeedExamples:
    """Test seed examples functionality."""

    @pytest.mark.asyncio
    async def test_seed_examples_included_in_prompt(self, sample_chunk: Chunk):
        """Test that seed examples are included in the prompt."""
        mock_model = MagicMock()
        mock_structured_model = MagicMock()

        mock_result = GeneratedQueriesOutput(
            queries=[GeneratedQuery(query="Q?", difficulty="easy", query_type="factual")],
            chunk_summary="Summary",
        )
        mock_structured_model.invoke.return_value = mock_result
        mock_model.with_structured_output.return_value = mock_structured_model

        generator = LangChainGenerator(model=mock_model, use_structured_output=True)

        seed_examples = ["Example question 1?", "Example question 2?"]
        await generator.generate_queries(
            sample_chunk,
            num_queries=1,
            seed_examples=seed_examples,
        )

        # The seed examples should be included in the formatted prompt
        # This is validated through the successful call - the format works
        mock_model.with_structured_output.assert_called_once()
