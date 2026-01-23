"""Tests for AlquimiaGenerator and AlquimiaChatModel."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fair_forge.generators import (
    AlquimiaGenerator,
    LocalMarkdownLoader,
    create_alquimia_generator,
)
from fair_forge.generators.alquimia_generator import AlquimiaChatModel
from fair_forge.schemas.common import Batch, Dataset
from fair_forge.schemas.generators import (
    Chunk,
    GeneratedQueriesOutput,
    GeneratedQuery,
)


def _create_mock_alquimia_module(mock_client):
    """Create a mock alquimia_client module with the given client class."""
    mock_module = ModuleType("alquimia_client")
    mock_module.AlquimiaClient = MagicMock(return_value=mock_client)
    return mock_module


class TestAlquimiaGeneratorInitialization:
    """Test suite for AlquimiaGenerator initialization."""

    def test_generator_initialization(self, mock_alquimia_config: dict):
        """Test generator initializes with correct values."""
        generator = AlquimiaGenerator(**mock_alquimia_config)

        assert generator.base_url == "https://api.alquimia.ai"
        assert generator.api_key == "test-api-key"
        assert generator.agent_id == "test-agent"
        assert generator.channel_id == "test-channel"
        assert generator.api_version == ""

    def test_generator_strips_trailing_slash(self):
        """Test generator strips trailing slash from base_url."""
        generator = AlquimiaGenerator(
            base_url="https://api.alquimia.ai/",
            api_key="test",
            agent_id="test-agent",
            channel_id="test-channel",
        )

        assert generator.base_url == "https://api.alquimia.ai"

    def test_generator_creates_chat_model(self, mock_alquimia_config: dict):
        """Test generator creates an AlquimiaChatModel internally."""
        generator = AlquimiaGenerator(**mock_alquimia_config)

        assert isinstance(generator.model, AlquimiaChatModel)

    def test_factory_function_creates_generator(self, mock_alquimia_config: dict):
        """Test create_alquimia_generator factory function."""
        generator = create_alquimia_generator(**mock_alquimia_config)

        assert isinstance(generator, AlquimiaGenerator)
        assert generator.base_url == "https://api.alquimia.ai"


class TestAlquimiaChatModel:
    """Test suite for AlquimiaChatModel."""

    def test_chat_model_initialization(self, mock_alquimia_config: dict):
        """Test AlquimiaChatModel initializes correctly."""
        model = AlquimiaChatModel(**mock_alquimia_config)

        assert model.base_url == "https://api.alquimia.ai"
        assert model.api_key == "test-api-key"
        assert model.agent_id == "test-agent"
        assert model._llm_type == "alquimia"

    def test_chat_model_identifying_params(self, mock_alquimia_config: dict):
        """Test _identifying_params returns correct values."""
        model = AlquimiaChatModel(**mock_alquimia_config)

        params = model._identifying_params
        assert params["base_url"] == "https://api.alquimia.ai"
        assert params["agent_id"] == "test-agent"
        assert params["channel_id"] == "test-channel"


def _create_mock_alquimia_client(response_content: str):
    """Helper to create a mocked AlquimiaClient."""
    mock_client = AsyncMock()

    # Mock infer to return stream_id
    mock_client.infer.return_value = {"stream_id": "test-stream-123"}

    # Mock stream to return events with response
    async def mock_stream(stream_id):
        yield {"response": {"data": {"content": response_content}}}

    mock_client.stream = mock_stream

    # Setup async context manager
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    return mock_client


class TestAlquimiaGeneratorGenerateQueries:
    """Test suite for AlquimiaGenerator.generate_queries method."""

    @pytest.mark.asyncio
    async def test_generate_queries_calls_agent(
        self, mock_alquimia_config: dict, sample_chunk: Chunk, mock_alquimia_response: str
    ):
        """Test that generate_queries calls the Alquimia agent."""
        generator = AlquimiaGenerator(**mock_alquimia_config)

        mock_client = _create_mock_alquimia_client(mock_alquimia_response)
        mock_module = _create_mock_alquimia_module(mock_client)

        with patch.dict(sys.modules, {"alquimia_client": mock_module}):
            queries = await generator.generate_queries(sample_chunk, num_queries=2)

            assert len(queries) == 2
            assert all(isinstance(q, GeneratedQuery) for q in queries)

    @pytest.mark.asyncio
    async def test_generate_queries_handles_agent_error(self, mock_alquimia_config: dict, sample_chunk: Chunk):
        """Test that generate_queries raises on agent error."""
        generator = AlquimiaGenerator(**mock_alquimia_config)

        mock_client = AsyncMock()
        mock_client.infer.side_effect = Exception("Agent Error")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        mock_module = _create_mock_alquimia_module(mock_client)

        with patch.dict(sys.modules, {"alquimia_client": mock_module}):
            with pytest.raises(Exception, match="Agent Error"):
                await generator.generate_queries(sample_chunk, num_queries=2)

    @pytest.mark.asyncio
    async def test_generate_queries_handles_empty_response(self, mock_alquimia_config: dict, sample_chunk: Chunk):
        """Test that generate_queries raises on empty response."""
        generator = AlquimiaGenerator(**mock_alquimia_config)

        mock_client = AsyncMock()
        mock_client.infer.return_value = {"stream_id": "test-stream-123"}

        async def mock_stream(stream_id):
            yield {"response": {"data": {"content": ""}}}

        mock_client.stream = mock_stream
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        mock_module = _create_mock_alquimia_module(mock_client)

        with patch.dict(sys.modules, {"alquimia_client": mock_module}):
            with pytest.raises(ValueError, match="Empty response"):
                await generator.generate_queries(sample_chunk, num_queries=2)

    @pytest.mark.asyncio
    async def test_generate_queries_handles_no_stream_id(self, mock_alquimia_config: dict, sample_chunk: Chunk):
        """Test that generate_queries raises when no stream_id returned."""
        generator = AlquimiaGenerator(**mock_alquimia_config)

        mock_client = AsyncMock()
        mock_client.infer.return_value = {}  # No stream_id
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        mock_module = _create_mock_alquimia_module(mock_client)

        with patch.dict(sys.modules, {"alquimia_client": mock_module}):
            with pytest.raises(ValueError, match="No stream_id"):
                await generator.generate_queries(sample_chunk, num_queries=2)


class TestAlquimiaStructuredOutputParsing:
    """Test suite for AlquimiaGenerator response parsing via structured output."""

    def test_parse_json_in_markdown(self, mock_alquimia_config: dict):
        """Test parsing JSON wrapped in markdown code blocks."""
        from fair_forge.generators.alquimia_generator import AlquimiaStructuredOutputModel

        model = AlquimiaChatModel(**mock_alquimia_config)
        structured = AlquimiaStructuredOutputModel(model, GeneratedQueriesOutput)

        response = '```json\n{"queries": [{"query": "Test question?"}], "chunk_summary": "Summary"}\n```'

        result = structured._parse_response(response)

        assert len(result.queries) == 1
        assert result.queries[0].query == "Test question?"

    def test_parse_raw_json(self, mock_alquimia_config: dict):
        """Test parsing raw JSON response."""
        from fair_forge.generators.alquimia_generator import AlquimiaStructuredOutputModel

        model = AlquimiaChatModel(**mock_alquimia_config)
        structured = AlquimiaStructuredOutputModel(model, GeneratedQueriesOutput)

        response = '{"queries": [{"query": "Test question?", "difficulty": "easy"}], "chunk_summary": "Summary"}'

        result = structured._parse_response(response)

        assert len(result.queries) == 1
        assert result.queries[0].query == "Test question?"
        assert result.queries[0].difficulty == "easy"

    def test_parse_plain_text_questions(self, mock_alquimia_config: dict):
        """Test parsing plain text questions when JSON fails."""
        from fair_forge.generators.alquimia_generator import AlquimiaStructuredOutputModel

        model = AlquimiaChatModel(**mock_alquimia_config)
        structured = AlquimiaStructuredOutputModel(model, GeneratedQueriesOutput)

        response = """Here are some questions:
1. What is the main feature?
2. How does it work?
- Why is this important?
"""
        result = structured._parse_plain_text_response(response)

        assert len(result.queries) == 3
        assert "What is the main feature?" in result.queries[0].query
        assert "How does it work?" in result.queries[1].query
        assert "Why is this important?" in result.queries[2].query


class TestAlquimiaGeneratorGenerateDataset:
    """Test suite for AlquimiaGenerator.generate_dataset method."""

    @pytest.mark.asyncio
    async def test_generate_dataset_creates_valid_dataset(
        self,
        mock_alquimia_config: dict,
        temp_markdown_file: Path,
        mock_alquimia_response: str,
    ):
        """Test that generate_dataset creates a valid Dataset object."""
        generator = AlquimiaGenerator(**mock_alquimia_config)
        loader = LocalMarkdownLoader()

        mock_client = _create_mock_alquimia_client(mock_alquimia_response)
        mock_module = _create_mock_alquimia_module(mock_client)

        with patch.dict(sys.modules, {"alquimia_client": mock_module}):
            datasets = await generator.generate_dataset(
                context_loader=loader,
                source=str(temp_markdown_file),
                assistant_id="test-assistant",
                num_queries_per_chunk=2,
                language="english",
            )

            assert len(datasets) > 0
            dataset = datasets[0]
            assert isinstance(dataset, Dataset)
            assert dataset.assistant_id == "test-assistant"
            assert dataset.language == "english"
            assert len(dataset.conversation) > 0

    @pytest.mark.asyncio
    async def test_generate_dataset_batches_have_correct_structure(
        self,
        mock_alquimia_config: dict,
        temp_markdown_file: Path,
        mock_alquimia_response: str,
    ):
        """Test that generated batches have correct structure."""
        generator = AlquimiaGenerator(**mock_alquimia_config)
        loader = LocalMarkdownLoader()

        mock_client = _create_mock_alquimia_client(mock_alquimia_response)
        mock_module = _create_mock_alquimia_module(mock_client)

        with patch.dict(sys.modules, {"alquimia_client": mock_module}):
            datasets = await generator.generate_dataset(
                context_loader=loader,
                source=str(temp_markdown_file),
                assistant_id="test-assistant",
                num_queries_per_chunk=2,
            )

            dataset = datasets[0]
            for batch in dataset.conversation:
                assert isinstance(batch, Batch)
                assert batch.qa_id
                assert batch.query
                assert batch.assistant == ""  # Empty, to be filled by runner
                assert batch.ground_truth_assistant == ""
                assert "Generated from chunk" in batch.observation
                assert "chunk_id" in batch.agentic

    @pytest.mark.asyncio
    async def test_generate_dataset_unique_qa_ids(
        self,
        mock_alquimia_config: dict,
        temp_markdown_file: Path,
        mock_alquimia_response: str,
    ):
        """Test that all generated qa_ids are unique."""
        generator = AlquimiaGenerator(**mock_alquimia_config)
        loader = LocalMarkdownLoader()

        mock_client = _create_mock_alquimia_client(mock_alquimia_response)
        mock_module = _create_mock_alquimia_module(mock_client)

        with patch.dict(sys.modules, {"alquimia_client": mock_module}):
            datasets = await generator.generate_dataset(
                context_loader=loader,
                source=str(temp_markdown_file),
                assistant_id="test-assistant",
                num_queries_per_chunk=2,
            )

            dataset = datasets[0]
            qa_ids = [batch.qa_id for batch in dataset.conversation]
            assert len(qa_ids) == len(set(qa_ids)), "All qa_ids should be unique"

    @pytest.mark.asyncio
    async def test_generate_dataset_includes_context(
        self,
        mock_alquimia_config: dict,
        temp_markdown_file: Path,
        mock_alquimia_response: str,
    ):
        """Test that dataset includes combined context from all chunks."""
        generator = AlquimiaGenerator(**mock_alquimia_config)
        loader = LocalMarkdownLoader()

        mock_client = _create_mock_alquimia_client(mock_alquimia_response)
        mock_module = _create_mock_alquimia_module(mock_client)

        with patch.dict(sys.modules, {"alquimia_client": mock_module}):
            datasets = await generator.generate_dataset(
                context_loader=loader,
                source=str(temp_markdown_file),
                assistant_id="test-assistant",
            )

            dataset = datasets[0]
            # Context should be non-empty and contain content from the file
            assert dataset.context
            assert len(dataset.context) > 0


class TestAlquimiaGeneratorIntegration:
    """Integration tests for AlquimiaGenerator."""

    @pytest.mark.asyncio
    async def test_full_generation_flow(
        self,
        mock_alquimia_config: dict,
        temp_markdown_file: Path,
        mock_alquimia_response: str,
    ):
        """Test the full generation flow from file to dataset."""
        generator = AlquimiaGenerator(**mock_alquimia_config)
        loader = LocalMarkdownLoader()

        mock_client = _create_mock_alquimia_client(mock_alquimia_response)
        mock_module = _create_mock_alquimia_module(mock_client)

        with patch.dict(sys.modules, {"alquimia_client": mock_module}):
            # Generate datasets (now returns list)
            datasets = await generator.generate_dataset(
                context_loader=loader,
                source=str(temp_markdown_file),
                assistant_id="integration-test-assistant",
                num_queries_per_chunk=2,
                language="english",
            )

            # Verify complete dataset structure
            assert len(datasets) > 0
            dataset = datasets[0]
            assert dataset.session_id  # Should be a valid UUID
            assert dataset.assistant_id == "integration-test-assistant"
            assert dataset.language == "english"
            assert dataset.context  # Combined context
            assert len(dataset.conversation) > 0

            # Verify batches
            for batch in dataset.conversation:
                assert batch.qa_id
                assert batch.query
                assert batch.assistant == ""
                assert batch.agentic.get("chunk_id")

    def test_generator_has_required_attributes(self, mock_alquimia_config: dict):
        """Test that generator has all required attributes."""
        generator = AlquimiaGenerator(**mock_alquimia_config)

        assert hasattr(generator, "base_url")
        assert hasattr(generator, "api_key")
        assert hasattr(generator, "agent_id")
        assert hasattr(generator, "channel_id")
        assert hasattr(generator, "api_version")
        assert hasattr(generator, "model")
        assert isinstance(generator.model, AlquimiaChatModel)


class TestAlquimiaChatModelWithStructuredOutput:
    """Test AlquimiaChatModel.with_structured_output functionality."""

    def test_with_structured_output_returns_wrapper(self, mock_alquimia_config: dict):
        """Test that with_structured_output returns AlquimiaStructuredOutputModel."""
        from fair_forge.generators.alquimia_generator import AlquimiaStructuredOutputModel

        model = AlquimiaChatModel(**mock_alquimia_config)
        structured = model.with_structured_output(GeneratedQueriesOutput)

        assert isinstance(structured, AlquimiaStructuredOutputModel)
        assert structured._schema == GeneratedQueriesOutput
