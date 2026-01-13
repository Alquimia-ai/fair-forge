"""Fixtures for generator tests."""

import pytest
from pathlib import Path
import tempfile

from fair_forge.schemas.generators import Chunk, GeneratedQuery


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Fixture providing sample chunks."""
    return [
        Chunk(
            content="Alquimia AI is an enterprise platform for building and deploying AI assistants. It provides tools for creating conversational agents that can understand context and respond appropriately.",
            chunk_id="about_alquimia",
            metadata={"header": "About Alquimia", "chunking_method": "header"},
        ),
        Chunk(
            content="The seven principles guide all decisions in the platform: transparency, fairness, accountability, privacy, security, reliability, and interpretability.",
            chunk_id="principles",
            metadata={"header": "Principles", "chunking_method": "header"},
        ),
    ]


@pytest.fixture
def sample_chunk() -> Chunk:
    """Fixture providing a single sample chunk."""
    return Chunk(
        content="Fair Forge is a performance-measurement library for evaluating AI models. It provides metrics for fairness, toxicity, bias, and conversational quality.",
        chunk_id="fair_forge_intro",
        metadata={"header": "Introduction", "chunking_method": "header"},
    )


@pytest.fixture
def sample_generated_queries() -> list[GeneratedQuery]:
    """Fixture providing sample generated queries."""
    return [
        GeneratedQuery(
            query="What is the main purpose of Fair Forge?",
            difficulty="easy",
            query_type="factual",
        ),
        GeneratedQuery(
            query="How does Fair Forge evaluate AI models for bias?",
            difficulty="medium",
            query_type="inferential",
        ),
        GeneratedQuery(
            query="Compare the toxicity and fairness metrics in Fair Forge.",
            difficulty="hard",
            query_type="comparative",
        ),
    ]


@pytest.fixture
def sample_markdown_content() -> str:
    """Fixture providing sample markdown content."""
    return """# Introduction

This is an introduction to our platform. It provides powerful tools for AI evaluation.

## Features

Our platform has many features including:
- Feature A: Automated testing
- Feature B: Metric collection
- Feature C: Report generation

## Getting Started

Follow these steps to begin using the platform.

### Prerequisites

You need Python 3.11+ installed on your system.

### Installation

Run the following command to install:
```
pip install fair-forge
```

## Advanced Usage

For advanced users, there are additional configuration options available.
"""


@pytest.fixture
def sample_markdown_no_headers() -> str:
    """Fixture providing markdown content without headers."""
    return """This is a plain text document without any markdown headers.

It contains multiple paragraphs of content that should be chunked by size.

The content discusses various topics but doesn't use any heading structure.

Each paragraph provides different information about the subject matter.
"""


@pytest.fixture
def sample_markdown_long_section() -> str:
    """Fixture providing markdown with a very long section."""
    long_content = "This is a very long paragraph. " * 200
    return f"""# Short Section

This is a short introduction.

## Long Section

{long_content}

## Another Short Section

This is the conclusion.
"""


@pytest.fixture
def temp_markdown_file(sample_markdown_content: str):
    """Fixture providing a temporary markdown file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(sample_markdown_content)
        f.flush()
        yield Path(f.name)
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_markdown_no_headers_file(sample_markdown_no_headers: str):
    """Fixture providing a temporary markdown file without headers."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(sample_markdown_no_headers)
        f.flush()
        yield Path(f.name)
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_markdown_long_section_file(sample_markdown_long_section: str):
    """Fixture providing a temporary markdown file with a long section."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(sample_markdown_long_section)
        f.flush()
        yield Path(f.name)
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def mock_llm_response() -> dict:
    """Fixture providing mock LLM response in JSON format."""
    return {
        "queries": [
            {"query": "What is the main purpose?", "difficulty": "easy", "query_type": "factual"},
            {"query": "How does it work?", "difficulty": "medium", "query_type": "inferential"},
        ],
        "chunk_summary": "Introduction to the platform.",
    }


@pytest.fixture
def mock_alquimia_response() -> str:
    """Fixture providing mock Alquimia agent response string."""
    return '```json\n{"queries": [{"query": "What is the main purpose?", "difficulty": "easy", "query_type": "factual"}, {"query": "How does it work?", "difficulty": "medium", "query_type": "inferential"}], "chunk_summary": "Introduction to the platform."}\n```'


@pytest.fixture
def mock_alquimia_config() -> dict:
    """Fixture providing mock Alquimia generator configuration."""
    return {
        "base_url": "https://api.alquimia.ai",
        "api_key": "test-api-key",
        "agent_id": "test-agent",
        "channel_id": "test-channel",
        "api_version": "",
    }
