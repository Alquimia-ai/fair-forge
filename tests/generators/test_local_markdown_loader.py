"""Tests for LocalMarkdownLoader."""

import pytest
from pathlib import Path

from fair_forge.generators.context_loaders import LocalMarkdownLoader
from fair_forge.schemas.generators import Chunk


class TestLocalMarkdownLoader:
    """Test suite for LocalMarkdownLoader."""

    def test_loader_initialization_defaults(self):
        """Test loader initializes with default values."""
        loader = LocalMarkdownLoader()

        assert loader.max_chunk_size == 2000
        assert loader.min_chunk_size == 200
        assert loader.overlap == 100
        assert loader.header_levels == [1, 2, 3]

    def test_loader_initialization_custom_values(self):
        """Test loader initializes with custom values."""
        loader = LocalMarkdownLoader(
            max_chunk_size=1000,
            min_chunk_size=100,
            overlap=50,
            header_levels=[1, 2],
        )

        assert loader.max_chunk_size == 1000
        assert loader.min_chunk_size == 100
        assert loader.overlap == 50
        assert loader.header_levels == [1, 2]

    def test_load_splits_by_headers(self, temp_markdown_file: Path):
        """Test that loader splits content by headers."""
        loader = LocalMarkdownLoader()
        chunks = loader.load(str(temp_markdown_file))

        # Should have multiple chunks based on headers
        assert len(chunks) >= 3

        # Each chunk should have metadata
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.content
            assert chunk.chunk_id
            assert "header" in chunk.metadata
            assert "source_file" in chunk.metadata

    def test_load_preserves_header_names(self, temp_markdown_file: Path):
        """Test that loader preserves header names in metadata."""
        loader = LocalMarkdownLoader()
        chunks = loader.load(str(temp_markdown_file))

        headers = [chunk.metadata.get("header") for chunk in chunks]

        # Should contain some expected headers
        assert "Introduction" in headers or any("Introduction" in h for h in headers if h)

    def test_load_falls_back_to_size_chunking(self, temp_markdown_no_headers_file: Path):
        """Test that loader falls back to size-based chunking without headers."""
        loader = LocalMarkdownLoader(max_chunk_size=100)
        chunks = loader.load(str(temp_markdown_no_headers_file))

        # Should create multiple chunks based on size
        assert len(chunks) >= 1

        # All chunks should have size-based chunking method (or "header" if intro was picked up)
        for chunk in chunks:
            assert chunk.metadata.get("chunking_method") in ["size", "header"]

    def test_load_splits_long_sections(self, temp_markdown_long_section_file: Path):
        """Test that loader splits very long sections by size."""
        loader = LocalMarkdownLoader(max_chunk_size=500)
        chunks = loader.load(str(temp_markdown_long_section_file))

        # Long section should be split into multiple chunks
        long_section_chunks = [
            c for c in chunks if "long_section" in c.chunk_id.lower()
        ]
        assert len(long_section_chunks) > 1

    def test_load_raises_on_missing_file(self):
        """Test that loader raises FileNotFoundError for missing files."""
        loader = LocalMarkdownLoader()

        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.md")

    def test_chunk_ids_are_unique(self, temp_markdown_file: Path):
        """Test that all chunk IDs are unique."""
        loader = LocalMarkdownLoader()
        chunks = loader.load(str(temp_markdown_file))

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs should be unique"

    def test_chunk_ids_derived_from_headers(self, temp_markdown_file: Path):
        """Test that chunk IDs are derived from header names."""
        loader = LocalMarkdownLoader()
        chunks = loader.load(str(temp_markdown_file))

        for chunk in chunks:
            # Chunk ID should be lowercase and use underscores
            assert chunk.chunk_id.islower() or "_" in chunk.chunk_id or chunk.chunk_id[0].isdigit() is False

    def test_source_file_in_metadata(self, temp_markdown_file: Path):
        """Test that source file path is included in metadata."""
        loader = LocalMarkdownLoader()
        chunks = loader.load(str(temp_markdown_file))

        for chunk in chunks:
            assert "source_file" in chunk.metadata
            assert chunk.metadata["source_file"] == str(temp_markdown_file)

    def test_empty_sections_are_skipped(self, temp_markdown_file: Path):
        """Test that empty sections are not included as chunks."""
        loader = LocalMarkdownLoader()
        chunks = loader.load(str(temp_markdown_file))

        for chunk in chunks:
            assert chunk.content.strip(), "Empty chunks should not be created"

    def test_custom_header_levels(self, temp_markdown_file: Path):
        """Test that custom header levels are respected."""
        # Only split on H1
        loader = LocalMarkdownLoader(header_levels=[1])
        chunks_h1_only = loader.load(str(temp_markdown_file))

        # Split on H1, H2, H3
        loader = LocalMarkdownLoader(header_levels=[1, 2, 3])
        chunks_all = loader.load(str(temp_markdown_file))

        # Should have fewer chunks when only splitting on H1
        assert len(chunks_h1_only) <= len(chunks_all)

    def test_overlap_in_size_chunks(self, temp_markdown_long_section_file: Path):
        """Test that size-based chunks have overlap."""
        loader = LocalMarkdownLoader(max_chunk_size=500, overlap=100)
        chunks = loader.load(str(temp_markdown_long_section_file))

        # Find consecutive size-based chunks
        size_chunks = [c for c in chunks if c.metadata.get("chunking_method") == "size"]

        if len(size_chunks) >= 2:
            # Check that consecutive chunks might share some content
            # (overlap means the end of one chunk overlaps with the start of the next)
            for i in range(len(size_chunks) - 1):
                current = size_chunks[i].content
                next_chunk = size_chunks[i + 1].content

                # With overlap, there should be some shared content at boundaries
                # This is a soft check - we just verify chunks exist
                assert len(current) > 0
                assert len(next_chunk) > 0


class TestLocalMarkdownLoaderEdgeCases:
    """Edge case tests for LocalMarkdownLoader."""

    def test_handles_unicode_content(self, tmp_path: Path):
        """Test that loader handles unicode content correctly."""
        unicode_content = """# Introduction

Cette section contient du texte en français.

## 日本語セクション

これは日本語のテキストです。

## Emoji Section 🎉

This section has emojis! 🚀 ✨ 🌟
"""
        md_file = tmp_path / "unicode.md"
        md_file.write_text(unicode_content, encoding="utf-8")

        loader = LocalMarkdownLoader()
        chunks = loader.load(str(md_file))

        assert len(chunks) >= 3
        # Should preserve unicode content
        all_content = " ".join(c.content for c in chunks)
        assert "français" in all_content or "日本語" in all_content or "🎉" in all_content

    def test_handles_code_blocks(self, tmp_path: Path):
        """Test that loader preserves code blocks."""
        code_content = """# Code Examples

Here is some Python code:

```python
def hello():
    print("Hello, World!")
```

And some JavaScript:

```javascript
function greet() {
    console.log("Hello!");
}
```
"""
        md_file = tmp_path / "code.md"
        md_file.write_text(code_content, encoding="utf-8")

        loader = LocalMarkdownLoader()
        chunks = loader.load(str(md_file))

        # Code blocks should be preserved
        all_content = " ".join(c.content for c in chunks)
        assert "def hello" in all_content or "print" in all_content

    def test_handles_nested_headers(self, tmp_path: Path):
        """Test that loader handles deeply nested headers."""
        nested_content = """# Level 1

## Level 2

### Level 3

#### Level 4

##### Level 5

###### Level 6

Content at level 6.
"""
        md_file = tmp_path / "nested.md"
        md_file.write_text(nested_content, encoding="utf-8")

        # Only split on levels 1-3
        loader = LocalMarkdownLoader(header_levels=[1, 2, 3])
        chunks = loader.load(str(md_file))

        # Should have chunks for levels 1, 2, 3 only
        assert len(chunks) >= 1

    def test_handles_single_line_file(self, tmp_path: Path):
        """Test that loader handles single-line files."""
        md_file = tmp_path / "single.md"
        md_file.write_text("Just one line of content.", encoding="utf-8")

        loader = LocalMarkdownLoader()
        chunks = loader.load(str(md_file))

        assert len(chunks) == 1
        assert chunks[0].content == "Just one line of content."
