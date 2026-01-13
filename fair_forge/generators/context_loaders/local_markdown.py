"""Local markdown file context loader with hybrid chunking."""

import re
from pathlib import Path
from typing import Optional

from loguru import logger

from fair_forge.schemas.generators import BaseContextLoader, Chunk


class LocalMarkdownLoader(BaseContextLoader):
    """Context loader for local markdown files with hybrid chunking.

    Uses a hybrid strategy:
    1. Primary: Split by markdown headers (H1, H2, H3, etc.)
    2. Fallback: Split by character count for long sections without headers

    Args:
        max_chunk_size: Maximum characters per chunk (default: 2000)
        min_chunk_size: Minimum characters per chunk (default: 200)
        overlap: Character overlap between size-based chunks (default: 100)
        header_levels: Header levels to split on (default: [1, 2, 3])
    """

    def __init__(
        self,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 200,
        overlap: int = 100,
        header_levels: Optional[list[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        self.header_levels = header_levels or [1, 2, 3]

    def _split_by_headers(self, content: str) -> list[tuple[str, str]]:
        """Split content by markdown headers.

        Args:
            content: Full markdown content

        Returns:
            list[tuple[str, str]]: List of (header, section_content) tuples
        """
        sections = []
        current_header = "Introduction"
        current_content: list[str] = []

        lines = content.split("\n")
        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                level = len(header_match.group(1))
                if level in self.header_levels:
                    # Save previous section
                    if current_content:
                        sections.append((current_header, "\n".join(current_content)))
                    current_header = header_match.group(2).strip()
                    current_content = []
                else:
                    current_content.append(line)
            else:
                current_content.append(line)

        # Don't forget the last section
        if current_content:
            sections.append((current_header, "\n".join(current_content)))

        return sections

    def _split_by_size(self, content: str, base_id: str) -> list[Chunk]:
        """Split content by character size with overlap.

        Args:
            content: Text content to split
            base_id: Base ID for chunk naming

        Returns:
            list[Chunk]: Size-based chunks
        """
        chunks = []
        start = 0
        part = 1

        while start < len(content):
            end = start + self.max_chunk_size

            # Try to break at a sentence or paragraph boundary
            if end < len(content):
                # Look for paragraph break
                para_break = content.rfind("\n\n", start, end)
                if para_break > start + self.min_chunk_size:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    sentence_break = content.rfind(". ", start, end)
                    if sentence_break > start + self.min_chunk_size:
                        end = sentence_break + 2

            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append(
                    Chunk(
                        content=chunk_content,
                        chunk_id=f"{base_id}_part{part}",
                        metadata={"chunking_method": "size"},
                    )
                )
                part += 1

            start = end - self.overlap if end < len(content) else end

        return chunks

    def load(self, source: str) -> list[Chunk]:
        """Load and chunk a markdown file.

        Args:
            source: Path to the markdown file

        Returns:
            list[Chunk]: Chunked content

        Raises:
            FileNotFoundError: If the markdown file does not exist
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {source}")

        if path.suffix.lower() not in [".md", ".markdown"]:
            logger.warning(f"File {source} may not be markdown format")

        logger.info(f"Loading markdown file: {source}")
        content = path.read_text(encoding="utf-8")

        # First, try header-based splitting
        sections = self._split_by_headers(content)

        chunks = []
        for i, (header, section_content) in enumerate(sections):
            section_content = section_content.strip()
            if not section_content:
                continue

            # Create base ID from header
            base_id = re.sub(r"[^a-zA-Z0-9]+", "_", header.lower()).strip("_")
            if not base_id:
                base_id = f"section_{i + 1}"

            # If section is too long, apply size-based splitting
            if len(section_content) > self.max_chunk_size:
                logger.debug(
                    f"Section '{header}' ({len(section_content)} chars) "
                    f"exceeds max size, splitting further"
                )
                sub_chunks = self._split_by_size(section_content, base_id)
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata["header"] = header
                    sub_chunk.metadata["source_file"] = str(path)
                chunks.extend(sub_chunks)
            else:
                chunks.append(
                    Chunk(
                        content=section_content,
                        chunk_id=base_id,
                        metadata={
                            "header": header,
                            "chunking_method": "header",
                            "source_file": str(path),
                        },
                    )
                )

        # If no header-based chunks were created, fall back to pure size-based
        if not chunks:
            logger.info("No header-based chunks created, using size-based chunking")
            chunks = self._split_by_size(content, path.stem)
            for chunk in chunks:
                chunk.metadata["source_file"] = str(path)

        logger.info(f"Created {len(chunks)} chunks from {source}")
        return chunks


__all__ = ["LocalMarkdownLoader"]
