"""Regulatory embedder for chunk retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as torch_functional
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from fair_forge.connectors import RegulatoryDocument


@dataclass
class Chunk:
    """A text chunk with source metadata."""

    text: str
    source: str
    chunk_index: int


@dataclass
class RetrievedChunk(Chunk):
    """A retrieved chunk with similarity score."""

    similarity: float = 0.0


@dataclass
class ChunkerConfig:
    """Configuration for text chunking."""

    chunk_size: int = 1000
    chunk_overlap: int = 100


@dataclass
class EmbedderConfig:
    """Configuration for the regulatory embedder."""

    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    max_length: int = 8192
    batch_size: int = 32
    top_k: int = 10
    similarity_threshold: float = 0.3
    task: str = "Given a user query, retrieve relevant passages that answer the query"
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)


class RegulatoryEmbedder:
    """
    Embeds regulatory documents and performs similarity-based retrieval.

    Uses Qwen3-Embedding for generating embeddings and retrieval.
    """

    def __init__(self, config: EmbedderConfig | None = None):
        """
        Initialize the regulatory embedder.

        Args:
            config: Embedder configuration. Uses defaults if not provided.
        """
        self.config = config or EmbedderConfig()
        self._tokenizer = None
        self._model = None
        self._chunks: list[Chunk] = []
        self._doc_embeddings: torch.Tensor | None = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                padding_side="left",
            )
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModel.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self._model.eval()
        return self._model

    def _chunk_text(self, text: str, source: str) -> list[Chunk]:
        """Split text into overlapping chunks."""
        text = re.sub(r"\n{3,}", "\n\n", text)

        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.config.chunker.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=source,
                        chunk_index=idx,
                    )
                )
            start += self.config.chunker.chunk_size - self.config.chunker.chunk_overlap
            idx += 1

        return chunks

    def load_corpus(self, documents: list[RegulatoryDocument]) -> int:
        """
        Load and chunk regulatory documents.

        Args:
            documents: List of regulatory documents to process.

        Returns:
            Total number of chunks created.
        """
        self._chunks = []
        for doc in documents:
            doc_chunks = self._chunk_text(doc.text, doc.source)
            self._chunks.extend(doc_chunks)

        self._doc_embeddings = None
        return len(self._chunks)

    def _last_token_pool(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the last token embedding using Qwen3 pooling."""
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def _embed_texts(self, texts: list[str], is_query: bool = False) -> torch.Tensor:
        """Encode texts using the embedding model."""
        if is_query:
            texts = [f"Instruct: {self.config.task}\nQuery:{t}" for t in texts]

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.model.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = self.model(**batch)
            embeddings = self._last_token_pool(
                outputs.last_hidden_state,
                batch["attention_mask"],
            )
            embeddings = torch_functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu()

    def _ensure_embeddings(self) -> None:
        """Ensure document embeddings are computed."""
        if self._doc_embeddings is not None:
            return

        if not self._chunks:
            self._doc_embeddings = torch.empty(0)
            return

        chunk_texts = [c.text for c in self._chunks]
        all_embeddings = []

        for i in range(0, len(chunk_texts), self.config.batch_size):
            emb = self._embed_texts(
                chunk_texts[i : i + self.config.batch_size],
                is_query=False,
            )
            all_embeddings.append(emb)

        self._doc_embeddings = torch.cat(all_embeddings, dim=0)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The query text.

        Returns:
            List of retrieved chunks above similarity threshold.
        """
        self._ensure_embeddings()

        if self._doc_embeddings is None or len(self._doc_embeddings) == 0:
            return []

        query_embedding = self._embed_texts([query], is_query=True)
        scores = (query_embedding @ self._doc_embeddings.T).squeeze(0).tolist()

        ranked = sorted(
            [(score, chunk) for score, chunk in zip(scores, self._chunks, strict=True)],
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, chunk in ranked[: self.config.top_k]:
            if score >= self.config.similarity_threshold:
                results.append(
                    RetrievedChunk(
                        text=chunk.text,
                        source=chunk.source,
                        chunk_index=chunk.chunk_index,
                        similarity=round(score, 4),
                    )
                )

        return results

    def retrieve_merged(
        self,
        user_query: str,
        agent_response: str,
    ) -> list[RetrievedChunk]:
        """
        Retrieve chunks for both user query and agent response, merged and deduplicated.

        Args:
            user_query: The user's query.
            agent_response: The agent's response.

        Returns:
            Merged list of unique chunks with maximum similarity scores.
        """
        query_chunks = self.retrieve(user_query)
        response_chunks = self.retrieve(agent_response)

        seen: dict[tuple[str, int], RetrievedChunk] = {}
        for chunk in query_chunks + response_chunks:
            key = (chunk.source, chunk.chunk_index)
            if key not in seen or chunk.similarity > seen[key].similarity:
                seen[key] = chunk

        return sorted(seen.values(), key=lambda c: c.similarity, reverse=True)


__all__ = [
    "Chunk",
    "ChunkerConfig",
    "EmbedderConfig",
    "RegulatoryEmbedder",
    "RetrievedChunk",
]
