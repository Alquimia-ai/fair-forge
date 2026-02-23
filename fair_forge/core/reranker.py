"""Regulatory reranker for contradiction detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from fair_forge.core.embedder import RetrievedChunk


@dataclass
class RankedChunk:
    """A chunk with reranker verdict."""

    text: str
    source: str
    chunk_index: int
    similarity: float
    reranker_score: float
    verdict: Literal["SUPPORTS", "CONTRADICTS"]


@dataclass
class RerankerConfig:
    """Configuration for the regulatory reranker."""

    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    max_length: int = 8192
    contradiction_threshold: float = 0.6


class RegulatoryReranker:
    """
    Reranks retrieved chunks to detect contradictions with agent responses.

    Uses Qwen3-Reranker to determine if chunks support or contradict responses.
    """

    def __init__(self, config: RerankerConfig | None = None):
        """
        Initialize the regulatory reranker.

        Args:
            config: Reranker configuration. Uses defaults if not provided.
        """
        self.config = config or RerankerConfig()
        self._tokenizer = None
        self._model = None

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
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self._model.eval()
        return self._model

    def _format_input(self, instruction: str, query: str, doc: str) -> str:
        """Format input for the reranker model."""
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _compute_scores(self, pairs: list[str]) -> list[float]:
        """Compute reranker scores for input pairs."""
        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Query meets the requirements based on the Document "
            'and the Instruct provided. Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        effective_max = self.config.max_length - len(prefix_tokens) - len(suffix_tokens)

        inputs_enc = self.tokenizer(
            pairs,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            max_length=effective_max,
        )
        for i, ids in enumerate(inputs_enc["input_ids"]):
            inputs_enc["input_ids"][i] = prefix_tokens + ids + suffix_tokens

        padded = self.tokenizer.pad(
            inputs_enc,
            padding=True,
            return_tensors="pt",
            max_length=self.config.max_length,
        )
        padded = {k: v.to(self.model.device) for k, v in padded.items()}

        with torch.no_grad():
            logits = self.model(**padded).logits[:, -1, :]
            true_v = logits[:, token_true_id]
            false_v = logits[:, token_false_id]
            log_probs = torch.nn.functional.log_softmax(
                torch.stack([false_v, true_v], dim=1),
                dim=1,
            )

        return log_probs[:, 1].exp().tolist()

    def check_contradictions(
        self,
        agent_response: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[RankedChunk]:
        """
        Check if retrieved chunks contradict the agent response.

        Args:
            agent_response: The agent's response to check.
            retrieved_chunks: List of retrieved chunks to evaluate.

        Returns:
            List of ranked chunks with support/contradiction verdicts.
        """
        if not retrieved_chunks:
            return []

        instruction = (
            "Given the agent response as the Query, determine whether the Document "
            "SUPPORTS (yes) or CONTRADICTS (no) the agent response."
        )

        pairs = [
            self._format_input(instruction, agent_response, chunk.text)
            for chunk in retrieved_chunks
        ]

        scores = self._compute_scores(pairs)

        results = []
        for chunk, score in zip(retrieved_chunks, scores, strict=True):
            verdict: Literal["SUPPORTS", "CONTRADICTS"] = (
                "SUPPORTS" if score >= self.config.contradiction_threshold else "CONTRADICTS"
            )
            results.append(
                RankedChunk(
                    text=chunk.text,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    similarity=chunk.similarity,
                    reranker_score=round(score, 4),
                    verdict=verdict,
                )
            )

        return results


__all__ = [
    "RankedChunk",
    "RegulatoryReranker",
    "RerankerConfig",
]
