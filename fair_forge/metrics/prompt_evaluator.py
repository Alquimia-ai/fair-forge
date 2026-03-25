"""PromptEvaluator metric: score a system prompt using distributional signals."""

import math
from collections import Counter

import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from fair_forge.core import FairForge, Retriever
from fair_forge.core.embedder import Embedder
from fair_forge.metrics.constraints import Constraint
from fair_forge.prompt_optimizer.protocols import Evaluator, Executor
from fair_forge.schemas import Batch
from fair_forge.schemas.prompt_evaluator import PromptEvaluatorMetric, QuerySampleMetrics
from fair_forge.statistical import FrequentistMode, StatisticalMode
from fair_forge.utils.math import cosine_similarity


class PromptEvaluator(FairForge):
    """Score a system prompt by measuring the distributional stability of its responses.

    For each query in the dataset the prompt is executed K times. The K responses are
    semantically clustered (cosine similarity ≥ τ). Two mandatory signals are computed:
      - CSR (Consistency/Stability Rate): fraction of responses in the dominant cluster.
      - Stability (1 - SE_n): inverted normalized Semantic Entropy over the cluster distribution.

    Two optional signals are added when their dependencies are available:
      - RSS (Reference Similarity Score): average cosine similarity to the reference response.
        Enabled automatically when ground_truth_assistant is present in the dataset.
      - JQ (Judge Quality): LLM-as-judge score. Enabled via jq_enabled=True.

    Args:
        retriever: Retriever class supplying the dataset.
        model: LangChain BaseChatModel used as executor and optional JQ judge.
        seed_prompt: The system prompt under evaluation.
        embedder: Embedder used for semantic clustering and RSS computation.
        k: Number of samples generated per query (default 10).
        tau: Cosine similarity threshold for semantic clustering (default 0.80).
        constraints: List of Constraint objects for ICR computation. Activated when provided.
        jq_enabled: If True, an LLM judge scores each response against the reference.
        objective: Criteria for the JQ judge — required when jq_enabled=True.
        statistical_mode: Statistical computation mode (default FrequentistMode).
        executor: Custom callable (prompt, query, context) → str. Defaults to LLM invocation.
    """

    def __init__(
        self,
        retriever: type[Retriever],
        model: BaseChatModel,
        seed_prompt: str,
        embedder: Embedder,
        k: int = 10,
        tau: float = 0.80,
        constraints: list[Constraint] | None = None,
        jq_enabled: bool = False,
        objective: str = "",
        statistical_mode: StatisticalMode | None = None,
        executor: Executor | None = None,
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)
        self._seed_prompt = seed_prompt
        self._embedder = embedder
        self._k = k
        self._tau = tau
        self._constraints = constraints or []
        self._statistical_mode = statistical_mode or FrequentistMode()
        self._executor = executor or self._default_executor(model)
        self._jq_evaluator: Evaluator | None = None
        if jq_enabled:
            from fair_forge.prompt_optimizer.evaluators import LLMEvaluator

            self._jq_evaluator = LLMEvaluator(model=model, criteria=objective)
        self._session_data: dict[str, dict] = {}

    @staticmethod
    def _default_executor(model: BaseChatModel) -> Executor:
        def _execute(prompt: str, query: str, context: str) -> str:
            system = f"{prompt}\n\n{context}" if context else prompt
            response = model.invoke([SystemMessage(content=system), HumanMessage(content=query)])
            return str(response.content)

        return _execute

    def _cluster(self, embeddings: np.ndarray) -> list[int]:
        """Assign cluster labels via union-find over cosine similarity threshold τ."""
        n = len(embeddings)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                if cosine_similarity(embeddings[i], embeddings[j]) >= self._tau:
                    root_i, root_j = find(i), find(j)
                    if root_i != root_j:
                        parent[root_i] = root_j

        root_map: dict[int, int] = {}
        labels = []
        for i in range(n):
            root = find(i)
            if root not in root_map:
                root_map[root] = len(root_map)
            labels.append(root_map[root])

        return labels

    def _csr(self, labels: list[int]) -> float:
        """Compute CSR using the configured statistical mode."""
        counts = Counter(labels)
        n_dominant = max(counts.values())
        result = self._statistical_mode.rate_estimation(n_dominant, self._k)
        return result if isinstance(result, float) else result["mean"]

    def _se_n(self, labels: list[int]) -> float:
        """Compute normalized Semantic Entropy (SE_n) from cluster labels."""
        if self._k <= 1:
            return 0.0
        counts = Counter(labels)
        entropy = -sum((n / self._k) * math.log(n / self._k) for n in counts.values() if n > 0)
        return entropy / math.log(self._k)

    def _rss(self, response_embeddings: np.ndarray, reference_embedding: np.ndarray) -> float:
        """Compute average cosine similarity between K responses and the reference."""
        sims = [cosine_similarity(emb, reference_embedding) for emb in response_embeddings]
        return sum(sims) / len(sims)

    def _icr(self, responses: list[str]) -> float | None:
        """Compute mean Instruction Compliance Rate across K responses."""
        if not self._constraints:
            return None
        scores = [sum(c.check(r) for c in self._constraints) / len(self._constraints) for r in responses]
        return round(sum(scores) / len(scores), 4)

    def _evaluate_query(self, qa_id: str, query: str, context: str, ground_truth: str | None) -> QuerySampleMetrics:
        """Generate K responses for a query and compute distributional metrics."""
        responses = [self._executor(self._seed_prompt, query, context) for _ in range(self._k)]

        texts_to_embed = responses + ([ground_truth] if ground_truth else [])
        all_embeddings = self._embedder.encode(texts_to_embed)
        response_embeddings = all_embeddings[: self._k]

        labels = self._cluster(response_embeddings)
        csr = self._csr(labels)
        stability = 1.0 - self._se_n(labels)

        rss = None
        if ground_truth:
            rss = round(self._rss(response_embeddings, all_embeddings[self._k]), 4)

        icr = self._icr(responses)

        jq = None
        if self._jq_evaluator and ground_truth:
            jq_scores = [self._jq_evaluator(r, ground_truth, query, context) for r in responses]
            jq = round(sum(jq_scores) / len(jq_scores), 4)

        return QuerySampleMetrics(
            qa_id=qa_id,
            k=self._k,
            csr=round(csr, 4),
            stability=round(stability, 4),
            rss=rss,
            icr=icr,
            jq=jq,
            n_clusters=len(set(labels)),
        )

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: str | None = "english",
    ):
        if session_id not in self._session_data:
            self._session_data[session_id] = {"assistant_id": assistant_id, "interactions": []}

        for item in batch:
            self.logger.debug(f"QA ID: {item.qa_id}")
            query_metrics = self._evaluate_query(item.qa_id, item.query, context, item.ground_truth_assistant)
            self.logger.debug(f"CSR: {query_metrics.csr:.4f}  Stability: {query_metrics.stability:.4f}")
            self._session_data[session_id]["interactions"].append(query_metrics)

    def on_process_complete(self):
        for session_id, data in self._session_data.items():
            interactions = data["interactions"]
            n = len(interactions)

            csr = round(sum(i.csr for i in interactions) / n, 4)
            stability = round(sum(i.stability for i in interactions) / n, 4)

            rss_values = [i.rss for i in interactions if i.rss is not None]
            rss = round(sum(rss_values) / len(rss_values), 4) if rss_values else None

            icr_values = [i.icr for i in interactions if i.icr is not None]
            icr = round(sum(icr_values) / len(icr_values), 4) if icr_values else None

            jq_values = [i.jq for i in interactions if i.jq is not None]
            jq = round(sum(jq_values) / len(jq_values), 4) if jq_values else None

            self.metrics.append(
                PromptEvaluatorMetric(
                    session_id=session_id,
                    assistant_id=data["assistant_id"],
                    seed_prompt=self._seed_prompt,
                    k=self._k,
                    tau=self._tau,
                    csr=csr,
                    stability=stability,
                    rss=rss,
                    icr=icr,
                    jq=jq,
                    n_queries=n,
                    interactions=interactions,
                )
            )
