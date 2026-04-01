"""Pydantic schemas for the PromptEvaluator metric."""

from pydantic import BaseModel

from fair_forge.schemas.metrics import BaseMetric

_W = 58


def _bar(value: float, width: int = 10) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def _score_line(label: str, value: float) -> str:
    return f"  {label:<12} {value:.2f}  {_bar(value)}"


class QuerySampleMetrics(BaseModel):
    """Per-query distributional metrics computed from K samples.

    csr is the fraction of responses in the dominant semantic cluster.
    stability is 1 - SE_n, where SE_n is the normalized Semantic Entropy.
    rss is the average cosine similarity to the reference response (when available).
    icr is the average fraction of constraints satisfied across K responses (when constraints are provided).
    jq is the average Judge Quality score (when explicitly enabled).
    """

    qa_id: str
    k: int
    csr: float
    stability: float
    rss: float | None = None
    icr: float | None = None
    jq: float | None = None
    n_clusters: int


class PromptEvaluatorMetric(BaseMetric):
    """Result of evaluating a system prompt over a dataset using distributional signals.

    csr is the mean Consistency/Stability Rate — how often the model's responses
    fall into the same semantic cluster across K samples.
    stability is the mean (1 - normalized Semantic Entropy) — low entropy indicates
    the model produces a focused distribution of meanings.
    rss is the mean Reference Similarity Score (included when reference responses are available).
    icr is the mean Instruction Compliance Rate (included when constraints are provided).
    jq is the mean Judge Quality score (included when jq_enabled=True).
    """

    seed_prompt: str
    k: int
    tau: float
    csr: float
    stability: float
    rss: float | None = None
    icr: float | None = None
    jq: float | None = None
    n_queries: int
    interactions: list[QuerySampleMetrics]

    def __str__(self) -> str:
        sep = "─" * _W
        lines = [
            "=" * _W,
            " PROMPT EVALUATOR",
            "=" * _W,
            "",
            " METRICS",
            f" {sep}",
            "  CSR  (Consistency/Stability Rate)          always active",
            "       How often the model gives the same meaning across K runs.",
            "       1.0 = perfectly consistent  |  0.0 = completely scattered",
            "",
            "  Stability  (1 - Semantic Entropy)          always active",
            "       How focused the distribution of meanings is.",
            "       1.0 = all responses converge  |  0.0 = maximum spread",
            "",
            "  RSS  (Reference Similarity Score)          optional — auto",
            "       Similarity between responses and the reference answer.",
            "       Activated automatically when ground_truth_assistant is in your dataset.",
            "",
            "  ICR  (Instruction Compliance Rate)         optional — auto",
            "       Fraction of verifiable prompt constraints satisfied per response.",
            "       Activated automatically when constraints are passed to PromptEvaluator.",
            "",
            "  JQ   (Judge Quality)                       optional — manual",
            "       LLM-as-judge score per response.",
            "       Activated with: jq_enabled=True, objective='your criteria'",
            "",
            "=" * _W,
            f"  Session:   {self.session_id}",
            f"  Assistant: {self.assistant_id}",
            f"  K = {self.k}  |  τ = {self.tau}  |  Queries evaluated: {self.n_queries}",
            "=" * _W,
            "",
            " OVERALL SCORES",
            f" {sep}",
            _score_line("CSR", self.csr),
            _score_line("Stability", self.stability),
        ]

        if self.rss is not None:
            lines.append(_score_line("RSS", self.rss))
        else:
            lines.append("  RSS          —     add ground_truth_assistant to your dataset to activate")

        if self.icr is not None:
            lines.append(_score_line("ICR", self.icr))
        else:
            lines.append("  ICR          —     pass constraints=[...] to activate")

        if self.jq is not None:
            lines.append(_score_line("JQ", self.jq))
        else:
            lines.append("  JQ           —     set jq_enabled=True to activate")

        lines += [
            "",
            " BREAKDOWN BY QUERY",
            f" {sep}",
        ]

        for i in self.interactions:
            row = f"  {i.qa_id:<14}  CSR {i.csr:.2f}  Stability {i.stability:.2f}  clusters: {i.n_clusters}"
            if i.rss is not None:
                row += f"  RSS {i.rss:.2f}"
            if i.icr is not None:
                row += f"  ICR {i.icr:.2f}"
            if i.jq is not None:
                row += f"  JQ {i.jq:.2f}"
            lines.append(row)

        lines.append("")
        return "\n".join(lines)

    def display(self) -> None:
        print(self)
