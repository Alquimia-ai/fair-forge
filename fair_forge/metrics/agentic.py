"""Agentic metric for evaluating agent responses with pass@K and tool correctness."""

from collections import defaultdict
from typing import Any

import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from fair_forge.core import FairForge, Retriever
from fair_forge.llm import Judge
from fair_forge.schemas import Batch
from fair_forge.schemas.agentic import AgenticConversation, AgenticMetric, ToolCorrectnessScore
from fair_forge.statistical import FrequentistMode, StatisticalMode


class AnswerCorrectnessOutput(BaseModel):
    """Structured output for answer correctness evaluation."""

    model_config = ConfigDict(extra="forbid")

    correctness_score: float = Field(ge=0.0, le=1.0, description="Correctness score (0.0-1.0)")
    reasoning: str = Field(description="Brief explanation of the evaluation")


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k: probability of ≥1 correct conversation in k independent attempts.

    Uses the Bernoulli model: p = c/n is the estimated success rate from evaluation,
    and 1 - (1-p)^k is the probability of at least one success in k independent attempts.

    Args:
        n: Total conversations evaluated
        c: Fully correct conversations
        k: Number of independent attempts (not bounded by n)

    Returns:
        Probability between 0.0 and 1.0
    """
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    return 1.0 - (1.0 - c / n) ** k


def pass_pow_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass^k: probability of k consecutive correct conversations.

    Uses p = c/n as the estimated success rate: (c/n)^k is the probability
    that k independent attempts are all correct.

    Args:
        n: Total conversations evaluated
        c: Fully correct conversations
        k: Number of consecutive attempts (not bounded by n)

    Returns:
        Probability between 0.0 and 1.0
    """
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0

    return (c / n) ** k


class Agentic(FairForge):
    """
    Agentic metric for evaluating complete agent conversations with pass@K/pass^K formulas.

    Evaluates conversations as complete units where a conversation is correct only if ALL
    its interactions are correct. pass@K and pass^K are computed globally across all
    evaluated conversations using p = c/n (fully correct / total).

    Metrics:
    - pass@K: Probability of ≥1 correct conversation when attempting K different conversations (0.0-1.0)
    - pass^K: Probability of K consecutive correct conversations (0.0-1.0)
    - Tool Correctness: Evaluates correct tool usage per interaction (selection, parameters, sequence, utilization)

    Uses an LLM judge for answer correctness, and direct dictionary comparison for tool correctness.

    Formulas:
        pass@k = 1 - (1 - p)^k  # Prob. of ≥1 correct conversation, p = c/n
        pass^k = p^k             # Prob. of all k conversations correct

    Where:
        n = total conversations evaluated
        c = fully correct conversations (all interactions correct)
        p = c/n, estimated success rate
        k = number of conversation attempts (required, user-specified)

    Args:
        retriever: Retriever class for loading datasets (each Dataset = 1 conversation)
        model: LangChain BaseChatModel instance for evaluation
        k: Number of independent attempts for pass@K/pass^K computation (required)
        use_structured_output: If True, use LangChain's with_structured_output()
        bos_json_clause: Opening marker for JSON blocks
        eos_json_clause: Closing marker for JSON blocks
        threshold: Similarity threshold for answer correctness (default: 0.7)
        tool_threshold: Threshold for tool correctness (default: 1.0)
        tool_weights: Weights for tool correctness components (default: 0.25 each)
        **kwargs: Additional arguments passed to FairForge base class

    Example:
        >>> from langchain_groq import ChatGroq
        >>> model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        >>> result = Agentic.run(MyRetriever, model=model, k=3, threshold=0.8)
        >>> print(f"pass@3={result.pass_at_k:.3f}, pass^3={result.pass_pow_k:.3f}")
    """

    def __init__(
        self,
        retriever: type[Retriever],
        model: BaseChatModel,
        k: int,
        use_structured_output: bool = False,
        strict: bool = True,
        bos_json_clause: str = "```json",
        eos_json_clause: str = "```",
        threshold: float = 0.7,
        tool_threshold: float = 1.0,
        tool_weights: dict[str, float] | None = None,
        statistical_mode: StatisticalMode | None = None,
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)

        self.model = model
        self.k = k
        self.use_structured_output = use_structured_output
        self.strict = strict
        self.bos_json_clause = bos_json_clause
        self.eos_json_clause = eos_json_clause
        self.threshold = threshold
        self.tool_threshold = tool_threshold

        if tool_weights is None:
            tool_weights = {
                "selection": 0.25,
                "parameters": 0.25,
                "sequence": 0.25,
                "utilization": 0.25,
            }
        self.tool_weights = tool_weights
        self.statistical_mode = statistical_mode if statistical_mode is not None else FrequentistMode()

        self.logger.info(f"Initialized Agentic metric with model: {model.__class__.__name__}")
        self.logger.info(f"Thresholds - Answer: {threshold}, Tool: {tool_threshold}")
        self.logger.info(f"Statistical mode: {self.statistical_mode.get_result_type()}")

    @classmethod
    def run(cls, retriever: type[Retriever], k: int, **kwargs) -> AgenticMetric:
        return cls(retriever, k=k, **kwargs)._process()

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: str | None = "english",
    ):
        """Process batch - actual evaluation happens in _process() by grouping qa_ids."""
        for interaction in batch:
            self.logger.debug(f"QA ID: {interaction.qa_id}, Assistant: {assistant_id}")

    def _evaluate_answer_correctness(self, judge: Judge, query: str, answer: str, ground_truth: str) -> float:
        """Evaluate answer correctness using LLM judge. Returns score 0.0-1.0."""
        system_prompt = """You are a STRICT evaluator. Your task is to determine if an agent's answer is correct compared to the ground truth.

**Agent's Answer:** {answer}

**Ground Truth:** {ground_truth}

Evaluate the correctness with STRICT criteria:

1. **Factual Accuracy** (most important): Is the core information factually correct?
2. **Precision**: Spelling errors, typos, or incorrect formatting should be penalized
3. **Completeness**: Does it answer what was asked?
4. **Format**: Natural language variations are acceptable ONLY if facts are perfect

IMPORTANT SCORING RULES:
- 1.0: Identical or perfectly correct with natural rephrasing (same facts, perfect spelling)
- 0.85-0.95: Correct facts with slightly more verbose explanation
- 0.65-0.75: Correct core fact BUT has typo/spelling error (e.g., "Poris" instead of "Paris")
- 0.5-0.65: Mostly correct but missing important details
- 0.3-0.5: Partially correct with significant errors
- 0.0-0.3: Wrong answer or completely incorrect

**Typo/Spelling Penalty**:
- Single character typo in short answer: MAX score 0.75
- Multiple typos: MAX score 0.5
- Wrong word entirely: score below 0.3

**Wrong Answer**: Factually incorrect information must score below 0.3

Examples:
- Q: "Capital of France?", A: "Paris", GT: "Paris" → 1.0 (perfect match)
- Q: "Capital of France?", A: "The capital of France is Paris", GT: "Paris" → 0.95 (correct, verbose)
- Q: "Capital of France?", A: "Poris", GT: "Paris" → 0.7 (TYPO PENALTY - core fact known but misspelled)
- Q: "Capital of France?", A: "Lyon", GT: "Paris" → 0.0 (completely wrong city)
- Q: "Capital of France?", A: "London", GT: "Paris" → 0.0 (wrong country)

After your reasoning, respond with ONLY this:
<result>
{{"correctness_score": 0.95, "reasoning": "brief explanation"}}
</result>
Replace 0.95 with the actual score (0.0-1.0). Do not include anything outside the <result> tags."""

        data = {"answer": answer, "ground_truth": ground_truth}

        for attempt in range(3):
            try:
                judge.chat_history = []
                _reasoning, result = judge.check(system_prompt, query, data)

                if result is not None:
                    score = float(result.get("correctness_score", 0.0))
                    self.logger.debug(f"✓ Judge score: {score}")
                    return score

                self.logger.warning(f"Retry {attempt + 1}/3 - judge returned None")
            except Exception:
                self.logger.exception(f"❌ Error on attempt {attempt + 1}")

        self.logger.error("❌ Judge failed after 3 attempts")
        return 0.0

    def _evaluate_tool_correctness(
        self, agentic: dict[str, Any], ground_truth_agentic: dict[str, Any]
    ) -> ToolCorrectnessScore:
        """Evaluate tool usage correctness by comparing selection, parameters, sequence, and utilization."""
        tools_used = agentic.get("tools_used", [])
        final_answer_uses_tools = agentic.get("final_answer_uses_tools", False)
        expected_tools = ground_truth_agentic.get("expected_tools", [])
        sequence_matters = ground_truth_agentic.get("tool_sequence_matters", True)

        reasoning_parts = []

        used_tool_names = {tool.get("tool_name") for tool in tools_used}
        expected_tool_names = {tool.get("tool_name") for tool in expected_tools}

        if expected_tool_names == used_tool_names:
            tool_selection = 1.0
            reasoning_parts.append("✓ Tool selection: correct")
        elif used_tool_names.issubset(expected_tool_names):
            tool_selection = len(used_tool_names) / len(expected_tool_names)
            missing = expected_tool_names - used_tool_names
            reasoning_parts.append(f"⚠ Tool selection: missing {missing}")
        elif expected_tool_names.issubset(used_tool_names):
            tool_selection = len(expected_tool_names) / len(used_tool_names)
            extra = used_tool_names - expected_tool_names
            reasoning_parts.append(f"⚠ Tool selection: extra tools {extra}")
        else:
            overlap = len(used_tool_names.intersection(expected_tool_names))
            total = len(used_tool_names.union(expected_tool_names))
            tool_selection = overlap / total if total > 0 else 0.0
            reasoning_parts.append(f"✗ Tool selection: used {used_tool_names}, expected {expected_tool_names}")

        used_tools_map: dict[str, list] = defaultdict(list)
        expected_tools_map: dict[str, list] = defaultdict(list)

        for tool in tools_used:
            used_tools_map[tool.get("tool_name")].append(tool)
        for tool in expected_tools:
            expected_tools_map[tool.get("tool_name")].append(tool)

        param_matches = []
        for tool_name in expected_tool_names:
            if tool_name not in used_tools_map:
                param_matches.append(0.0)
                continue

            used_list = used_tools_map[tool_name]
            expected_list = expected_tools_map[tool_name]

            for exp_tool in expected_list:
                exp_params = exp_tool.get("parameters", {})
                used_tool = used_list[0] if used_list else {}
                used_params = used_tool.get("parameters", {})

                if exp_params == used_params:
                    param_matches.append(1.0)
                else:
                    all_keys = set(exp_params.keys()).union(used_params.keys())
                    matching = sum(1 for k in all_keys if exp_params.get(k) == used_params.get(k))
                    param_matches.append(matching / len(all_keys) if all_keys else 0.0)

        parameter_accuracy = sum(param_matches) / len(param_matches) if param_matches else 0.0

        if parameter_accuracy == 1.0:
            reasoning_parts.append("✓ Parameters: correct")
        elif parameter_accuracy > 0.7:
            reasoning_parts.append(f"⚠ Parameters: mostly correct ({parameter_accuracy:.2f})")
        else:
            reasoning_parts.append(f"✗ Parameters: incorrect ({parameter_accuracy:.2f})")

        if not sequence_matters:
            sequence_correct = 1.0
            reasoning_parts.append("✓ Sequence: not required")
        else:
            sequence_matches = []
            for tool_name in expected_tool_names:
                if tool_name not in used_tools_map:
                    sequence_matches.append(0.0)
                    continue

                used_list = used_tools_map[tool_name]
                expected_list = expected_tools_map[tool_name]

                for i, exp_tool in enumerate(expected_list):
                    exp_step = exp_tool.get("step")
                    used_tool = used_list[i] if i < len(used_list) else None
                    used_step = used_tool.get("step") if used_tool else None

                    if exp_step == used_step:
                        sequence_matches.append(1.0)
                    else:
                        sequence_matches.append(0.0)

            sequence_correct = sum(sequence_matches) / len(sequence_matches) if sequence_matches else 0.0

            if sequence_correct == 1.0:
                reasoning_parts.append("✓ Sequence: correct")
            else:
                reasoning_parts.append(f"✗ Sequence: incorrect ({sequence_correct:.2f})")

        if final_answer_uses_tools:
            result_utilization = 1.0
            reasoning_parts.append("✓ Utilization: tools used in answer")
        else:
            result_utilization = 0.0
            reasoning_parts.append("✗ Utilization: tools not used in answer")

        overall = (
            self.tool_weights["selection"] * tool_selection
            + self.tool_weights["parameters"] * parameter_accuracy
            + self.tool_weights["sequence"] * sequence_correct
            + self.tool_weights["utilization"] * result_utilization
        )

        is_correct = overall >= self.tool_threshold

        reasoning = "; ".join(reasoning_parts)

        return ToolCorrectnessScore(
            tool_selection_correct=tool_selection,
            parameter_accuracy=parameter_accuracy,
            sequence_correct=sequence_correct,
            result_utilization=result_utilization,
            overall_correctness=overall,
            is_correct=is_correct,
            reasoning=reasoning,
        )

    def _process(self) -> AgenticMetric:
        """Evaluate each conversation, then compute global pass@K and pass^K from aggregate success rate."""
        self.logger.info(f"[Agentic] Evaluating {len(self.dataset)} conversations")

        judge = Judge(
            model=self.model,
            use_structured_output=self.use_structured_output,
            strict=self.strict,
            bos_json_clause="<result>",
            eos_json_clause="</result>",
            verbose=self.verbose,
            chat_history=False,
        )

        conversations: list[AgenticConversation] = []

        for dataset_idx, dataset in enumerate(self.dataset, 1):
            total_interactions = len(dataset.conversation)
            self.logger.info(
                f"[Agentic] Evaluating conversation {dataset_idx}/{len(self.dataset)}: "
                f"{dataset.session_id} ({total_interactions} interactions)"
            )

            correctness_scores = []
            correct_indices = []
            tool_correctness_scores = []

            for i, batch in enumerate(dataset.conversation):
                self.logger.debug(f"  Interaction {i + 1}/{total_interactions} (qa_id: {batch.qa_id})")

                score = self._evaluate_answer_correctness(
                    judge=judge,
                    query=batch.query,
                    answer=batch.assistant,
                    ground_truth=batch.ground_truth_assistant,
                )

                correctness_scores.append(score)

                if score >= self.threshold:
                    correct_indices.append(i)
                    self.logger.debug(f"    Answer score: {score:.3f} ✅ CORRECT")
                else:
                    self.logger.debug(f"    Answer score: {score:.3f} ❌ INCORRECT")

                if batch.ground_truth_agentic:
                    if batch.agentic and batch.agentic.get("tools_used"):
                        tool_correctness = self._evaluate_tool_correctness(
                            agentic=batch.agentic, ground_truth_agentic=batch.ground_truth_agentic
                        )
                        tool_correctness_scores.append(tool_correctness)
                        self.logger.debug(
                            f"    Tool correctness: {tool_correctness.overall_correctness:.3f}, "
                            f"Correct={tool_correctness.is_correct}"
                        )
                    else:
                        tool_correctness_scores.append(None)
                        self.logger.debug("    No tools used")
                else:
                    tool_correctness_scores.append(None)

            correct_interactions = len(correct_indices)
            is_fully_correct = correct_interactions == total_interactions

            status = (
                "✅ FULLY CORRECT" if is_fully_correct else f"❌ PARTIAL ({correct_interactions}/{total_interactions})"
            )
            self.logger.info(f"  Conversation result: {status}")

            conversations.append(
                AgenticConversation(
                    session_id=dataset.session_id,
                    assistant_id=dataset.assistant_id,
                    total_interactions=total_interactions,
                    correct_interactions=correct_interactions,
                    is_fully_correct=is_fully_correct,
                    threshold=self.threshold,
                    correctness_scores=correctness_scores,
                    correct_indices=correct_indices,
                    tool_correctness_scores=tool_correctness_scores if tool_correctness_scores else [],
                )
            )

        n = len(conversations)
        c = sum(1 for conv in conversations if conv.is_fully_correct)
        p = c / n if n > 0 else 0.0

        self.logger.info(
            f"[Agentic] Completed evaluation. "
            f"{c}/{n} conversations fully correct (p = {p:.1%}). "
            f"pass@{self.k} = {pass_at_k(n, c, self.k):.4f}, "
            f"pass^{self.k} = {pass_pow_k(n, c, self.k):.4f}"
        )

        p_result = self.statistical_mode.rate_estimation(c, n)

        if self.statistical_mode.get_result_type() == "point_estimate":
            return AgenticMetric(
                k=self.k,
                n=n,
                c=c,
                p=p,
                pass_at_k=pass_at_k(n, c, self.k),
                pass_pow_k=pass_pow_k(n, c, self.k),
                conversations=conversations,
            )

        p_samples = p_result["samples"]
        pass_at_k_samples = 1.0 - (1.0 - p_samples) ** self.k
        pass_pow_k_samples = p_samples**self.k

        alpha = (1.0 - getattr(self.statistical_mode, "ci_level", 0.95)) / 2.0
        return AgenticMetric(
            k=self.k,
            n=n,
            c=c,
            p=p,
            pass_at_k=float(np.mean(pass_at_k_samples)),
            pass_at_k_ci_low=float(np.quantile(pass_at_k_samples, alpha)),
            pass_at_k_ci_high=float(np.quantile(pass_at_k_samples, 1.0 - alpha)),
            pass_pow_k=float(np.mean(pass_pow_k_samples)),
            pass_pow_k_ci_low=float(np.quantile(pass_pow_k_samples, alpha)),
            pass_pow_k_ci_high=float(np.quantile(pass_pow_k_samples, 1.0 - alpha)),
            conversations=conversations,
        )

    @staticmethod
    def summary(result: AgenticMetric) -> None:
        """Print global pass@K results and per-conversation breakdown."""
        print("=" * 70)
        print(f"{'AGENTIC EVALUATION RESULTS':^70}")
        print("=" * 70)
        print(f"\n  Conversations : {result.c}/{result.n} fully correct  (p = {result.p:.1%})")

        if result.pass_at_k_ci_low is not None:
            print(
                f"  pass@{result.k}        = {result.pass_at_k:.4f}  [{result.pass_at_k_ci_low:.4f}, {result.pass_at_k_ci_high:.4f}]"
            )
            print(
                f"  pass^{result.k}        = {result.pass_pow_k:.4f}  [{result.pass_pow_k_ci_low:.4f}, {result.pass_pow_k_ci_high:.4f}]"
            )
        else:
            print(f"  pass@{result.k}        = {result.pass_at_k:.4f}")
            print(f"  pass^{result.k}        = {result.pass_pow_k:.4f}")
        print("\n" + "-" * 70)
        for i, conv in enumerate(result.conversations, 1):
            status = "✅ CORRECT" if conv.is_fully_correct else "❌ FAILED"
            print(f"\n  [{i}] {conv.session_id} / {conv.assistant_id}")
            print(f"       Status : {status}")
            print(f"       Turns  : {conv.correct_interactions}/{conv.total_interactions} correct")
            print(f"       Scores : {[round(s, 2) for s in conv.correctness_scores]}")
        print("\n" + "=" * 70)

    @staticmethod
    def compare_k(result: AgenticMetric, k_max: int = 10) -> None:
        """Print pass@K and pass^K for K from 1 to k_max using the global success rate."""
        print(f"\n{'K':<6} {'pass@K':<14} {'pass^K':<14}")
        print("-" * 34)
        for k in range(1, k_max + 1):
            pak = pass_at_k(result.n, result.c, k)
            ppk = pass_pow_k(result.n, result.c, k)
            print(f"k={k:<4} {pak:<14.4f} {ppk:<14.4f}")

    @staticmethod
    def plot(result: AgenticMetric, k_max: int = 10, save_path: str | None = None) -> None:
        """Plot pass@K and pass^K curves based on the global success rate."""
        import matplotlib.pyplot as plt

        k_range = list(range(1, k_max + 1))
        pak = [pass_at_k(result.n, result.c, k) for k in k_range]
        ppk = [pass_pow_k(result.n, result.c, k) for k in k_range]
        color = "#2ecc71"
        label = f"Agent  p = {result.p:.0%}  ({result.c}/{result.n} conversations correct)"

        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(k_range, pak, "o-", color=color, linewidth=2, markersize=6, label=label)
        ax1.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95% target")
        ax1.set_xlabel("K (number of attempts)")
        ax1.set_ylabel("Probability")
        ax1.set_title("pass@K  =  1 - (1 - p)^K\nP(at least 1 correct in K attempts)", fontweight="bold")
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)

        ax2.plot(k_range, ppk, "o-", color=color, linewidth=2, markersize=6, label=label)
        ax2.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="70% target")
        ax2.set_xlabel("K (number of attempts)")
        ax2.set_ylabel("Probability")
        ax2.set_title("pass^K  =  p^K\nP(all K attempts correct)", fontweight="bold")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)

        plt.suptitle("Agentic Metric — pass@K curves", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_tools(result: AgenticMetric, save_path: str | None = None) -> None:
        """Plot per-interaction tool correctness breakdown for each conversation."""
        import matplotlib.pyplot as plt

        dim_colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

        scored = [
            (conv, [t for t in conv.tool_correctness_scores if t is not None])
            for conv in result.conversations
            if any(t is not None for t in conv.tool_correctness_scores)
        ]

        if not scored:
            print("No tool correctness data — add agentic/ground_truth_agentic fields to your Batch objects.")
            return

        _fig, axes = plt.subplots(len(scored), 1, figsize=(12, 4 * len(scored)))
        if len(scored) == 1:
            axes = [axes]

        for ax, (conv, tool_scores) in zip(axes, scored, strict=False):
            n = len(tool_scores)
            x = np.arange(n)
            bar_w = 0.18
            dim_values = {
                "Selection": [t.tool_selection_correct for t in tool_scores],
                "Parameters": [t.parameter_accuracy for t in tool_scores],
                "Sequence": [t.sequence_correct for t in tool_scores],
                "Utilization": [t.result_utilization for t in tool_scores],
            }
            for di, (dim, vals) in enumerate(dim_values.items()):
                ax.bar(x + di * bar_w, vals, bar_w, label=dim, color=dim_colors[di], alpha=0.85)

            overall = [t.overall_correctness for t in tool_scores]
            ax.plot(
                x + 1.5 * bar_w, overall, "D--", color="#2c3e50", markersize=7, linewidth=1.5, label="Overall", zorder=5
            )
            ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
            ax.set_xlabel("Interaction index")
            ax.set_ylabel("Score")
            ax.set_title(f"Tool correctness — {conv.assistant_id}", fontsize=11)
            ax.set_xticks(x + 1.5 * bar_w)
            ax.set_xticklabels([f"#{i}" for i in range(n)])
            ax.set_ylim(0, 1.15)
            ax.legend(fontsize=9, loc="lower right")
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle("Tool Correctness — per-interaction breakdown", fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
