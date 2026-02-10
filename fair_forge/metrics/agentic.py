"""Agentic metric for evaluating agent responses with pass@K and tool correctness."""

from collections import defaultdict
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from fair_forge.core import FairForge, Retriever
from fair_forge.llm import Judge
from fair_forge.schemas import Batch
from fair_forge.schemas.agentic import AgenticMetric, ToolCorrectnessScore


class AnswerCorrectnessOutput(BaseModel):
    """Structured output for answer correctness evaluation."""

    correctness_score: float = Field(ge=0.0, le=1.0, description="Correctness score (0.0-1.0)")
    reasoning: str = Field(description="Brief explanation of the evaluation")


class Agentic(FairForge):
    """
    Agentic metric for evaluating AI agent responses.

    Evaluates multiple agent responses using:
    - pass@K: At least one of K responses is correct
    - pass^K: All K responses are correct
    - Tool Correctness: Evaluates correct tool usage (selection, parameters, sequence, utilization)

    Uses an LLM judge for answer correctness, and direct dictionary comparison for tool correctness.

    Args:
        retriever: Retriever class for loading datasets
        model: LangChain BaseChatModel instance for evaluation
        use_structured_output: If True, use LangChain's with_structured_output()
        bos_json_clause: Opening marker for JSON blocks
        eos_json_clause: Closing marker for JSON blocks
        threshold: Similarity threshold for answer correctness (default: 0.7)
        tool_threshold: Threshold for tool correctness (default: 0.75)
        tool_weights: Weights for tool correctness components (default: 0.25 each)
        **kwargs: Additional arguments passed to FairForge base class

    Example:
        >>> from langchain_groq import ChatGroq
        >>> model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        >>> results = Agentic.run(MyRetriever, model=model, threshold=0.8)
    """

    def __init__(
        self,
        retriever: type[Retriever],
        model: BaseChatModel,
        use_structured_output: bool = False,
        bos_json_clause: str = "```json",
        eos_json_clause: str = "```",
        threshold: float = 0.7,
        tool_threshold: float = 0.75,
        tool_weights: dict[str, float] | None = None,
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)

        self.model = model
        self.use_structured_output = use_structured_output
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

        self.logger.info(f"Initialized Agentic metric with model: {model.__class__.__name__}")
        self.logger.info(f"Thresholds - Answer: {threshold}, Tool: {tool_threshold}")

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

**Question:** {query}

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
- Q: "Capital of France?", A: "Pariis", GT: "Paris" → 0.7 (TYPO PENALTY)
- Q: "Capital of France?", A: "Lyon", GT: "Paris" → 0.0 (completely wrong city)
- Q: "Capital of France?", A: "London", GT: "Paris" → 0.0 (wrong country)
"""

        data = {"query": query, "answer": answer, "ground_truth": ground_truth}

        try:
            _reasoning, result = judge.check(
                system_prompt, "Evaluate the answer correctness.", data, output_schema=AnswerCorrectnessOutput
            )

            if result is None:
                self.logger.error("No valid response from judge for answer correctness")
                return 0.0

            if isinstance(result, dict):
                return float(result.get("correctness_score", 0.0))
            return float(result.correctness_score)

        except Exception:
            self.logger.exception("Error evaluating answer correctness")
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

    def _process(self) -> list[AgenticMetric]:
        """Group responses by qa_id and evaluate pass@K, pass^K, and tool correctness."""
        self.logger.info("[Agentic] Grouping datasets by qa_id for pass@K evaluation")

        judge = Judge(
            model=self.model,
            use_structured_output=self.use_structured_output,
            bos_json_clause=self.bos_json_clause,
            eos_json_clause=self.eos_json_clause,
            verbose=self.verbose,
        )

        qa_groups: dict[str, list] = defaultdict(list)
        for dataset in self.dataset:
            for batch in dataset.conversation:
                qa_groups[batch.qa_id].append(
                    {
                        "session_id": dataset.session_id,
                        "assistant_id": dataset.assistant_id,
                        "query": batch.query,
                        "answer": batch.assistant,
                        "ground_truth": batch.ground_truth_assistant,
                        "agentic": batch.agentic,
                        "ground_truth_agentic": batch.ground_truth_agentic,
                    }
                )

        self.logger.info(f"[Agentic] Found {len(qa_groups)} unique qa_ids")

        for qa_id, responses in qa_groups.items():
            k = len(responses)
            self.logger.info(f"[Agentic] Evaluating qa_id={qa_id} with K={k} responses")

            ground_truth = responses[0]["ground_truth"]
            ground_truth_agentic = responses[0].get("ground_truth_agentic")

            correctness_scores = []
            correct_indices = []

            for i, response in enumerate(responses):
                self.logger.debug(f"  Evaluating response {i + 1}/{k} (assistant: {response['assistant_id']})")

                score = self._evaluate_answer_correctness(
                    judge=judge, query=response["query"], answer=response["answer"], ground_truth=ground_truth
                )

                correctness_scores.append(score)

                if score >= self.threshold:
                    correct_indices.append(i)
                    self.logger.debug(f"    Score: {score:.3f} ✅ CORRECT")
                else:
                    self.logger.debug(f"    Score: {score:.3f} ❌ INCORRECT")

            pass_at_k = len(correct_indices) > 0
            pass_pow_k = len(correct_indices) == k

            self.logger.info(f"  pass@{k}: {pass_at_k}, pass^{k}: {pass_pow_k} ({len(correct_indices)}/{k} correct)")

            tool_correctness = None
            if ground_truth_agentic:
                response_to_eval = responses[correct_indices[0]] if correct_indices else responses[0]

                if response_to_eval.get("agentic"):
                    self.logger.debug("  Evaluating tool correctness...")
                    tool_correctness = self._evaluate_tool_correctness(
                        agentic=response_to_eval["agentic"], ground_truth_agentic=ground_truth_agentic
                    )
                    self.logger.debug(
                        f"    Overall: {tool_correctness.overall_correctness:.3f}, "
                        f"Correct: {tool_correctness.is_correct}"
                    )

            metric = AgenticMetric(
                session_id=responses[0]["session_id"],
                assistant_id=responses[0]["assistant_id"],
                qa_id=qa_id,
                k=k,
                threshold=self.threshold,
                correctness_scores=correctness_scores,
                pass_at_k=pass_at_k,
                pass_pow_k=pass_pow_k,
                correct_indices=correct_indices,
                tool_correctness=tool_correctness,
            )

            self.metrics.append(metric)

        self.logger.info(f"[Agentic] Completed evaluation. Total metrics: {len(self.metrics)}")
        return self.metrics
