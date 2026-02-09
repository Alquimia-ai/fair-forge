"""Agentic metric for evaluating agent responses with pass@K and tool correctness."""

import json
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


class ToolCorrectnessOutput(BaseModel):
    """Structured output for tool correctness evaluation."""

    tool_selection_correct: float = Field(ge=0.0, le=1.0, description="Tool selection score")
    parameter_accuracy: float = Field(ge=0.0, le=1.0, description="Parameter accuracy score")
    sequence_correct: float = Field(ge=0.0, le=1.0, description="Sequence correctness score")
    result_utilization: float = Field(ge=0.0, le=1.0, description="Result utilization score")
    reasoning: str = Field(description="Explanation of the evaluation")


class Agentic(FairForge):
    """
    Agentic metric for evaluating AI agent responses.

    Evaluates multiple agent responses using:
    - pass@K: At least one of K responses is correct
    - pass^K: All K responses are correct
    - Tool Correctness: Evaluates correct tool usage (selection, parameters, sequence, utilization)

    Uses an LLM judge for evaluation.

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

        # Tool correctness weights (default: equal weight 0.25 each)
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
        """Process batch - actual processing happens in _process()."""
        for interaction in batch:
            self.logger.debug(f"QA ID: {interaction.qa_id}, Assistant: {assistant_id}")

    def _evaluate_answer_correctness(self, judge: Judge, query: str, answer: str, ground_truth: str) -> float:
        """
        Use LLM judge to evaluate if an answer is correct.

        Args:
            judge: Judge instance for evaluation
            query: The question asked
            answer: The agent's answer
            ground_truth: The expected correct answer

        Returns:
            float: Correctness score between 0.0 (incorrect) and 1.0 (perfect)
        """
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

            # Handle both Pydantic model and dict results
            if isinstance(result, dict):
                return float(result.get("correctness_score", 0.0))
            return float(result.correctness_score)

        except Exception:
            self.logger.exception("Error evaluating answer correctness")
            return 0.0

    def _evaluate_tool_correctness(
        self, judge: Judge, agentic: dict[str, Any], ground_truth_agentic: dict[str, Any]
    ) -> ToolCorrectnessScore:
        """
        Use LLM judge to evaluate correct tool usage.

        Args:
            judge: Judge instance for evaluation
            agentic: Dict with 'tools_used' and 'final_answer_uses_tools'
            ground_truth_agentic: Dict with 'expected_tools' and 'tool_sequence_matters'

        Returns:
            ToolCorrectnessScore with individual and overall scores
        """
        tools_used = agentic.get("tools_used", [])
        final_answer_uses_tools = agentic.get("final_answer_uses_tools", False)
        expected_tools = ground_truth_agentic.get("expected_tools", [])
        sequence_matters = ground_truth_agentic.get("tool_sequence_matters", True)

        system_prompt = """You are a STRICT evaluator of AI agent tool usage. Evaluate how correctly an agent used tools.

**Tools Used by Agent:**
{tools_used_json}

**Final Answer Uses Tools:** {final_answer_uses_tools}

**Expected Tools:**
{expected_tools_json}

**Sequence Matters:** {sequence_matters}

Evaluate the following aspects with STRICT criteria (each scored 0.0 to 1.0):

1. **tool_selection_correct**:
   - Did the agent select the correct tools (by tool_name)?
   - 1.0 = all correct tools, 0.0 = wrong tools

2. **parameter_accuracy**:
   - Were the parameters correct for each tool?
   - Compare parameter values EXACTLY
   - 1.0 = all parameters perfect, 0.0 = wrong parameters

3. **sequence_correct**:
   - Were tools called in the correct order?
   - If sequence_matters=True: Compare tool_step for EACH tool
   - For each tool_name, check if its tool_step matches expected
   - Example: calculator at step 1 when expected at step 2 = WRONG ORDER
   - 1.0 = ALL tool_steps match, 0.0 = wrong order
   - If sequence_matters=False: Always return 1.0

4. **result_utilization**:
   - Did the agent use the tool results properly?
   - Check if final_answer_uses_tools=True
   - 1.0 = excellent use, 0.0 = didn't use

CRITICAL FOR SEQUENCE:
- Compare each tool's tool_step with its expected tool_step
- "calculator" at step 1 when expected at step 2 = WRONG (0.0)
- "web_search" at step 2 when expected at step 1 = WRONG (0.0)
- Swapped steps = COMPLETELY WRONG ORDER (0.0)
"""

        data = {
            "tools_used_json": json.dumps(tools_used, indent=2),
            "final_answer_uses_tools": final_answer_uses_tools,
            "expected_tools_json": json.dumps(expected_tools, indent=2),
            "sequence_matters": sequence_matters,
        }

        try:
            _reasoning, result = judge.check(
                system_prompt, "Evaluate the tool usage correctness.", data, output_schema=ToolCorrectnessOutput
            )

            if result is None:
                self.logger.error("No valid response from judge for tool correctness")
                return ToolCorrectnessScore(
                    tool_selection_correct=0.0,
                    parameter_accuracy=0.0,
                    sequence_correct=0.0,
                    result_utilization=0.0,
                    overall_correctness=0.0,
                    is_correct=False,
                    reasoning="Error: No valid response from judge",
                )

            # Handle both Pydantic model and dict results
            if isinstance(result, dict):
                tool_selection = result.get("tool_selection_correct", 0.0)
                parameter_acc = result.get("parameter_accuracy", 0.0)
                sequence = result.get("sequence_correct", 0.0)
                utilization = result.get("result_utilization", 0.0)
                reason = result.get("reasoning", "")
            else:
                tool_selection = result.tool_selection_correct
                parameter_acc = result.parameter_accuracy
                sequence = result.sequence_correct
                utilization = result.result_utilization
                reason = result.reasoning

            # If sequence doesn't matter, don't penalize
            if not sequence_matters:
                sequence = 1.0

            # Calculate overall (weighted average)
            overall = (
                self.tool_weights["selection"] * tool_selection
                + self.tool_weights["parameters"] * parameter_acc
                + self.tool_weights["sequence"] * sequence
                + self.tool_weights["utilization"] * utilization
            )

            # Determine if correct based on threshold
            is_correct = overall >= self.tool_threshold

            return ToolCorrectnessScore(
                tool_selection_correct=tool_selection,
                parameter_accuracy=parameter_acc,
                sequence_correct=sequence,
                result_utilization=utilization,
                overall_correctness=overall,
                is_correct=is_correct,
                reasoning=reason,
            )

        except Exception:
            self.logger.exception("Error evaluating tool correctness")
            return ToolCorrectnessScore(
                tool_selection_correct=0.0,
                parameter_accuracy=0.0,
                sequence_correct=0.0,
                result_utilization=0.0,
                overall_correctness=0.0,
                is_correct=False,
                reasoning="Error during evaluation",
            )

    def _process(self) -> list[AgenticMetric]:
        """
        Process datasets grouped by qa_id to evaluate K responses per question.

        Override base processing to group multiple responses (different assistant_ids)
        for the same qa_id, then evaluate pass@K, pass^K, and tool correctness.
        """
        self.logger.info("[Agentic] Grouping datasets by qa_id for pass@K evaluation")

        # Initialize judge
        judge = Judge(
            model=self.model,
            use_structured_output=self.use_structured_output,
            bos_json_clause=self.bos_json_clause,
            eos_json_clause=self.eos_json_clause,
            verbose=self.verbose,
        )

        # Group datasets by qa_id (each qa_id has K responses from different assistants)
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

        # Process each qa_id group
        for qa_id, responses in qa_groups.items():
            k = len(responses)
            self.logger.info(f"[Agentic] Evaluating qa_id={qa_id} with K={k} responses")

            # Get ground truth from first response (should be same for all)
            ground_truth = responses[0]["ground_truth"]
            ground_truth_agentic = responses[0].get("ground_truth_agentic")

            # Evaluate correctness of each response
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

            # Calculate pass@K and pass^K
            pass_at_k = len(correct_indices) > 0  # At least one correct
            pass_pow_k = len(correct_indices) == k  # All correct

            self.logger.info(f"  pass@{k}: {pass_at_k}, pass^{k}: {pass_pow_k} ({len(correct_indices)}/{k} correct)")

            # Evaluate tool correctness (if applicable)
            tool_correctness = None
            if ground_truth_agentic:
                # Evaluate the first correct response (or first if none correct)
                response_to_eval = responses[correct_indices[0]] if correct_indices else responses[0]

                if response_to_eval.get("agentic"):
                    self.logger.debug("  Evaluating tool correctness...")
                    tool_correctness = self._evaluate_tool_correctness(
                        judge=judge, agentic=response_to_eval["agentic"], ground_truth_agentic=ground_truth_agentic
                    )
                    self.logger.debug(
                        f"    Overall: {tool_correctness.overall_correctness:.3f}, "
                        f"Correct: {tool_correctness.is_correct}"
                    )

            # Create metric result
            metric = AgenticMetric(
                session_id=responses[0]["session_id"],
                assistant_id=responses[0]["assistant_id"],  # Use first assistant as reference
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
