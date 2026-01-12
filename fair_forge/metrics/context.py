"""Context metric for evaluating AI response alignment with provided context."""
from typing import Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel

from fair_forge.core import FairForge, Retriever
from fair_forge.llm import ContextJudgeOutput, Judge
from fair_forge.llm.prompts import (
    context_reasoning_system_prompt,
    context_reasoning_system_prompt_observation,
)
from fair_forge.schemas import Batch
from fair_forge.schemas.context import ContextMetric


class Context(FairForge):
    """Metric for evaluating how well AI responses align with provided context.

    Args:
        retriever: Retriever class for loading datasets
        model: LangChain BaseChatModel instance for evaluation
        use_structured_output: If True, use LangChain's with_structured_output()
        bos_json_clause: Opening marker for JSON blocks
        eos_json_clause: Closing marker for JSON blocks
        **kwargs: Additional arguments passed to FairForge base class
    """

    def __init__(
        self,
        retriever: Type[Retriever],
        model: BaseChatModel,
        use_structured_output: bool = False,
        bos_json_clause: str = "```json",
        eos_json_clause: str = "```",
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)
        self.model = model
        self.use_structured_output = use_structured_output
        self.bos_json_clause = bos_json_clause
        self.eos_json_clause = eos_json_clause

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: Optional[str] = "english",
    ):
        judge = Judge(
            model=self.model,
            use_structured_output=self.use_structured_output,
            bos_json_clause=self.bos_json_clause,
            eos_json_clause=self.eos_json_clause,
        )
        for interaction in batch:
            self.logger.debug(f"QA ID: {interaction.qa_id}")

            query = interaction.query
            data = {
                "context": context,
                "assistant_answer": interaction.assistant,
            }
            if interaction.observation:
                reasoning, result = judge.check(
                    context_reasoning_system_prompt_observation,
                    query,
                    {"observation": interaction.observation, **data},
                    output_schema=ContextJudgeOutput,
                )
            else:
                reasoning, result = judge.check(
                    context_reasoning_system_prompt,
                    query,
                    {"ground_truth_assistant": interaction.assistant, **data},
                    output_schema=ContextJudgeOutput,
                )

            if result is None:
                raise ValueError(
                    f"[FAIR FORGE/CONTEXT] No valid response from judge for QA ID: {interaction.qa_id}"
                )

            if self.use_structured_output:
                insight = result.insight
                score = result.score
            else:
                insight = result["insight"]
                score = result["score"]

            metric = ContextMetric(
                context_insight=insight,
                context_thinkings=reasoning,
                context_awareness=score,
                session_id=session_id,
                assistant_id=assistant_id,
                qa_id=interaction.qa_id,
            )
            self.logger.debug(f"Context insight: {metric.context_insight}")
            self.logger.debug(f"Context awareness: {metric.context_awareness}")
            self.metrics.append(metric)
