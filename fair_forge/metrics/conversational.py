"""Conversational metric for evaluating dialogue quality using Grice's maxims."""
from typing import Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel

from fair_forge.core import FairForge, Retriever
from fair_forge.llm import ConversationalJudgeOutput, Judge
from fair_forge.llm.prompts import (
    conversational_reasoning_system_prompt,
    conversational_reasoning_system_prompt_observation,
)
from fair_forge.schemas import Batch
from fair_forge.schemas.conversational import ConversationalMetric


class Conversational(FairForge):
    """Metric for evaluating conversational quality using Grice's maxims.

    Evaluates memory recall, language appropriateness, and Grice's maxims:
    quality, quantity, relation, manner, and sensibleness.

    Args:
        retriever: Retriever class for loading datasets
        model: LangChain BaseChatModel instance for evaluation
        use_structured_output: If True, use LangChain's with_structured_output()
        bos_think_token: Begin token for chain-of-thought extraction
        eos_think_token: End token for chain-of-thought extraction
        bos_json_clause: Opening marker for JSON blocks
        eos_json_clause: Closing marker for JSON blocks
        **kwargs: Additional arguments passed to FairForge base class
    """

    def __init__(
        self,
        retriever: Type[Retriever],
        model: BaseChatModel,
        use_structured_output: bool = False,
        bos_think_token: Optional[str] = None,
        eos_think_token: Optional[str] = None,
        bos_json_clause: str = "```json",
        eos_json_clause: str = "```",
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)
        self.model = model
        self.use_structured_output = use_structured_output
        self.bos_think_token = bos_think_token
        self.eos_think_token = eos_think_token
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
            bos_think_token=self.bos_think_token,
            eos_think_token=self.eos_think_token,
            bos_json_clause=self.bos_json_clause,
            eos_json_clause=self.eos_json_clause,
        )
        for interaction in batch:
            self.logger.debug(f"QA ID: {interaction.qa_id}")

            data = {
                "preferred_language": language,
                "assistant_answer": interaction.assistant,
            }
            if interaction.observation:
                thinking, result = judge.check(
                    conversational_reasoning_system_prompt_observation,
                    interaction.query,
                    {"observation": interaction.observation, **data},
                    output_schema=ConversationalJudgeOutput,
                )
            else:
                thinking, result = judge.check(
                    conversational_reasoning_system_prompt,
                    interaction.query,
                    {"ground_truth_assistant": interaction.assistant, **data},
                    output_schema=ConversationalJudgeOutput,
                )

            if result is None:
                raise ValueError(
                    f"[FAIR FORGE/CONVERSATIONAL] No valid response from judge for QA ID: {interaction.qa_id}"
                )

            if self.use_structured_output:
                metric = ConversationalMetric(
                    session_id=session_id,
                    qa_id=interaction.qa_id,
                    assistant_id=assistant_id,
                    conversational_insight=result.insight,
                    conversational_memory=result.memory,
                    conversational_language=result.language,
                    conversational_quality_maxim=result.quality_maxim,
                    conversational_quantity_maxim=result.quantity_maxim,
                    conversational_relation_maxim=result.relation_maxim,
                    conversational_manner_maxim=result.manner_maxim,
                    conversational_sensibleness=result.sensibleness,
                    conversational_thinkings=thinking,
                )
            else:
                metric = ConversationalMetric(
                    session_id=session_id,
                    qa_id=interaction.qa_id,
                    assistant_id=assistant_id,
                    conversational_insight=result["insight"],
                    conversational_memory=result["memory"],
                    conversational_language=result["language"],
                    conversational_quality_maxim=result["quality_maxim"],
                    conversational_quantity_maxim=result["quantity_maxim"],
                    conversational_relation_maxim=result["relation_maxim"],
                    conversational_manner_maxim=result["manner_maxim"],
                    conversational_sensibleness=result["sensibleness"],
                    conversational_thinkings=thinking,
                )

            self.logger.debug(f"Conversational memory: {metric.conversational_memory}")
            self.logger.debug(f"Conversational language: {metric.conversational_language}")
            self.logger.debug(f"Conversational quality maxim: {metric.conversational_quality_maxim}")
            self.logger.debug(f"Conversational quantity maxim: {metric.conversational_quantity_maxim}")
            self.logger.debug(f"Conversational relation maxim: {metric.conversational_relation_maxim}")
            self.logger.debug(f"Conversational manner maxim: {metric.conversational_manner_maxim}")
            self.logger.debug(f"Conversational sensibleness: {metric.conversational_sensibleness}")
            self.metrics.append(metric)
