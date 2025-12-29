from fair_forge.core import FairForge, Retriever
from typing import Optional, Type
from pydantic import SecretStr
from fair_forge.schemas import Batch, ContextMetric
from fair_forge.prompts import (
    context_reasoning_system_prompt,
    context_reasoning_system_prompt_observation,
)
from fair_forge.helpers.judge import Judge


class Context(FairForge):
    def __init__(
        self,
        retriever: Type[Retriever],
        judge_bos_think_token: str = "<think>",
        judege_eos_think_token: str = "</think>",
        judge_base_url: str = "https://api.groq.com/openai/v1",
        judge_api_key: SecretStr = SecretStr(""),
        judge_model: str = "deepseek-r1-distill-llama-70b",
        judge_temperature: float = 0,
        judge_bos_json_clause: str = "```json",
        judge_eos_json_clause: str = "```",
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)
        self.judge_url = judge_base_url
        self.judge_api_key = judge_api_key
        self.judge_model = judge_model
        self.judge_temperature = judge_temperature
        self.judge_bos_think_token = judge_bos_think_token
        self.judge_eos_think_token = judege_eos_think_token
        self.judge_bos_json_clause = judge_bos_json_clause
        self.judge_eos_json_clause = judge_eos_json_clause

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: Optional[str] = "english",
    ):
        judge = Judge(
            bos_think_token=self.judge_bos_think_token,
            eos_think_token=self.judge_eos_think_token,
            base_url=self.judge_url,
            api_key=self.judge_api_key,
            model=self.judge_model,
            temperature=self.judge_temperature,
            bos_json_clause=self.judge_bos_json_clause,
            eos_json_clause=self.judge_eos_json_clause,
        )
        for interaction in batch:
            self.logger.debug(f"QA ID: {interaction.qa_id}")

            query = interaction.query
            data = {
                "context": context,
                "assistant_answer": interaction.assistant,
            }
            if interaction.observation:
                thinking, json = judge.check(
                    context_reasoning_system_prompt_observation,
                    query,
                    {"observation": interaction.observation, **data},
                )
            else:
                thinking, json = judge.check(
                    context_reasoning_system_prompt,
                    query,
                    {"ground_truth_assistant": interaction.assistant, **data},
                )

            if json is None:
                raise ValueError(
                    f"[FAIR FORGE/CONTEXT] No JSON found {self.judge_bos_json_clause} {self.judge_eos_json_clause} "
                )
            metric = ContextMetric(
                    context_insight=json["insight"],
                    context_thinkings=thinking,
                    context_awareness=json["score"],
                    session_id=session_id,
                    assistant_id=assistant_id,
                    qa_id=interaction.qa_id,
                )
            self.logger.debug(f"Context insight: {metric.context_insight}")
            self.logger.debug(f"Context awareness: {metric.context_awareness}")
            self.metrics.append(metric)
