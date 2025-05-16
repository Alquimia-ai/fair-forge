from fair_forge import FairForge, Retriever
from typing import Type, Optional
from pydantic import SecretStr
from fair_forge.schemas import Batch, ConversationalMetric
from fair_forge.helpers.judge import Judge
from fair_forge.prompts import (
    conversational_reasoning_system_prompt,
    conversational_reasoning_system_prompt_observation,
)


class Conversational(FairForge):
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
            data = {
                "preferred_language": language,
                "assistant_answer": interaction.assistant,
            }
            if interaction.observation:
                thinking, json = judge.check(
                    conversational_reasoning_system_prompt_observation,
                    interaction.query,
                    {"observation": interaction.observation, **data},
                )
            else:
                thinking, json = judge.check(
                    conversational_reasoning_system_prompt,
                    interaction.query,
                    {"ground_truth_assistant": interaction.assistant, **data},
                )
            if json is None:
                raise ValueError(
                    f"[FAIR FORGE/CONVERSATIONAL] No JSON found {self.judge_bos_json_clause} {self.judge_eos_json_clause} "
                )

            self.metrics.append(
                ConversationalMetric(
                    session_id=session_id,
                    qa_id=interaction.qa_id,
                    assistant_id=assistant_id,
                    conversational_insight=json["insight"],
                    conversational_memory=json["memory"],
                    conversational_language=json["language"],
                    conversational_quality_maxim=json["quality_maxim"],
                    conversational_quantity_maxim=json["quantity_maxim"],
                    conversational_relation_maxim=json["relation_maxim"],
                    conversational_manner_maxim=json["manner_maxim"],
                    conversational_sensibleness=json["sensibleness"],
                    conversational_thinkings=thinking,
                )
            )
