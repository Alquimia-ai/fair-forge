from fair_forge import FairForge, Retriever
from typing import Type, Optional
from fair_forge.schemas import Batch
from fair_forge.helpers.judge import Judge
from pydantic import SecretStr

class Arena(FairForge):
    def __init__(self, retriever: Type[Retriever],
        judge_base_url: str = "https://api.groq.com/openai/v1",
        judge_api_key: SecretStr = SecretStr(""),
        judge_model: str = "deepseek-r1-distill-llama-70b",
        judge_temperature: float = 0,
        judge_bos_json_clause: str = "```json",
        judge_eos_json_clause: str = "```",
        judge_bos_think_token: Optional[str] = None,
        judge_eos_think_token: Optional[str] = None,
        **kwargs,):
        super().__init__(retriever, **kwargs)
        self.judge_url = judge_base_url
        self.judge_api_key = judge_api_key
        self.judge_model = judge_model
        self.judge_temperature = judge_temperature
        self.judge_bos_think_token = judge_bos_think_token
        self.judge_eos_think_token = judge_eos_think_token
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
        for interaction in batch:
            pass
