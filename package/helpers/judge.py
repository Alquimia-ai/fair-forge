from pydantic import SecretStr
from cot import CoT
import re
import logging
import json


class Judge:
    def __init__(
        self,
        bos_think_token: str = "<think>",
        eos_think_token: str = "</think>",
        base_url: str = "https://api.groq.com/openai",
        api_key: SecretStr = SecretStr(""),
        model: str = "deepseek-r1-distill-llama-70b",
        temperature: float = 0,
        bos_json_clause: str = "```json",
        eos_json_clause: str = "```",
    ):
        self.cot = CoT(
            bos_think_token=bos_think_token,
            eos_think_token=eos_think_token,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
        self.bos_json_clause = bos_json_clause
        self.eos_json_clause = eos_json_clause

    def check(self, system_prompt: str, query: str, data: dict):
        """
        Judge an assistant response using Chain of Thought (CoT) reasoning. using an specific criteria
        """
        response = self.cot.reason(
            system_prompt=system_prompt,
            query=query,
            **data,
        )
        ## the judge answer is always a json
        json_match = re.search(
            self.bos_json_clause + r"\s*(\{.*?\})\s*" + self.eos_json_clause,
            response.answer,
            re.DOTALL,
        )
        json_str = json_match.group(1).strip() if json_match else ""
        json_data = None
        if json_str:
            try:
                json_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                json_data = None
                logging.error(f"[FAIR FORGE/JUDGE] JSON decoding error: {e}")
        else:
            logging.error(
                f"[FAIR FORGE/JUDGE] No JSON found {self.bos_json_clause} {self.eos_json_clause} "
            )
        return response.thought, json_data
