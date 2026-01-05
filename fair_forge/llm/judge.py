from typing import Optional
from pydantic import SecretStr
from fair_forge.llm.cot import ChainOfThought, CoT
import re
import logging
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class Judge:
    def __init__(
        self,
        bos_think_token: Optional[str] =None,
        eos_think_token: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai",
        api_key: SecretStr = SecretStr(""),
        model: str = "deepseek-r1-distill-llama-70b",
        temperature: float = 0,
        bos_json_clause = "```json",
        eos_json_clause = "```",
    ):
        if bos_think_token is not None:
            # if bos_think_token is not None, then we need to use the CoT model
            self.cot = CoT(
                bos_think_token=bos_think_token,
                eos_think_token=eos_think_token,
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
            )
        else:
            self.cot = None
            self.chat_history = []
            self.model = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            )
        self.bos_json_clause = bos_json_clause
        self.eos_json_clause = eos_json_clause

    def direct_inference(self,system_prompt:str, query: str, **kwargs):
        self.chat_history.append(("human", query))
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), *self.chat_history]
        )
        
        chain = prompt | self.model
        response = chain.invoke(kwargs)
        return ChainOfThought(thought="", answer=response.content)


    def check(self, system_prompt: str, query: str, data: dict):
        """
        Judge a response using an specific criteria
        """
        if self.cot is not None:
            response = self.cot.reason(
                system_prompt=system_prompt,
                    query=query,
                    **data,
            )
        else:
            ## Get response directly from the model, without using the CoT model
            response = self.direct_inference(system_prompt, query, **data)

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
