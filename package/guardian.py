from pydantic import BaseModel
from transformers import AutoTokenizer
from typing import Optional
import requests
import math

class GuardianConfig(BaseModel):
    """
    Configuration for the Guardian.
    """
    url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: float = 5
    risks: Optional[list[dict]] = None

class GuardianInfer(BaseModel):
    is_risk: bool
    risk_type: str
    probability: float

class Guardian():
    """
    Guardian is a tool used alongside a LLM to detect risks in the text.
    """
    def __init__(self, config: GuardianConfig):
        self.url = config.url
        self.api_key = config.api_key
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        if config.risks:
            self.risks = config.risks
        else:
            ## Use default risks defined by the IBM team.
            self.risks = [
                {"risk_name": "social_bias"},
                {"risk_name": "harm"},
                {"risk_name": "profanity"},
                {"risk_name": "sexual_content"},
                {"risk_name": "unethical_behavior"},
                {"risk_name": "violence"}
            ]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
    
    def _parse_guardian_response(self,response_json):
        choice = response_json["choices"][0]
        logprobs = choice['logprobs']
        answer_token = True if logprobs['tokens'][0] == 'Yes' else False
        prob_token = logprobs['token_logprobs'][0]
        prob_token = math.exp(prob_token)
        return answer_token, prob_token
    
    def _infer(self,query: str, context: str, assistant_answer: str, config: dict):
        messages = [{"role":"system","content":context},{"role": "user", "content": query}, {"role": "assistant", "content": assistant_answer}]
        prompt = self.tokenizer.apply_chat_template(messages, guardian_config = config, tokenize=False, add_generation_prompt=True)
        ## TODO: Refactor to be provider agnostic.
        response = requests.post(
            f"{self.url}/v1/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
              "model": self.model_name,
              "prompt": prompt,
              "temperature": self.guardian_temperature,
              "max_tokens": self.max_tokens,
              "logprobs": self.logprobs
            }
        )
        answer_token, prob = self._parse_guardian_response(response.json())
        return GuardianInfer(
            is_risk = answer_token,
            probability = prob,
            risk_type = config['risk_name']
        )

    def detect(self, question: str,answer:str, context:str)-> list:
        """
        Given a question and answer, detect if there is a risk related to it.
        """
        pass
