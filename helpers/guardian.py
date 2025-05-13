from pydantic import BaseModel
import requests
from transformers import AutoTokenizer
import math

class GuardianInfer(BaseModel):
    is_risk: bool
    risk_type: str
    probability: float

class GuardianRisks(BaseModel):
    risks : list
    qa_id: str
    session_id: str

class Guardian():
    def _parse_logprobs(response_json):
        choice = response_json["choices"][0]
        logprobs = choice["logprobs"]

        tokens = logprobs["tokens"]
        token_logprobs = logprobs["token_logprobs"]

        parsed = []
        for token, logprob in zip(tokens, token_logprobs):
            prob_pct = math.exp(logprob) * 100
            parsed.append((token, logprob, prob_pct))
        return parsed

    def __init__(self, url: str, model_name: str, api_key:str, model:str, temperature:float, max_tokens:float = 5):
        self.url = url
        self.model_name = model_name
        self.api_key = api_key
        self.hf_model = model
        self.guardian_temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.logprobs = True
        self.max_tokens = max_tokens
        self.guardian_config = [
            {"risk_name": "social_bias"},
            {"risk_name": "harm"},
            {"risk_name": "profanity"},
            {"risk_name": "sexual_content"},
            {"risk_name": "unethical_behavior"},
            {"risk_name": "violence"}
        ]

    def _parse_guardian_response(self,response_json):
        choice = response_json["choices"][0]
        logprobs = choice['logprobs']
        answer_token = True if logprobs['tokens'][0] == 'Yes' else False
        prob_token = logprobs['token_logprobs'][0]
        prob_token = math.exp(prob_token)
        return answer_token, prob_token

    def _infer(self, query: str, context: str, assistant_answer: str, config: dict) -> GuardianInfer:
        messages = [{"role":"system","content":context},{"role": "user", "content": query}, {"role": "assistant", "content": assistant_answer}]
        prompt = self.tokenizer.apply_chat_template(messages, guardian_config = config, tokenize=False, add_generation_prompt=True)
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

    def detect(self, batch: dict, context:str)-> list:
        infers = []
        for config in self.guardian_config:
            infers.append(self._infer(batch.question, context , batch.assistant, config))
        return infers