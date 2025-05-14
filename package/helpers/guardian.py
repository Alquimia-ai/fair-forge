from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Optional, Type
import requests
import math
from abc import ABC, abstractmethod
import torch
from functools import partial


class GuardianInfer(BaseModel):
    is_risk: bool
    risk_type: str
    probability: float


class GuardianProvider(BaseModel):
    probability: float
    is_risk: bool


class Provider:
    def __init__(
        self,
        url,
        api_key,
        model,
        tokenizer,
        temperature: float = 0,
        max_tokens: int = 5,
        **kwargs,
    ):
        self.url = url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logprobs = True
        self.tokenizer = tokenizer

    @abstractmethod
    def infer(self, prompt) -> GuardianProvider:
        raise NotImplementedError("You should implement this method.")


class VllmProvider(Provider):
    """
    Uses OpenAI SLA to detect risks.
    """

    def _parse_guardian_response(self, response_json):
        choice = response_json["choices"][0]
        logprobs = choice["logprobs"]
        answer_token = True if logprobs["tokens"][0] == "Yes" else False
        prob_token = logprobs["token_logprobs"][0]
        prob_token = math.exp(prob_token)
        return answer_token, prob_token

    def infer(self, prompt) -> GuardianProvider:
        response = requests.post(
            f"{self.url}/v1/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "logprobs": self.logprobs,
            },
        )
        answer_token, prob = self._parse_guardian_response(response.json())
        return GuardianProvider(probability=prob, is_risk=answer_token)


class HuggingFaceProvider(Provider):
    """
    HuggingFaceProvider is a provider that uses Hugging Face models to detect risks.
    It uses the transformers library to load the model and tokenizer.
    """

    safe_token = "No"
    unsafe_token = "Yes"

    def _parse_output(self, output: Any, input_len):
        nlogprobs = 20
        is_risk, prob_of_risk = False, None

        if nlogprobs > 0:
            list_index_logprobs_i = [
                torch.topk(token_i, k=nlogprobs, largest=True, sorted=True)
                for token_i in list(output.scores)[:-1]
            ]
            if list_index_logprobs_i is not None:
                prob = self._get_probabilities(list_index_logprobs_i)
                prob_of_risk = prob[1]

        res = self.tokenizer.decode(
            output.sequences[:, input_len:][0], skip_special_tokens=True
        ).strip()

        is_risk = True if res.lower() == self.unsafe_token.lower() else False

        if prob_of_risk is None:
            raise ValueError(
                "Probabilities are not available. Please check the model output."
            )

        return is_risk, prob_of_risk.item()

    def _get_probabilities(self, logprobs):
        safe_token_prob = 1e-50
        unsafe_token_prob = 1e-50
        for gen_token_i in logprobs:
            for logprob, index in zip(
                gen_token_i.values.tolist()[0], gen_token_i.indices.tolist()[0]
            ):
                decoded_token = self.tokenizer.convert_ids_to_tokens(index)
                if decoded_token.strip().lower() == self.safe_token.lower():
                    safe_token_prob += math.exp(logprob)
                if decoded_token.strip().lower() == self.unsafe_token.lower():
                    unsafe_token_prob += math.exp(logprob)

        probabilities = torch.softmax(
            torch.tensor([math.log(safe_token_prob), math.log(unsafe_token_prob)]),
            dim=0,
        )

        return probabilities

    def infer(self, prompt: partial) -> GuardianProvider:
        model = AutoModelForCausalLM.from_pretrained(
            self.model, device_map="auto", torch_dtype=torch.bfloat16
        )
        prompt = partial(prompt, return_tensors="pt")
        input_ids = prompt().to(self.model.device)
        input_len = input_ids.shape[1]
        model.eval()

        with torch.no_grad():
            output = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=20,
                return_dict_in_generate=True,
                output_scores=True,
            )

        is_risk, prob_of_risk = self._parse_output(output, input_len)
        return GuardianProvider(probability=prob_of_risk, is_risk=is_risk)


class GuardianConfig(BaseModel):
    """
    Configuration for the Guardian.
    """

    url: Optional[str]
    api_key: Optional[str]
    model: Optional[str]
    temperature: float = 0
    max_tokens: float = 5
    risks: Optional[list[dict]] = None
    provider: Optional[Type[Provider]] = VllmProvider


class Guardian:
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
                {"risk_name": "violence"},
            ]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.provider = config.provider(
            self.url, self.api_key, self.model, self.tokenizer
        )

    def _infer_risk(
        self, query: str, context: str, assistant_answer: str, config: dict
    ):
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant_answer},
        ]
        prompt = partial(
            self.tokenizer.apply_chat_template,
            messages=messages,
            guardian_config=config,
            tokenize=False,
            add_generation_prompt=True,
        )
        infer = self.provider.infer(prompt)

        return GuardianInfer(
            is_risk=infer.is_risk,
            risk_type=config["risk_name"],
            probability=infer.probability,
        )

    def has_any_risk(self, question: str, answer: str, context: str) -> list:
        """
        Given a question and answer, detect if there is a risk related to it.
        """
        risks = []
        for risk in self.risks:
            infer = self._infer_risk(question, context, answer, config=risk)
            risks.append(infer)
        return risks

    def is_context_aware(self):
        raise NotImplementedError(
            "The Guardian is not context aware. Please implement this method."
        )
