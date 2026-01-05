## Process Chain of Thought models
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr
from langchain_core.prompts import ChatPromptTemplate
import re

class ChainOfThought(BaseModel):
    thought: str
    answer: str

class CoT:
    def __init__(
        self,
        bos_think_token: str = "<think>",
        eos_think_token: str = "</think>",
        base_url: str = "https://api.groq.com/openai",
        api_key: SecretStr = SecretStr(""),
        model: str = "deepseek-r1-distill-llama-70b",
        temperature: float = 0,
    ):
        self.bos_think_token = bos_think_token
        self.eos_think_token = eos_think_token
        self.chat_history = []
        self.reasoning_model = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
        )

    def reason(self, system_prompt: str, query: str, **kwargs):
        self.chat_history.append(("human", query))
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), *self.chat_history]
        )
        chain = prompt | self.reasoning_model
        response = chain.invoke(kwargs)
        reasoning = str(response.content)

        # Split the text into thinking and non-thinking sections
        pattern = rf"{self.bos_think_token}(.*?){self.eos_think_token}(.*)"
        match = re.search(pattern, reasoning, re.DOTALL)

        if match:
            think_content = match.group(1).strip()
            non_think_content = match.group(2).strip()
        else:
            think_content = ""
            non_think_content = reasoning.strip()

        return ChainOfThought(thought=think_content, answer=non_think_content)
