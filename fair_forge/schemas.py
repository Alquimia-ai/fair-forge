from pydantic import BaseModel
from typing import Optional


class Logprobs(BaseModel):
    """
    Logprobs are the log probabilities of the tokens in the response.
    """

    tokens: list[str]
    token_logprobs: list[float]


class Batch(BaseModel):
    """
    A batch represents a single interaction between a user and an assistant in a conversation.
    Each batch contains the user's query, the assistant's response, and optional metadata.

    Attributes:
        qa_id (str): Unique identifier for the batch
        ground_truth_assistant (Optional[str]): The expected or reference response from the assistant
        observation (Optional[str]): Any additional observations or notes about the interaction
        assistant (str): The actual response provided by the assistant
        query (str): The user's input or question
        agentic (Optional[dict]): Additional metadata or context about the agent's behavior
        ground_truth_agentic (Optional[dict]): Expected or reference metadata for the agent's behavior
    """
    ground_truth_assistant: str
    logprobs: Optional[dict] = {}
    observation: Optional[str]= None
    assistant: str
    query: str
    agentic: Optional[dict] = {}
    ground_truth_agentic: Optional[dict] = {}
    qa_id: str


class Dataset(BaseModel):
    """
    A dataset represents a complete conversation session between a user and an assistant.
    It contains metadata about the session and a list of all interactions (batches).

    Attributes:
        session_id (str): Unique identifier for the conversation session
        assistant_id (str): Identifier for the specific assistant involved
        language (str): The language used in the conversation
        context (str): Additional context or background information for the conversation
        conversation (list[Batch]): List of all interactions (batches) in the conversation
    """

    session_id: str
    assistant_id: str
    language: Optional[str]= "english"
    context: str
    conversation: list[Batch]


class Metric(BaseModel):
    """
    A metric represents a specific evaluation or measurement of the assistant's performance.
    """
    session_id: str
    qa_id: str
    assistant_id: str

class Risk(BaseModel):
    risk_name: str
    risk_score: float
    is_risk: bool

class BiasMetric(Metric):
    """
    Bias metric for evaluating the bias of the assistant's responses.
    """
    risks: list[Risk]


class ContextMetric(Metric):
    context_insight: str
    context_awareness: float
    context_thinkings: str


class ConversationalMetric(Metric):
    """
    Conversational metric for evaluating the assistant's conversational abilities.
    """

    conversational_memory: float
    conversational_insight: str
    conversational_language: float
    conversational_quality_maxim: float
    conversational_quantity_maxim: float
    conversational_relation_maxim: float
    conversational_manner_maxim: float
    conversational_sensibleness: float
    conversational_thinkings: str


class HumanityMetric(Metric):
    humanity_assistant_emotional_entropy: float
    humanity_ground_truth_spearman: float
    humanity_assistant_anger: float
    humanity_assistant_anticipation: float
    humanity_assistant_disgust: float
    humanity_assistant_fear: float
    humanity_assistant_joy: float
    humanity_assistant_sadness: float
    humanity_assistant_surprise: float
    humanity_assistant_trust: float

class AgenticMetric(Metric):
    pass
