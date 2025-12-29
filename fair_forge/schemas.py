from pydantic import BaseModel
from typing import Any, Dict, List, Literal, Optional
from enum import Enum
from typing import Type
from abc import ABC, abstractmethod
from functools import partial
from transformers import AutoTokenizer

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


class BaseMetric(BaseModel):
    """
    A metric represents a specific evaluation or measurement of the assistant's performance.
    """
    session_id: str
    assistant_id: str


class GuardianBias(BaseModel):
    """
    A data model that represents the result of a bias detection analysis.

    Attributes:
            is_biased (bool): Indicates whether bias was detected in the interaction
            attribute (dict): The specific attribute(s) that were analyzed for bias
            certainty (Optional[float]): A confidence score for the bias detection, if available
    """
    is_biased: bool
    attribute: str
    certainty: Optional[float]

class BiasMetric(BaseMetric):
    """
    Bias metric for evaluating the bias of the assistant's responses.
    """
    class ConfidenceInterval(BaseModel):
        lower_bound: float
        upper_bound: float
        probability: float
        samples: int
        k_success: int
        alpha: float
        confidence_level: float
        protected_attribute: str

    class GuardianInteraction(GuardianBias):
        qa_id: str

    confidence_intervals: list[ConfidenceInterval]
    guardian_interactions: dict[str, list[GuardianInteraction]]


class ProtectedAttribute(BaseModel):
    class Attribute(str,Enum):
        age = "age"
        gender = "gender"
        race = "race"
        religion = "religion"
        nationality = "nationality"
        sexual_orientation = "sexual_orientation"
        
    attribute: Attribute
    description: str

class ContextMetric(BaseMetric):
    context_insight: str
    context_awareness: float
    context_thinkings: str
    qa_id: str


class ConversationalMetric(BaseMetric):
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
    qa_id: str



class HumanityMetric(BaseMetric):
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
    qa_id: str



class LLMGuardianProviderInfer(BaseModel):
    is_bias: bool
    probability: float

class LLMGuardianProvider(ABC):
    def __init__(self,
                 model:str,
                 tokenizer:AutoTokenizer,
                 api_key:Optional[str] = None,
                 url:Optional[str] = None,
                 temperature:float=0.0,
                 safe_token: str = "No",
                 unsafe_token: str = "Yes",
                 max_tokens:int = 5 ,
                 logprobs:bool=True,
                 
                 **kwargs):
        
        self.model = model
        self.api_key = api_key
        self.url = url
        self.temperature = temperature
        self.safe_token = safe_token
        self.unsafe_token = unsafe_token
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.logprobs = logprobs
        
    @abstractmethod
    def infer(self,prompt: partial) -> LLMGuardianProviderInfer:
        raise NotImplementedError("Subclass must implement this method")

class GuardianLLMConfig(BaseModel):
    model:str
    api_key:Optional[str] = None
    url:Optional[str] = None
    temperature:float
    logprobs:bool = False
    provider:Type[LLMGuardianProvider]

class AgenticMetric(BaseMetric):
    pass


class BestOfContest(BaseModel):
    """
    A single contest within the best-of tournament bracket.
    """
    round: int
    left_id: str
    right_id: str
    winner_id: str
    confidence: float | None = None
    verdict: str | None = None
    reasoning: dict | None = None
    thinkings: str | None = None


class BestOfMetric(BaseMetric):
    """
    Best-of tournament metric capturing the final winner and all contests that led to it.
    """
    bestof_winner_id: str
    bestof_contests: list[BestOfContest]
    qa_id: str


class ToxicityDataset(BaseModel):
    word:str
    category: Optional[str] = None



class GroupProfilingFrequentist(BaseModel):
    DR: float
    ASB: float
    DTO: float
    DIDT: float


class BayesianSummary(BaseModel):
    mean: float
    ci_low: float
    ci_high: float


class GroupProfilingBayesian(BaseModel):
    priors: Dict[str, float]
    mc_samples: int
    ci_level: float
    q_i: Optional[Dict[str, float]] = None
    summary: Dict[str, BayesianSummary]  # keys: DR/ASB/DTO/DIDT


class GroupProfiling(BaseModel):
    MetricMode = Literal["frequentist", "bayesian"]
    mode: MetricMode
    weights: Dict[str, float]  # w_DR, w_ASB, w_DTO
    groups: List[str]

    N_i: Dict[str, int]
    K_i: Dict[str, int]
    p_i: Dict[str, float]
    q_i: Dict[str, float]
    T_i: Dict[str, float]

    frequentist: GroupProfilingFrequentist
    bayesian: Optional[GroupProfilingBayesian] = None

    toxicity_threshold: float = 0.0


class ToxicityMetric(BaseMetric):
    """
    Toxicity metric for evaluating toxic language in assistant responses.
    """
    class AssistantSpace(BaseModel):
        latent_space: list
        embeddings: list
        cluster_labels: list

    cluster_profiling: dict[float,float]
    group_profiling: Optional[GroupProfiling] = None
    assistant_space: AssistantSpace


class GroupDetection(BaseModel):
    """Result for a single group."""
    present: bool
    score: float
    best_prototype: str
    best_prototype_index: int


class BaseGroupExtractor(ABC):
    """
    Abstract base class for group extractors.
    
    Group extractors detect whether texts mention specific groups using
    various detection methods (e.g., embedding similarity, keyword matching, etc.).
    """
    
    @abstractmethod
    def detect_one(self, text: str) -> Dict[str, GroupDetection]:
        """
        Detect groups in a single text.

        Args:
            text: The text to analyze

        Returns:
            dict[group -> GroupDetection]
        """
        raise NotImplementedError("Subclass must implement detect_one")
    
    @abstractmethod
    def detect_batch(self, texts: List[str]) -> List[Dict[str, GroupDetection]]:
        """
        Detect groups for a batch of texts.

        Args:
            texts: List of texts to analyze

        Returns:
            list of dicts, aligned with `texts`. Each dict maps group -> GroupDetection
        """
        raise NotImplementedError("Subclass must implement detect_batch")
    
    def to_presence_table(self, detections: List[Dict[str, GroupDetection]]) -> List[Dict[str, Any]]:
        """
        Converts detections into row dicts:
          {"group": ..., "present": ..., "score": ..., "best_prototype": ...}

        Args:
            detections: Output of detect_batch()

        Returns:
            list of per-text dictionaries (one per input text) mapping group->fields
        """
        rows: List[Dict[str, Any]] = []
        for det in detections:
            row: Dict[str, Any] = {}
            for g, r in det.items():
                row[f"{g}_present"] = r.present
                row[f"{g}_score"] = r.score
                row[f"{g}_best_prototype"] = r.best_prototype
            rows.append(row)
        return rows


