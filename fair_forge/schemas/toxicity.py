"""Toxicity metric schemas."""
from pydantic import BaseModel
from typing import Dict, List, Literal, Optional
from .metrics import BaseMetric


class ToxicityDataset(BaseModel):
    """A toxicity word entry."""
    word: str
    category: Optional[str] = None


class GroupDetection(BaseModel):
    """Result for a single group detection."""
    present: bool
    score: float
    best_prototype: str
    best_prototype_index: int


class GroupProfilingFrequentist(BaseModel):
    """Frequentist statistics for group profiling."""
    DR: float
    ASB: float
    DTO: float
    DIDT: float


class BayesianSummary(BaseModel):
    """Bayesian posterior summary."""
    mean: float
    ci_low: float
    ci_high: float


class GroupProfilingBayesian(BaseModel):
    """Bayesian statistics for group profiling."""
    priors: Dict[str, float]
    mc_samples: int
    ci_level: float
    q_i: Optional[Dict[str, float]] = None
    summary: Dict[str, BayesianSummary]  # keys: DR/ASB/DTO/DIDT


class GroupProfiling(BaseModel):
    """Group profiling results for toxicity analysis."""
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

    cluster_profiling: dict[float, float]
    group_profiling: Optional[GroupProfiling] = None
    assistant_space: AssistantSpace
