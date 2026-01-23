"""
Fair-Forge: AI Evaluation Framework

A modular framework for measuring fairness, quality, and safety of AI assistant responses.
"""

# Core abstractions
from .core import (
    BaseGroupExtractor,
    FairForge,
    Guardian,
    Retriever,
    ToxicityLoader,
)

# Common schemas
from .schemas import (
    BaseMetric,
    Batch,
    Dataset,
)

# Statistical modes
from .statistical import BayesianMode, FrequentistMode, StatisticalMode

# Version
__version__ = "0.1.1"

__all__ = [
    # Core
    "FairForge",
    "Retriever",
    "Guardian",
    "ToxicityLoader",
    "BaseGroupExtractor",
    # Schemas
    "Batch",
    "Dataset",
    "BaseMetric",
    # Statistical
    "StatisticalMode",
    "FrequentistMode",
    "BayesianMode",
]
