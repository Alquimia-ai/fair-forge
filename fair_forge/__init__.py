"""
Fair-Forge: AI Evaluation Framework

A modular framework for measuring fairness, quality, and safety of AI assistant responses.
"""

# Core abstractions
from .core import (
    FairForge,
    Retriever,
    Guardian,
    ToxicityLoader,
    BaseGroupExtractor,
)

# Common schemas
from .schemas import (
    Batch,
    Dataset,
    BaseMetric,
)

# Statistical modes
from .statistical import StatisticalMode, FrequentistMode, BayesianMode

# Version
__version__ = '0.0.1'

__all__ = [
    # Core
    'FairForge',
    'Retriever',
    'Guardian',
    'ToxicityLoader',
    'BaseGroupExtractor',
    # Schemas
    'Batch',
    'Dataset',
    'BaseMetric',
    # Statistical
    'StatisticalMode',
    'FrequentistMode',
    'BayesianMode',
]
