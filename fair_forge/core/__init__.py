"""Core abstractions and base classes for Fair Forge.

Regulatory components (embedder, reranker) require torch and should be imported directly:
    from fair_forge.core.embedder import RegulatoryEmbedder, EmbedderConfig
    from fair_forge.core.reranker import RegulatoryReranker, RerankerConfig
"""

from .base import FairForge
from .exceptions import (
    FairForgeError,
    GuardianError,
    LoaderError,
    MetricError,
    RetrieverError,
    StatisticalModeError,
)
from .extractor import BaseGroupExtractor
from .guardian import Guardian
from .loader import ToxicityLoader
from .retriever import Retriever
from .sentiment import SentimentAnalyzer

__all__ = [
    "BaseGroupExtractor",
    "FairForge",
    "FairForgeError",
    "Guardian",
    "GuardianError",
    "LoaderError",
    "MetricError",
    "Retriever",
    "RetrieverError",
    "SentimentAnalyzer",
    "StatisticalModeError",
    "ToxicityLoader",
]
