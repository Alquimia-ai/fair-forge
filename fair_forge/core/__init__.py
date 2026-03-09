"""Core abstractions and base classes for Fair Forge.

Pipeline components (document_retriever, contradiction_checker) require numpy and should
be imported directly:
    from fair_forge.core.document_retriever import DocumentRetriever, DocumentRetrieverConfig
    from fair_forge.core.contradiction_checker import ContradictionChecker
"""

from .base import FairForge
from .embedder import Embedder
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
from .reranker import Reranker
from .retriever import Retriever
from .sentiment import SentimentAnalyzer

__all__ = [
    "BaseGroupExtractor",
    "Embedder",
    "FairForge",
    "FairForgeError",
    "Guardian",
    "GuardianError",
    "LoaderError",
    "MetricError",
    "Reranker",
    "Retriever",
    "RetrieverError",
    "SentimentAnalyzer",
    "StatisticalModeError",
    "ToxicityLoader",
]
