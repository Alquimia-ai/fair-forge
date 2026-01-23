"""Custom exceptions for Fair Forge."""


class FairForgeError(Exception):
    """Base exception for Fair Forge."""


class RetrieverError(FairForgeError):
    """Exception raised when a retriever fails to load data."""


class MetricError(FairForgeError):
    """Exception raised when a metric calculation fails."""


class GuardianError(FairForgeError):
    """Exception raised when a guardian fails to detect bias."""


class LoaderError(FairForgeError):
    """Exception raised when a loader fails to load data."""


class StatisticalModeError(FairForgeError):
    """Exception raised when a statistical mode calculation fails."""
