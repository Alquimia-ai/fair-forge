"""Custom exceptions for Fair Forge."""


class FairForgeError(Exception):
    """Base exception for Fair Forge."""
    pass


class RetrieverError(FairForgeError):
    """Exception raised when a retriever fails to load data."""
    pass


class MetricError(FairForgeError):
    """Exception raised when a metric calculation fails."""
    pass


class GuardianError(FairForgeError):
    """Exception raised when a guardian fails to detect bias."""
    pass


class LoaderError(FairForgeError):
    """Exception raised when a loader fails to load data."""
    pass


class StatisticalModeError(FairForgeError):
    """Exception raised when a statistical mode calculation fails."""
    pass
