"""Storage interfaces for Fair Forge."""

from abc import ABC, abstractmethod
from datetime import datetime

from .common import Dataset


class BaseStorage(ABC):
    """
    Abstract base class for test dataset loading and result storage.

    Storage implementations handle loading test datasets from various sources
    (local filesystem, cloud storage, etc.) and saving execution results.
    """

    @abstractmethod
    def load_datasets(self) -> list[Dataset]:
        """
        Load test datasets from storage.

        Returns:
            list[Dataset]: List of test datasets loaded from storage
        """

    @abstractmethod
    def save_results(self, datasets: list[Dataset], run_id: str, timestamp: datetime) -> str:
        """
        Save executed test results to storage.

        Args:
            datasets: List of datasets with filled assistant responses
            run_id: Unique identifier for this test run
            timestamp: Timestamp of test execution

        Returns:
            str: Path or identifier where results were saved
        """


__all__ = ["BaseStorage"]
