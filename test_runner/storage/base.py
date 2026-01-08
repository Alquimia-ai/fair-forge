# test_runner/storage/base.py

from abc import ABC, abstractmethod
from datetime import datetime
from fair_forge.schemas.common import Dataset


class TestStorage(ABC):
    """Abstract base class for test suite loading and result storage."""

    @abstractmethod
    def load_datasets(self) -> list[Dataset]:
        """
        Load test datasets from storage.

        Returns:
            list[Dataset]: List of test datasets
        """
        pass

    @abstractmethod
    def save_results(self, datasets: list[Dataset], run_id: str, timestamp: datetime) -> str:
        """
        Save executed test results to storage.

        Args:
            datasets: List of datasets with filled assistant responses
            run_id: Unique identifier for this test run
            timestamp: Timestamp of test execution

        Returns:
            str: Path where results were saved
        """
        pass
