"""Local filesystem storage implementation."""

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from fair_forge.schemas.common import Dataset
from fair_forge.schemas.storage import BaseStorage


class LocalStorage(BaseStorage):
    """Local filesystem implementation for test storage."""

    def __init__(self, tests_dir: Path, results_dir: Path, enabled_suites: list[str] | None = None):
        """
        Initialize local storage.

        Args:
            tests_dir: Directory containing test dataset JSON files
            results_dir: Directory for saving test results
            enabled_suites: List of enabled test suite names (None or empty = all)
        """
        self.tests_dir = Path(tests_dir)
        self.results_dir = Path(results_dir)
        self.enabled_suites = enabled_suites or []

    def load_datasets(self) -> list[Dataset]:
        """
        Load test datasets from local filesystem.

        Returns:
            list[Dataset]: List of test datasets
        """
        datasets: list[Dataset] = []

        if not self.tests_dir.exists():
            logger.warning(f"Tests directory does not exist: {self.tests_dir}")
            return datasets

        logger.info(f"Loading test datasets from {self.tests_dir}")

        # Find all JSON files in tests directory
        json_files = list(self.tests_dir.glob("*.json"))

        if not json_files:
            logger.warning(f"No JSON test files found in {self.tests_dir}")
            return datasets

        logger.info(f"Found {len(json_files)} JSON test file(s)")

        for json_file in json_files:
            file_name = json_file.stem

            # Skip if we have a filter and this suite is not in it
            if self.enabled_suites and file_name not in self.enabled_suites:
                logger.info(f"Skipping test suite: {file_name} (not in enabled_suites)")
                continue

            logger.info(f"Loading test dataset: {file_name}")

            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                # Support loading multiple datasets from a single file
                if isinstance(data, list):
                    # File contains array of datasets
                    for dataset_data in data:
                        dataset = Dataset.model_validate(dataset_data)
                        datasets.append(dataset)
                    logger.success(f"Loaded {len(data)} dataset(s) from {file_name}")
                else:
                    # File contains single dataset
                    dataset = Dataset.model_validate(data)
                    datasets.append(dataset)
                    logger.success(f"Loaded dataset from {file_name}")

            except Exception as e:
                logger.error(f"Failed to load test dataset {file_name}: {e}")
                continue

        if not datasets:
            logger.warning("No test datasets loaded!")
        else:
            total_batches = sum(len(ds.conversation) for ds in datasets)
            logger.info(f"Loaded {len(datasets)} dataset(s) with {total_batches} total test case(s)")

        return datasets

    def save_results(self, datasets: list[Dataset], run_id: str, timestamp: datetime) -> str:
        """
        Save test results to local filesystem.

        Args:
            datasets: List of datasets with filled assistant responses
            run_id: Unique identifier for this test run
            timestamp: Timestamp of test execution

        Returns:
            str: Path where results were saved
        """
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        file_name = f"test_run_{timestamp_str}_{run_id}.json"
        output_path = self.results_dir / file_name

        logger.info(f"Saving results locally: {output_path}")

        try:
            # Convert datasets to JSON
            results_data = [dataset.model_dump() for dataset in datasets]

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)

            logger.success(f"Results saved locally to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to save results locally: {e}")
            raise


__all__ = ["LocalStorage"]
