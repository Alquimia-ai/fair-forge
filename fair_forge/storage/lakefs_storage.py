"""LakeFS storage implementation."""

import json
from datetime import datetime

import lakefs
from lakefs.client import Client as LakeFSClient
from loguru import logger

from fair_forge.schemas.common import Dataset
from fair_forge.schemas.storage import BaseStorage


class LakeFSStorage(BaseStorage):
    """LakeFS implementation for test storage."""

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        repo_id: str,
        enabled_suites: list[str] | None = None,
        tests_prefix: str = "tests/",
        results_prefix: str = "results/",
        branch_name: str = "main",
    ):
        """
        Initialize LakeFS storage.

        Args:
            host: LakeFS server URL
            username: LakeFS username
            password: LakeFS password
            repo_id: LakeFS repository ID
            enabled_suites: List of enabled test suite names (None or empty = all)
            tests_prefix: Path prefix for test datasets in LakeFS
            results_prefix: Path prefix for results in LakeFS
            branch_name: Branch name to use (default: "main")
        """
        if not all([host, username, password, repo_id]):
            raise ValueError("LakeFS credentials are incomplete. Required: host, username, password, repo_id")

        self.client = LakeFSClient(
            username=username,
            password=password,
            host=host,
        )
        self.repo = lakefs.Repository(repository_id=repo_id, client=self.client)
        self.ref = self.repo.ref(branch_name)
        self.branch = self.repo.branch(branch_name)
        self.enabled_suites = enabled_suites or []
        self.tests_prefix = tests_prefix
        self.results_prefix = results_prefix

    def load_datasets(self) -> list[Dataset]:
        """
        Load test datasets from LakeFS.

        Returns:
            list[Dataset]: List of test datasets
        """
        datasets: list[Dataset] = []

        logger.info("Loading test datasets from LakeFS...")

        try:
            all_objects = list(self.ref.objects(prefix=self.tests_prefix))
            json_files = [obj for obj in all_objects if obj.path.endswith(".json")]

            logger.info(f"Found {len(json_files)} JSON test file(s) in LakeFS")

            for obj in json_files:
                remote_path = obj.path
                file_name = remote_path.split("/")[-1].replace(".json", "")

                # Skip if we have a filter and this suite is not in it
                if self.enabled_suites and file_name not in self.enabled_suites:
                    logger.info(f"Skipping test suite: {file_name} (not in enabled_suites)")
                    continue

                logger.info(f"Loading test dataset: {file_name}")

                try:
                    with self.ref.object(remote_path).reader(mode="rb") as r:
                        data = json.loads(r.read().decode("utf-8"))

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

        except Exception as e:
            logger.error(f"Failed to list objects from LakeFS: {e}")
            raise

        return datasets

    def save_results(self, datasets: list[Dataset], run_id: str, timestamp: datetime) -> str:
        """
        Save test results to LakeFS.

        Args:
            datasets: List of datasets with filled assistant responses
            run_id: Unique identifier for this test run
            timestamp: Timestamp of test execution

        Returns:
            str: Path where results were saved
        """
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        file_name = f"test_run_{timestamp_str}_{run_id}.json"
        remote_path = f"{self.results_prefix}{file_name}"

        logger.info(f"Saving results to LakeFS: {remote_path}")

        try:
            # Convert datasets to JSON
            results_data = [dataset.model_dump() for dataset in datasets]
            results_json = json.dumps(results_data, indent=2, ensure_ascii=False)

            # Write to LakeFS
            with self.branch.object(remote_path).writer(mode="wb") as writer:
                writer.write(results_json.encode("utf-8"))

            logger.success(f"Results saved to {remote_path}")
            return remote_path

        except Exception as e:
            logger.error(f"Failed to save results to LakeFS: {e}")
            raise


__all__ = ["LakeFSStorage"]
