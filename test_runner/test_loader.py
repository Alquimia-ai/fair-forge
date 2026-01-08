# test_runner/test_loader.py

from loguru import logger
from storage import get_storage
from fair_forge.schemas.common import Dataset


def load_datasets() -> list[Dataset]:
    """
    Load test datasets from configured storage backend.

    The storage backend is determined by the TEST_STORAGE_BACKEND
    environment variable (local or lakefs).

    Returns:
        list[Dataset]: List of test datasets to execute
    """
    logger.info("Loading test datasets...")

    storage = get_storage()
    datasets = storage.load_datasets()

    return datasets
