# test_runner/storage/__init__.py

from loguru import logger
from .base import TestStorage
from .local_storage import LocalTestStorage
from .lakefs_storage import LakeFSTestStorage


def get_storage() -> TestStorage:
    """
    Get storage backend based on configuration.

    Returns:
        TestStorage: Configured storage backend instance

    Raises:
        ValueError: If storage backend is unknown or configuration is invalid
    """
    # Import here to avoid circular dependency
    from config import (
        TEST_STORAGE_BACKEND,
        TESTS_DIR,
        RESULTS_DIR,
        ENABLED_TEST_SUITES,
        LAKEFS_HOST,
        LAKEFS_USERNAME,
        LAKEFS_PASSWORD,
        LAKEFS_REPO_ID,
    )

    if TEST_STORAGE_BACKEND == "local":
        logger.info("Using local filesystem storage")
        return LocalTestStorage(
            tests_dir=TESTS_DIR,
            results_dir=RESULTS_DIR,
            enabled_suites=ENABLED_TEST_SUITES,
        )
    elif TEST_STORAGE_BACKEND == "lakefs":
        logger.info("Using LakeFS storage")
        return LakeFSTestStorage(
            host=LAKEFS_HOST,
            username=LAKEFS_USERNAME,
            password=LAKEFS_PASSWORD,
            repo_id=LAKEFS_REPO_ID,
            enabled_suites=ENABLED_TEST_SUITES,
        )
    else:
        raise ValueError(
            f"Unknown storage backend: {TEST_STORAGE_BACKEND}. "
            f"Valid options: 'local', 'lakefs'"
        )


__all__ = ["TestStorage", "get_storage", "LocalTestStorage", "LakeFSTestStorage"]
