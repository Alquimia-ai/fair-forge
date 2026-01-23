"""Storage module for Fair Forge.

This module provides storage implementations for loading test datasets
and saving execution results from various sources (local filesystem, LakeFS, etc.).
"""

from pathlib import Path
from typing import Literal

from loguru import logger

from fair_forge.schemas.storage import BaseStorage

from .lakefs_storage import LakeFSStorage
from .local_storage import LocalStorage


def create_local_storage(
    tests_dir: str | Path,
    results_dir: str | Path,
    enabled_suites: list[str] | None = None,
) -> LocalStorage:
    """
    Create a local filesystem storage backend.

    Args:
        tests_dir: Directory containing test dataset JSON files
        results_dir: Directory for saving test results
        enabled_suites: List of enabled test suite names (None or empty = all)

    Returns:
        LocalStorage: Configured local storage instance
    """
    logger.info("Creating local filesystem storage")
    return LocalStorage(
        tests_dir=Path(tests_dir),
        results_dir=Path(results_dir),
        enabled_suites=enabled_suites,
    )


def create_lakefs_storage(
    host: str,
    username: str,
    password: str,
    repo_id: str,
    enabled_suites: list[str] | None = None,
    tests_prefix: str = "tests/",
    results_prefix: str = "results/",
    branch_name: str = "main",
) -> LakeFSStorage:
    """
    Create a LakeFS storage backend.

    Args:
        host: LakeFS server URL
        username: LakeFS username
        password: LakeFS password
        repo_id: LakeFS repository ID
        enabled_suites: List of enabled test suite names (None or empty = all)
        tests_prefix: Path prefix for test datasets in LakeFS
        results_prefix: Path prefix for results in LakeFS
        branch_name: Branch name to use (default: "main")

    Returns:
        LakeFSStorage: Configured LakeFS storage instance

    Raises:
        ValueError: If credentials are incomplete
    """
    logger.info("Creating LakeFS storage")
    return LakeFSStorage(
        host=host,
        username=username,
        password=password,
        repo_id=repo_id,
        enabled_suites=enabled_suites,
        tests_prefix=tests_prefix,
        results_prefix=results_prefix,
        branch_name=branch_name,
    )


def create_storage(
    backend: Literal["local", "lakefs"],
    **kwargs,
) -> BaseStorage:
    """
    Factory function to create a storage backend based on type.

    Args:
        backend: Storage backend type ("local" or "lakefs")
        **kwargs: Backend-specific configuration parameters

    Returns:
        BaseStorage: Configured storage backend instance

    Raises:
        ValueError: If backend type is unknown

    Examples:
        >>> # Create local storage
        >>> storage = create_storage(
        ...     backend="local",
        ...     tests_dir="./tests",
        ...     results_dir="./results",
        ... )
        >>>
        >>> # Create LakeFS storage
        >>> storage = create_storage(
        ...     backend="lakefs",
        ...     host="http://lakefs:8000",
        ...     username="admin",
        ...     password="secret",
        ...     repo_id="my-repo",
        ... )
    """
    if backend == "local":
        return create_local_storage(**kwargs)
    if backend == "lakefs":
        return create_lakefs_storage(**kwargs)
    raise ValueError(f"Unknown storage backend: {backend}. Valid options: 'local', 'lakefs'")


__all__ = [
    "BaseStorage",
    "LakeFSStorage",
    "LocalStorage",
    "create_lakefs_storage",
    "create_local_storage",
    "create_storage",
]
