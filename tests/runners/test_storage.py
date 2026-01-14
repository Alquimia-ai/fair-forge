"""Tests for storage implementations."""
import json
import pytest
from pathlib import Path
from datetime import datetime
from fair_forge.storage import LocalStorage, create_local_storage, create_storage
from fair_forge.schemas.common import Dataset, Batch


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create temporary test directories."""
    tests_dir = tmp_path / "tests"
    results_dir = tmp_path / "results"
    tests_dir.mkdir()
    results_dir.mkdir()
    return tests_dir, results_dir


@pytest.fixture
def mock_dataset_json(temp_test_dir):
    """Create a mock dataset JSON file."""
    tests_dir, _ = temp_test_dir
    dataset_data = {
        "session_id": "test_session_001",
        "assistant_id": "test_assistant",
        "language": "english",
        "context": "Test context",
        "conversation": [
            {
                "qa_id": "qa_001",
                "query": "What is AI?",
                "assistant": "",
                "ground_truth_assistant": "AI is artificial intelligence",
                "observation": "Test question",
                "agentic": {},
                "ground_truth_agentic": {},
            }
        ],
    }

    json_file = tests_dir / "test_suite.json"
    with open(json_file, "w") as f:
        json.dump(dataset_data, f)

    return json_file


def test_local_storage_initialization(temp_test_dir):
    """Test LocalStorage initialization."""
    tests_dir, results_dir = temp_test_dir

    storage = LocalStorage(
        tests_dir=tests_dir,
        results_dir=results_dir,
        enabled_suites=None,
    )

    assert storage.tests_dir == tests_dir
    assert storage.results_dir == results_dir
    assert storage.enabled_suites == []


def test_local_storage_load_datasets(temp_test_dir, mock_dataset_json):
    """Test loading datasets from local storage."""
    tests_dir, results_dir = temp_test_dir

    storage = LocalStorage(
        tests_dir=tests_dir,
        results_dir=results_dir,
        enabled_suites=None,
    )

    datasets = storage.load_datasets()

    assert len(datasets) == 1
    assert datasets[0].session_id == "test_session_001"
    assert len(datasets[0].conversation) == 1


def test_local_storage_load_datasets_with_filter(temp_test_dir, mock_dataset_json):
    """Test loading datasets with enabled_suites filter."""
    tests_dir, results_dir = temp_test_dir

    # Test with filter that includes the suite
    storage = LocalStorage(
        tests_dir=tests_dir,
        results_dir=results_dir,
        enabled_suites=["test_suite"],
    )

    datasets = storage.load_datasets()
    assert len(datasets) == 1

    # Test with filter that excludes the suite
    storage = LocalStorage(
        tests_dir=tests_dir,
        results_dir=results_dir,
        enabled_suites=["other_suite"],
    )

    datasets = storage.load_datasets()
    assert len(datasets) == 0


def test_local_storage_load_datasets_empty_directory(temp_test_dir):
    """Test loading datasets from empty directory."""
    tests_dir, results_dir = temp_test_dir

    storage = LocalStorage(
        tests_dir=tests_dir,
        results_dir=results_dir,
        enabled_suites=None,
    )

    datasets = storage.load_datasets()
    assert len(datasets) == 0


def test_local_storage_save_results(temp_test_dir, sample_dataset):
    """Test saving results to local storage."""
    tests_dir, results_dir = temp_test_dir

    storage = LocalStorage(
        tests_dir=tests_dir,
        results_dir=results_dir,
        enabled_suites=None,
    )

    run_id = "test_run_001"
    timestamp = datetime(2024, 1, 1, 12, 0, 0)

    result_path = storage.save_results(
        datasets=[sample_dataset],
        run_id=run_id,
        timestamp=timestamp,
    )

    # Check that file was created
    assert Path(result_path).exists()
    assert "test_run_20240101_120000" in result_path
    assert run_id in result_path

    # Check file contents
    with open(result_path) as f:
        data = json.load(f)

    assert len(data) == 1
    assert data[0]["session_id"] == sample_dataset.session_id


def test_local_storage_save_multiple_datasets(temp_test_dir, sample_dataset):
    """Test saving multiple datasets to local storage."""
    tests_dir, results_dir = temp_test_dir

    storage = LocalStorage(
        tests_dir=tests_dir,
        results_dir=results_dir,
        enabled_suites=None,
    )

    # Create second dataset
    dataset2 = sample_dataset.model_copy(update={"session_id": "session_002"})

    result_path = storage.save_results(
        datasets=[sample_dataset, dataset2],
        run_id="test_run_002",
        timestamp=datetime.now(),
    )

    # Check file contents
    with open(result_path) as f:
        data = json.load(f)

    assert len(data) == 2
    assert data[0]["session_id"] == sample_dataset.session_id
    assert data[1]["session_id"] == "session_002"


def test_create_local_storage_factory():
    """Test create_local_storage factory function."""
    storage = create_local_storage(
        tests_dir="./tests",
        results_dir="./results",
        enabled_suites=["suite1", "suite2"],
    )

    assert isinstance(storage, LocalStorage)
    assert storage.tests_dir == Path("./tests")
    assert storage.results_dir == Path("./results")
    assert storage.enabled_suites == ["suite1", "suite2"]


def test_create_storage_factory_local():
    """Test create_storage factory with local backend."""
    storage = create_storage(
        backend="local",
        tests_dir="./tests",
        results_dir="./results",
    )

    assert isinstance(storage, LocalStorage)


def test_create_storage_factory_invalid_backend():
    """Test create_storage factory with invalid backend."""
    with pytest.raises(ValueError, match="Unknown storage backend"):
        create_storage(backend="invalid", tests_dir="./tests", results_dir="./results")


def test_local_storage_load_multiple_datasets_from_array(temp_test_dir):
    """Test loading multiple datasets from a single JSON file with array."""
    tests_dir, results_dir = temp_test_dir

    # Create JSON file with array of datasets
    datasets_data = [
        {
            "session_id": "session_001",
            "assistant_id": "assistant_001",
            "language": "english",
            "context": "Context 1",
            "conversation": [
                {
                    "qa_id": "qa_001",
                    "query": "Query 1",
                    "assistant": "",
                    "ground_truth_assistant": "Answer 1",
                    "observation": None,
                    "agentic": {},
                    "ground_truth_agentic": {},
                }
            ],
        },
        {
            "session_id": "session_002",
            "assistant_id": "assistant_001",
            "language": "english",
            "context": "Context 2",
            "conversation": [
                {
                    "qa_id": "qa_002",
                    "query": "Query 2",
                    "assistant": "",
                    "ground_truth_assistant": "Answer 2",
                    "observation": None,
                    "agentic": {},
                    "ground_truth_agentic": {},
                }
            ],
        },
    ]

    json_file = tests_dir / "multiple_datasets.json"
    with open(json_file, "w") as f:
        json.dump(datasets_data, f)

    storage = LocalStorage(
        tests_dir=tests_dir,
        results_dir=results_dir,
        enabled_suites=None,
    )

    datasets = storage.load_datasets()

    assert len(datasets) == 2
    assert datasets[0].session_id == "session_001"
    assert datasets[1].session_id == "session_002"
