"""Tests for BaseRunner interface."""

from typing import Any

import pytest

from fair_forge.schemas.common import Batch, Dataset
from fair_forge.schemas.runner import BaseRunner


class MockRunner(BaseRunner):
    """Mock runner implementation for testing."""

    async def run_batch(self, batch: Batch, session_id: str, **kwargs: Any) -> tuple[Batch, bool, float]:
        """Mock implementation that returns a fixed response."""
        mock_response = f"Mock response for: {batch.query}"
        updated_batch = batch.model_copy(update={"assistant": mock_response})
        return updated_batch, True, 10.0

    async def run_dataset(self, dataset: Dataset, **kwargs: Any) -> tuple[Dataset, dict[str, Any]]:
        """Mock implementation that processes all batches."""
        updated_batches = []
        for batch in dataset.conversation:
            updated_batch, _, _ = await self.run_batch(batch, dataset.session_id)
            updated_batches.append(updated_batch)

        updated_dataset = dataset.model_copy(update={"conversation": updated_batches})

        summary = {
            "session_id": dataset.session_id,
            "total_batches": len(dataset.conversation),
            "successes": len(dataset.conversation),
            "failures": 0,
            "total_execution_time_ms": 100.0,
            "avg_batch_time_ms": 10.0,
        }

        return updated_dataset, summary


@pytest.mark.asyncio
async def test_base_runner_run_batch(sample_batch):
    """Test that BaseRunner can execute a single batch."""
    runner = MockRunner()
    session_id = "test_session_001"

    updated_batch, success, exec_time = await runner.run_batch(sample_batch, session_id)

    assert success is True
    assert exec_time > 0
    assert updated_batch.assistant == f"Mock response for: {sample_batch.query}"
    assert updated_batch.qa_id == sample_batch.qa_id


@pytest.mark.asyncio
async def test_base_runner_run_dataset(sample_dataset):
    """Test that BaseRunner can execute a complete dataset."""
    runner = MockRunner()

    updated_dataset, summary = await runner.run_dataset(sample_dataset)

    assert updated_dataset.session_id == sample_dataset.session_id
    assert len(updated_dataset.conversation) == len(sample_dataset.conversation)
    assert summary["session_id"] == sample_dataset.session_id
    assert summary["total_batches"] == len(sample_dataset.conversation)
    assert summary["successes"] == len(sample_dataset.conversation)
    assert summary["failures"] == 0
    assert summary["total_execution_time_ms"] > 0
    assert summary["avg_batch_time_ms"] > 0


@pytest.mark.asyncio
async def test_base_runner_batch_response_filled(sample_batch):
    """Test that runner fills in the assistant response."""
    runner = MockRunner()
    session_id = "test_session_002"

    # Create batch with empty assistant field for testing
    empty_batch = sample_batch.model_copy(update={"assistant": ""})
    assert empty_batch.assistant == ""

    updated_batch, _, _ = await runner.run_batch(empty_batch, session_id)

    # Updated batch should have filled assistant field
    assert updated_batch.assistant != ""
    assert "Mock response" in updated_batch.assistant


@pytest.mark.asyncio
async def test_base_runner_dataset_all_batches_filled(sample_dataset):
    """Test that runner fills all batches in a dataset."""
    runner = MockRunner()

    # Create dataset with empty assistant fields for testing
    empty_batches = [batch.model_copy(update={"assistant": ""}) for batch in sample_dataset.conversation]
    empty_dataset = sample_dataset.model_copy(update={"conversation": empty_batches})

    # All batches should start with empty assistant fields
    for batch in empty_dataset.conversation:
        assert batch.assistant == ""

    updated_dataset, _ = await runner.run_dataset(empty_dataset)

    # All batches should now have filled assistant fields
    for batch in updated_dataset.conversation:
        assert batch.assistant != ""
        assert "Mock response" in batch.assistant


@pytest.mark.asyncio
async def test_base_runner_preserves_batch_metadata(sample_batch):
    """Test that runner preserves original batch metadata."""
    runner = MockRunner()
    session_id = "test_session_003"

    updated_batch, _, _ = await runner.run_batch(sample_batch, session_id)

    # Check that metadata is preserved
    assert updated_batch.qa_id == sample_batch.qa_id
    assert updated_batch.query == sample_batch.query
    assert updated_batch.ground_truth_assistant == sample_batch.ground_truth_assistant
    assert updated_batch.observation == sample_batch.observation


@pytest.mark.asyncio
async def test_base_runner_preserves_dataset_metadata(sample_dataset):
    """Test that runner preserves original dataset metadata."""
    runner = MockRunner()

    updated_dataset, _ = await runner.run_dataset(sample_dataset)

    # Check that metadata is preserved
    assert updated_dataset.session_id == sample_dataset.session_id
    assert updated_dataset.assistant_id == sample_dataset.assistant_id
    assert updated_dataset.language == sample_dataset.language
    assert updated_dataset.context == sample_dataset.context
