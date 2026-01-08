# test_runner/main.py

import asyncio
import uuid
import time
from loguru import logger
from datetime import datetime

from test_loader import load_datasets
from test_runner import run_dataset
from storage import get_storage
from config import ENABLED_TEST_SUITES, TEST_STORAGE_BACKEND


async def main():
    """
    Main test execution function.
    """
    run_id = str(uuid.uuid4())
    start_time = time.time()
    timestamp = datetime.now()

    logger.info("=" * 70)
    logger.info("Fair Forge Agent Test Runner")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Timestamp: {timestamp.isoformat()}")
    logger.info(f"Storage Backend: {TEST_STORAGE_BACKEND}")

    if ENABLED_TEST_SUITES:
        logger.info(f"Enabled test suites: {', '.join(ENABLED_TEST_SUITES)}")
    else:
        logger.info("Running all available test datasets")

    logger.info("=" * 70)

    # Load test datasets
    datasets = load_datasets()

    if not datasets:
        logger.error("No test datasets loaded. Exiting.")
        return 1

    # Count total batches
    total_batches = sum(len(ds.conversation) for ds in datasets)
    logger.info(f"\nLoaded {len(datasets)} dataset(s) with {total_batches} total test case(s)\n")

    # Run datasets
    executed_datasets = []
    dataset_summaries = []
    total_successes = 0
    total_failures = 0

    for i, dataset in enumerate(datasets, 1):
        logger.info(f"[{i}/{len(datasets)}] Dataset: {dataset.session_id}")

        try:
            executed_dataset, summary = await run_dataset(dataset)
            executed_datasets.append(executed_dataset)
            dataset_summaries.append(summary)

            total_successes += summary["successes"]
            total_failures += summary["failures"]

        except Exception as e:
            logger.error(f"Failed to execute dataset {dataset.session_id}: {e}")
            # Still append the dataset even if execution failed
            executed_datasets.append(dataset)
            total_failures += len(dataset.conversation)

        logger.info("")

    total_execution_time = (time.time() - start_time) * 1000

    # Print summary
    logger.info("=" * 70)
    logger.info("TEST RUN SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total datasets:          {len(datasets)}")
    logger.info(f"Total test cases:        {total_batches}")
    logger.info(f"Successful:              {total_successes} ✓")
    logger.info(f"Failed:                  {total_failures} ✗")

    success_rate = (total_successes / total_batches * 100) if total_batches > 0 else 0
    logger.info(f"Success rate:            {success_rate:.1f}%")
    logger.info(f"Total execution time:    {total_execution_time/1000:.2f}s")
    logger.info("=" * 70)

    # Save results
    try:
        storage = get_storage()
        result_path = storage.save_results(executed_datasets, run_id, timestamp)
        logger.success(f"\n✅ Results saved: {result_path}")
    except Exception as e:
        logger.error(f"\n❌ Failed to save results: {e}")
        return 1

    # Final status
    if total_failures == 0:
        logger.success("\n✅ All tests executed successfully!")
        return 0
    else:
        logger.warning(f"\n⚠️  {total_failures} test case(s) failed to execute")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
