"""Alquimia AI runner implementation."""

import time
from datetime import datetime
from typing import Any

from alquimia_client import AlquimiaClient
from loguru import logger

from fair_forge.schemas.common import Batch, Dataset
from fair_forge.schemas.runner import BaseRunner


class AlquimiaRunner(BaseRunner):
    """
    Runner implementation for Alquimia AI agents.

    This runner executes test batches by invoking Alquimia AI agents through
    the Alquimia Client API.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        agent_id: str,
        channel_id: str,
        api_version: str = "",
    ):
        """
        Initialize Alquimia runner.

        Args:
            base_url: Alquimia API base URL
            api_key: Alquimia API key
            agent_id: Agent/assistant identifier
            channel_id: Channel identifier
            api_version: API version (optional)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.agent_id = agent_id
        self.channel_id = channel_id
        self.api_version = api_version

    async def _invoke_agent(self, query: str, session_id: str, extra_data: dict[str, Any]) -> str:
        """
        Send a query to the agent and return the complete response.

        Args:
            query: User query to send to the agent
            session_id: Session identifier for conversation context
            extra_data: Additional keyword arguments to pass to the agent

        Returns:
            str: Complete response from the agent

        Raises:
            ValueError: If no stream_id is returned or response is empty
            Exception: For other API errors
        """
        async with AlquimiaClient(
            base_url=self.base_url,
            api_key=self.api_key,
            api_version=self.api_version,
        ) as client:
            result = await client.infer(
                assistant_id=self.agent_id,
                session_id=session_id,
                channel=self.channel_id,
                query=query,
                date=datetime.now().strftime("%Y-%m-%d"),
                **extra_data,
            )

            stream_id = result.get("stream_id")
            if not stream_id:
                raise ValueError("No stream_id returned from agent")

            response = ""
            async for event in client.stream(stream_id):
                response = event["response"]["data"]["content"]

            if not response:
                raise ValueError("Empty response from agent")

            return response

    async def run_batch(self, batch: Batch, session_id: str, **kwargs: Any) -> tuple[Batch, bool, float]:
        """
        Execute a single batch (test case) and return the updated batch with response.

        Args:
            batch: Batch object with query to execute
            session_id: Session identifier for conversation context
            **kwargs: Additional runner-specific arguments (unused)

        Returns:
            tuple: (updated_batch, success, execution_time_ms)
        """
        start_time = time.time()

        try:
            # Extract agentic data to pass as extra kwargs
            extra_data = batch.agentic if batch.agentic else {}

            response = await self._invoke_agent(batch.query, session_id, extra_data)
            execution_time = (time.time() - start_time) * 1000

            # Create updated batch with response
            updated_batch = batch.model_copy(update={"assistant": response})

            logger.debug(f"  ✓ Batch {batch.qa_id} completed ({execution_time:.1f}ms)")
            return updated_batch, True, execution_time

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = str(e)
            logger.error(f"  ✗ Batch {batch.qa_id} failed: {error_msg}")

            # Store error in observation field
            error_observation = f"{batch.observation or ''}\nError: {error_msg}".strip()
            updated_batch = batch.model_copy(
                update={"assistant": f"[ERROR] {error_msg}", "observation": error_observation}
            )

            return updated_batch, False, execution_time

    async def run_dataset(self, dataset: Dataset, **kwargs: Any) -> tuple[Dataset, dict[str, Any]]:
        """
        Execute all batches in a dataset and return updated dataset with responses.

        Args:
            dataset: Dataset object containing conversation batches to execute
            **kwargs: Additional runner-specific arguments (unused)

        Returns:
            tuple: (updated_dataset, execution_summary)
        """
        start_time = time.time()
        num_batches = len(dataset.conversation)

        logger.info(f"Running dataset: {dataset.session_id} ({num_batches} batch(es))")

        updated_batches = []
        successes = 0
        failures = 0
        total_batch_time = 0.0

        # Execute each batch in the conversation
        for i, batch in enumerate(dataset.conversation, 1):
            logger.debug(f"  Batch {i}/{num_batches}: {batch.qa_id}")

            updated_batch, success, batch_time = await self.run_batch(batch, dataset.session_id)
            updated_batches.append(updated_batch)
            total_batch_time += batch_time

            if success:
                successes += 1
            else:
                failures += 1

        total_time = (time.time() - start_time) * 1000

        # Create updated dataset with filled batches
        updated_dataset = dataset.model_copy(update={"conversation": updated_batches})

        # Create execution summary
        summary: dict[str, Any] = {
            "session_id": dataset.session_id,
            "total_batches": num_batches,
            "successes": successes,
            "failures": failures,
            "total_execution_time_ms": total_time,
            "avg_batch_time_ms": total_batch_time / num_batches if num_batches > 0 else 0,
        }

        status = "✓" if failures == 0 else "⚠"
        logger.info(
            f"{status} Dataset {dataset.session_id} completed: "
            f"{successes}/{num_batches} passed ({total_time:.1f}ms)"
        )

        return updated_dataset, summary


__all__ = ["AlquimiaRunner"]
