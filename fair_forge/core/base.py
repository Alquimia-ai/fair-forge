"""FairForge base class for metrics."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

from fair_forge.utils.logging import VerboseLogger

if TYPE_CHECKING:
    from fair_forge.core.retriever import Retriever
    from fair_forge.schemas.common import Batch


class FairForge(ABC):
    """
    Abstract base class for implementing fairness metrics and analysis.

    This class provides the framework for processing datasets and computing fairness metrics.
    Subclasses should implement the batch method to define specific metric calculations.

    Attributes:
        retriever (Type[Retriever]): The retriever class to use for loading data.
        metrics (list): List to store computed metrics.
        dataset (list[Dataset]): The loaded dataset for processing.
        verbose (bool): Whether to enable verbose logging.
        logger (VerboseLogger): Logger instance for verbose logging.
    """

    def __init__(self, retriever: type["Retriever"], verbose: bool = False, **kwargs):
        """
        Initialize FairForge with a data retriever.

        Args:
            retriever (Type[Retriever]): The retriever class to use for loading data.
            verbose (bool): Whether to enable verbose logging.
            **kwargs: Additional configuration parameters.
        """
        self.retriever = retriever(**kwargs)
        self.metrics = []
        self.verbose = verbose
        self.logger = VerboseLogger(verbose)

        self.dataset = self.retriever.load_dataset()
        self.level = self.retriever.iteration_level

        if isinstance(self.dataset, Iterator) and self.level.value == "full_dataset":
            raise ValueError(
                "When using a generator, you must explicitly set 'iteration_level' ('stream_sessions' or 'stream_batches') in the Retriever."
            )

        strategies = {
            "full_dataset": self._process_dataset,
            "stream_sessions": self._process_dataset,
            "stream_batches": self._process_qa,
        }

        self._iteration_processor = strategies.get(self.level.value)
        if not self._iteration_processor:
            raise ValueError(f"Unknown iteration_level: {self.level}")

    @abstractmethod
    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list["Batch"],
        language: str | None,
    ):
        """
        Process a single batch of conversation data.

        This method should be implemented by subclasses to define how each batch
        of conversation data is processed and what metrics are computed.

        Args:
            session_id (str): Unique identifier for the conversation session.
            context (str): Contextual information for the conversation.
            assistant_id (str): Identifier for the AI assistant.
            batch (list[Batch]): List of conversation batches to process.
            language (Optional[str]): Language of the conversation, if specified.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Should be implemented by each metric")

    @classmethod
    def run(cls, retriever: type["Retriever"], **kwargs) -> list:
        """
        Run the metric analysis on the entire dataset.

        This class method provides a convenient way to instantiate and run the metric
        analysis in one step.

        Args:
            retriever (Type[Retriever]): The retriever class to use for loading data.
            **kwargs: Additional configuration parameters.

        Returns:
            list: The computed metrics for the entire dataset.
        """
        return cls(retriever, **kwargs)._process()

    def _process_dataset(self, data):
        self.logger.info("Processing using dataset/session level parsing")
        for element in data:
            self.logger.info(f"Session ID: {element.session_id}, Assistant ID: {element.assistant_id}")
            self.batch(
                session_id=element.session_id,
                context=element.context,
                assistant_id=element.assistant_id,
                batch=element.conversation,
                language=element.language,
            )

    def _process_qa(self, data):
        self.logger.info("Processing using QA batch level parsing")
        for streamed in data:
            self.logger.info(
                f"Session ID: {streamed.metadata.session_id}, Assistant ID: {streamed.metadata.assistant_id}"
            )
            self.batch(
                session_id=streamed.metadata.session_id,
                context=streamed.metadata.context,
                assistant_id=streamed.metadata.assistant_id,
                batch=[streamed.batch],
                language=streamed.metadata.language,
            )

    def on_process_complete(self):  # noqa: B027
        """Optional hook evaluated after all dataset elements are processed. Useful for accumulator metrics."""

    def _process(self) -> list:
        """
        Process the entire dataset and compute metrics using the configured strategy.
        """
        self.logger.info("Starting to process dataset")

        self._iteration_processor(self.dataset)
        self.on_process_complete()

        self.logger.info(f"Completed processing. Total metrics collected: {len(self.metrics)}")
        return self.metrics
