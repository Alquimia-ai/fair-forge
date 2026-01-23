"""FairForge base class for metrics."""

from abc import ABC, abstractmethod
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
        self.retriever = retriever
        self.metrics = []
        self.verbose = verbose
        self.logger = VerboseLogger(verbose)

        self.dataset = self.retriever(**kwargs).load_dataset()
        self.logger.info(f"Loaded dataset with {len(self.dataset)} batches")

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

    def _process(self) -> list:
        """
        Process the entire dataset and compute metrics.

        This internal method iterates through the dataset and processes each batch
        using the implemented batch method.

        Returns:
            list: The computed metrics for all batches in the dataset.
        """
        self.logger.info("Starting to process dataset")

        for batch in self.dataset:
            self.logger.info(f"Session ID: {batch.session_id}, Assistant ID: {batch.assistant_id}")

            self.batch(
                session_id=batch.session_id,
                context=batch.context,
                assistant_id=batch.assistant_id,
                batch=batch.conversation,
                language=batch.language,
            )

        self.logger.info(f"Completed processing all batches. Total metrics collected: {len(self.metrics)}")
        return self.metrics
