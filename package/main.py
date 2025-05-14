from .schemas import Batch, Dataset
from abc import ABC, abstractmethod
from typing import Type
from typing import Optional


class Retriever:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def load_dataset(self) -> list[Dataset]:
        """
        Load from the cold storage the dataset and return it as a Dataset object.
        """
        raise Exception(
            "You should implement this method according to the type of storage you are using."
        )


class FairForge(ABC):
    def __init__(self, retriever: Type[Retriever], **kwargs):
        self.retriever = retriever
        self.metrics = []
        self.dataset = self.retriever().load_dataset()

    @abstractmethod
    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: Optional[str],
    ):
        """
        Process each individual batch.
        This method is the interface for each metric.
        """
        raise Exception("Should be implemented by each metric")

    def __call__(self):
        """
        Process the entire dataset and saves the metric data under the metrics attribute.
        """

        for batch in self.dataset:
            self.batch(
                session_id=batch.session_id,
                context=batch.context,
                assistant_id=batch.assistant_id,
                batch=batch.conversation,
                language=batch.language,
            )

        return self.metrics
