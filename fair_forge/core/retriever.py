"""Retriever abstract base class for loading datasets."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fair_forge.schemas.common import Dataset


class Retriever(ABC):
    """
    Abstract base class for data retrieval from cold storage.

    This class serves as a template for implementing specific data retrieval strategies.
    Subclasses should implement the load_dataset method to fetch data from their respective storage systems.

    Attributes:
        kwargs (dict): Additional configuration parameters passed during initialization.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Retriever with optional configuration parameters.

        Args:
            **kwargs: Arbitrary keyword arguments for configuration.
        """
        self.kwargs = kwargs

    @abstractmethod
    def load_dataset(self) -> list["Dataset"]:
        """
        Load dataset from cold storage.

        This method should be implemented by subclasses to handle specific storage systems
        and return the data in the required Dataset format.

        Returns:
            list[Dataset]: A list of Dataset objects containing the retrieved data.

        Raises:
            Exception: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("You should implement this method according to the type of storage you are using.")
