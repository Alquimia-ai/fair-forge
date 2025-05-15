from .schemas import Batch, Dataset
from abc import ABC, abstractmethod
from typing import Type
from typing import Optional


class Retriever:
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
    def load_dataset(self) -> list[Dataset]:
        """
        Load dataset from cold storage.
        
        This method should be implemented by subclasses to handle specific storage systems
        and return the data in the required Dataset format.
        
        Returns:
            list[Dataset]: A list of Dataset objects containing the retrieved data.
            
        Raises:
            Exception: If the method is not implemented by a subclass.
        """
        raise Exception(
            "You should implement this method according to the type of storage you are using."
        )


class FairForge(ABC):
    """
    Abstract base class for implementing fairness metrics and analysis.
    
    This class provides the framework for processing datasets and computing fairness metrics.
    Subclasses should implement the batch method to define specific metric calculations.
    
    Attributes:
        retriever (Type[Retriever]): The retriever class to use for loading data.
        metrics (list): List to store computed metrics.
        dataset (list[Dataset]): The loaded dataset for processing.
    """

    def __init__(self, retriever: Type[Retriever], **kwargs):
        """
        Initialize FairForge with a data retriever.
        
        Args:
            retriever (Type[Retriever]): The retriever class to use for loading data.
            **kwargs: Additional configuration parameters.
        """
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
            Exception: If the method is not implemented by a subclass.
        """
        raise Exception("Should be implemented by each metric")
    
    @classmethod
    def run(cls, retriever: Type[Retriever], **kwargs) -> list:
        """
        Run the  metric analysis on the entire dataset.
        
        This class method provides a convenient way to instantiate and run the metric
        analysis in one step.
        
        Args:
            retriever (Type[Retriever]): The retriever class to use for loading data.
            **kwargs: Additional configuration parameters.
            
        Returns:
            list: The computed metrics for the entire dataset.
        """
        return cls(retriever, **kwargs)._process()

    def _process(self)->list:
        """
        Process the entire dataset and compute metrics.
        
        This internal method iterates through the dataset and processes each batch
        using the implemented batch method.
        
        Returns:
            list: The computed metrics for all batches in the dataset.
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
