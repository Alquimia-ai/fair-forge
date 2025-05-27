from .schemas import Batch, Dataset, GuardianBias,ProtectedAttribute, ToxicityDataset
from abc import ABC, abstractmethod
from typing import Type
from typing import Optional
import logging

class VerboseLogger:
    """
    Custom logger class that handles verbose logging.
    Only logs messages when verbose mode is enabled.
    """
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        self.logger.setLevel(logging.DEBUG)

    def info(self, message: str):
        """Log info message if verbose is enabled"""
        if self.verbose:
            self.logger.info(message)

    def debug(self, message: str):
        """Log debug message if verbose is enabled"""
        if self.verbose:
            self.logger.debug(message)

    def warning(self, message: str):
        """Log warning message if verbose is enabled"""
        if self.verbose:
            self.logger.warning(message)

    def error(self, message: str):
        """Log error message if verbose is enabled"""
        if self.verbose:
            self.logger.error(message)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        verbose (bool): Whether to enable verbose logging.
        logger (VerboseLogger): Logger instance for verbose logging.
    """

    def __init__(self, retriever: Type[Retriever], verbose: bool = False, **kwargs):
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
        
        self.dataset = self.retriever().load_dataset()
        self.logger.info(f"Loaded dataset with {len(self.dataset)} batches")

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
    def run(cls, retriever: Type[Retriever],  **kwargs) -> list:
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
        return cls(retriever,**kwargs)._process()

    def _process(self)->list:
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
    

class Guardian(ABC):
    """
    An abstract base class that serves as a framework for implementing bias detection in LLM interactions.
    
    This class is designed to be implemented by different bias detection mechanisms that can analyze
    question-answer pairs for potential biases towards specific attributes. It provides a standardized
    interface for bias detection across different implementations.

    The class uses a nested BiasInfer model to structure the bias detection results in a consistent format.
    """
    def __init__(self,**kwargs):
        """
        Initialize the Guardian with a VerboseLogger for detailed logging of bias detection operations.
        """
        self.logger = VerboseLogger()
        
    @abstractmethod
    def is_biased(self, question: str, answer: str, attribute: ProtectedAttribute , context: Optional[str] = None) -> GuardianBias:
        """
        Analyze a question-answer interaction for potential bias towards specific attributes.

        This abstract method must be implemented by concrete Guardian classes to define their
        specific bias detection logic. The implementation should analyze the given question and
        answer pair for potential bias towards the specified attributes.

        Args:
            question (str): The question being analyzed
            answer (str): The answer being analyzed
            attribute (dict): A dictionary specifying the attributes to check for bias
            context (Optional[str]): Additional context that might be relevant for bias detection

        Returns:
            GuardianBias: A GuardianBias object containing:
                - is_biased: Whether bias was detected
                - attribute: The specific attribute(s) that showed bias
                - certainty: Optional confidence score for the detection

        Raises:
            NotImplementedError: If the concrete class does not implement this method
        """
        raise NotImplementedError("You should implement this method.")
    


class ToxicityLoader(ABC):
    """Abstract base class for loading toxicity datasets.

    This class serves as a template for implementing custom toxicity dataset loaders.
    It provides a standardized interface for loading toxicity-related datasets that
    can be used for training or evaluation of toxicity detection models.

    The class is designed to be extended by concrete implementations that handle
    specific data sources or formats. Each implementation must provide its own
    logic for loading and processing the toxicity data.

    Attributes:
        kwargs (dict): Configuration parameters passed during initialization.
            These parameters can be used by concrete implementations to customize
            the loading behavior.

    Example:
        To create a custom loader:
        ```python
        class CustomToxicityLoader(ToxicityLoader):
            def load(self,language:str) -> list[ToxicityDataset]:
                # Implementation specific to your data source
                pass
        ```
    """
    def __init__(self, **kwargs):
        """
        Initialize the ToxicityLoader with optional configuration parameters.
        
        Args:
            **kwargs: Arbitrary keyword arguments for configuration.
        """
        self.kwargs = kwargs

    @abstractmethod
    def load(self,language:str) -> list[ToxicityDataset]:
        """Load and return a list of toxicity datasets.

        This method must be implemented by concrete subclasses to provide
        the actual dataset loading logic.

        Returns:
            list[ToxicityDataset]: A list of loaded toxicity datasets.

        Raises:
            NotImplementedError: If the concrete class does not implement this method.
        """
        raise NotImplementedError
