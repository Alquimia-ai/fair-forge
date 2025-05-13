from .schemas import Batch,Dataset
from abc import ABC, abstractmethod

class Retriever():
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        
    @abstractmethod
    def load_dataset(self) -> Dataset:
        """
        Load from the cold storage the dataset and return it as a Dataset object.
        """
        raise Exception("You should implement this method according to the type of storage you are using.")


class FairForge(ABC):
    def __init__(self,retriever: Retriever):
        self.retriever = retriever
        self.metrics = []
    
    @abstractmethod
    def batch(self, dataset: list[Batch]):
        """
        Process each individual batch. 
        This method is the interface for each metric.
        """
        raise Exception("Should be implemented by each metric")

    def process(self):
        """
        Process the entire dataset and saves the metric data under the metrics attribute.
        """
        self.dataset = self.retriever().load_dataset()

        for batch in self.dataset:
            self.batch(batch)

        return self.metrics
    

