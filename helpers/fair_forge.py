from .dataset import load_dataset, Conversation
from abc import ABC, abstractmethod

class FairForge(ABC):
    def __init__(self):
        self.metrics = []
    
    @abstractmethod
    def process(self, thread: Conversation):
        raise Exception("Should be implemented by each metric")

    def pipeline(self):
        self.dataset = load_dataset()

        for thread in self.dataset:
            self.process(thread)
        return self.metrics