from fair_forge import FairForge, Retriever
from typing import Type, Optional
from fair_forge.schemas import Batch

class Agentic(FairForge):
    def __init__(self, retriever: Type[Retriever], **kwargs):
        super().__init__(retriever, **kwargs)

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: Optional[str] = "english",
    ):  
        for interaction in batch:
            pass
