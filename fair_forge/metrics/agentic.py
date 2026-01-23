from fair_forge.core import FairForge, Retriever
from fair_forge.schemas import Batch


class Agentic(FairForge):
    def __init__(self, retriever: type[Retriever], **kwargs):
        super().__init__(retriever, **kwargs)

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: str | None = "english",
    ):
        for _interaction in batch:
            pass
