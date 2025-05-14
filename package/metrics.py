from main import FairForge, Retriever
from typing import Optional
from schemas import Batch, BiasMetric
from helpers.guardian import Guardian, Provider, GuardianConfig, VllmProvider
from typing import Type


class Bias(FairForge):
    def __init__(
        self,
        retriever: Type[Retriever],
        guardian_url: Optional[str],
        guardian_api_key: Optional[str],
        guardian_model: Optional[str],
        guardian_temperature: float = 0,
        guardian_max_tokens: float = 5,
        guardian_risks: Optional[list[dict]] = None,
        guardian_provider: Type[Provider] = VllmProvider,
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)
        self.guardian = Guardian(
            GuardianConfig(
                url=guardian_url,
                api_key=guardian_api_key,
                model=guardian_model,
                temperature=guardian_temperature,
                max_tokens=guardian_max_tokens,
                risks=guardian_risks,
                provider=guardian_provider,
            )
        )

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
        for interaction in batch:
            self.metrics.append(
                BiasMetric(
                    session_id=session_id,
                    qa_id=interaction.qa_id,
                    assistant_id=assistant_id,
                    risks=self.guardian.has_any_risk(
                        question=interaction.query,
                        answer=interaction.ground_truth_assistant,
                        context=context,
                    ),
                )
            )


class Context(FairForge):
    def __init__(self, retriever: Type[Retriever], **kwargs):
        super().__init__(retriever, **kwargs)

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: Optional[str],
    ):
        pass


class Agentic(FairForge):
    def __init__(self, retriever: Type[Retriever], **kwargs):
        super().__init__(retriever, **kwargs)

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: Optional[str],
    ):
        pass


class Conversational(FairForge):
    def __init__(self, retriever: Type[Retriever], **kwargs):
        super().__init__(retriever, **kwargs)

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: Optional[str],
    ):
        pass


class Humanity(FairForge):
    def __init__(self, retriever: Type[Retriever], **kwargs):
        super().__init__(retriever, **kwargs)

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: Optional[str],
    ):
        pass
