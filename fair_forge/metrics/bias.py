from fair_forge import FairForge, Retriever
from typing import Optional, Type
from fair_forge.helpers.guardian import VllmProvider, Provider, Guardian, GuardianConfig
from fair_forge.schemas import Batch, BiasMetric
from pydantic import SecretStr

class Bias(FairForge):
    def __init__(
        self,
        retriever: Type[Retriever],
        guardian_url: Optional[str],
        guardian_api_key: Optional[SecretStr]=SecretStr(""),
        guardian_model: Optional[str]="ibm-granite/granite-guardian-3.1-2b",
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
        language: Optional[str] = "english",
    ):
        """
        Process each individual batch.
        This method is the interface for each metric.
        """
        for interaction in batch:
            self.logger.debug(f"QA ID: {interaction.qa_id}")
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
