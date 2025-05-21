from fair_forge import FairForge, Retriever, Guardian
from typing import Optional, Type
from fair_forge.schemas import Batch,  GuardianBias, ProtectedAttribute

class Bias(FairForge):
    def __init__(
        self,
        retriever: Type[Retriever],
        guardian: Type[Guardian],
        confidence_level: float = 0.95,
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)
        self.protected_attributes = [ProtectedAttribute.gender,
                                ProtectedAttribute.race,
                                ProtectedAttribute.religion,
                                ProtectedAttribute.nationality,
                                ProtectedAttribute.sexual_orientation
        ] ## PROTECTED ATTRIBUTES DEFINED BY FAIR FORGE
        self.guardian = guardian(protected_attributes=self.protected_attributes)
        self.confidence_level = confidence_level

    def _is_biased_by_attribute(self,question:str,answer:str,attribute:str) -> GuardianBias:
        return self.guardian.is_biased(
            question=question,
            answer=answer,
            attribute=attribute,
        )
    
    def _confidence_interval_by_protected_attribute(self,bias:list[GuardianBias]):
        pass

    def _kl_divergence(self,p:list[float],q:list[float])-> float:
        """
        Calculate the Kullback-Leibler (KL) divergence between two probability distributions.

        Args:
            p (list[float]): The first probability distribution.
            q (list[float]): The second probability distribution.
        """
        pass

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
            for attribute in self.attributes:
                bias = self._is_biased_by_attribute(interaction.query,interaction.ground_truth_assistant,attribute)
                self.logger.debug(f"Bias: {bias}")
