from fair_forge import FairForge, Retriever
from typing import Type, Optional
from fair_forge.schemas import Batch, ContextMetric

class LevenshteinDistance(FairForge):
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
        # Implement your metric logic here
        for interaction in batch:
            # Process each interaction
            distance = self._compute_distance(interaction.ground_truth_assistant, interaction.assistant)

            metric = LevenshteinDistanceMetric(
                    levenshtein_distance=distance,
                    session_id=session_id,
                    assistant_id=assistant_id,
                    qa_id=interaction.qa_id,
                )

            self.metrics.append(metric)

    def _compute_distance(self, s1: str, s2: str) -> int:
        """
        Computes the Levenshtein distance between two sentences.
        This measures how many single-character edits (insertions, deletions, substitutions)
        are needed to transform one sentence into the other.
        """
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]