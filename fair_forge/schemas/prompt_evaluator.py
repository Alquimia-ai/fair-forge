"""Pydantic schemas for the PromptEvaluator metric."""

from pydantic import BaseModel

from fair_forge.schemas.metrics import BaseMetric


class PromptInteractionScore(BaseModel):
    qa_id: str
    score: float
    passed: bool


class PromptEvaluatorMetric(BaseMetric):
    """Result of evaluating a single prompt against a dataset.

    prompt_score is the average judge score across all interactions (0.0–1.0).
    pass_rate is the fraction of interactions that scored at or above the threshold.
    """

    seed_prompt: str
    objective: str
    prompt_score: float
    pass_rate: float
    threshold: float
    n_interactions: int
    interactions: list[PromptInteractionScore]

    def display(self) -> None:
        """Print a human-readable summary of the prompt evaluation."""
        print(f"Session: {self.session_id}  |  Assistant: {self.assistant_id}")
        print(f"Prompt score:  {self.prompt_score:.2f}")
        print(f"Pass rate:     {self.pass_rate:.0%}  ({sum(1 for i in self.interactions if i.passed)}/{self.n_interactions} interactions passed @ threshold={self.threshold})")
        print()
        for i in self.interactions:
            label = "pass" if i.passed else "FAIL"
            print(f"  {i.qa_id}  score={i.score:.2f}  {label}")
        print()
