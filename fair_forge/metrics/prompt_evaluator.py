"""PromptEvaluator metric: score a system prompt against a dataset."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from fair_forge.core import FairForge, Retriever
from fair_forge.prompt_optimizer.protocols import Evaluator, Executor
from fair_forge.schemas import Batch
from fair_forge.schemas.prompt_evaluator import PromptEvaluatorMetric, PromptInteractionScore


class PromptEvaluator(FairForge):
    """Score a system prompt by running it against a dataset and evaluating the responses.

    The metric executes the seed_prompt on each query in the dataset, evaluates the
    generated response against the ground truth, and aggregates the results into a
    prompt_score per session.

    Args:
        retriever: Retriever class supplying the dataset (needs query, context, ground_truth_assistant)
        model: LangChain BaseChatModel used as executor and default evaluator judge
        seed_prompt: The system prompt being evaluated
        objective: Business goal the prompt is trying to achieve — used by the default evaluator
        threshold: Minimum score for an interaction to be considered passing (default 0.6)
        executor: Custom function (prompt, query, context) -> response. Defaults to simple LLM invocation.
        evaluator: Custom function (actual, expected, query, context) -> float. Defaults to LLMEvaluator.
    """

    def __init__(
        self,
        retriever: type[Retriever],
        model: BaseChatModel,
        seed_prompt: str,
        objective: str,
        threshold: float = 0.6,
        executor: Executor | None = None,
        evaluator: Evaluator | None = None,
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)
        from fair_forge.prompt_optimizer.evaluators import LLMEvaluator

        self._seed_prompt = seed_prompt
        self._objective = objective
        self._threshold = threshold
        self._executor = executor or self._default_executor(model)
        self._evaluator = evaluator or LLMEvaluator(model=model, criteria=objective)
        self._session_data: dict[str, dict] = {}

    @staticmethod
    def _default_executor(model: BaseChatModel) -> Executor:
        def _execute(prompt: str, query: str, context: str) -> str:
            system = f"{prompt}\n\n{context}" if context else prompt
            response = model.invoke([SystemMessage(content=system), HumanMessage(content=query)])
            return str(response.content)

        return _execute

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: str | None = "english",
    ):
        if session_id not in self._session_data:
            self._session_data[session_id] = {"assistant_id": assistant_id, "interactions": []}

        for item in batch:
            self.logger.debug(f"QA ID: {item.qa_id}")
            actual = self._executor(self._seed_prompt, item.query, context)
            score = self._evaluator(actual, item.ground_truth_assistant, item.query, context)
            self.logger.debug(f"Score: {score:.4f}")
            self._session_data[session_id]["interactions"].append(
                PromptInteractionScore(qa_id=item.qa_id, score=round(score, 4), passed=score >= self._threshold)
            )

    def on_process_complete(self):
        for session_id, data in self._session_data.items():
            interactions = data["interactions"]
            scores = [i.score for i in interactions]
            prompt_score = round(sum(scores) / len(scores), 4)
            pass_rate = round(sum(1 for i in interactions if i.passed) / len(interactions), 4)

            self.metrics.append(
                PromptEvaluatorMetric(
                    session_id=session_id,
                    assistant_id=data["assistant_id"],
                    seed_prompt=self._seed_prompt,
                    objective=self._objective,
                    prompt_score=prompt_score,
                    pass_rate=pass_rate,
                    threshold=self._threshold,
                    n_interactions=len(interactions),
                    interactions=interactions,
                )
            )
