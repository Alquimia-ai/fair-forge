"""Unit tests for PromptEvaluator metric."""

from unittest.mock import MagicMock, patch

import pytest

from fair_forge.metrics.prompt_evaluator import PromptEvaluator
from fair_forge.schemas.prompt_evaluator import PromptEvaluatorMetric, PromptInteractionScore
from tests.fixtures.mock_data import create_sample_batch, create_sample_dataset
from tests.fixtures.mock_retriever import MockRetriever, PromptEvaluatorDatasetRetriever

_SEED_PROMPT = "You are a helpful assistant. Answer using only the provided context."
_OBJECTIVE = "Answer the user's question using only the information provided in the context."


def _mock_evaluator(scores: list[float]):
    mock = MagicMock()
    mock.side_effect = scores
    return mock


def _mock_executor(responses: list[str]):
    mock = MagicMock()
    mock.side_effect = responses
    return mock


def _make_retriever(batches):
    dataset = create_sample_dataset(
        session_id="test_session",
        assistant_id="test_bot",
        context="Test context.",
        conversation=batches,
    )
    return type("R", (MockRetriever,), {"load_dataset": lambda self: [dataset]})


def _run(batches, scores, threshold=0.6):
    retriever = _make_retriever(batches)
    metric = PromptEvaluator(retriever, model=MagicMock(), seed_prompt=_SEED_PROMPT, objective=_OBJECTIVE, threshold=threshold)
    metric._executor = _mock_executor(["response"] * len(scores))
    metric._evaluator = _mock_evaluator(scores)
    metric.batch(session_id="test_session", context="ctx", assistant_id="bot", batch=batches)
    metric.on_process_complete()
    return metric.metrics[0]


class TestPromptEvaluatorInit:
    def test_default_threshold(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(prompt_evaluator_dataset_retriever, model=MagicMock(), seed_prompt=_SEED_PROMPT, objective=_OBJECTIVE)
        assert metric._threshold == 0.6

    def test_custom_threshold(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(prompt_evaluator_dataset_retriever, model=MagicMock(), seed_prompt=_SEED_PROMPT, objective=_OBJECTIVE, threshold=0.8)
        assert metric._threshold == 0.8

    def test_stores_seed_prompt_and_objective(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(prompt_evaluator_dataset_retriever, model=MagicMock(), seed_prompt=_SEED_PROMPT, objective=_OBJECTIVE)
        assert metric._seed_prompt == _SEED_PROMPT
        assert metric._objective == _OBJECTIVE

    def test_accepts_custom_executor(self, prompt_evaluator_dataset_retriever):
        custom_executor = MagicMock()
        metric = PromptEvaluator(prompt_evaluator_dataset_retriever, model=MagicMock(), seed_prompt=_SEED_PROMPT, objective=_OBJECTIVE, executor=custom_executor)
        assert metric._executor is custom_executor

    def test_accepts_custom_evaluator(self, prompt_evaluator_dataset_retriever):
        custom_evaluator = MagicMock()
        metric = PromptEvaluator(prompt_evaluator_dataset_retriever, model=MagicMock(), seed_prompt=_SEED_PROMPT, objective=_OBJECTIVE, evaluator=custom_evaluator)
        assert metric._evaluator is custom_evaluator


class TestPromptEvaluatorScoring:
    def test_perfect_score(self):
        batches = [create_sample_batch(qa_id=f"qa_{i:03d}") for i in range(3)]
        result = _run(batches, [1.0, 1.0, 1.0])
        assert result.prompt_score == pytest.approx(1.0)
        assert result.pass_rate == pytest.approx(1.0)

    def test_zero_score(self):
        batches = [create_sample_batch(qa_id=f"qa_{i:03d}") for i in range(3)]
        result = _run(batches, [0.0, 0.0, 0.0])
        assert result.prompt_score == pytest.approx(0.0)
        assert result.pass_rate == pytest.approx(0.0)

    def test_mixed_scores(self):
        batches = [create_sample_batch(qa_id=f"qa_{i:03d}") for i in range(3)]
        result = _run(batches, [1.0, 0.5, 0.0])
        assert result.prompt_score == pytest.approx(0.5, abs=0.01)

    def test_pass_rate(self):
        batches = [create_sample_batch(qa_id=f"qa_{i:03d}") for i in range(3)]
        result = _run(batches, [0.9, 0.7, 0.3], threshold=0.6)
        assert result.pass_rate == pytest.approx(2 / 3, abs=0.01)

    def test_threshold_boundary(self):
        batches = [create_sample_batch(qa_id=f"qa_{i:03d}") for i in range(2)]
        result = _run(batches, [0.6, 0.59], threshold=0.6)
        assert result.interactions[0].passed is True
        assert result.interactions[1].passed is False

    def test_executor_called_with_seed_prompt(self):
        batches = [create_sample_batch(qa_id="qa_001")]
        retriever = _make_retriever(batches)
        metric = PromptEvaluator(retriever, model=MagicMock(), seed_prompt=_SEED_PROMPT, objective=_OBJECTIVE)
        executor = _mock_executor(["response"])
        evaluator = _mock_evaluator([0.8])
        metric._executor = executor
        metric._evaluator = evaluator
        metric.batch(session_id="test_session", context="ctx", assistant_id="bot", batch=batches)
        executor.assert_called_once_with(_SEED_PROMPT, batches[0].query, "ctx")

    def test_evaluator_called_with_generated_response(self):
        batches = [create_sample_batch(qa_id="qa_001")]
        retriever = _make_retriever(batches)
        metric = PromptEvaluator(retriever, model=MagicMock(), seed_prompt=_SEED_PROMPT, objective=_OBJECTIVE)
        metric._executor = _mock_executor(["generated response"])
        evaluator = _mock_evaluator([0.8])
        metric._evaluator = evaluator
        metric.batch(session_id="test_session", context="ctx", assistant_id="bot", batch=batches)
        evaluator.assert_called_once_with("generated response", batches[0].ground_truth_assistant, batches[0].query, "ctx")

    def test_result_schema(self):
        batches = [create_sample_batch(qa_id="qa_000")]
        result = _run(batches, [0.8])
        assert isinstance(result, PromptEvaluatorMetric)
        assert result.seed_prompt == _SEED_PROMPT
        assert result.objective == _OBJECTIVE
        assert result.session_id == "test_session"


class TestPromptEvaluatorRun:
    def test_run_method(self, prompt_evaluator_dataset_retriever):
        with patch.object(PromptEvaluator, "batch"):
            with patch.object(PromptEvaluator, "on_process_complete"):
                metric = PromptEvaluator(
                    prompt_evaluator_dataset_retriever,
                    model=MagicMock(),
                    seed_prompt=_SEED_PROMPT,
                    objective=_OBJECTIVE,
                )
                metric.metrics = [PromptEvaluatorMetric(
                    session_id="s", assistant_id="a", seed_prompt=_SEED_PROMPT,
                    objective=_OBJECTIVE, prompt_score=0.8, pass_rate=1.0,
                    threshold=0.6, n_interactions=1,
                    interactions=[PromptInteractionScore(qa_id="q", score=0.8, passed=True)],
                )]
        assert isinstance(metric.metrics[0], PromptEvaluatorMetric)

    def test_multiple_sessions(self):
        dataset_a = create_sample_dataset(session_id="session_a", conversation=[create_sample_batch()])
        dataset_b = create_sample_dataset(session_id="session_b", conversation=[create_sample_batch()])
        retriever = type("R", (MockRetriever,), {"load_dataset": lambda self: [dataset_a, dataset_b]})
        metric = PromptEvaluator(retriever, model=MagicMock(), seed_prompt=_SEED_PROMPT, objective=_OBJECTIVE)
        metric._executor = _mock_executor(["r1", "r2"])
        metric._evaluator = _mock_evaluator([0.8, 0.7])
        metric.batch(session_id="session_a", context="ctx", assistant_id="bot", batch=[create_sample_batch()])
        metric.batch(session_id="session_b", context="ctx", assistant_id="bot", batch=[create_sample_batch()])
        metric.on_process_complete()
        assert len(metric.metrics) == 2
