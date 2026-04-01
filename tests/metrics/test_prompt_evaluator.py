"""Unit tests for PromptEvaluator metric."""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fair_forge.metrics.constraints import JsonConstraint, KeywordConstraint, RegexConstraint, WordCountConstraint
from fair_forge.metrics.prompt_evaluator import PromptEvaluator
from fair_forge.schemas.prompt_evaluator import PromptEvaluatorMetric, QuerySampleMetrics
from fair_forge.statistical import BayesianMode, FrequentistMode
from tests.fixtures.mock_data import create_sample_batch, create_sample_dataset
from tests.fixtures.mock_retriever import MockRetriever

_SEED_PROMPT = "You are a helpful assistant. Answer using only the provided context."


def _mock_embedder(embeddings_map: dict[str, np.ndarray] | None = None):
    """Return an embedder that maps text → embedding vector."""
    mock = MagicMock()

    def encode(sentences: list[str]) -> np.ndarray:
        if embeddings_map:
            return np.array([embeddings_map.get(s, np.array([1.0, 0.0, 0.0])) for s in sentences])
        return np.tile(np.array([1.0, 0.0, 0.0]), (len(sentences), 1))

    mock.encode.side_effect = encode
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


def _run(batches, responses, embeddings_map=None, k=3, tau=0.80, **kwargs):
    retriever = _make_retriever(batches)
    embedder = _mock_embedder(embeddings_map)
    metric = PromptEvaluator(
        retriever,
        model=MagicMock(),
        seed_prompt=_SEED_PROMPT,
        embedder=embedder,
        k=k,
        tau=tau,
        **kwargs,
    )
    metric._executor = _mock_executor(responses)
    metric.batch(session_id="test_session", context="ctx", assistant_id="bot", batch=batches)
    metric.on_process_complete()
    return metric.metrics[0]


class TestPromptEvaluatorInit:
    def test_default_parameters(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
        )
        assert metric._k == 10
        assert metric._tau == 0.80
        assert isinstance(metric._statistical_mode, FrequentistMode)

    def test_custom_k_and_tau(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
            k=5,
            tau=0.85,
        )
        assert metric._k == 5
        assert metric._tau == 0.85

    def test_stores_seed_prompt(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
        )
        assert metric._seed_prompt == _SEED_PROMPT

    def test_jq_disabled_by_default(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
        )
        assert metric._jq_evaluator is None

    def test_jq_enabled(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
            jq_enabled=True,
            objective="Answer accurately.",
        )
        assert metric._jq_evaluator is not None

    def test_custom_executor(self, prompt_evaluator_dataset_retriever):
        custom_executor = MagicMock()
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
            executor=custom_executor,
        )
        assert metric._executor is custom_executor

    def test_custom_statistical_mode(self, prompt_evaluator_dataset_retriever):
        mode = BayesianMode()
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
            statistical_mode=mode,
        )
        assert metric._statistical_mode is mode


class TestClustering:
    def test_identical_responses_form_one_cluster(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
            k=3,
            tau=0.80,
        )
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        labels = metric._cluster(embeddings)
        assert len(set(labels)) == 1

    def test_orthogonal_responses_form_separate_clusters(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
            k=3,
            tau=0.80,
        )
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        labels = metric._cluster(embeddings)
        assert len(set(labels)) == 3

    def test_two_similar_one_different(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
            k=3,
            tau=0.80,
        )
        embeddings = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        labels = metric._cluster(embeddings)
        assert len(set(labels)) == 2
        assert labels[0] == labels[1]
        assert labels[2] != labels[0]


class TestCSRAndStability:
    def test_all_same_cluster_csr_is_1(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
            k=3,
        )
        labels = [0, 0, 0]
        assert metric._csr(labels) == pytest.approx(1.0)

    def test_all_different_clusters_csr_is_1_over_k(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
            k=3,
        )
        labels = [0, 1, 2]
        assert metric._csr(labels) == pytest.approx(1 / 3)

    def test_max_entropy_yields_zero_stability(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
            k=3,
        )
        labels = [0, 1, 2]
        se_n = metric._se_n(labels)
        assert se_n == pytest.approx(1.0, abs=1e-6)

    def test_single_cluster_yields_max_stability(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
            k=3,
        )
        labels = [0, 0, 0]
        se_n = metric._se_n(labels)
        assert se_n == pytest.approx(0.0)

    def test_partial_entropy(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
            k=3,
        )
        labels = [0, 0, 1]
        expected_entropy = -(2 / 3) * math.log(2 / 3) - (1 / 3) * math.log(1 / 3)
        expected_se_n = expected_entropy / math.log(3)
        assert metric._se_n(labels) == pytest.approx(expected_se_n, abs=1e-6)


class TestRSS:
    def test_rss_identical_embeddings(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
        )
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
        reference = np.array([1.0, 0.0])
        assert metric._rss(embeddings, reference) == pytest.approx(1.0)

    def test_rss_orthogonal_embeddings(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
        )
        embeddings = np.array([[0.0, 1.0], [0.0, 1.0]])
        reference = np.array([1.0, 0.0])
        assert metric._rss(embeddings, reference) == pytest.approx(0.0)


class TestEvaluateQuery:
    def test_rss_absent_without_ground_truth(self, prompt_evaluator_dataset_retriever):
        """RSS is None when no ground truth is provided."""
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=_mock_embedder(),
            k=3,
        )
        metric._executor = _mock_executor(["r1", "r2", "r3"])
        result = metric._evaluate_query("qa_001", "query?", "ctx", None)
        assert result.rss is None

    def test_rss_present_with_ground_truth(self, prompt_evaluator_dataset_retriever):
        """RSS is computed when ground truth is present."""
        embeddings_map = {
            "r1": np.array([1.0, 0.0, 0.0]),
            "r2": np.array([1.0, 0.0, 0.0]),
            "r3": np.array([1.0, 0.0, 0.0]),
            "gt": np.array([1.0, 0.0, 0.0]),
        }
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=_mock_embedder(embeddings_map),
            k=3,
        )
        metric._executor = _mock_executor(["r1", "r2", "r3"])
        result = metric._evaluate_query("qa_001", "query?", "ctx", "gt")
        assert result.rss == pytest.approx(1.0, abs=1e-4)

    def test_jq_absent_when_disabled(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=_mock_embedder(),
            k=3,
        )
        metric._executor = _mock_executor(["r1", "r2", "r3"])
        result = metric._evaluate_query("qa_001", "query?", "ctx", "gt")
        assert result.jq is None

    def test_executor_called_k_times(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=_mock_embedder(),
            k=5,
        )
        executor = _mock_executor(["r"] * 5)
        metric._executor = executor
        metric._evaluate_query("qa_001", "query?", "ctx", None)
        assert executor.call_count == 5

    def test_executor_called_with_seed_prompt(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=_mock_embedder(),
            k=3,
        )
        executor = _mock_executor(["r"] * 3)
        metric._executor = executor
        metric._evaluate_query("qa_001", "my query", "my ctx", None)
        for call in executor.call_args_list:
            assert call[0][0] == _SEED_PROMPT
            assert call[0][1] == "my query"
            assert call[0][2] == "my ctx"

    def test_n_clusters_matches_cluster_output(self, prompt_evaluator_dataset_retriever):
        embeddings_map = {
            "r1": np.array([1.0, 0.0, 0.0]),
            "r2": np.array([1.0, 0.0, 0.0]),
            "r3": np.array([0.0, 1.0, 0.0]),
        }
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=_mock_embedder(embeddings_map),
            k=3,
        )
        metric._executor = _mock_executor(["r1", "r2", "r3"])
        result = metric._evaluate_query("qa_001", "q", "ctx", None)
        assert result.n_clusters == 2


class TestBatchAndAggregation:
    def test_result_is_prompt_evaluator_metric(self):
        batches = [create_sample_batch(qa_id="qa_001")]
        result = _run(batches, ["r"] * 3)
        assert isinstance(result, PromptEvaluatorMetric)

    def test_result_contains_seed_prompt(self):
        batches = [create_sample_batch(qa_id="qa_001")]
        result = _run(batches, ["r"] * 3)
        assert result.seed_prompt == _SEED_PROMPT

    def test_result_k_and_tau(self):
        batches = [create_sample_batch(qa_id="qa_001")]
        result = _run(batches, ["r"] * 3, k=3, tau=0.75)
        assert result.k == 3
        assert result.tau == pytest.approx(0.75)

    def test_interactions_count_matches_batches(self):
        batches = [create_sample_batch(qa_id=f"qa_{i:03d}") for i in range(3)]
        result = _run(batches, ["r"] * 9)
        assert result.n_queries == 3
        assert len(result.interactions) == 3

    def test_csr_is_mean_of_interactions(self):
        batches = [create_sample_batch(qa_id="qa_001"), create_sample_batch(qa_id="qa_002")]
        result = _run(batches, ["r"] * 6)
        expected_csr = round(sum(i.csr for i in result.interactions) / 2, 4)
        assert result.csr == pytest.approx(expected_csr)

    def test_stability_is_mean_of_interactions(self):
        batches = [create_sample_batch(qa_id="qa_001"), create_sample_batch(qa_id="qa_002")]
        result = _run(batches, ["r"] * 6)
        expected = round(sum(i.stability for i in result.interactions) / 2, 4)
        assert result.stability == pytest.approx(expected)

    def test_rss_is_none_when_no_ground_truth(self):
        batch = create_sample_batch(qa_id="qa_001", ground_truth_assistant="")
        result = _run([batch], ["r"] * 3)
        assert result.rss is None

    def test_jq_is_none_when_disabled(self):
        batches = [create_sample_batch(qa_id="qa_001")]
        result = _run(batches, ["r"] * 3)
        assert result.jq is None

    def test_multiple_sessions_produce_multiple_metrics(self):
        dataset_a = create_sample_dataset(session_id="session_a", conversation=[create_sample_batch()])
        dataset_b = create_sample_dataset(session_id="session_b", conversation=[create_sample_batch()])
        retriever = type("R", (MockRetriever,), {"load_dataset": lambda self: [dataset_a, dataset_b]})
        metric = PromptEvaluator(
            retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=_mock_embedder(),
            k=3,
        )
        metric._executor = _mock_executor(["r"] * 6)
        metric.batch(session_id="session_a", context="ctx", assistant_id="bot", batch=[create_sample_batch()])
        metric.batch(session_id="session_b", context="ctx", assistant_id="bot", batch=[create_sample_batch()])
        metric.on_process_complete()
        assert len(metric.metrics) == 2

    def test_high_consistency_prompt_has_high_csr(self):
        """All K responses cluster together → CSR = 1.0."""
        embeddings_map = {f"r{i}": np.array([1.0, 0.0, 0.0]) for i in range(3)}
        batches = [create_sample_batch(qa_id="qa_001")]
        result = _run(batches, [f"r{i}" for i in range(3)], embeddings_map=embeddings_map, k=3)
        assert result.csr == pytest.approx(1.0)

    def test_scattered_prompt_has_low_csr(self):
        """All K responses in different clusters → CSR = 1/K."""
        embeddings_map = {
            "r0": np.array([1.0, 0.0, 0.0]),
            "r1": np.array([0.0, 1.0, 0.0]),
            "r2": np.array([0.0, 0.0, 1.0]),
        }
        batches = [create_sample_batch(qa_id="qa_001")]
        result = _run(batches, ["r0", "r1", "r2"], embeddings_map=embeddings_map, k=3)
        assert result.csr == pytest.approx(1 / 3, abs=0.01)


class TestStatisticalMode:
    def test_frequentist_mode_returns_float_csr(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
            k=3,
            statistical_mode=FrequentistMode(),
        )
        labels = [0, 0, 1]
        csr = metric._csr(labels)
        assert isinstance(csr, float)
        assert csr == pytest.approx(2 / 3)

    def test_bayesian_mode_returns_float_csr(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=MagicMock(),
            k=3,
            statistical_mode=BayesianMode(mc_samples=100),
        )
        labels = [0, 0, 1]
        csr = metric._csr(labels)
        assert isinstance(csr, float)
        assert 0.0 <= csr <= 1.0


class TestICR:
    def test_icr_is_none_without_constraints(self):
        batches = [create_sample_batch(qa_id="qa_001")]
        result = _run(batches, ["r"] * 3)
        assert result.icr is None

    def test_icr_none_in_interactions_without_constraints(self, prompt_evaluator_dataset_retriever):
        metric = PromptEvaluator(
            prompt_evaluator_dataset_retriever,
            model=MagicMock(),
            seed_prompt=_SEED_PROMPT,
            embedder=_mock_embedder(),
            k=3,
        )
        metric._executor = _mock_executor(["r1", "r2", "r3"])
        result = metric._evaluate_query("qa_001", "q", "ctx", None)
        assert result.icr is None

    def test_icr_is_1_when_all_constraints_pass(self):
        constraint = MagicMock()
        constraint.check.return_value = True
        batches = [create_sample_batch(qa_id="qa_001")]
        result = _run(batches, ["r"] * 3, constraints=[constraint])
        assert result.icr == pytest.approx(1.0)

    def test_icr_is_0_when_no_constraints_pass(self):
        constraint = MagicMock()
        constraint.check.return_value = False
        batches = [create_sample_batch(qa_id="qa_001")]
        result = _run(batches, ["r"] * 3, constraints=[constraint])
        assert result.icr == pytest.approx(0.0)

    def test_icr_partial_compliance(self):
        always_pass = MagicMock()
        always_pass.check.return_value = True
        always_fail = MagicMock()
        always_fail.check.return_value = False
        batches = [create_sample_batch(qa_id="qa_001")]
        result = _run(batches, ["r"] * 3, constraints=[always_pass, always_fail])
        assert result.icr == pytest.approx(0.5)

    def test_icr_averaged_across_queries(self):
        constraint = MagicMock()
        constraint.check.return_value = True
        batches = [create_sample_batch(qa_id="qa_001"), create_sample_batch(qa_id="qa_002")]
        result = _run(batches, ["r"] * 6, constraints=[constraint])
        expected = round(sum(i.icr for i in result.interactions if i.icr is not None) / 2, 4)
        assert result.icr == pytest.approx(expected)


class TestBuiltinConstraints:
    def test_json_constraint_valid_json(self):
        assert JsonConstraint().check('{"key": "value"}') is True

    def test_json_constraint_invalid_json(self):
        assert JsonConstraint().check("not json") is False

    def test_word_count_within_limit(self):
        assert WordCountConstraint(max_words=10).check("hello world") is True

    def test_word_count_exceeds_limit(self):
        assert WordCountConstraint(max_words=2).check("one two three four") is False

    def test_keyword_present_case_insensitive(self):
        assert KeywordConstraint("hello").check("Hello World") is True

    def test_keyword_absent(self):
        assert KeywordConstraint("missing").check("Hello World") is False

    def test_keyword_case_sensitive_match(self):
        assert KeywordConstraint("Hello", case_sensitive=True).check("Hello World") is True

    def test_keyword_case_sensitive_no_match(self):
        assert KeywordConstraint("hello", case_sensitive=True).check("Hello World") is False

    def test_regex_match(self):
        assert RegexConstraint(r"\d{3}").check("my code is 123") is True

    def test_regex_no_match(self):
        assert RegexConstraint(r"\d{3}").check("no digits here") is False


class TestRunMethod:
    def test_run_returns_list(self, prompt_evaluator_dataset_retriever):
        with patch.object(PromptEvaluator, "batch"):
            with patch.object(PromptEvaluator, "on_process_complete"):
                metric = PromptEvaluator(
                    prompt_evaluator_dataset_retriever,
                    model=MagicMock(),
                    seed_prompt=_SEED_PROMPT,
                    embedder=MagicMock(),
                )
                metric.metrics = [
                    PromptEvaluatorMetric(
                        session_id="s",
                        assistant_id="a",
                        seed_prompt=_SEED_PROMPT,
                        k=3,
                        tau=0.80,
                        csr=0.9,
                        stability=0.8,
                        n_queries=1,
                        interactions=[QuerySampleMetrics(qa_id="q", k=3, csr=0.9, stability=0.8, n_clusters=1)],
                    )
                ]
        assert isinstance(metric.metrics[0], PromptEvaluatorMetric)
