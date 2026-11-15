import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_cosine_baseline.py"


def load_script():
    spec = importlib.util.spec_from_file_location("evaluate_cosine_baseline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ranking_and_standard_metrics() -> None:
    module = load_script()
    ranking = ["a", "b", "c", "d"]
    assert module.recall_at_k(ranking, {"a", "c"}, 1) == 0.5
    assert module.recall_at_k(ranking, {"a", "c"}, 3) == 1.0
    assert module.reciprocal_rank(ranking, {"c"}) == 1 / 3
    assert module.first_rank(ranking, {"b"}) == 2
    assert module.first_rank(ranking, {"x"}) is None


def test_ndcg_ignores_negative_gain() -> None:
    module = load_script()
    relevance = {"positive": 3, "misleading": -1}
    ideal = module.ndcg_at_k(["positive", "misleading"], relevance, 2)
    reversed_score = module.ndcg_at_k(["misleading", "positive"], relevance, 2)
    assert ideal == 1.0
    assert 0.0 < reversed_score < ideal


def test_exact_rankings_only_use_each_query_haystack() -> None:
    module = load_script()
    documents = [
        {"document_id": "a", "question_id": "q1"},
        {"document_id": "b", "question_id": "q1"},
        {"document_id": "c", "question_id": "q2"},
    ]
    queries = [{"query_id": "q1"}, {"query_id": "q2"}]
    document_ids = ["a", "b", "c"]
    query_ids = ["q1", "q2"]
    document_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    query_vectors = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    rankings = module.exact_rankings(
        documents,
        queries,
        document_ids,
        query_ids,
        document_vectors,
        query_vectors,
    )
    assert [document_id for document_id, _ in rankings["q1"]] == ["a", "b"]
    assert [document_id for document_id, _ in rankings["q2"]] == ["c"]
