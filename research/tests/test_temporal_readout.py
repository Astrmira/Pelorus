import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_temporal_readout as temporal


def test_local_metric_equals_weighted_euclidean_projections():
    torch.manual_seed(12)
    model = temporal.Probe("local_metric", 7)
    with torch.no_grad():
        model.gate.weight.normal_()
    q, documents = torch.randn(7), torch.randn(3, 7)
    delta = documents - q
    weights = model.gate(q).softmax(-1)
    explicit = temporal.EPSILON * torch.sum(delta ** 2, -1)
    for k in range(4):
        explicit = explicit + weights[k] * torch.sum((delta @ model.projections[k].T) ** 2, -1)
    torch.testing.assert_close(-model.raw_score(q, documents), explicit)


def test_metric_field_is_positive_definite():
    model = temporal.Probe("local_metric", 7).double()
    x = torch.randn(7, dtype=torch.float64)
    weights = model.gate(x).softmax(-1)
    matrix = temporal.EPSILON * torch.eye(7, dtype=torch.float64)
    for k in range(4):
        matrix = matrix + weights[k] * model.projections[k].T @ model.projections[k]
    assert torch.linalg.eigvalsh(matrix).min() >= temporal.EPSILON - 1e-10


def test_constant_field_path_integral_matches_quadratic_distance():
    torch.manual_seed(2)
    path = temporal.Probe("path_metric", 7)
    local = temporal.Probe("local_metric", 7)
    local.load_state_dict(path.state_dict())
    q, docs = torch.randn(7), torch.randn(3, 7)
    torch.testing.assert_close(path.raw_score(q, docs), local.raw_score(q, docs), rtol=1e-5, atol=1e-5)


def test_path_is_symmetric_candidate_independent_and_has_finite_gradients():
    torch.manual_seed(2)
    model = temporal.Probe("path_metric", 7).double()
    with torch.no_grad():
        model.gate.weight.normal_()
    points = torch.randn(4, 7, dtype=torch.float64)
    a = model.raw_score(points[0], points[1:2])[0]
    b = model.raw_score(points[1], points[:1])[0]
    many = model.raw_score(points[0], points[1:])[0]
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(a, many, rtol=1e-6, atol=1e-6)
    a.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_geometry_and_generic_controls_have_matched_or_stronger_capacity():
    count = lambda name: sum(parameter.numel() for parameter in temporal.Probe(name, 1536).parameters())
    assert count("local_metric") == count("path_metric") == 18436
    for name in ("bilinear", "shared_nonlinear", "pair_mlp_matched"):
        assert abs(count(name) - count("local_metric")) < 20
    assert count("pair_mlp_stronger") > 2 * count("local_metric")


def test_annotations_split_by_topic_and_evidence_without_candidate_position_leakage():
    cases = json.loads((temporal.DATA / "annotations.json").read_text(encoding="utf-8"))
    topics, fingerprints = defaultdict(set), defaultdict(set)
    target_positions = []
    for case in cases:
        topics[case["group"]].add(case["split"])
        assert len(case["candidate_ids"]) == 2
        for evidence in case["evidence"]:
            fingerprints[evidence["user_fingerprint"]].add(case["split"])
        for style in ("seen", "paraphrase", "ordinary"):
            queries = [q for q in case["queries"] if q["style"] == style]
            assert {q["target"] for q in queries} == set(case["candidate_ids"])
        earlier = next(q for q in case["queries"] if q["style"] == "seen" and q["state"] == "earlier")
        target_positions.append(case["candidate_ids"].index(earlier["target"]))
        if case["split"] == "holdout":
            assert not case["previously_used_source"]
            assert not any(row["previously_encoded_pair"] for row in case["evidence"])
    assert all(len(splits) == 1 for splits in topics.values())
    assert all(len(splits) == 1 for splits in fingerprints.values())
    assert set(target_positions) == {0, 1}


def test_date_rule_uses_query_text_not_hidden_state_label():
    cases = json.loads((temporal.DATA / "annotations.json").read_text(encoding="utf-8"))
    for case in cases:
        for query in case["queries"]:
            query["state"] = "被打乱的隐藏标签"
    assert temporal.date_rule(cases, "seen")["summary"]["paired_accuracy"] == 1
    assert temporal.date_rule(cases, "paraphrase")["summary"]["paired_accuracy"] == 1


def test_neutral_query_cannot_get_both_opposite_targets_correct():
    model = temporal.Probe("cosine_raw", 3)
    corpus = {"raw_queries": torch.tensor([[1., 0., 0.]]), "raw_documents": torch.eye(3)[:2]}
    case = {"id": "示例", "neutral_index": 0, "document_indices": [0, 1],
            "queries": [{"style": "seen", "query_index": 0, "target_index": target} for target in (0, 1)]}
    result = temporal.style_metrics(model, corpus, [case], "seen", neutral=True)
    assert result["summary"]["paired_accuracy"] == 0
    assert result["summary"]["single_accuracy"] == 0.5
