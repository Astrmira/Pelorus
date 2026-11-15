import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_readout as diagnostic
import run_low_capacity_experiment as experiment


def test_capture_preserves_scores_and_parameter_gradients():
    torch.manual_seed(7)
    query, documents = torch.randn(8), torch.randn(5, 8)
    mask = torch.tensor([True, False, True, False, False])
    for name, model_class in experiment.MODEL_CLASSES.items():
        if name.startswith("B0"):
            continue
        model = model_class(8, 4)
        original = model.score(query, documents)
        original_grads = torch.autograd.grad(experiment.multi_positive_loss(original, mask), tuple(model.parameters()))
        captured, geometric = diagnostic.captured_score(model, query, documents)
        captured_grads = torch.autograd.grad(experiment.multi_positive_loss(captured, mask), tuple(model.parameters()))
        torch.testing.assert_close(captured, original, rtol=0, atol=0)
        for actual, expected in zip(captured_grads, original_grads, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert "combine" not in model.__dict__
        assert geometric.shape == (5,)


def test_exact_saturation_blocks_geometry_with_nonzero_loss_gradient():
    model = experiment.SharedMahalanobis(2, 2)
    with torch.no_grad():
        model.projection.weight.copy_(torch.eye(2))
    split = {
        "tensor_queries": torch.tensor([[10.0, 0.0]]),
        "tensor_documents": torch.tensor([[0.0, 10.0], [-10.0, 0.0]]),
        "examples": [{"query_id": "示例", "query_index": 0, "candidate_indices": [0, 1],
                      "positive_mask": np.array([False, True])}],
    }
    result = diagnostic.inspect_split(model, split, gradients=True)
    assert result["tanh_exact_saturation"] == 1
    assert result["gradients"]["projection.weight"]["per_query_l2"]["max"] == 0
    assert result["loss_to_combined_score_gradient_l2"]["min"] > 0
    assert result["loss_to_geometry_gradient_l2"]["max"] == 0
    assert result["ranking"]["ranking_changed"] == 0


def test_shared_coordinates_preserve_ranking_but_independent_rotation_breaks_it():
    documents = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    queries = np.array([[1, 0]], dtype=np.float32)
    rotation = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    split = {"examples": [{"query_id": "示例", "query_index": 0, "candidate_indices": [0, 1, 2],
                           "positive_mask": np.array([True, False, False])}]}
    shared = diagnostic.evaluate_coordinates(split, documents @ rotation, queries @ rotation, documents, queries)
    independent = diagnostic.evaluate_coordinates(split, documents, queries @ rotation, documents, queries)
    assert shared["summary"]["mrr"] == 1
    assert shared["summary"]["rank_spearman"] == 1
    assert independent["summary"]["mrr"] < 1
    assert independent["summary"]["top1_changed"] == 1


def test_recall_counts_all_positives_instead_of_only_first():
    scores = torch.arange(12, dtype=torch.float32)
    mask = np.zeros(12, dtype=bool)
    mask[[0, 11]] = True
    row, _ = diagnostic.ranking_diagnostics(scores, scores, mask)
    assert row["mrr"] == 1
    assert row["recall@10"] == 0.5
    assert row["evidence_set_recall@10"] == 0
