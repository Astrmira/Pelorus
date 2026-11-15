import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_methodology_retest as repaired


def toy_split():
    torch.manual_seed(10)
    return {"tensor_queries": torch.randn(2, 8), "tensor_documents": torch.randn(8, 8),
            "examples": [{"query_id": str(i), "query_index": i, "candidate_indices": list(range(i*4, i*4+4)),
                          "positive_mask": np.array([True, True, False, False])} for i in range(2)]}


def test_all_positive_loss_keeps_gradient_for_overlooked_positive():
    scores = torch.tensor([10.0, -5.0, 0.0], requires_grad=True)
    mask = torch.tensor([True, True, False])
    all_grad = torch.autograd.grad(repaired.retrieval_loss(scores, mask, "all"), scores)[0]
    any_grad = torch.autograd.grad(repaired.retrieval_loss(scores, mask, "any"), scores)[0]
    assert all_grad[1] < -0.49
    assert abs(any_grad[1]) < 1e-6


def test_scale_calibration_is_label_independent_and_avoids_initial_saturation():
    train = toy_split()
    model = repaired.RepairedScorer("B1", 8, repaired.VARIANTS["R1_scale"], 4)
    model.calibrate(train)
    expected = model.scale.clone()
    altered = copy.deepcopy(train)
    for example in altered["examples"]:
        example["positive_mask"] = ~example["positive_mask"]
    model.calibrate(altered)
    torch.testing.assert_close(model.scale, expected, atol=0, rtol=0)
    q, d, _ = repaired.tensors(train, train["examples"][0])
    scores, raw, residual, derivative = model.parts(q, d)
    assert torch.isfinite(scores).all()
    assert torch.all((residual > -1) & (residual <= 0))
    assert torch.all(derivative > 0)
    assert torch.equal(torch.argsort(raw), torch.argsort(residual))


def test_shared_scoring_is_symmetric_and_independent_of_other_candidates():
    torch.manual_seed(11)
    points = torch.randn(5, 8)
    for name in repaired.MODELS:
        config = repaired.VARIANTS["R5_squared_h1" if name == "H1" else "R4_shared"]
        model = repaired.RepairedScorer(name, 8, config, 4)
        forward = model.score(points[0], points[1:2])[0]
        reverse = model.score(points[1], points[0:1])[0]
        with_extra = model.score(points[0], points[1:])[0]
        torch.testing.assert_close(forward, reverse, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(forward, with_extra, atol=1e-6, rtol=1e-6)


def test_shared_geometry_capacity_matches_and_alignment_initializes_exactly():
    counts = []
    for name in ("B2", "H1", "H2"):
        torch.manual_seed(11)
        shared = repaired.RepairedScorer(name, 8, repaired.VARIANTS["R4_shared"], 4)
        assert shared.inner.query_projection is shared.inner.document_projection
        assert shared.inner.query_radius is shared.inner.document_radius
        counts.append(sum(parameter.numel() for parameter in shared.parameters()))
        aligned = repaired.RepairedScorer(name, 8, repaired.VARIANTS["R2_aligned"], 4)
        assert aligned.inner.query_projection is not aligned.inner.document_projection
        torch.testing.assert_close(aligned.inner.query_projection.weight, aligned.inner.document_projection.weight, atol=0, rtol=0)
    assert len(set(counts)) == 1


def test_map_sees_missing_second_evidence_and_selector_preserves_b0_on_ties():
    mask = torch.tensor([True, True, False, False])
    good, _ = repaired.ranking_metrics(torch.tensor([4.0, 3.0, 2.0, 1.0]), mask)
    partial, _ = repaired.ranking_metrics(torch.tensor([4.0, 1.0, 3.0, 2.0]), mask)
    assert good["mrr"] == partial["mrr"]
    assert good["map"] > partial["map"]
    assert repaired.better(good, partial)
    assert not repaired.better(good, good)
    split = toy_split()
    baseline = repaired.evaluate(None, split, baseline=True)
    model = repaired.RepairedScorer("H1", 8, repaired.VARIANTS["R4_shared"], 4)
    bypassed = repaired.evaluate(model, split, baseline=True)
    assert baseline == bypassed


def test_audit_fingerprint_ignores_labels_and_whitespace_but_keeps_roles():
    from audit_readout_splits import content_fingerprint
    left = [{"role": "user", "content": "相同  文本", "has_answer": True}]
    right = [{"role": "user", "content": "相同\n文本"}]
    assert content_fingerprint(left) == content_fingerprint(right)
    assert content_fingerprint(left) != content_fingerprint([{**right[0], "role": "assistant"}])
