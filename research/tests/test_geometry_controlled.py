"""验证标准测地公式、数值稳定性和合成实验边界。"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from geometry_readouts import GeometryReadout, exp_origin, poincare_squared, tangent_distance_squared
from run_geometry_controlled import generate


@pytest.mark.parametrize('c', [0.1, 1.0, 4.0])
def test_poincare_formula_agrees_with_acosh_and_origin_radius(c):
    torch.manual_seed(8)
    v = torch.randn(15, 2, dtype=torch.float64)
    w = torch.randn(15, 2, dtype=torch.float64)
    x, y = exp_origin(v, c), exp_origin(w, c)
    arg = 1 + 2 * c * (x-y).square().sum(-1) / ((1-c*x.square().sum(-1))*(1-c*y.square().sum(-1)))
    reference = torch.acosh(arg).square() / c
    torch.testing.assert_close(poincare_squared(x, y, c), reference)
    torch.testing.assert_close(poincare_squared(x, torch.zeros_like(x), c), v.square().sum(-1))


def test_zero_curvature_and_small_curvature_limit():
    x = torch.tensor([[1., 2.], [-.5, 1.]], dtype=torch.float64)
    y = x.flip(0)
    expected = (x-y).square().sum(-1)
    torch.testing.assert_close(tangent_distance_squared(x, y, 0), expected)
    torch.testing.assert_close(tangent_distance_squared(x, y, 1e-9), expected, rtol=1e-7, atol=1e-9)


def test_zero_distance_has_finite_gradients_including_curvature():
    for v in ([0., 0.], [3., 4.]):
        x = torch.tensor([v], dtype=torch.float64, requires_grad=True)
        c = torch.tensor(1., dtype=torch.float64, requires_grad=True)
        loss = tangent_distance_squared(x, x, c).sum()
        assert float(loss.detach()) == 0
        loss.backward()
        assert torch.isfinite(x.grad).all() and torch.isfinite(c.grad).all()


def test_symmetry_triangle_and_rotation_invariance():
    torch.manual_seed(91)
    a, b, c = torch.randn(3, 20, 2, dtype=torch.float64)
    d = lambda x, y: tangent_distance_squared(x, y, 1).sqrt()
    torch.testing.assert_close(d(a, b), d(b, a))
    assert (d(a, b) <= d(a, c) + d(c, b) + 1e-9).all()
    rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    torch.testing.assert_close(d(a @ rotation, b @ rotation), d(a, b))


def test_equal_radius_poincare_ranking_equals_cosine():
    angles = torch.linspace(.1, 3., 20, dtype=torch.float64)
    docs = 2 * torch.stack([angles.cos(), angles.sin()], -1)
    q = torch.tensor([[1., 0.]], dtype=torch.float64)
    for c in (.1, 1., 4.):
        torch.testing.assert_close(tangent_distance_squared(q, docs, c).argsort(), (-docs[:, 0]).argsort())


def test_candidate_order_and_stress_distribution():
    projection = np.linalg.qr(np.random.default_rng(9).normal(size=(16, 2)))[0].T
    natural = generate(200, 31, projection)
    stress = generate(100, 32, projection, pressure=True)
    for space in ('euclidean', 'hyperbolic'):
        assert .3 < natural[space + '_target'].mean() < .7
    assert np.all(stress['euclidean_target'] != stress['hyperbolic_target'])
    assert not np.array_equal(natural['observed'][:100], stress['observed'])
    np.testing.assert_allclose(natural['observed'] @ projection.T, natural['latent'], atol=1e-12)


def test_matched_parameter_count_and_learnable_distance_gradients():
    count = lambda n: sum(p.numel() for p in GeometryReadout(n, 16).parameters())
    assert count('shared_euclidean') == count('poincare_1') == 32
    assert count('bilinear') >= count('poincare_1')
    assert count('nonlinear_euclidean') >= count('poincare_learned')
    model = GeometryReadout('poincare_learned', 16).double()
    model(torch.randn(5, 16, dtype=torch.float64), torch.randn(5, 2, 16, dtype=torch.float64)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
