import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_low_capacity_experiment.py"


def load_script():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("run_low_capacity_experiment", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multi_positive_loss_rewards_any_positive() -> None:
    module = load_script()
    mask = torch.tensor([True, True, False])
    good = module.multi_positive_loss(torch.tensor([4.0, 1.0, 0.0]), mask)
    bad = module.multi_positive_loss(torch.tensor([0.0, 1.0, 4.0]), mask)
    assert good < bad


def test_model_scores_are_finite() -> None:
    module = load_script()
    query = torch.randn(8)
    documents = torch.randn(5, 8)
    for model_class in module.MODEL_CLASSES.values():
        model = (
            model_class()
            if model_class is module.FixedWhitenedCosine
            else model_class(input_dim=8, output_dim=4)
        )
        scores = model.score(query, documents)
        assert scores.shape == (5,)
        assert torch.isfinite(scores).all()


def test_hyperbolic_radii_are_bounded() -> None:
    module = load_script()
    model = module.HyperbolicTwoTower(input_dim=8, output_dim=4)
    rows = torch.randn(20, 8) * 100
    tangent = model.tangent(rows, model.document_projection, model.document_radius)
    radii = torch.linalg.norm(tangent, dim=-1)
    assert torch.all(radii >= module.RADIUS_MIN - 1e-6)
    assert torch.all(radii <= module.RADIUS_BUDGET + 1e-6)
