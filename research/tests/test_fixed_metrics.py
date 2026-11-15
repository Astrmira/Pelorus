import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_fixed_metrics.py"


def load_script():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("evaluate_fixed_metrics", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_radius_lorentz_preserves_cosine_order() -> None:
    module = load_script()
    rng = np.random.default_rng(42)
    documents = module.normalize(rng.normal(size=(100, 16)))
    query = module.normalize(rng.normal(size=(1, 16)))[0]
    cosine_order = np.argsort(-(documents @ query), kind="stable")
    for curvature in (0.1, 1.0, 5.0):
        scores = module.negative_lorentz_distance(query, documents, curvature)
        assert np.array_equal(np.argsort(-scores, kind="stable"), cosine_order)


def test_normalized_l2_preserves_cosine_order() -> None:
    module = load_script()
    rng = np.random.default_rng(7)
    documents = module.normalize(rng.normal(size=(100, 8)))
    query = module.normalize(rng.normal(size=(1, 8)))[0]
    cosine_order = np.argsort(-(documents @ query), kind="stable")
    l2_order = np.argsort(-module.negative_l2(query, documents), kind="stable")
    assert np.array_equal(l2_order, cosine_order)


def test_radial_squash_is_bounded_and_monotonic() -> None:
    module = load_script()
    rows = np.asarray([[1.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    transformed = module.radial_squash(rows, scale=2.0, radius_budget=3.0)
    norms = np.linalg.norm(transformed, axis=1)
    assert np.all(np.diff(norms) > 0)
    assert np.all(norms < 3.0)
