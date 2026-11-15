"""确保开发实验不重复打开测试集，并按链保留时间条件配对。"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_temporal_geometry_development import assemble, metrics
from geometry_readouts import GeometryReadout


@pytest.mark.parametrize('split', ['holdout', 'fresh_test'])
def test_refuses_test_cases(split):
    with pytest.raises(ValueError, match='只允许'):
        assemble([{'split': split}], {})


def test_condition_blind_query_cannot_solve_both_targets():
    model = GeometryReadout('raw_cosine', 2).double()
    q = torch.tensor([[1., 0.]], dtype=torch.float64).repeat(6, 1)
    d = torch.eye(2, dtype=torch.float64).repeat(6, 1, 1)
    result = metrics(model, (q, d, torch.tensor([0, 1, 0, 1, 0, 1])))
    assert all(m['single'] == .5 and m['paired'] == 0 for m in result.values())
