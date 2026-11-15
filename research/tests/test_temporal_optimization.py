"""保护优化诊断的数据边界。"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_temporal_optimization as diagnostic


@pytest.mark.parametrize("split", ["validation", "holdout"])
def test_rejects_non_training_cases(split):
    with pytest.raises(ValueError, match="禁止"):
        diagnostic.examples_for([{"split": "train", "queries": []}, {"split": split, "queries": []}])


def test_diagnostic_excludes_unseen_templates_and_ordinary_controls():
    case = {"split": "train", "queries": [{"style": name} for name in ("seen", "paraphrase", "ordinary")]}
    examples = diagnostic.examples_for([case])
    assert len(examples) == 1
    assert examples[0][1]["style"] == "seen"
