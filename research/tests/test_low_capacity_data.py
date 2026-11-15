import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_low_capacity_training_data.py"
DATA_DIR = ROOT / "data" / "processed" / "longmemeval_low_capacity"
PILOT = ROOT / "data" / "processed" / "longmemeval_pilot" / "pilot_annotations.json"


def load_script():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("prepare_low_capacity_training_data", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_splits_are_question_isolated() -> None:
    train = {row["id"] for row in read_jsonl(DATA_DIR / "train_queries.jsonl")}
    validation = {row["id"] for row in read_jsonl(DATA_DIR / "validation_queries.jsonl")}
    pilot = {
        row["question_id"]
        for row in json.loads(PILOT.read_text(encoding="utf-8"))["queries"]
    }
    assert not (train & validation)
    assert not (train & pilot)
    assert not (validation & pilot)


def test_candidate_pools_have_all_positives_and_fixed_size() -> None:
    for split, expected_queries in (("train", 80), ("validation", 16)):
        metadata = read_jsonl(DATA_DIR / f"{split}_metadata.jsonl")
        qrels = read_jsonl(DATA_DIR / f"{split}_qrels.jsonl")
        positives = {}
        for row in qrels:
            positives.setdefault(row["query_id"], set()).add(row["document_id"])
        assert len(metadata) == expected_queries
        for row in metadata:
            assert len(row["candidate_ids"]) == 32
            assert len(set(row["candidate_ids"])) == 32
            assert set(row["positive_ids"]) == positives[row["query_id"]]
            assert set(row["positive_ids"]) <= set(row["candidate_ids"])


def test_generation_is_reproducible() -> None:
    module = load_script()
    first = module.build()
    second = module.build()
    assert first == second
