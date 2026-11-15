"""验证独立时间终检的数据边界与冻结输入。"""

import json
from pathlib import Path

import run_low_capacity_experiment as old


ROOT = old.ROOT
DATA = ROOT / "data" / "processed" / "temporal_fresh_test"
OLD_DATA = ROOT / "data" / "processed" / "temporal_readout"


def test_fresh_test_has_expected_shape_and_split():
    cases = json.loads((DATA / "annotations.json").read_text(encoding="utf-8"))
    documents = (DATA / "documents.jsonl").read_text(encoding="utf-8").splitlines()
    queries = (DATA / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(cases) == 8
    assert len(documents) == 16
    assert len(queries) == 32
    assert {case["split"] for case in cases} == {"fresh_test"}
    assert all({query["style"] for query in case["queries"]} == {"paraphrase", "ordinary"} for case in cases)


def test_fresh_source_ids_are_disjoint_from_old_temporal_readout():
    fresh = {case["id"] for case in json.loads((DATA / "annotations.json").read_text(encoding="utf-8"))}
    old_ids = {case["id"] for case in json.loads((OLD_DATA / "annotations.json").read_text(encoding="utf-8"))}
    assert fresh.isdisjoint(old_ids)


def test_fresh_manifest_matches_text_inputs():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["input_sha256"].items():
        assert old.sha256_file(DATA / f"{name}.jsonl") == expected
