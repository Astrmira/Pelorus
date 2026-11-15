import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "encode_embeddings_api.py"


def load_script():
    spec = importlib.util.spec_from_file_location("encode_embeddings_api", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_batches_respects_count_and_token_limits() -> None:
    module = load_script()
    rows = [
        {"id": "a", "text": "a" * 8},
        {"id": "b", "text": "b" * 8},
        {"id": "c", "text": "c" * 20},
    ]
    batches = module.make_batches(rows, batch_size=2, max_batch_tokens=5)
    assert [[row["id"] for row in batch] for batch in batches] == [["a", "b"], ["c"]]


def test_cache_round_trip_and_materialize(tmp_path: Path) -> None:
    module = load_script()
    cache_path = tmp_path / "cache.jsonl"
    records = [
        {"id": "a", "embedding": [1.0, 0.0]},
        {"id": "b", "embedding": [0.0, 1.0]},
    ]
    module.append_cache(cache_path, records)
    cache = module.load_cache(cache_path, expected_dimensions=2)
    vectors_path = tmp_path / "vectors.npy"
    ids_path = tmp_path / "ids.json"
    stats = module.materialize(
        [{"id": "b", "text": "二"}, {"id": "a", "text": "一"}],
        cache,
        vectors_path,
        ids_path,
    )
    vectors = np.load(vectors_path)
    assert vectors.tolist() == [[0.0, 1.0], [1.0, 0.0]]
    assert stats["count"] == 2
    assert stats["dimensions"] == 2


def test_cache_rejects_wrong_dimensions(tmp_path: Path) -> None:
    module = load_script()
    cache_path = tmp_path / "cache.jsonl"
    module.append_cache(cache_path, [{"id": "a", "embedding": [1.0]}])
    try:
        module.load_cache(cache_path, expected_dimensions=2)
    except ValueError as error:
        assert "维度无效" in str(error)
    else:
        raise AssertionError("应拒绝维度不匹配的缓存")
