import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_longmemeval_encoding_dataset.py"
OUTPUT_DIR = ROOT / "data" / "processed" / "longmemeval_turn_pairs"


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_longmemeval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(name: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (OUTPUT_DIR / name).read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_pair_turns_preserves_irregular_roles() -> None:
    module = load_script()
    session = [
        {"role": "assistant", "content": "开场"},
        {"role": "user", "content": "问题一"},
        {"role": "user", "content": "问题二"},
        {"role": "assistant", "content": "回答"},
        {"role": "tool", "content": "结果"},
        {"role": "user", "content": "结尾"},
    ]
    pairs = module.pair_turns(session)
    assert [[turn["role"] for turn in pair] for pair in pairs] == [
        ["assistant"],
        ["user"],
        ["user", "assistant"],
        ["tool"],
        ["user"],
    ]
    assert sum(len(pair) for pair in pairs) == len(session)


def test_generation_is_reproducible_and_qrels_are_valid() -> None:
    module = load_script()
    first = module.build()
    second = module.build()
    assert first["outputs_sha256"] == second["outputs_sha256"]

    documents = read_jsonl("documents.jsonl")
    document_ids = {row["document_id"] for row in documents}
    assert len(document_ids) == len(documents)
    assert all(row["text"] and row["content_sha256"] for row in documents)

    for name in ("official_qrels.jsonl", "pilot_qrels.jsonl"):
        qrels = read_jsonl(name)
        assert qrels
        assert all(row["document_id"] in document_ids for row in qrels)
        assert {row["query_id"] for row in qrels} == {
            row["query_id"] for row in read_jsonl("queries.jsonl")
        }


def test_encoding_inputs_do_not_contain_label_fields() -> None:
    documents = read_jsonl("encoding_documents.jsonl")
    queries = read_jsonl("encoding_queries.jsonl")
    assert all(set(row) == {"id", "text"} for row in documents + queries)
    forbidden_keys = {
        "answer",
        "answer_session_ids",
        "has_answer",
        "candidate_id",
        "relevance",
        "roles",
        "rationale_zh",
        "is_official_evidence_session",
        "session_id",
    }
    assert all(not (set(row) & forbidden_keys) for row in documents + queries)


def test_ids_and_hashes_are_stable() -> None:
    module = load_script()
    assert module.document_id("q", "s", 0) == module.document_id("q", "s", 0)
    assert module.document_id("q", "s", 0) != module.document_id("q", "s", 1)
    for row in read_jsonl("documents.jsonl"):
        assert module.sha256_text(row["text"]) == row["content_sha256"]
