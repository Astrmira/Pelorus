import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_longmemeval_encoding_dataset import document_id, pair_text, pair_turns


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "longmemeval_s_cleaned.json"
PILOT = ROOT / "data" / "processed" / "longmemeval_pilot" / "pilot_annotations.json"
OUTPUT_DIR = ROOT / "data" / "processed" / "longmemeval_low_capacity"
SEED = 20260912
TRAIN_COUNT = 80
VALIDATION_COUNT = 16
CANDIDATES_PER_QUERY = 32
INSTRUCTION = (
    "Instruct: Given a personal conversation history, retrieve the passages that provide "
    "the evidence needed to answer the question.\nQuery: "
)


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def overlap(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    return len(query_tokens & tokenize(text)) / max(1, len(query_tokens))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def has_consistent_evidence(item: dict[str, Any]) -> bool:
    marked = {
        str(session_id)
        for session_id, session in zip(
            item["haystack_session_ids"], item["haystack_sessions"], strict=True
        )
        if any(bool(turn.get("has_answer")) for turn in session)
    }
    return marked == set(map(str, item["answer_session_ids"])) and bool(marked)


def stratified_select(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item["question_type"])].append(item)
    for question_type in grouped:
        grouped[question_type].sort(key=lambda row: str(row["question_id"]))
        random.Random(f"{SEED}:{question_type}").shuffle(grouped[question_type])

    types = sorted(grouped)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for index, question_type in enumerate(types):
        train_n = TRAIN_COUNT // len(types) + int(index < TRAIN_COUNT % len(types))
        validation_n = VALIDATION_COUNT // len(types) + int(index < VALIDATION_COUNT % len(types))
        rows = grouped[question_type]
        train.extend(rows[:train_n])
        validation.extend(rows[train_n : train_n + validation_n])
    return sorted(train, key=lambda row: str(row["question_id"])), sorted(
        validation, key=lambda row: str(row["question_id"])
    )


def build_candidates(item: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    question_id = str(item["question_id"])
    query = str(item["question"])
    candidates: list[dict[str, Any]] = []
    positive_ids: set[str] = set()
    for session_index, (session_id, date, session) in enumerate(
        zip(
            item["haystack_session_ids"],
            item["haystack_dates"],
            item["haystack_sessions"],
            strict=True,
        )
    ):
        for pair_index, turns in enumerate(pair_turns(session)):
            text = pair_text(str(date), turns)
            doc_id = document_id(question_id, str(session_id), pair_index)
            is_positive = any(bool(turn.get("has_answer")) for turn in turns)
            if is_positive:
                positive_ids.add(doc_id)
            candidates.append(
                {
                    "id": doc_id,
                    "text": text,
                    "question_id": question_id,
                    "session_index": session_index,
                    "pair_index": pair_index,
                    "is_positive": is_positive,
                    "overlap": overlap(query, text),
                }
            )
    if not positive_ids:
        raise ValueError(f"问题没有 Turn-Pair 正证据：{question_id}")
    if len(positive_ids) > CANDIDATES_PER_QUERY:
        raise ValueError(f"正证据超过候选池容量：{question_id}")

    selected = {row["id"]: row for row in candidates if row["is_positive"]}
    hard_negatives = sorted(
        (row for row in candidates if not row["is_positive"]),
        key=lambda row: (-float(row["overlap"]), int(row["session_index"]), int(row["pair_index"])),
    )
    hard_target = max(0, CANDIDATES_PER_QUERY - len(selected)) * 3 // 4
    for row in hard_negatives[:hard_target]:
        selected[row["id"]] = row

    remaining = [row for row in candidates if row["id"] not in selected]
    random.Random(f"{SEED}:{question_id}:negatives").shuffle(remaining)
    for row in remaining:
        if len(selected) >= CANDIDATES_PER_QUERY:
            break
        selected[row["id"]] = row
    output = sorted(selected.values(), key=lambda row: str(row["id"]))
    return output, positive_ids


def build() -> dict[str, Any]:
    dataset = json.loads(SOURCE.read_text(encoding="utf-8"))
    pilot_ids = {
        str(row["question_id"])
        for row in json.loads(PILOT.read_text(encoding="utf-8"))["queries"]
    }
    eligible = [
        item
        for item in dataset
        if str(item["question_id"]) not in pilot_ids
        and not str(item["question_id"]).endswith("_abs")
        and has_consistent_evidence(item)
    ]
    train, validation = stratified_select(eligible)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "seed": SEED,
        "source_sha256": sha256_file(SOURCE),
        "pilot_annotations_sha256": sha256_file(PILOT),
        "candidate_policy": {
            "per_query": CANDIDATES_PER_QUERY,
            "positives": "保留所有 has_answer Turn Pair。",
            "negatives": "约四分之三按 Query 词面重合选 hard negatives，其余确定性随机补齐。",
            "notice": "候选选择不使用向量、qrels 分数或 Pilot 标签；只用于训练和验证。",
        },
        "splits": {},
    }
    for split, items in (("train", train), ("validation", validation)):
        documents: list[dict[str, Any]] = []
        queries: list[dict[str, Any]] = []
        qrels: list[dict[str, Any]] = []
        metadata: list[dict[str, Any]] = []
        for item in items:
            question_id = str(item["question_id"])
            candidates, positive_ids = build_candidates(item)
            documents.extend({"id": row["id"], "text": row["text"]} for row in candidates)
            queries.append({"id": question_id, "text": INSTRUCTION + str(item["question"])})
            qrels.extend(
                {"query_id": question_id, "document_id": document_id, "relevance": 1}
                for document_id in sorted(positive_ids)
            )
            metadata.append(
                {
                    "query_id": question_id,
                    "question_type": item["question_type"],
                    "candidate_ids": [row["id"] for row in candidates],
                    "positive_ids": sorted(positive_ids),
                }
            )
        paths = {
            "documents": OUTPUT_DIR / f"{split}_documents.jsonl",
            "queries": OUTPUT_DIR / f"{split}_queries.jsonl",
            "qrels": OUTPUT_DIR / f"{split}_qrels.jsonl",
            "metadata": OUTPUT_DIR / f"{split}_metadata.jsonl",
        }
        for key, rows in (
            ("documents", documents),
            ("queries", queries),
            ("qrels", qrels),
            ("metadata", metadata),
        ):
            write_jsonl(paths[key], rows)
        manifest["splits"][split] = {
            "query_count": len(queries),
            "document_count": len(documents),
            "positive_count": len(qrels),
            "question_types": dict(Counter(str(item["question_type"]) for item in items)),
            "estimated_tokens": sum((len(row["text"]) + 3) // 4 for row in documents + queries),
            "question_ids": [row["id"] for row in queries],
            "sha256": {key: sha256_file(path) for key, path in paths.items()},
        }
    train_ids = set(manifest["splits"]["train"]["question_ids"])
    validation_ids = set(manifest["splits"]["validation"]["question_ids"])
    if train_ids & validation_ids or train_ids & pilot_ids or validation_ids & pilot_ids:
        raise ValueError("Train、Validation 与 Pilot Test 存在问题级泄漏")
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    print(json.dumps(build(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
