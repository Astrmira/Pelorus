import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "longmemeval_s_cleaned.json"
OUTPUT_DIR = ROOT / "data" / "processed" / "longmemeval_pilot"
SEED = 20260910
NON_ABSTENTION_PER_TYPE = 2
ABSTENTION_COUNT = 2
CANDIDATES_PER_QUERY = 8


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2
    }


def session_text(session: list[dict[str, object]]) -> str:
    return "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in session
    )


def overlap_score(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def has_consistent_evidence(item: dict[str, object]) -> bool:
    if str(item["question_id"]).endswith("_abs"):
        return True
    marked = {
        session_id
        for session_id, session in zip(
            item["haystack_session_ids"], item["haystack_sessions"], strict=True
        )
        if any(bool(turn.get("has_answer")) for turn in session)
    }
    return marked == set(item["answer_session_ids"])


def choose_items(dataset: list[dict[str, object]]) -> list[dict[str, object]]:
    rng = random.Random(SEED)
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in dataset:
        if str(item["question_id"]).endswith("_abs") or not has_consistent_evidence(item):
            continue
        grouped.setdefault(str(item["question_type"]), []).append(item)

    chosen: list[dict[str, object]] = []
    for question_type in sorted(grouped):
        candidates = sorted(grouped[question_type], key=lambda row: str(row["question_id"]))
        chosen.extend(rng.sample(candidates, NON_ABSTENTION_PER_TYPE))

    abstention = sorted(
        (item for item in dataset if str(item["question_id"]).endswith("_abs")),
        key=lambda row: str(row["question_id"]),
    )
    chosen.extend(rng.sample(abstention, ABSTENTION_COUNT))
    return chosen


def build_candidate_pool(item: dict[str, object]) -> list[dict[str, object]]:
    rng = random.Random(f"{SEED}:{item['question_id']}")
    sessions = []
    for index, (session_id, date, session) in enumerate(
        zip(
            item["haystack_session_ids"],
            item["haystack_dates"],
            item["haystack_sessions"],
            strict=True,
        )
    ):
        text = session_text(session)
        sessions.append(
            {
                "session_id": session_id,
                "date": date,
                "index": index,
                "text": text,
                "official_evidence": session_id in item["answer_session_ids"]
                and not str(item["question_id"]).endswith("_abs"),
                "overlap": overlap_score(str(item["question"]), text),
            }
        )

    selected: dict[str, dict[str, object]] = {}
    for session in sessions:
        if session["official_evidence"]:
            selected[str(session["session_id"])] = session

    lexical = sorted(sessions, key=lambda row: (-float(row["overlap"]), int(row["index"])))
    for session in lexical:
        if len(selected) >= CANDIDATES_PER_QUERY - 2:
            break
        selected.setdefault(str(session["session_id"]), session)

    evidence_indices = [int(row["index"]) for row in sessions if row["official_evidence"]]
    if evidence_indices:
        for evidence_index in evidence_indices:
            for neighbor_index in (evidence_index - 1, evidence_index + 1):
                if 0 <= neighbor_index < len(sessions):
                    neighbor = sessions[neighbor_index]
                    selected.setdefault(str(neighbor["session_id"]), neighbor)

    remaining = [row for row in sessions if str(row["session_id"]) not in selected]
    rng.shuffle(remaining)
    for session in remaining:
        if len(selected) >= CANDIDATES_PER_QUERY:
            break
        selected[str(session["session_id"])] = session

    candidates = list(selected.values())[:CANDIDATES_PER_QUERY]
    rng.shuffle(candidates)
    return [
        {
            "candidate_id": f"c{index + 1:02d}",
            "date": row["date"],
            "text": row["text"],
            "_session_id_for_key": row["session_id"],
        }
        for index, row in enumerate(candidates)
    ]


def main() -> None:
    dataset = json.loads(SOURCE.read_text(encoding="utf-8"))
    chosen = choose_items(dataset)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    blind_items = []
    key_items = []
    for item in chosen:
        candidates = build_candidate_pool(item)
        blind_items.append(
            {
                "question_id": item["question_id"],
                "question_type": item["question_type"],
                "question": item["question"],
                "reference_answer": item["answer"],
                "question_date": item["question_date"],
                "is_abstention": str(item["question_id"]).endswith("_abs"),
                "candidates": [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key != "_session_id_for_key"
                    }
                    for candidate in candidates
                ],
            }
        )
        key_items.append(
            {
                "question_id": item["question_id"],
                "official_answer_session_ids": item["answer_session_ids"],
                "candidate_mapping": {
                    candidate["candidate_id"]: candidate["_session_id_for_key"]
                    for candidate in candidates
                },
            }
        )

    (OUTPUT_DIR / "blind_annotation_pack.json").write_text(
        json.dumps(blind_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "official_evidence_key.json").write_text(
        json.dumps(key_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "seed": SEED,
        "query_count": len(blind_items),
        "candidate_count": sum(len(item["candidates"]) for item in blind_items),
        "question_types": Counter(item["question_type"] for item in blind_items),
        "selection": {
            "non_abstention_per_type": NON_ABSTENTION_PER_TYPE,
            "abstention_count": ABSTENTION_COUNT,
            "excluded_inconsistent_non_abstention_items": sum(
                1
                for item in dataset
                if not str(item["question_id"]).endswith("_abs")
                and not has_consistent_evidence(item)
            ),
        },
        "candidate_pool_notice": "候选池不是完整 qrels；仅用于冻结标注规范的 Pilot。",
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=dict), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
