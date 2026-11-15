import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "longmemeval_s_cleaned.json"
PILOT_DIR = ROOT / "data" / "processed" / "longmemeval_pilot"
ANNOTATIONS = PILOT_DIR / "pilot_annotations.json"
OUTPUT_DIR = ROOT / "data" / "processed" / "longmemeval_turn_pairs"
SCHEMA_VERSION = "0.1-turn-pair"

# Pilot 非零 Session 标签经逐对话阅读后定位到实际证据 Pair。
# 值允许覆盖 Session 级标签，以表达同一 Session 内不同 Pair 的证据强度。
PILOT_PAIR_LABELS: dict[tuple[str, str], dict[int, dict[str, Any]]] = {
    ("2698e78f", "answer_9282283d_2"): {2: {}},
    ("2698e78f", "answer_9282283d_1"): {1: {}},
    ("6a1eabeb", "answer_a25d4a91_2"): {0: {}},
    ("6a1eabeb", "answer_a25d4a91_1"): {2: {}},
    ("157a136e", "answer_8de18468_1"): {1: {}},
    ("157a136e", "answer_8de18468_2"): {0: {}},
    ("2e6d26dc", "answer_fa526fc0_4"): {0: {}},
    ("2e6d26dc", "answer_fa526fc0_2"): {0: {}},
    ("2e6d26dc", "answer_fa526fc0_1"): {0: {}},
    ("2e6d26dc", "answer_fa526fc0_3"): {0: {}},
    ("cc539528", "answer_ultrachat_374124"): {3: {}},
    ("65240037", "answer_ultrachat_403752"): {4: {}},
    ("a89d7624", "answer_8f15ac24"): {2: {}},
    ("32260d93", "answer_0250ae1c"): {0: {}},
    ("76d63226", "answer_bbdc7b0a"): {2: {}},
    ("0862e8bf", "answer_c6fd8ebd"): {5: {}},
    ("6e984301", "answer_88841f26_1"): {0: {}},
    ("6e984301", "answer_88841f26_2"): {1: {}},
    ("b29f3365", "answer_436d4309_1"): {
        0: {"relevance": 2, "roles": ["bridge"]},
        3: {"relevance": 2, "roles": ["bridge"]},
    },
    ("b29f3365", "answer_436d4309_2"): {0: {}},
    ("19b5f2b3_abs", "answer_5ff494b9_abs"): {1: {}},
    ("19b5f2b3_abs", "e44a5733_1"): {5: {}},
    ("eeda8a6d_abs", "answer_3e5fea0e_abs_1"): {0: {}},
    ("eeda8a6d_abs", "answer_3e5fea0e_abs_2"): {0: {}},
}

# 拒答题没有 has_answer 标记，官方 Session 需人工定位到支持拒答的 Pair。
ABSTENTION_OFFICIAL_PAIRS: dict[tuple[str, str], list[int]] = {
    ("19b5f2b3_abs", "answer_5ff494b9_abs"): [1],
    ("eeda8a6d_abs", "answer_3e5fea0e_abs_1"): [0],
    ("eeda8a6d_abs", "answer_3e5fea0e_abs_2"): [0],
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def pair_turns(session: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """将 user 与紧随其后的一个 assistant 配对，其他 Turn 单独保留。"""
    pairs: list[list[dict[str, Any]]] = []
    pending_user: dict[str, Any] | None = None
    for turn in session:
        role = str(turn.get("role", "unknown")).lower()
        if role == "user":
            if pending_user is not None:
                pairs.append([pending_user])
            pending_user = turn
        elif role == "assistant" and pending_user is not None:
            pairs.append([pending_user, turn])
            pending_user = None
        else:
            if pending_user is not None:
                pairs.append([pending_user])
                pending_user = None
            pairs.append([turn])
    if pending_user is not None:
        pairs.append([pending_user])
    return pairs


def pair_text(date: str, turns: list[dict[str, Any]]) -> str:
    lines = [f"Date: {date}"]
    for turn in turns:
        role = str(turn.get("role", "unknown")).lower()
        label = {"user": "User", "assistant": "Assistant"}.get(role, "Other")
        lines.append(f"{label}: {str(turn.get('content', '')).strip()}")
    return "\n".join(lines)


def document_id(question_id: str, session_id: str, pair_index: int) -> str:
    source_key = f"{question_id}\0{session_id}\0{pair_index}"
    return f"lmep_{sha256_text(source_key)[:24]}"


def approximate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def build() -> dict[str, Any]:
    dataset = json.loads(SOURCE.read_text(encoding="utf-8"))
    pilot = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    pilot_queries = {row["question_id"]: row for row in pilot["queries"]}
    source_items = {row["question_id"]: row for row in dataset}

    documents: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    encoding_documents: list[dict[str, Any]] = []
    encoding_queries: list[dict[str, Any]] = []
    official_qrels: list[dict[str, Any]] = []
    pilot_qrels: list[dict[str, Any]] = []
    source_key_rows: list[dict[str, Any]] = []
    doc_lookup: dict[tuple[str, str, int], str] = {}

    for question_id in sorted(pilot_queries):
        item = source_items[question_id]
        query = {
            "query_id": question_id,
            "text": str(item["question"]).strip(),
            "question_type": item["question_type"],
            "question_date": item["question_date"],
            "is_abstention": question_id.endswith("_abs"),
        }
        queries.append(query)
        encoding_queries.append({"id": question_id, "text": query["text"]})

        for session_index, (session_id, date, session) in enumerate(
            zip(
                item["haystack_session_ids"],
                item["haystack_dates"],
                item["haystack_sessions"],
                strict=True,
            )
        ):
            session_hash = sha256_text(f"{question_id}\0{session_id}")
            for pair_index, turns in enumerate(pair_turns(session)):
                text = pair_text(str(date), turns)
                doc_id = document_id(question_id, str(session_id), pair_index)
                content_hash = sha256_text(text)
                doc_lookup[(question_id, str(session_id), pair_index)] = doc_id
                documents.append(
                    {
                        "document_id": doc_id,
                        "text": text,
                        "content_sha256": content_hash,
                        "question_id": question_id,
                        "question_type": item["question_type"],
                        "date": date,
                        "session_index": session_index,
                        "pair_index": pair_index,
                        "source_session_sha256": session_hash,
                        "turn_roles": [str(turn.get("role", "unknown")) for turn in turns],
                        "character_count": len(text),
                        "approximate_token_count": approximate_tokens(text),
                    }
                )
                encoding_documents.append({"id": doc_id, "text": text})
                source_key_rows.append(
                    {
                        "document_id": doc_id,
                        "question_id": question_id,
                        "session_id": session_id,
                        "pair_index": pair_index,
                        "content_sha256": content_hash,
                    }
                )

        official_ids = set(item["answer_session_ids"])
        for session_id, session in zip(
            item["haystack_session_ids"], item["haystack_sessions"], strict=True
        ):
            if session_id not in official_ids:
                continue
            pairs = pair_turns(session)
            if question_id.endswith("_abs"):
                selected = ABSTENTION_OFFICIAL_PAIRS[(question_id, str(session_id))]
            else:
                selected = [
                    index
                    for index, turns in enumerate(pairs)
                    if any(bool(turn.get("has_answer")) for turn in turns)
                ]
            if not selected:
                raise ValueError(f"官方证据未定位到 Pair：{question_id}/{session_id}")
            for pair_index in selected:
                official_qrels.append(
                    {
                        "query_id": question_id,
                        "document_id": doc_lookup[(question_id, str(session_id), pair_index)],
                        "relevance": 1,
                        "provenance": "longmemeval_official",
                    }
                )

        annotations = {
            row["session_id"]: row
            for row in pilot_queries[question_id]["annotations"]
            if row["relevance"] != 0
        }
        expected = {(question_id, session_id) for session_id in annotations}
        actual = {key for key in PILOT_PAIR_LABELS if key[0] == question_id}
        if expected != actual:
            raise ValueError(f"Pilot Pair 定位不完整：{question_id}")
        for (_, session_id), pair_labels in sorted(PILOT_PAIR_LABELS.items()):
            if _ != question_id:
                continue
            session_label = annotations[session_id]
            for pair_index, overrides in pair_labels.items():
                pilot_qrels.append(
                    {
                        "query_id": question_id,
                        "document_id": doc_lookup[(question_id, session_id, pair_index)],
                        "relevance": overrides.get("relevance", session_label["relevance"]),
                        "roles": overrides.get("roles", session_label["roles"]),
                        "support_strength": session_label["support_strength"],
                        "rationale_zh": session_label["rationale_zh"],
                        "provenance": "pilot_manual_turn_pair",
                    }
                )

    documents.sort(key=lambda row: (row["question_id"], row["session_index"], row["pair_index"]))
    queries.sort(key=lambda row: row["query_id"])
    encoding_documents.sort(key=lambda row: row["id"])
    encoding_queries.sort(key=lambda row: row["id"])
    official_qrels.sort(key=lambda row: (row["query_id"], row["document_id"]))
    pilot_qrels.sort(key=lambda row: (row["query_id"], row["document_id"]))
    source_key_rows.sort(key=lambda row: row["document_id"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "documents.jsonl": documents,
        "queries.jsonl": queries,
        "encoding_documents.jsonl": encoding_documents,
        "encoding_queries.jsonl": encoding_queries,
        "official_qrels.jsonl": official_qrels,
        "pilot_qrels.jsonl": pilot_qrels,
        "source_key.jsonl": source_key_rows,
    }
    for name, rows in outputs.items():
        write_jsonl(OUTPUT_DIR / name, rows)

    role_patterns = Counter(tuple(row["turn_roles"]) for row in documents)
    lengths = sorted(row["approximate_token_count"] for row in documents)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "pilot_annotations_sha256": hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest(),
        "query_count": len(queries),
        "document_count": len(documents),
        "official_qrel_count": len(official_qrels),
        "pilot_qrel_count": len(pilot_qrels),
        "pairing_policy": "每个 user Turn 与紧随其后的一个 assistant Turn 组成文档；连续 user、开头 assistant、结尾 user 和非标准角色不会丢失，必要时形成单独文档。",
        "text_policy": "待编码文档仅含日期、User/Assistant/Other 原文；待编码 Query 仅含问题文本。",
        "token_estimate_policy": "按 UTF-8 文本字符数除以 4 向上取整，仅用于调用 tokenizer 前的截断风险粗筛。",
        "approximate_tokens": {
            "min": lengths[0],
            "median": lengths[len(lengths) // 2],
            "p95": lengths[int((len(lengths) - 1) * 0.95)],
            "max": lengths[-1],
            "over_8192": sum(length > 8192 for length in lengths),
        },
        "turn_role_patterns": {"+".join(key): value for key, value in sorted(role_patterns.items())},
        "sensitive_files": ["source_key.jsonl", "official_qrels.jsonl", "pilot_qrels.jsonl"],
        "encoding_files": ["encoding_documents.jsonl", "encoding_queries.jsonl"],
        "outputs_sha256": {
            name: hashlib.sha256((OUTPUT_DIR / name).read_bytes()).hexdigest()
            for name in sorted(outputs)
        },
    }
    (OUTPUT_DIR / "encoding_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    print(json.dumps(build(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
