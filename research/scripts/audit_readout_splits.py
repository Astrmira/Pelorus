"""核验已缓存候选的来源，生成保守的训练内容隔离清单。"""

import hashlib
import json
from collections import defaultdict

import run_low_capacity_experiment as original
from prepare_longmemeval_encoding_dataset import document_id, pair_text, pair_turns


OUTPUT = original.ROOT / "artifacts" / "experiments" / "methodology_retest_v4"
SOURCE = original.ROOT / "data" / "raw" / "longmemeval_s_cleaned.json"


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_fingerprint(turns):
    # 不使用日期、Session ID 或 has_answer，以识别跨题复用的原文。
    return digest(json.dumps([(str(turn.get("role", "")).lower(), " ".join(str(turn.get("content", "")).split()))
                              for turn in turns], ensure_ascii=False))


def audit():
    metadata = {name: original.read_jsonl(original.DATA_DIR / f"{name}_metadata.jsonl") for name in ("train", "validation")}
    dev_queries = original.read_jsonl(original.TEST_DATA_DIR / "queries.jsonl")
    split_queries = {name: {row["query_id"] for row in rows} for name, rows in metadata.items()}
    split_queries["development"] = {row["query_id"] for row in dev_queries}
    wanted = {name: {value for row in rows for value in row["candidate_ids"]} for name, rows in metadata.items()}
    wanted["development"] = {row["document_id"] for row in original.read_jsonl(original.TEST_DATA_DIR / "documents.jsonl")}
    sources, by_split = {}, {name: {} for name in wanted}
    question_fingerprints = defaultdict(set)
    for item in json.loads(SOURCE.read_text(encoding="utf-8")):
        qid = str(item["question_id"])
        split = next((name for name, ids in split_queries.items() if qid in ids), None)
        if split is None:
            continue
        question_fingerprints[digest(" ".join(item["question"].lower().split()))].add((split, qid))
        for session_id, date, turns in zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"], strict=True):
            session_hash = content_fingerprint(turns)
            for pair_index, pair in enumerate(pair_turns(turns)):
                doc_id = document_id(qid, str(session_id), pair_index)
                if doc_id not in wanted[split]:
                    continue
                row = {"query_id": qid, "session_hash": session_hash, "pair_hash": content_fingerprint(pair),
                       "text_hash": digest(pair_text(str(date), pair)),
                       "has_answer": any(bool(turn.get("has_answer")) for turn in pair)}
                sources[doc_id] = row
                by_split[split][doc_id] = row
    for split in wanted:
        if set(by_split[split]) != wanted[split]:
            raise ValueError(f"部分候选无法映射回原始数据：{split}")
        cached_texts = original.read_jsonl(original.DATA_DIR / f"{split}_documents.jsonl") if split != "development" else original.read_jsonl(original.TEST_DATA_DIR / "documents.jsonl")
        for row in cached_texts:
            doc_id = row.get("id", row.get("document_id"))
            if digest(row["text"]) != sources[doc_id]["text_hash"]:
                raise ValueError("源数据重建文本与缓存输入不一致")
    overlaps = {}
    for a, b in (("train", "validation"), ("train", "development"), ("validation", "development")):
        row = {}
        for key in ("session_hash", "pair_hash"):
            common = {r[key] for r in by_split[a].values()} & {r[key] for r in by_split[b].values()}
            row[key] = {"shared_hash_count": len(common),
                        "left_document_count": sum(r[key] in common for r in by_split[a].values()),
                        "right_document_count": sum(r[key] in common for r in by_split[b].values()),
                        "left_positive_count": sum(r[key] in common and r["has_answer"] for r in by_split[a].values()),
                        "right_positive_count": sum(r[key] in common and r["has_answer"] for r in by_split[b].values())}
        overlaps[f"{a}/{b}"] = row
    blocked_sessions = {row["session_hash"] for name in ("validation", "development") for row in by_split[name].values()}
    blocked_pairs = {row["pair_hash"] for name in ("validation", "development") for row in by_split[name].values()}
    blocked_questions = {qid for entries in question_fingerprints.values() if len({name for name, _ in entries}) > 1
                         for name, qid in entries if name == "train"}
    kept, excluded, removed_candidates = [], [], []
    for row in metadata["train"]:
        blocked = {doc_id for doc_id in row["candidate_ids"]
                   if sources[doc_id]["session_hash"] in blocked_sessions or sources[doc_id]["pair_hash"] in blocked_pairs}
        if blocked & set(row["positive_ids"]) or row["query_id"] in blocked_questions:
            excluded.append({"query_id": row["query_id"], "reason": "正证据或相同问题跨集合重复"})
            continue
        candidates = [doc_id for doc_id in row["candidate_ids"] if doc_id not in blocked]
        if len(candidates) <= len(row["positive_ids"]):
            excluded.append({"query_id": row["query_id"], "reason": "移除共享候选后没有负候选"})
            continue
        kept.append({**row, "candidate_ids": candidates})
        removed_candidates.extend(sorted(blocked))
    if len(kept) < 16:
        raise ValueError("内容隔离后训练问题不足16题，需要重新设计切分")
    contradictions = []
    for row in kept:
        positive_hashes = {sources[doc_id]["pair_hash"] for doc_id in row["positive_ids"]}
        duplicate_negatives = [doc_id for doc_id in row["candidate_ids"]
                              if doc_id not in row["positive_ids"] and sources[doc_id]["pair_hash"] in positive_hashes]
        if duplicate_negatives:
            contradictions.append({"query_id": row["query_id"], "removed_unmarked_duplicates": duplicate_negatives})
            row["candidate_ids"] = [doc_id for doc_id in row["candidate_ids"] if doc_id not in duplicate_negatives]
    validation_kept, validation_excluded, validation_removed = [], [], []
    dev_sessions = {row["session_hash"] for row in by_split["development"].values()}
    dev_pairs = {row["pair_hash"] for row in by_split["development"].values()}
    for row in metadata["validation"]:
        blocked = {doc_id for doc_id in row["candidate_ids"] if sources[doc_id]["session_hash"] in dev_sessions or sources[doc_id]["pair_hash"] in dev_pairs}
        if blocked & set(row["positive_ids"]):
            validation_excluded.append(row["query_id"])
            continue
        candidates = [doc_id for doc_id in row["candidate_ids"] if doc_id not in blocked]
        validation_kept.append({**row, "candidate_ids": candidates})
        validation_removed.extend(sorted(blocked))
    post = {}
    retained = {"train": {doc_id for row in kept for doc_id in row["candidate_ids"]},
                "validation": {doc_id for row in validation_kept for doc_id in row["candidate_ids"]},
                "development": wanted["development"]}
    for a, b in (("train", "validation"), ("train", "development"), ("validation", "development")):
        post[f"{a}/{b}"] = {key: len({sources[doc_id][key] for doc_id in retained[a]} & {sources[doc_id][key] for doc_id in retained[b]})
                              for key in ("session_hash", "pair_hash")}
        if any(post[f"{a}/{b}"].values()):
            raise ValueError("清洗后仍有内容重叠")
    result = {
        "policy": "训练候选若与验证/开发的已缓存候选共享完整Session或标准化Pair原文，则移除；涉及训练正例则排除整题。只读文本身份，不使用留出集相关性来清洗。",
        "source_sha256": original.sha256_file(SOURCE), "overlaps_before": overlaps,
        "train_queries_before": len(metadata["train"]), "train_queries_after": len(kept),
        "train_candidates_after": sum(len(row["candidate_ids"]) for row in kept),
        "validation_queries_after": len(validation_kept),
        "validation_candidates_after": sum(len(row["candidate_ids"]) for row in validation_kept),
        "validation_excluded_queries": validation_excluded,
        "validation_removed_shared_negative_candidates": validation_removed,
        "validation_metadata": validation_kept, "overlaps_after": post,
        "excluded_queries": excluded, "removed_shared_negative_candidates": removed_candidates,
        "within_query_duplicate_label_conflicts": contradictions,
        "limitations": "仅核验精确/空白标准化重复，不等于主题或事实链隔离；验证仅有32候选缓存，未缓存文本仍可能共享。语义等价但措辞不同的假负例尚未人工穷举审计。",
        "training_metadata": kept,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "split_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("overlaps_before", "train_queries_after", "train_candidates_after", "excluded_queries")}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    audit()
