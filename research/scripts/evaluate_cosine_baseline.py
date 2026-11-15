import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "longmemeval_turn_pairs"
PILOT_ANNOTATIONS = ROOT / "data" / "processed" / "longmemeval_pilot" / "pilot_annotations.json"
EMBEDDING_DIR = ROOT / "artifacts" / "embeddings" / "longmemeval_turn_pairs"
OUTPUT_DIR = ROOT / "artifacts" / "evaluations" / "cosine_baseline"
TOP_K = (1, 5, 10, 20)
DIAGNOSTIC_TOP_K = 20


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_qrels(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["query_id"])][str(row["document_id"])] = row
    return dict(grouped)


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    for rank, document_id in enumerate(ranking, 1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranking: list[str], relevance: dict[str, int], k: int) -> float:
    return sum(
        (2 ** max(0, relevance.get(document_id, 0)) - 1) / math.log2(rank + 1)
        for rank, document_id in enumerate(ranking[:k], 1)
    )


def ndcg_at_k(ranking: list[str], relevance: dict[str, int], k: int) -> float:
    ideal = sorted((max(0, value) for value in relevance.values()), reverse=True)
    ideal_dcg = sum((2**value - 1) / math.log2(rank + 1) for rank, value in enumerate(ideal[:k], 1))
    return dcg_at_k(ranking, relevance, k) / ideal_dcg if ideal_dcg else 0.0


def first_rank(ranking: list[str], targets: set[str]) -> int | None:
    for rank, document_id in enumerate(ranking, 1):
        if document_id in targets:
            return rank
    return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def exact_rankings(
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    document_ids: list[str],
    query_ids: list[str],
    document_vectors: np.ndarray,
    query_vectors: np.ndarray,
) -> dict[str, list[tuple[str, float]]]:
    document_index = {document_id: index for index, document_id in enumerate(document_ids)}
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    documents_by_query: dict[str, list[str]] = defaultdict(list)
    for document in documents:
        documents_by_query[str(document["question_id"])].append(str(document["document_id"]))

    rankings: dict[str, list[tuple[str, float]]] = {}
    for query in queries:
        query_id = str(query["query_id"])
        candidates = documents_by_query[query_id]
        candidate_indices = np.asarray([document_index[document_id] for document_id in candidates])
        scores = document_vectors[candidate_indices] @ query_vectors[query_index[query_id]]
        order = np.argsort(-scores, kind="stable")
        rankings[query_id] = [
            (candidates[int(position)], float(scores[int(position)])) for position in order
        ]
    return rankings


def evaluate_official(
    rankings: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    per_query: dict[str, dict[str, Any]] = {}
    for query_id, scored in rankings.items():
        ranking = [document_id for document_id, _ in scored]
        relevant = set(qrels[query_id])
        row: dict[str, Any] = {
            "evidence_count": len(relevant),
            "mrr": reciprocal_rank(ranking, relevant),
            "first_evidence_rank": first_rank(ranking, relevant),
            "all_evidence_rank": max(ranking.index(document_id) + 1 for document_id in relevant),
        }
        for k in TOP_K:
            row[f"recall@{k}"] = recall_at_k(ranking, relevant, k)
            row[f"hit@{k}"] = float(bool(set(ranking[:k]) & relevant))
            row[f"evidence_set_recall@{k}"] = float(relevant <= set(ranking[:k]))
        per_query[query_id] = row

    summary = {
        key: mean([float(row[key]) for row in per_query.values()])
        for key in ["mrr"]
        + [metric for k in TOP_K for metric in (f"recall@{k}", f"hit@{k}", f"evidence_set_recall@{k}")]
    }
    return summary, per_query


def evaluate_pilot(
    rankings: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, dict[str, Any]]],
    query_metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    per_query: dict[str, dict[str, Any]] = {}
    for query_id, scored in rankings.items():
        ranking = [document_id for document_id, _ in scored]
        labels = qrels[query_id]
        relevance = {document_id: int(row["relevance"]) for document_id, row in labels.items()}
        positive = {document_id for document_id, value in relevance.items() if value > 0}
        decisive = {document_id for document_id, value in relevance.items() if value == 3}
        misleading = {document_id for document_id, value in relevance.items() if value < 0}
        stale = {
            document_id
            for document_id, row in labels.items()
            if "stale" in row.get("roles", [])
        }
        abstention_support = {
            document_id
            for document_id, row in labels.items()
            if "abstention_support" in row.get("roles", [])
        }
        row: dict[str, Any] = {
            "positive_count": len(positive),
            "misleading_count": len(misleading),
            "first_positive_rank": first_rank(ranking, positive),
            "first_decisive_rank": first_rank(ranking, decisive),
            "first_misleading_rank": first_rank(ranking, misleading),
        }
        for k in TOP_K:
            row[f"ndcg@{k}"] = ndcg_at_k(ranking, relevance, k)
            row[f"positive_recall@{k}"] = recall_at_k(ranking, positive, k)
            row[f"positive_set_recall@{k}"] = float(positive <= set(ranking[:k]))
            row[f"decisive_hit@{k}"] = float(bool(decisive & set(ranking[:k]))) if decisive else None
            row[f"misleading_hit@{k}"] = float(bool(misleading & set(ranking[:k]))) if misleading else None
            row[f"stale_hit@{k}"] = float(bool(stale & set(ranking[:k]))) if stale else None
            row[f"abstention_support_recall@{k}"] = (
                recall_at_k(ranking, abstention_support, k) if abstention_support else None
            )
        per_query[query_id] = row

    summary: dict[str, float] = {}
    always = [metric for k in TOP_K for metric in (f"ndcg@{k}", f"positive_recall@{k}", f"positive_set_recall@{k}")]
    conditional = [
        metric
        for k in TOP_K
        for metric in (
            f"decisive_hit@{k}",
            f"misleading_hit@{k}",
            f"stale_hit@{k}",
            f"abstention_support_recall@{k}",
        )
    ]
    for key in always + conditional:
        values = [float(row[key]) for row in per_query.values() if row.get(key) is not None]
        summary[key] = mean(values)
    summary["abstention_query_count"] = float(sum(bool(row["is_abstention"]) for row in query_metadata.values()))
    return summary, per_query


def excerpt(text: str, limit: int = 500) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[:limit] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description="评估固定文档向量上的精确 Cosine 检索。")
    parser.add_argument("--query-vectors", default="queries.npy")
    parser.add_argument("--query-ids", default="query_ids.json")
    parser.add_argument("--output-name", default="cosine_baseline")
    parser.add_argument("--variant", default="raw_query")
    args = parser.parse_args()

    documents_path = DATA_DIR / "documents.jsonl"
    queries_path = DATA_DIR / "queries.jsonl"
    official_path = DATA_DIR / "official_qrels.jsonl"
    pilot_path = DATA_DIR / "pilot_qrels.jsonl"
    document_vectors_path = EMBEDDING_DIR / "documents.npy"
    query_vectors_path = EMBEDDING_DIR / args.query_vectors
    document_ids_path = EMBEDDING_DIR / "document_ids.json"
    query_ids_path = EMBEDDING_DIR / args.query_ids
    output_dir = ROOT / "artifacts" / "evaluations" / args.output_name

    documents = read_jsonl(documents_path)
    queries = read_jsonl(queries_path)
    official_qrels = group_qrels(read_jsonl(official_path))
    pilot_qrels = group_qrels(read_jsonl(pilot_path))
    document_ids = json.loads(document_ids_path.read_text(encoding="utf-8"))
    query_ids = json.loads(query_ids_path.read_text(encoding="utf-8"))
    document_vectors = np.load(document_vectors_path)
    query_vectors = np.load(query_vectors_path)

    if document_vectors.shape[0] != len(document_ids) or query_vectors.shape[0] != len(query_ids):
        raise ValueError("向量数量与 ID 映射不一致")
    if document_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("文档与 Query 向量维度不一致")

    rankings = exact_rankings(
        documents, queries, document_ids, query_ids, document_vectors, query_vectors
    )
    query_metadata = {str(row["query_id"]): row for row in queries}
    document_metadata = {str(row["document_id"]): row for row in documents}
    pilot_source = json.loads(PILOT_ANNOTATIONS.read_text(encoding="utf-8"))
    reference_answers = {
        str(row["question_id"]): row.get("reference_answer") for row in pilot_source["queries"]
    }

    official_summary, official_per_query = evaluate_official(rankings, official_qrels)
    pilot_summary, pilot_per_query = evaluate_pilot(rankings, pilot_qrels, query_metadata)

    diagnostics = []
    for query_id in query_ids:
        official_labels = official_qrels[query_id]
        pilot_labels = pilot_qrels[query_id]
        top_rows = []
        for rank, (document_id, score) in enumerate(rankings[query_id][:DIAGNOSTIC_TOP_K], 1):
            document = document_metadata[document_id]
            pilot_label = pilot_labels.get(document_id)
            top_rows.append(
                {
                    "rank": rank,
                    "document_id": document_id,
                    "cosine": score,
                    "date": document["date"],
                    "session_index": document["session_index"],
                    "pair_index": document["pair_index"],
                    "is_official_evidence": document_id in official_labels,
                    "pilot_relevance": pilot_label["relevance"] if pilot_label else 0,
                    "pilot_roles": pilot_label["roles"] if pilot_label else ["none"],
                    "text_excerpt": excerpt(str(document["text"])),
                }
            )
        diagnostics.append(
            {
                **query_metadata[query_id],
                "reference_answer": reference_answers[query_id],
                "haystack_document_count": len(rankings[query_id]),
                "official_metrics": official_per_query[query_id],
                "pilot_metrics": pilot_per_query[query_id],
                "top_20": top_rows,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "metric": "exact_cosine",
        "query_variant": args.variant,
        "scope": "每个 Query 仅扫描其所属 LongMemEval Haystack 的全部 Turn-Pair 文档",
        "query_count": len(queries),
        "document_count": len(documents),
        "dimensions": int(document_vectors.shape[1]),
        "top_k": list(TOP_K),
        "official": official_summary,
        "pilot": pilot_summary,
        "notes": {
            "negative_labels": "Pilot relevance=-1 不计入 nDCG 增益，单独以 misleading/stale hit 指标报告。",
            "macro_average": "汇总指标均为 Query 级宏平均；条件指标只在具有对应标签的 Query 上平均。",
        },
        "input_sha256": {
            "documents": sha256_file(documents_path),
            "queries": sha256_file(queries_path),
            "official_qrels": sha256_file(official_path),
            "pilot_qrels": sha256_file(pilot_path),
            "document_vectors": sha256_file(document_vectors_path),
            "query_vectors": sha256_file(query_vectors_path),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "per_query.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
