import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from evaluate_cosine_baseline import evaluate_official, evaluate_pilot, group_qrels, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "longmemeval_turn_pairs"
EMBEDDING_DIR = ROOT / "artifacts" / "embeddings" / "longmemeval_turn_pairs"
OUTPUT_DIR = ROOT / "artifacts" / "evaluations" / "fixed_metrics"
RADIUS_BUDGET = 3.0
CURVATURES = (0.1, 1.0, 5.0)
EPS = 1e-12


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= EPS):
        raise ValueError("变换产生了零向量，无法归一化")
    return rows / norms


def radial_squash(rows: np.ndarray, scale: float, radius_budget: float = RADIUS_BUDGET) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    directions = rows / np.maximum(norms, EPS)
    radii = radius_budget * np.tanh(norms / scale)
    return directions * radii


def negative_l2(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    return -np.linalg.norm(documents - query[None, :], axis=1)


def negative_lorentz_distance(
    query_tangent: np.ndarray,
    document_tangent: np.ndarray,
    curvature: float,
) -> np.ndarray:
    sqrt_c = math.sqrt(curvature)
    query_radius = np.linalg.norm(query_tangent)
    document_radii = np.linalg.norm(document_tangent, axis=1)
    query_direction = query_tangent / max(query_radius, EPS)
    document_directions = document_tangent / np.maximum(document_radii[:, None], EPS)
    cosines = np.clip(document_directions @ query_direction, -1.0, 1.0)
    alpha = (
        np.cosh(sqrt_c * query_radius) * np.cosh(sqrt_c * document_radii)
        - np.sinh(sqrt_c * query_radius) * np.sinh(sqrt_c * document_radii) * cosines
    )
    distances = np.arccosh(np.maximum(alpha, 1.0)) / sqrt_c
    return -distances


def rank_by_query_haystack(
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    document_ids: list[str],
    query_ids: list[str],
    document_vectors: np.ndarray,
    query_vectors: np.ndarray,
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
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
        indices = np.asarray([document_index[document_id] for document_id in candidates])
        scores = score_fn(query_vectors[query_index[query_id]], document_vectors[indices])
        order = np.argsort(-scores, kind="stable")
        rankings[query_id] = [
            (candidates[int(position)], float(scores[int(position)])) for position in order
        ]
    return rankings


def rank_ids(rankings: dict[str, list[tuple[str, float]]]) -> dict[str, list[str]]:
    return {
        query_id: [document_id for document_id, _ in ranking]
        for query_id, ranking in rankings.items()
    }


def ranking_agreement(
    left: dict[str, list[tuple[str, float]]],
    right: dict[str, list[tuple[str, float]]],
) -> dict[str, Any]:
    left_ids = rank_ids(left)
    right_ids = rank_ids(right)
    exact = {query_id: left_ids[query_id] == right_ids[query_id] for query_id in left_ids}
    return {
        "all_queries_exact": all(exact.values()),
        "exact_query_count": sum(exact.values()),
        "query_count": len(exact),
        "mismatched_queries": [query_id for query_id, same in exact.items() if not same],
    }


def evaluate_method(
    rankings: dict[str, list[tuple[str, float]]],
    official_qrels: dict[str, dict[str, dict[str, Any]]],
    pilot_qrels: dict[str, dict[str, dict[str, Any]]],
    query_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    official_summary, official_per_query = evaluate_official(rankings, official_qrels)
    pilot_summary, pilot_per_query = evaluate_pilot(rankings, pilot_qrels, query_metadata)
    return {
        "official": official_summary,
        "pilot": pilot_summary,
        "per_query": {
            query_id: {
                "official": official_per_query[query_id],
                "pilot": pilot_per_query[query_id],
            }
            for query_id in rankings
        },
    }


def main() -> None:
    documents_path = DATA_DIR / "documents.jsonl"
    queries_path = DATA_DIR / "queries.jsonl"
    official_path = DATA_DIR / "official_qrels.jsonl"
    pilot_path = DATA_DIR / "pilot_qrels.jsonl"
    document_vectors_path = EMBEDDING_DIR / "documents.npy"
    document_ids_path = EMBEDDING_DIR / "document_ids.json"

    documents = read_jsonl(documents_path)
    queries = read_jsonl(queries_path)
    official_qrels = group_qrels(read_jsonl(official_path))
    pilot_qrels = group_qrels(read_jsonl(pilot_path))
    query_metadata = {str(row["query_id"]): row for row in queries}
    document_ids = json.loads(document_ids_path.read_text(encoding="utf-8"))
    documents_raw = np.load(document_vectors_path).astype(np.float64)

    query_variants = {
        "raw_query": (
            np.load(EMBEDDING_DIR / "queries.npy").astype(np.float64),
            json.loads((EMBEDDING_DIR / "query_ids.json").read_text(encoding="utf-8")),
        ),
        "instructed_query_v1": (
            np.load(EMBEDDING_DIR / "queries_instructed.npy").astype(np.float64),
            json.loads((EMBEDDING_DIR / "query_ids_instructed.json").read_text(encoding="utf-8")),
        ),
    }

    mean = documents_raw.mean(axis=0)
    std = documents_raw.std(axis=0)
    safe_std = np.maximum(std, 1e-6)
    documents_centered = documents_raw - mean
    documents_whitened = documents_centered / safe_std
    radial_scale = float(np.median(np.linalg.norm(documents_whitened, axis=1)))
    documents_radial = radial_squash(documents_whitened, radial_scale)

    report: dict[str, Any] = {
        "protocol": {
            "supervision": "无监督固定参数探针；变换只根据全部文档向量估计，不读取 qrels。",
            "scope": "每个 Query 只在其所属 LongMemEval Haystack 内精确扫描。",
            "diagonal_whitening": "使用全局文档均值和逐维标准差；标准差下限为 1e-6。",
            "radial_mapping": "u(x)=direction(whiten(x))*3*tanh(norm(whiten(x))/document_median_norm)。",
            "curvatures": list(CURVATURES),
            "warning": "该探针是转导式语料统计实验，不能作为跨语料泛化结论。",
        },
        "corpus_statistics": {
            "document_count": len(documents),
            "dimensions": int(documents_raw.shape[1]),
            "whitened_norm_median": radial_scale,
            "radial_norm_min": float(np.linalg.norm(documents_radial, axis=1).min()),
            "radial_norm_max": float(np.linalg.norm(documents_radial, axis=1).max()),
        },
        "query_variants": {},
        "input_sha256": {
            "documents": sha256_file(documents_path),
            "queries": sha256_file(queries_path),
            "official_qrels": sha256_file(official_path),
            "pilot_qrels": sha256_file(pilot_path),
            "document_vectors": sha256_file(document_vectors_path),
        },
    }

    for variant_name, (queries_raw, query_ids) in query_variants.items():
        if list(query_ids) != [str(row["query_id"]) for row in queries]:
            raise ValueError(f"{variant_name} 的 Query ID 顺序不一致")
        queries_centered = queries_raw - mean
        queries_whitened = queries_centered / safe_std
        queries_radial = radial_squash(queries_whitened, radial_scale)

        methods: dict[str, tuple[np.ndarray, np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray]]] = {
            "cosine": (normalize(documents_raw), normalize(queries_raw), lambda q, d: d @ q),
            "normalized_l2": (normalize(documents_raw), normalize(queries_raw), negative_l2),
            "centered_cosine": (
                normalize(documents_centered),
                normalize(queries_centered),
                lambda q, d: d @ q,
            ),
            "diagonal_whitened_cosine": (
                normalize(documents_whitened),
                normalize(queries_whitened),
                lambda q, d: d @ q,
            ),
            "diagonal_whitened_radial_l2": (documents_radial, queries_radial, negative_l2),
        }
        for curvature in CURVATURES:
            methods[f"fixed_radius_lorentz_c{curvature:g}"] = (
                normalize(documents_raw),
                normalize(queries_raw),
                lambda q, d, c=curvature: negative_lorentz_distance(q, d, c),
            )
            methods[f"diagonal_whitened_radial_lorentz_c{curvature:g}"] = (
                documents_radial,
                queries_radial,
                lambda q, d, c=curvature: negative_lorentz_distance(q, d, c),
            )

        rankings: dict[str, dict[str, list[tuple[str, float]]]] = {}
        evaluations: dict[str, Any] = {}
        for method_name, (document_vectors, query_vectors, score_fn) in methods.items():
            method_rankings = rank_by_query_haystack(
                documents,
                queries,
                document_ids,
                query_ids,
                document_vectors,
                query_vectors,
                score_fn,
            )
            rankings[method_name] = method_rankings
            evaluations[method_name] = evaluate_method(
                method_rankings, official_qrels, pilot_qrels, query_metadata
            )

        equivalence = {
            "normalized_l2_vs_cosine": ranking_agreement(rankings["cosine"], rankings["normalized_l2"]),
        }
        for curvature in CURVATURES:
            name = f"fixed_radius_lorentz_c{curvature:g}"
            equivalence[f"{name}_vs_cosine"] = ranking_agreement(rankings["cosine"], rankings[name])

        baseline = evaluations["cosine"]
        summary_methods = {}
        for method_name, result in evaluations.items():
            summary_methods[method_name] = {
                "official": result["official"],
                "pilot": result["pilot"],
                "delta_vs_cosine": {
                    "official": {
                        key: float(value) - float(baseline["official"][key])
                        for key, value in result["official"].items()
                    },
                    "pilot": {
                        key: float(value) - float(baseline["pilot"][key])
                        for key, value in result["pilot"].items()
                    },
                },
            }

        report["query_variants"][variant_name] = {
            "methods": summary_methods,
            "equivalence_checks": equivalence,
            "per_query": {
                method_name: result["per_query"] for method_name, result in evaluations.items()
            },
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "fixed_metric_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    compact = {
        variant: {
            method: {
                "official_mrr": values["official"]["mrr"],
                "official_recall@10": values["official"]["recall@10"],
                "official_evidence_set_recall@10": values["official"]["evidence_set_recall@10"],
                "pilot_ndcg@10": values["pilot"]["ndcg@10"],
                "pilot_positive_recall@10": values["pilot"]["positive_recall@10"],
                "pilot_misleading_hit@10": values["pilot"]["misleading_hit@10"],
            }
            for method, values in variant_data["methods"].items()
        }
        for variant, variant_data in report["query_variants"].items()
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
