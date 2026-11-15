import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts" / "evaluations" / "fixed_metrics" / "fixed_metric_report.json"
OUTPUT = ROOT / "artifacts" / "evaluations" / "fixed_metrics" / "analysis.json"
KEY_METRICS = (
    ("official", "mrr"),
    ("official", "recall@10"),
    ("official", "evidence_set_recall@10"),
    ("pilot", "ndcg@10"),
    ("pilot", "positive_recall@10"),
)
COMPARISONS = (
    ("cosine", "centered_cosine"),
    ("cosine", "diagonal_whitened_cosine"),
    ("diagonal_whitened_radial_l2", "diagonal_whitened_radial_lorentz_c0.1"),
    ("diagonal_whitened_radial_l2", "diagonal_whitened_radial_lorentz_c1"),
    ("diagonal_whitened_radial_l2", "diagonal_whitened_radial_lorentz_c5"),
)


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    differences = differences[np.abs(differences) > 1e-15]
    if differences.size == 0:
        return 1.0
    observed = abs(float(differences.mean()))
    extreme = 0
    total = 2 ** differences.size
    for signs in itertools.product((-1.0, 1.0), repeat=differences.size):
        statistic = abs(float(np.mean(differences * np.asarray(signs))))
        extreme += statistic >= observed - 1e-15
    return extreme / total


def compare(
    methods: dict[str, dict[str, Any]], before: str, after: str, family: str, metric: str
) -> dict[str, Any]:
    query_ids = sorted(methods[before]["per_query"])
    differences = np.asarray(
        [
            float(methods[after]["per_query"][query_id][family][metric])
            - float(methods[before]["per_query"][query_id][family][metric])
            for query_id in query_ids
        ]
    )
    return {
        "before": before,
        "after": after,
        "metric": f"{family}.{metric}",
        "mean_delta": float(differences.mean()),
        "median_delta": float(np.median(differences)),
        "improved_queries": int(np.sum(differences > 1e-15)),
        "unchanged_queries": int(np.sum(np.abs(differences) <= 1e-15)),
        "degraded_queries": int(np.sum(differences < -1e-15)),
        "exact_sign_flip_p_two_sided": exact_sign_flip_pvalue(differences),
        "per_query_delta": {
            query_id: float(delta) for query_id, delta in zip(query_ids, differences, strict=True)
        },
    }


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    analysis: dict[str, Any] = {
        "notice": (
            "双侧精确符号翻转检验仅用于小样本诊断；未做多重比较校正，"
            "且 Whitening 使用当前完整语料统计，因此不能作为跨语料泛化显著性结论。"
        ),
        "variants": {},
    }
    for variant, variant_data in report["query_variants"].items():
        methods = {
            method_name: {
                **method_summary,
                "per_query": variant_data["per_query"][method_name],
            }
            for method_name, method_summary in variant_data["methods"].items()
        }
        comparisons = [
            compare(methods, before, after, family, metric)
            for before, after in COMPARISONS
            for family, metric in KEY_METRICS
        ]
        analysis["variants"][variant] = {
            "equivalence_checks": variant_data["equivalence_checks"],
            "comparisons": comparisons,
        }
    OUTPUT.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    compact = {}
    for variant, data in analysis["variants"].items():
        compact[variant] = [
            {
                "comparison": f"{row['before']} -> {row['after']}",
                "metric": row["metric"],
                "mean_delta": row["mean_delta"],
                "wins_ties_losses": [
                    row["improved_queries"],
                    row["unchanged_queries"],
                    row["degraded_queries"],
                ],
                "p": row["exact_sign_flip_p_two_sided"],
            }
            for row in data["comparisons"]
            if row["metric"] in {"official.mrr", "pilot.ndcg@10"}
        ]
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
