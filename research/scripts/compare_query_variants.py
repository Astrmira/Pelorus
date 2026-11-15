import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "artifacts" / "evaluations" / "cosine_baseline"
INSTRUCTED_DIR = ROOT / "artifacts" / "evaluations" / "cosine_instructed_query"
OUTPUT = ROOT / "artifacts" / "evaluations" / "query_variant_comparison.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(after[key]) - float(value)
        for key, value in before.items()
        if key in after and isinstance(value, (int, float)) and isinstance(after[key], (int, float))
    }


def nullable_rank_delta(before: Any, after: Any) -> int | None:
    if before is None or after is None:
        return None
    return int(after) - int(before)


def main() -> None:
    raw_summary = load(RAW_DIR / "summary.json")
    instructed_summary = load(INSTRUCTED_DIR / "summary.json")
    raw_queries = {row["query_id"]: row for row in load(RAW_DIR / "per_query.json")}
    instructed_queries = {row["query_id"]: row for row in load(INSTRUCTED_DIR / "per_query.json")}

    if raw_summary["input_sha256"]["document_vectors"] != instructed_summary["input_sha256"]["document_vectors"]:
        raise ValueError("两组实验使用了不同文档向量，不能直接比较")
    if set(raw_queries) != set(instructed_queries):
        raise ValueError("两组实验的 Query 集合不一致")

    per_query = []
    for query_id in sorted(raw_queries):
        before = raw_queries[query_id]
        after = instructed_queries[query_id]
        raw_official = before["official_metrics"]
        instructed_official = after["official_metrics"]
        raw_pilot = before["pilot_metrics"]
        instructed_pilot = after["pilot_metrics"]
        per_query.append(
            {
                "query_id": query_id,
                "question_type": before["question_type"],
                "question": before["text"],
                "is_abstention": before["is_abstention"],
                "official_first_evidence_rank": {
                    "raw": raw_official["first_evidence_rank"],
                    "instructed": instructed_official["first_evidence_rank"],
                    "delta": nullable_rank_delta(
                        raw_official["first_evidence_rank"], instructed_official["first_evidence_rank"]
                    ),
                },
                "official_all_evidence_rank": {
                    "raw": raw_official["all_evidence_rank"],
                    "instructed": instructed_official["all_evidence_rank"],
                    "delta": nullable_rank_delta(
                        raw_official["all_evidence_rank"], instructed_official["all_evidence_rank"]
                    ),
                },
                "pilot_first_positive_rank": {
                    "raw": raw_pilot["first_positive_rank"],
                    "instructed": instructed_pilot["first_positive_rank"],
                    "delta": nullable_rank_delta(
                        raw_pilot["first_positive_rank"], instructed_pilot["first_positive_rank"]
                    ),
                },
                "pilot_first_misleading_rank": {
                    "raw": raw_pilot["first_misleading_rank"],
                    "instructed": instructed_pilot["first_misleading_rank"],
                    "delta": nullable_rank_delta(
                        raw_pilot["first_misleading_rank"], instructed_pilot["first_misleading_rank"]
                    ),
                },
                "official_metric_deltas": numeric_deltas(raw_official, instructed_official),
                "pilot_metric_deltas": numeric_deltas(raw_pilot, instructed_pilot),
            }
        )

    report = {
        "comparison": "raw_query_vs_instructed_query_v1",
        "fixed_document_vectors_sha256": raw_summary["input_sha256"]["document_vectors"],
        "raw_query_vectors_sha256": raw_summary["input_sha256"]["query_vectors"],
        "instructed_query_vectors_sha256": instructed_summary["input_sha256"]["query_vectors"],
        "official_deltas": numeric_deltas(raw_summary["official"], instructed_summary["official"]),
        "pilot_deltas": numeric_deltas(raw_summary["pilot"], instructed_summary["pilot"]),
        "per_query": per_query,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
