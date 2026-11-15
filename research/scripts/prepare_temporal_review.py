"""生成自然时间状态事实链的阅读包，保留完整候选原文供标注核验。"""

import json
from pathlib import Path

from prepare_low_capacity_training_data import SOURCE
from prepare_longmemeval_encoding_dataset import pair_turns, pair_text, document_id
import run_low_capacity_experiment as old


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "processed" / "temporal_readout"


def main():
    used = set()
    for split in ("train", "validation"):
        used.update(row["query_id"] for row in old.read_jsonl(old.DATA_DIR / f"{split}_metadata.jsonl"))
    used.update(row["query_id"] for row in old.read_jsonl(old.TEST_DATA_DIR / "queries.jsonl"))
    rows = []
    for item in json.loads(SOURCE.read_text(encoding="utf-8")):
        if item["question_type"] != "knowledge-update" or str(item["question_id"]).endswith("_abs"):
            continue
        sessions = []
        for index, (sid, date, turns) in enumerate(zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"], strict=True)):
            if sid not in item["answer_session_ids"] and not any(turn.get("has_answer") for turn in turns):
                continue
            pairs = [{"document_id": document_id(str(item["question_id"]), str(sid), pi), "pair_index": pi,
                      "has_answer": any(turn.get("has_answer") for turn in pair), "text": pair_text(str(date), pair)}
                     for pi, pair in enumerate(pair_turns(turns))]
            sessions.append({"session_index": index, "session_id": sid, "date": date, "pairs": pairs})
        rows.append({"id": str(item["question_id"]), "question": item["question"], "answer": item["answer"],
                     "previously_used": str(item["question_id"]) in used, "sessions": sessions})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "review.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(json.dumps({key: row[key] for key in ("id", "question", "answer", "previously_used")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
