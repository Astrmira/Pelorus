"""冻结从未进入旧训练、验证或留出切分的新时间事实链终检。"""

import hashlib
import json
from pathlib import Path

from prepare_low_capacity_training_data import INSTRUCTION, SOURCE
from prepare_temporal_review import pair_text, pair_turns, document_id, OUTPUT as REVIEW_OUTPUT
import run_low_capacity_experiment as old


ROOT = old.ROOT
OUTPUT = ROOT / "data" / "processed" / "temporal_fresh_test"

# 这些来源题不在 temporal_readout 的36条事实链中；值均来自下方两条经核验用户陈述。
ANNOTATIONS = [
    ("9ea5eabc", "travel", "where I went on my family trip", "Hawaii", "Paris"),
    ("0f05491a", "rewards", "the number of stars needed for Starbucks Gold", "125 stars", "120 stars"),
    ("08e075c7", "health_tracking", "how long I had been using my Fitbit Charge 3", "6 months", "9 months"),
    ("6a27ffc2", "programming", "the number of Corey Schafer Python videos I had completed", "20 videos", "30 videos"),
    ("db467c8c", "family", "how long my parents had been staying with me in the US", "6 months", "9 months"),
    ("e61a7584", "pets", "how long I had had my cat Luna", "6 months", "9 months"),
    ("ba61f0b9", "work", "the number of women on Rachel's team", "5 women", "6 women"),
    ("0977f2af", "kitchen", "the kitchen gadget I had bought", "Instant Pot", "Air Fryer"),
]


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    review = {row["id"]: row for row in json.loads((REVIEW_OUTPUT / "review.json").read_text(encoding="utf-8"))}
    old_annotations = {row["id"] for row in json.loads((ROOT / "data/processed/temporal_readout/annotations.json").read_text(encoding="utf-8"))}
    source = {str(row["question_id"]): row for row in json.loads(SOURCE.read_text(encoding="utf-8"))}
    cases, documents, queries = [], [], []
    for qid, group, subject, early_value, late_value in ANNOTATIONS:
        if qid in old_annotations:
            raise ValueError(f"新终检题意外出现在旧事实链：{qid}")
        row = review[qid]
        marked = [(session, pair) for session in row["sessions"] for pair in session["pairs"] if pair["has_answer"]]
        if len(marked) != 2 or row["previously_used"]:
            raise ValueError(f"终检事实链必须恰好两条且未使用：{qid}")
        marked.sort(key=lambda item: item[0]["date"])
        evidence, state_docs = [], []
        for state, (session, pair), answer in zip(("earlier", "later"), marked, (early_value, late_value), strict=True):
            turns = source[qid]["haystack_sessions"][session["session_index"]]
            paired = pair_turns(turns)[pair["pair_index"]]
            users = [turn for turn in paired if turn["role"] == "user"]
            if len(users) != 1:
                raise ValueError(f"证据Pair必须只有一条用户陈述：{qid}")
            text = pair_text(session["date"], users)
            doc_id = "ftd_" + sha(qid + "\0" + text)[:24]
            documents.append({"id": doc_id, "text": text})
            state_docs.append(doc_id)
            evidence.append({"document_id": doc_id, "date": session["date"], "answer": answer, "state": state,
                             "support_text": text, "source_session_index": session["session_index"],
                             "source_pair_index": pair["pair_index"]})
        if evidence[0]["date"] >= evidence[1]["date"]:
            raise ValueError(f"事实链日期未严格递增：{qid}")
        case = {"id": qid, "group": group, "subject": subject, "candidate_ids": sorted(state_docs),
                "evidence": evidence, "queries": [], "split": "fresh_test"}
        # 与旧训练模板不同：先/后用自然语言表达，普通事实也改用新句式。
        for style in ("paraphrase", "ordinary"):
            for index, (state, value) in enumerate(zip(("earlier", "later"), (early_value, late_value), strict=True)):
                if style == "paraphrase":
                    text = (f"In the older of the two dated reports, what did I say about {subject}?"
                            if index == 0 else f"What did I say about {subject} in the newer dated report?")
                else:
                    text = f"Select the report that mentions {value} for {subject}."
                query_id = "ftq_" + sha(qid + style + state)[:24]
                queries.append({"id": query_id, "text": INSTRUCTION + text})
                case["queries"].append({"query_id": query_id, "style": style, "state": state,
                                        "target": state_docs[index], "text": text})
        cases.append(case)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT / "documents.jsonl", documents)
    write_jsonl(OUTPUT / "queries.jsonl", queries)
    (OUTPUT / "annotations.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "purpose": "在多模板训练与选择冻结后，评估从未进入旧事实链的独立时间读出终检",
        "source_sha256": old.sha256_file(SOURCE),
        "preparation_script_sha256": old.sha256_file(Path(__file__)),
        "chains": len(cases), "documents": len(documents), "queries": len(queries),
        "styles": ["paraphrase", "ordinary"],
        "labels": "每条链两条按来源日期严格递增的用户陈述；查询目标是指定早/晚状态或指定值的证据",
        "limitations": ["仍来自LongMemEval构造对话，而非真实纵向用户日志", "标签由单一AI阅读并核验", "只有两候选证据，不能替代全库检索评估"],
        "input_sha256": {name: old.sha256_file(OUTPUT / f"{name}.jsonl") for name in ("documents", "queries")},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"事实链": len(cases), "文档": len(documents), "查询": len(queries), "输出": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
