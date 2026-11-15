"""在不改动已冻结选择的前提下评估独立时间事实链。"""

import json
from pathlib import Path

import numpy as np
import torch

import run_low_capacity_experiment as old
import run_temporal_readout as base
import run_temporal_readout_v2 as v2


DATA = old.ROOT / "data" / "processed" / "temporal_fresh_test"
EMBEDDINGS = old.ROOT / "artifacts" / "embeddings" / "temporal_fresh_test"
OUTPUT = old.ROOT / "artifacts" / "experiments" / "temporal_fresh_test_v1"


def load_fresh():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["input_sha256"].items():
        if old.sha256_file(DATA / f"{name}.jsonl") != expected:
            raise ValueError("独立终检输入清单不一致")
    runtime = json.loads((EMBEDDINGS / "runtime_manifest.json").read_text(encoding="utf-8"))
    if runtime["dimensions"] != 1536 or runtime["request_model"] != "Qwen3-Embedding-8B":
        raise ValueError("独立终检向量模型或维度不一致")
    for name in ("documents", "queries"):
        if runtime["input_sha256"][name] != old.sha256_file(DATA / f"{name}.jsonl"):
            raise ValueError("独立终检向量与文本版本不一致")
    cases = json.loads((DATA / "annotations.json").read_text(encoding="utf-8"))
    document_ids = json.loads((EMBEDDINGS / "document_ids.json").read_text(encoding="utf-8"))
    query_ids = json.loads((EMBEDDINGS / "query_ids.json").read_text(encoding="utf-8"))
    documents = np.load(EMBEDDINGS / "documents.npy")
    queries = np.load(EMBEDDINGS / "queries.npy")
    di, qi = {value: i for i, value in enumerate(document_ids)}, {value: i for i, value in enumerate(query_ids)}
    mean = np.load(v2.OUTPUT / "whitening_mean.npy")
    std = np.load(v2.OUTPUT / "whitening_std.npy")
    normalized = lambda x: torch.from_numpy(x / np.linalg.norm(x, axis=1, keepdims=True))
    corpus = {"raw_documents": normalized(documents), "raw_queries": normalized(queries),
              "documents": normalized(old.apply_whitening(documents, mean, std)),
              "queries": normalized(old.apply_whitening(queries, mean, std))}
    for case in cases:
        case["document_indices"] = [di[doc_id] for doc_id in case["candidate_ids"]]
        for query in case["queries"]:
            query["query_index"] = qi[query["query_id"]]
            query["target_index"] = case["candidate_ids"].index(query["target"])
    return cases, corpus


def make_model(name, corpus):
    return base.Probe(name, corpus["documents"].shape[1])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cases, corpus = load_fresh()
    frozen = json.loads((v2.OUTPUT / "results.json").read_text(encoding="utf-8"))
    registry = json.loads((v2.OUTPUT / "selection_registry.json").read_text(encoding="utf-8"))
    if frozen["selection_registry_sha256"] != old.sha256_file(v2.OUTPUT / "selection_registry.json"):
        raise ValueError("多模板选择记录在独立终检前发生变化")
    rows = []
    baselines = {
        "cosine_raw": make_model("cosine_raw", corpus),
        "cosine_whitened": make_model("cosine_whitened", corpus),
    }
    for name, model in baselines.items():
        result = {style: base.style_metrics(model, corpus, cases, style) for style in ("paraphrase", "ordinary")}
        rows.append({"name": name, "seed": None, "lr": None, "selected_name": name, "evaluation": result})
    for entry in registry:
        model_name = entry["selected_name"]
        model = baselines.get(model_name)
        if model is None:
            model = make_model(entry["name"], corpus)
            state = torch.load(v2.OUTPUT / entry["checkpoint"], map_location="cpu", weights_only=True)
            model.load_state_dict(state["selected"])
        result = {style: base.style_metrics(model, corpus, cases, style) for style in ("paraphrase", "ordinary")}
        rows.append({**entry, "evaluation": result})
    base.save_json(OUTPUT / "results.json", {"protocol": {"source": "temporal_fresh_test", "selection_registry_sha256": frozen["selection_registry_sha256"],
                                                        "styles": ["paraphrase", "ordinary"], "chains": len(cases)}, "runs": rows})
    lines = ["# 独立时间事实链终检", "", "本报告只读取多模板训练的冻结选择记录；8条新事实链从未进入旧训练、验证或留出切分。没有根据本终检指标挑选模型。", "", "| 读出 | 学习率 | 种子 | 选中模型 | 新paraphrase成对 | 新ordinary单问 |", "|---|---:|---:|---|---:|---:|"]
    for row in rows:
        p = row["evaluation"]["paraphrase"]["summary"]["paired_accuracy"]
        o = row["evaluation"]["ordinary"]["summary"]["single_accuracy"]
        lines.append(f"| {row['name']} | {row.get('lr') if row.get('lr') is not None else '—'} | {row.get('seed') if row.get('seed') is not None else '—'} | {row['selected_name']} | {p:.1%} | {o:.1%} |")
    for name in v2.MODEL_NAMES:
        group = [row for row in rows if row.get("name") == name]
        if group:
            p = np.mean([row["evaluation"]["paraphrase"]["summary"]["paired_accuracy"] for row in group])
            o = np.mean([row["evaluation"]["ordinary"]["summary"]["single_accuracy"] for row in group])
            lines += ["", f"{name} 的三种子/两档学习率运行平均：新 paraphrase 成对 {p:.1%}，新 ordinary 单问 {o:.1%}。"]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("独立终检评估完成。", flush=True)


if __name__ == "__main__":
    main()
