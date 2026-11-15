"""比较冻结单向量的时间条件读出；路径积分不冒充最短测地距离。"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

import run_low_capacity_experiment as old
from prepare_temporal_review import OUTPUT as DATA
from diagnose_readout import save_json


OUTPUT = old.ROOT / "artifacts" / "experiments" / "temporal_readout_v1"
EMBEDDINGS = old.ROOT / "artifacts" / "embeddings" / "temporal_readout"
SEEDS = (11, 23, 47)
MODEL_NAMES = ("bilinear", "shared_nonlinear", "pair_mlp_matched", "pair_mlp_stronger", "fixed_metric", "local_metric", "path_metric")
EPSILON = 1e-3
EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 5e-4
TIE_MARGIN = 1e-6


class Probe(nn.Module):
    def __init__(self, name, dimensions):
        super().__init__()
        self.name = name
        self.register_buffer("score_scale", torch.tensor(1.0))
        if name == "bilinear":
            self.query_map = nn.Linear(dimensions, 6, bias=False)
            self.document_map = nn.Linear(dimensions, 6, bias=False)
            with torch.no_grad():
                self.document_map.weight.copy_(self.query_map.weight)
        elif name == "shared_nonlinear":
            self.map = nn.Linear(dimensions, 12)
        elif name.startswith("pair_mlp"):
            hidden = 3 if name == "pair_mlp_matched" else 8
            self.map = nn.Sequential(nn.Linear(4 * dimensions, hidden), nn.SiLU(), nn.Linear(hidden, 1, bias=False))
        elif name in ("fixed_metric", "local_metric", "path_metric"):
            self.projections = nn.Parameter(torch.randn(4, 2, dimensions) / math.sqrt(2))
            if name == "fixed_metric":
                self.weights = nn.Parameter(torch.zeros(4))
            else:
                self.gate = nn.Linear(dimensions, 4)
                nn.init.zeros_(self.gate.weight)
                nn.init.zeros_(self.gate.bias)
            nodes, weights = np.polynomial.legendre.leggauss(5)
            self.register_buffer("nodes", torch.tensor((nodes + 1) / 2, dtype=torch.float32))
            self.register_buffer("quadrature_weights", torch.tensor(weights / 2, dtype=torch.float32))

    def local_energy(self, position, delta):
        projected = torch.einsum("...d,krd->...kr", delta, self.projections)
        branch_energy = torch.sum(projected ** 2, dim=-1)
        gate = self.weights.softmax(-1) if self.name == "fixed_metric" else self.gate(position).softmax(-1)
        return EPSILON * torch.sum(delta ** 2, dim=-1) + torch.sum(gate * branch_energy, dim=-1)

    def raw_score(self, query, documents):
        q = query.expand_as(documents)
        if self.name.startswith("cosine"):
            return torch.sum(q * documents, dim=-1)
        if self.name == "bilinear":
            return torch.sum(self.query_map(q) * self.document_map(documents), dim=-1)
        if self.name == "shared_nonlinear":
            return -(EPSILON * torch.sum((q - documents) ** 2, dim=-1)
                     + torch.sum((torch.tanh(self.map(q)) - torch.tanh(self.map(documents))) ** 2, dim=-1))
        if self.name.startswith("pair_mlp"):
            features = torch.cat((q, documents, torch.abs(q - documents), q * documents), dim=-1)
            return self.map(features).squeeze(-1)
        delta = documents - q
        if self.name != "path_metric":
            return -self.local_energy(q, delta)
        positions = q[:, None, :] + self.nodes[None, :, None] * delta[:, None, :]
        energies = self.local_energy(positions, delta[:, None, :].expand_as(positions))
        # 积分固定直线路径上的长度，不优化路径，因此不是测地距离求解器。
        lengths = (torch.sqrt(energies.clamp_min(1e-12)) * self.quadrature_weights).sum(-1)
        return -(lengths ** 2)

    def score(self, query, documents):
        return self.raw_score(query, documents) / self.score_scale


def load_dataset():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["input_sha256"].items():
        if old.sha256_file(DATA / name) != expected:
            raise ValueError("标注或输入与冻结清单不一致")
    runtime = json.loads((EMBEDDINGS / "runtime_manifest.json").read_text(encoding="utf-8"))
    if runtime["dimensions"] != 1536 or runtime["request_model"] != "Qwen3-Embedding-8B":
        raise ValueError("向量模型或维度与协议不一致")
    for name in ("documents", "queries"):
        if runtime["input_sha256"][name] != old.sha256_file(DATA / f"{name}.jsonl"):
            raise ValueError("向量缓存对应的文本版本不一致")
    cases = json.loads((DATA / "annotations.json").read_text(encoding="utf-8"))
    document_ids = json.loads((EMBEDDINGS / "document_ids.json").read_text(encoding="utf-8"))
    query_ids = json.loads((EMBEDDINGS / "query_ids.json").read_text(encoding="utf-8"))
    documents = np.load(EMBEDDINGS / "documents.npy")
    queries = np.load(EMBEDDINGS / "queries.npy")
    di, qi = {value: i for i, value in enumerate(document_ids)}, {value: i for i, value in enumerate(query_ids)}
    for values in (documents, queries):
        if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) == 0):
            raise ValueError("向量非有限或包含零向量")
    train_indices = [di[doc] for case in cases if case["split"] == "train" for doc in case["candidate_ids"]]
    mean, std = old.fit_whitening(documents[train_indices])
    normalized = lambda x: torch.from_numpy(x / np.linalg.norm(x, axis=1, keepdims=True))
    corpus = {"raw_documents": normalized(documents), "raw_queries": normalized(queries),
              "documents": normalized(old.apply_whitening(documents, mean, std)),
              "queries": normalized(old.apply_whitening(queries, mean, std))}
    for case in cases:
        case["document_indices"] = [di[doc_id] for doc_id in case["candidate_ids"]]
        case["neutral_index"] = qi[case["neutral_query_id"]]
        for query in case["queries"]:
            query["query_index"] = qi[query["query_id"]]
            query["target_index"] = case["candidate_ids"].index(query["target"])
    return cases, corpus, mean, std


def tensors(model, corpus, case, query):
    prefix = "raw_" if model.name == "cosine_raw" else ""
    return corpus[prefix + "queries"][query["query_index"]], corpus[prefix + "documents"][case["document_indices"]]


def calibrate(model, corpus, cases):
    # 用训练查询的无标签分差RMS统一初始logit尺度；后续冻结，不读验证统计。
    gaps = []
    with torch.no_grad():
        for case in cases:
            for query in case["queries"]:
                if query["style"] == "seen":
                    q, docs = tensors(model, corpus, case, query)
                    scores = model.raw_score(q, docs)
                    gaps.append((scores[0] - scores[1]) ** 2)
        model.score_scale.copy_(torch.stack(gaps).mean().sqrt().clamp_min(0.01))


def style_metrics(model, corpus, cases, style, neutral=False):
    rows = []
    with torch.no_grad():
        for case in cases:
            correct, margins, losses, predictions = [], [], [], []
            for query in case["queries"]:
                if query["style"] != style:
                    continue
                query_used = {**query, "query_index": case["neutral_index"]} if neutral else query
                q, docs = tensors(model, corpus, case, query_used)
                scores = model.score(q, docs)
                if not torch.isfinite(scores).all():
                    raise ValueError("探针产生非有限分数")
                target = query["target_index"]
                margin = float(scores[target] - scores[1 - target])
                margins.append(margin)
                correct.append(margin > TIE_MARGIN)
                losses.append(float(torch.logsumexp(scores, 0) - scores[target]))
                predictions.append(int(torch.argmax(scores)))
            rows.append({"case_id": case["id"], "single_accuracy": float(np.mean(correct)),
                         "paired_accuracy": float(all(correct)), "reversal_rate": float(predictions[0] != predictions[1]),
                         "loss": float(np.mean(losses)), "margins": margins})
    return {"summary": {key: float(np.mean([row[key] for row in rows])) for key in ("single_accuracy", "paired_accuracy", "reversal_rate", "loss")},
            "per_chain": rows}


def validation_metrics(model, corpus, cases):
    # 留出改写模板不参与训练或选模。
    return {style: style_metrics(model, corpus, cases, style) for style in ("seen", "ordinary")}


def selection_key(metrics):
    return (metrics["seen"]["summary"]["paired_accuracy"], metrics["seen"]["summary"]["single_accuracy"],
            metrics["ordinary"]["summary"]["single_accuracy"], -metrics["seen"]["summary"]["loss"])


def improves(metrics, incumbent, ordinary_floor):
    if metrics["ordinary"]["summary"]["single_accuracy"] + 1e-10 < ordinary_floor:
        return False
    return selection_key(metrics) > selection_key(incumbent)


def train_probe(name, seed, corpus, train, validation, baseline_name, baseline_metrics):
    torch.manual_seed(seed)
    model = Probe(name, corpus["documents"].shape[1])
    calibrate(model, corpus, train)
    initial = copy.deepcopy(model.state_dict())
    selected = copy.deepcopy(initial)
    best, selected_name, selected_epoch = baseline_metrics, baseline_name, None
    floor = baseline_metrics["ordinary"]["summary"]["single_accuracy"]
    metrics = validation_metrics(model, corpus, validation)
    if improves(metrics, best, floor):
        best, selected_name, selected_epoch = metrics, name, 0
    history = [{"epoch": 0, "validation": metrics}]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    examples = [(case, query) for case in train for query in case["queries"] if query["style"] == "seen"]
    rng = np.random.RandomState(seed)
    stale = 0
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(examples))
        batch_losses, gradient_norms = [], []
        for start in range(0, len(order), 8):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in order[start:start+8]:
                case, query = examples[int(index)]
                q, docs = tensors(model, corpus, case, query)
                scores = model.score(q, docs)
                losses.append(torch.logsumexp(scores, 0) - scores[query["target_index"]])
            loss = torch.stack(losses).mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            gradient_norms.append(float(norm))
            batch_losses.append(float(loss.detach()))
        metrics = validation_metrics(model, corpus, validation)
        history.append({"epoch": epoch, "train_loss": float(np.mean(batch_losses)), "validation": metrics,
                        "gradient_l2_mean_before_clip": float(np.mean(gradient_norms))})
        if improves(metrics, best, floor):
            selected = copy.deepcopy(model.state_dict())
            best, selected_name, selected_epoch, stale = metrics, name, epoch, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    last = copy.deepcopy(model.state_dict())
    path = OUTPUT / f"{name}_seed{seed}.pt"
    torch.save({"initial": initial, "selected": selected, "last": last}, path)
    return {"name": name, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
            "selected_name": selected_name, "selected_epoch": selected_epoch, "selected_validation": best,
            "history": history, "checkpoint": path.name, "checkpoint_sha256": old.sha256_file(path)}


def date_rule(cases, style):
    rows = []
    for case in cases:
        evidence = {row["document_id"]: row for row in case["evidence"]}
        ordered = sorted(case["candidate_ids"], key=lambda doc_id: evidence[doc_id]["date"])
        correct = []
        for query in case["queries"]:
            if query["style"] != style:
                continue
            text = query["text"].lower()
            if "earlier of" in text or "first entry chronologically" in text:
                chosen = ordered[0]
            elif "later of" in text or "most recent entry chronologically" in text:
                chosen = ordered[-1]
            else:
                raise ValueError("日期规则无法解析时间条件")
            correct.append(chosen == query["target"])
        rows.append({"case_id": case["id"], "single_accuracy": float(np.mean(correct)), "paired_accuracy": float(all(correct))})
    return {"summary": {key: float(np.mean([row[key] for row in rows])) for key in ("single_accuracy", "paired_accuracy")}, "per_chain": rows}


def protocol():
    return {"models": MODEL_NAMES, "seeds": SEEDS, "epochs": EPOCHS, "patience": PATIENCE, "lr": LEARNING_RATE,
            "metric_field": "G(x)=0.001I+sum softmax(gate(x))_k L_k^T L_k；4分支，每分支秩2。",
            "path": "5点Gauss-Legendre积分直线路径长度，以负长度平方评分；不是最短测地距离。",
            "equivalence": "local_metric就是门控加权欧氏投影平方距离，不把其代数等价实现当作另一种几何胜利。",
            "selection": "仅验证seen模板和普通事实对照参与选择；普通事实准确率不得低于候选基线。依次比较成对正确率、单问正确率、普通准确率、负交叉熵。并列保留更早状态。",
            "fallback": "原始/Whitening Cosine基线先按验证指标选一个作为每个训练探针的回退候选。",
            "split": "同主题与共享证据隔离；仅训练22链seen模板，6链验证；8链和paraphrase模板均延后评估。",
            "limits": "仅两证据微型诊断，日期规则能直接完成任务；不报告Recall@10，不据此宣称全库或一般逻辑收益。",
            "capacity": "局部/路径模型完全同参数；普通基线约同参数，并额外加入约2.7倍参数的MLP，防止弱化欧氏对照。"}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cases, corpus, mean, std = load_dataset()
    splits = {name: [case for case in cases if case["split"] == name] for name in ("train", "validation", "holdout")}
    run_protocol = protocol()
    paths = [Path(__file__), DATA / "manifest.json", DATA / "annotations.json"]
    paths += [EMBEDDINGS / name for name in ("documents.npy", "queries.npy", "document_ids.json", "query_ids.json", "runtime_manifest.json")]
    run_protocol["input_sha256"] = {str(path.relative_to(old.ROOT)): old.sha256_file(path) for path in paths}
    save_json(OUTPUT / "protocol.json", run_protocol)
    np.save(OUTPUT / "whitening_mean.npy", mean)
    np.save(OUTPUT / "whitening_std.npy", std)
    baselines, validation = {}, {}
    for name in ("cosine_raw", "cosine_whitened"):
        model = Probe(name, corpus["documents"].shape[1])
        calibrate(model, corpus, splits["train"])
        baselines[name] = model
        validation[name] = validation_metrics(model, corpus, splits["validation"])
    baseline_name = max(validation, key=lambda name: selection_key(validation[name]))
    runs = []
    with threadpool_limits(limits=1):
        for name in MODEL_NAMES:
            for seed in SEEDS:
                print(f"开始时间读出：{name}，随机种子{seed}", flush=True)
                row = train_probe(name, seed, corpus, splits["train"], splits["validation"], baseline_name, validation[baseline_name])
                runs.append(row)
                save_json(OUTPUT / f"{name}_seed{seed}.json", row)
                print(f"选中{row['selected_name']}，轮次{row['selected_epoch']}，验证成对正确率{row['selected_validation']['seen']['summary']['paired_accuracy']:.3f}", flush=True)
        registry = [{key: row[key] for key in ("name", "seed", "selected_name", "selected_epoch", "checkpoint", "checkpoint_sha256")} for row in runs]
        save_json(OUTPUT / "selection_registry.json", registry)
        print("全部选择已冻结，开始留出链和未见改写的评估。", flush=True)
        baseline_results = {name: {split: {style: style_metrics(model, corpus, subset, style) for style in ("seen", "paraphrase", "ordinary")}
                                   for split, subset in splits.items()} for name, model in baselines.items()}
        for row in runs:
            model = baselines.get(row["selected_name"])
            if model is None:
                model = Probe(row["name"], corpus["documents"].shape[1])
                state = torch.load(OUTPUT / row["checkpoint"], map_location="cpu", weights_only=True)
                model.load_state_dict(state["selected"])
            row["evaluation"] = {split: {style: style_metrics(model, corpus, subset, style) for style in ("seen", "paraphrase", "ordinary")}
                                 for split, subset in splits.items()}
            row["condition_blind_holdout"] = style_metrics(model, corpus, splits["holdout"], "seen", neutral=True)
            save_json(OUTPUT / f"{row['name']}_seed{row['seed']}.json", row)
    results = {"protocol": run_protocol, "baseline_selected": baseline_name, "baselines": baseline_results,
               "date_rule": {style: date_rule(splits["holdout"], style) for style in ("seen", "paraphrase")},
               "runs": runs, "selection_registry_sha256": old.sha256_file(OUTPUT / "selection_registry.json")}
    save_json(OUTPUT / "results.json", results)
    print("时间读出对照完成，已保存全部结果。", flush=True)


if __name__ == "__main__":
    main()
