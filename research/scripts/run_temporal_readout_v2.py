"""多模板训练的时间条件读出；新测试集在本脚本冻结后才生成。"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run_temporal_readout as base
import run_low_capacity_experiment as old


OUTPUT = old.ROOT / "artifacts" / "experiments" / "temporal_readout_v2"
SEEDS = (11, 23, 47)
MODEL_NAMES = ("bilinear", "shared_nonlinear", "pair_mlp_matched", "pair_mlp_stronger", "fixed_metric", "local_metric", "path_metric")
LEARNING_RATES = (5e-4, 5e-3)
EPOCHS = 180
PATIENCE = 25
TRAIN_STYLES = ("seen", "paraphrase", "ordinary")


def validation_metrics(model, corpus, cases):
    return {style: base.style_metrics(model, corpus, cases, style) for style in TRAIN_STYLES}


def selection_key(metrics):
    return (
        metrics["seen"]["summary"]["paired_accuracy"],
        metrics["paraphrase"]["summary"]["paired_accuracy"],
        metrics["ordinary"]["summary"]["single_accuracy"],
        metrics["seen"]["summary"]["single_accuracy"],
        -metrics["seen"]["summary"]["loss"],
    )


def improves(metrics, incumbent, ordinary_floor):
    if metrics["ordinary"]["summary"]["single_accuracy"] + 1e-10 < ordinary_floor:
        return False
    return selection_key(metrics) > selection_key(incumbent)


def train_probe(name, seed, lr, corpus, train, validation, baseline_name, baseline_metrics):
    torch.manual_seed(seed)
    model = base.Probe(name, corpus["documents"].shape[1])
    base.calibrate(model, corpus, train)
    initial = copy.deepcopy(model.state_dict())
    selected = copy.deepcopy(initial)
    best, selected_name, selected_epoch = baseline_metrics, baseline_name, None
    floor = baseline_metrics["ordinary"]["summary"]["single_accuracy"]
    metrics = validation_metrics(model, corpus, validation)
    if improves(metrics, best, floor):
        best, selected_name, selected_epoch = metrics, name, 0
    history = [{"epoch": 0, "validation": metrics}]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    examples = [(case, query) for case in train for query in case["queries"] if query["style"] in TRAIN_STYLES]
    rng = np.random.RandomState(seed)
    stale = 0
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(examples))
        batch_losses, gradient_norms = [], []
        for start in range(0, len(order), 8):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in order[start:start + 8]:
                case, query = examples[int(index)]
                q, docs = base.tensors(model, corpus, case, query)
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
    path = OUTPUT / f"{name}_seed{seed}_lr{lr:g}.pt"
    torch.save({"initial": initial, "selected": selected, "last": last}, path)
    return {"name": name, "seed": seed, "lr": lr, "parameters": sum(p.numel() for p in model.parameters()),
            "selected_name": selected_name, "selected_epoch": selected_epoch, "selected_validation": best,
            "history": history, "checkpoint": path.name, "checkpoint_sha256": old.sha256_file(path)}


def protocol(paths):
    return {
        "models": MODEL_NAMES,
        "seeds": SEEDS,
        "learning_rates": LEARNING_RATES,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "train_styles": TRAIN_STYLES,
        "selection": "逐运行只看6条验证链的seen/paraphrase/ordinary；普通事实准确率不得低于Whitening Cosine基线。依次比较seen成对、paraphrase成对、ordinary单问、seen单问、seen负交叉熵；并列保留较早状态。所有选择在新测试集生成前冻结。",
        "metric_field": "G(x)=0.001I+sum softmax(gate(x))_k L_k^T L_k；4分支，每分支秩2。",
        "path": "5点Gauss-Legendre积分固定直线路径长度，以负长度平方评分；不是最短测地距离求解器。",
        "limits": "仅两证据微型诊断；训练集使用三种模板，独立测试使用新事实链和新措辞；不据此宣称全库或一般逻辑收益。",
        "input_sha256": {str(path.relative_to(old.ROOT)): old.sha256_file(path) for path in paths},
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cases, corpus, mean, std = base.load_dataset()
    splits = {name: [case for case in cases if case["split"] == name] for name in ("train", "validation")}
    paths = [Path(__file__), base.DATA / "manifest.json", base.DATA / "annotations.json"]
    paths += [base.EMBEDDINGS / name for name in ("documents.npy", "queries.npy", "document_ids.json", "query_ids.json", "runtime_manifest.json")]
    run_protocol = protocol(paths)
    base.save_json(OUTPUT / "protocol.json", run_protocol)
    np.save(OUTPUT / "whitening_mean.npy", mean)
    np.save(OUTPUT / "whitening_std.npy", std)
    baselines, validation = {}, {}
    for name in ("cosine_raw", "cosine_whitened"):
        model = base.Probe(name, corpus["documents"].shape[1])
        base.calibrate(model, corpus, splits["train"])
        baselines[name] = model
        validation[name] = validation_metrics(model, corpus, splits["validation"])
    baseline_name = max(validation, key=lambda name: selection_key(validation[name]))
    runs = []
    with threadpool_limits(limits=1):
        for name in MODEL_NAMES:
            for lr in LEARNING_RATES:
                for seed in SEEDS:
                    print(f"开始多模板时间读出：{name}，学习率{lr}，随机种子{seed}", flush=True)
                    row = train_probe(name, seed, lr, corpus, splits["train"], splits["validation"], baseline_name, validation[baseline_name])
                    runs.append(row)
                    base.save_json(OUTPUT / f"{name}_seed{seed}_lr{lr:g}.json", row)
                    print(f"选中{row['selected_name']}，轮次{row['selected_epoch']}，验证键值{selection_key(row['selected_validation'])}", flush=True)
    registry = [{key: row[key] for key in ("name", "seed", "lr", "selected_name", "selected_epoch", "checkpoint", "checkpoint_sha256")} for row in runs]
    base.save_json(OUTPUT / "selection_registry.json", registry)
    results = {"protocol": run_protocol, "baseline_selected": baseline_name,
               "baselines_validation": validation, "runs": runs,
               "selection_registry_sha256": old.sha256_file(OUTPUT / "selection_registry.json")}
    base.save_json(OUTPUT / "results.json", results)
    lines = ["# 时间读出多模板训练冻结记录", "", "本轮在旧训练/验证切分中使用seen、paraphrase、ordinary三种查询训练；旧holdout完全未读取。两档学习率、三个种子、七种读出均在运行前登记。新独立测试集将在本文件和selection_registry.json冻结后才生成。", "", "| 模型 | 学习率 | 种子 | 选中读出 | 选中轮次 | 验证seen成对 | 验证paraphrase成对 | 验证ordinary单问 |", "|---|---:|---:|---|---:|---:|---:|---:|"]
    for row in runs:
        m = row["selected_validation"]
        lines.append(f"| {row['name']} | {row['lr']} | {row['seed']} | {row['selected_name']} | {row['selected_epoch'] if row['selected_epoch'] is not None else '回退'} | {m['seen']['summary']['paired_accuracy']:.1%} | {m['paraphrase']['summary']['paired_accuracy']:.1%} | {m['ordinary']['summary']['single_accuracy']:.1%} |")
    lines += ["", f"Whitening Cosine 基线为 {baseline_name}。每个候选的模型状态在验证指标停止改善时冻结；测试阶段不得根据新数据挑选。", "", "详细逐轮训练、梯度、检查点哈希和输入哈希保存在 results.json。"]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("多模板时间读出训练完成，选择记录已冻结。", flush=True)


if __name__ == "__main__":
    main()
