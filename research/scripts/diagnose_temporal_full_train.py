"""仅使用完整训练集，诊断时间度量探针的优化预算。"""

import json
import math
from pathlib import Path

import numpy as np
import torch

import run_temporal_readout as base


OUTPUT = base.old.ROOT / "artifacts/diagnostics/temporal_optimization_full_train_v1"


def examples_for(cases):
    """只允许训练集的 seen 查询进入诊断。"""
    if any(case["split"] != "train" for case in cases):
        raise ValueError("完整训练诊断禁止使用验证或留出链")
    examples = [(case, query) for case in cases for query in case["queries"] if query["style"] == "seen"]
    if len(examples) != 44:
        raise ValueError(f"训练查询数应为44，实际为{len(examples)}")
    return examples


def losses_for(model, corpus, examples):
    losses, margins = [], []
    for case, query in examples:
        q, docs = base.tensors(model, corpus, case, query)
        scores = model.score(q, docs)
        target = query["target_index"]
        losses.append(torch.logsumexp(scores, 0) - scores[target])
        margins.append(scores[target] - scores[1 - target])
    return torch.stack(losses), torch.stack(margins)


def inspect(model, corpus, cases, examples):
    model.zero_grad(set_to_none=True)
    losses, margins = losses_for(model, corpus, examples)
    losses.mean().backward()
    gradients = {name: float(param.grad.norm()) for name, param in model.named_parameters() if param.grad is not None}
    with torch.no_grad():
        metric = base.style_metrics(model, corpus, cases, "seen")["summary"]
        result = {
            "loss": float(losses.mean()),
            "minimum_margin": float(margins.min()),
            "score_scale": float(model.score_scale),
            "gradient_norms": gradients,
            **metric,
        }
        if hasattr(model, "gate"):
            positions = torch.stack([base.tensors(model, corpus, case, query)[0] for case, query in examples])
            gates = model.gate(positions).softmax(-1)
            result["gate_entropy_mean"] = float(-(gates * gates.clamp_min(1e-12).log()).sum(-1).mean())
            result["gate_max_probability"] = float(gates.max())
            result["gate_query_pair_l1_mean"] = float((gates[::2] - gates[1::2]).abs().sum(-1).mean())
    return result


def run(name, seed, lr, corpus, cases, calibration_cases):
    torch.manual_seed(seed)
    model = base.Probe(name, corpus["documents"].shape[1])
    base.calibrate(model, corpus, calibration_cases)
    examples = examples_for(cases)
    initial = inspect(model, corpus, cases, examples)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history, sustained, first_success = [], 0, None
    for step in range(1, 1201):
        optimizer.zero_grad(set_to_none=True)
        losses, _ = losses_for(model, corpus, examples)
        losses.mean().backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            updated_loss, updated_margin = losses_for(model, corpus, examples)
            success = bool((updated_margin > 0.1).all())
        sustained = sustained + 1 if success else 0
        if success and first_success is None:
            first_success = step
        if step == 1 or step % 100 == 0 or sustained == 10:
            history.append({
                "step": step,
                "loss": float(updated_loss.mean()),
                "minimum_margin": float(updated_margin.min()),
                "gradient_norm_before_clip": float(norm),
            })
        if sustained == 10:
            break
    final = inspect(model, corpus, cases, examples)
    row = {
        "name": name,
        "seed": seed,
        "lr": lr,
        "steps": step,
        "first_success_step": first_success,
        "sustained_success": sustained == 10,
        "initial": initial,
        "final": final,
        "history": history,
    }
    if name == "path_metric":
        with torch.no_grad():
            _, five = losses_for(model, corpus, examples)
            nodes, weights = np.polynomial.legendre.leggauss(17)
            model.nodes = torch.tensor((nodes + 1) / 2, dtype=torch.float32)
            model.quadrature_weights = torch.tensor(weights / 2, dtype=torch.float32)
            _, seventeen = losses_for(model, corpus, examples)
            row["quadrature_margin_max_absolute_difference"] = float((five - seventeen).abs().max())
    base.save_json(OUTPUT / f"{name}_{seed}_{lr}.json", row)
    print(
        f"{name}，种子{seed}，学习率{lr}：成对正确率{final['paired_accuracy']:.1%}，"
        f"最小边际{final['minimum_margin']:.3f}，持续通过{row['sustained_success']}，{step}步",
        flush=True,
    )
    return row


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    all_cases, corpus, _, _ = base.load_dataset()
    train = sorted([case for case in all_cases if case["split"] == "train"], key=lambda case: case["id"])
    protocol = {
        "cases": [case["id"] for case in train],
        "selection": "完整训练集22条链，按ID排序；不根据指标挑选",
        "query_style": "seen",
        "models": ["bilinear", "fixed_metric", "local_metric", "path_metric"],
        "seeds": [11, 23, 47],
        "learning_rates": [0.0005, 0.005],
        "maximum_steps": 1200,
        "success": "44个训练查询的标准化正确边际均大于0.1，连续10步；不是泛化检验",
        "calibration": "沿用全部22条训练链seen查询的初始分差RMS；本轮不使用验证或留出",
        "script_sha256": base.old.sha256_file(Path(__file__)),
        "frozen_experiment_sha256": base.old.sha256_file(base.OUTPUT / "results.json"),
    }
    base.save_json(OUTPUT / "protocol.json", protocol)
    rows = [
        run(name, seed, lr, corpus, train, train)
        for name in protocol["models"]
        for lr in protocol["learning_rates"]
        for seed in protocol["seeds"]
    ]
    if protocol["frozen_experiment_sha256"] != base.old.sha256_file(base.OUTPUT / "results.json"):
        raise RuntimeError("上一轮冻结结果发生变化")
    base.save_json(OUTPUT / "results.json", {"protocol": protocol, "runs": rows})
    lines = [
        "# 完整训练集时间度量优化诊断",
        "",
        "本轮只使用训练集22条链的44个 seen 查询，不计算验证、改写或留出成绩。模型结构、冻结向量、Whitening、初始化评分尺度和权重衰减沿用上一轮；只扩大训练实例和优化预算。",
        "",
        "成功标准：44个训练查询的正确边际都超过0.1并连续保持10步。两档学习率和1200步上限在运行前登记。该结果只衡量完整训练集可拟合性，不衡量泛化。",
        "",
        "| 模型 | 学习率 | 持续成功 | 最终成对正确率均值 | 更新步数 |",
        "|---|---:|---:|---:|---|",
    ]
    for name in protocol["models"]:
        for lr in protocol["learning_rates"]:
            group = [row for row in rows if row["name"] == name and row["lr"] == lr]
            lines.append(
                f"| {name} | {lr} | {sum(row['sustained_success'] for row in group)}/3 | "
                f"{np.mean([row['final']['paired_accuracy'] for row in group]):.1%} | "
                + ", ".join(str(row["steps"]) for row in group)
                + " |"
            )
    lines += [
        "",
        "## 诊断解读",
        "",
        "如果完整训练集也不能稳定拟合，说明模型容量、目标冲突或数据可分性仍是主要瓶颈；如果能拟合而验证仍失败，后续应把重点放在模板外推和读出归纳偏差。任何情况下，本轮都不把训练拟合成绩当作泛化证据。",
        "",
        "训练链ID：" + ", ".join(protocol["cases"]),
        "",
        "初始与最终损失、逐参数梯度范数、评分尺度、门控熵及训练过程保存在 results.json。路径模型另存5点与17点直线路径积分的差异；这不是最短测地距离验证。",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
