"""仅使用训练链，诊断时间度量探针的小样本可拟合性。"""

import json
import math
from pathlib import Path
import numpy as np
import torch
import run_temporal_readout as base

OUTPUT = base.old.ROOT / "artifacts/diagnostics/temporal_optimization_v1"


def examples_for(cases):
    if any(case["split"] != "train" for case in cases):
        raise ValueError("优化诊断禁止使用验证或留出链")
    return [(case, query) for case in cases for query in case["queries"] if query["style"] == "seen"]


def losses_for(model, corpus, examples):
    losses, margins = [], []
    for case, query in examples:
        q, docs = base.tensors(model, corpus, case, query)
        scores = model.score(q, docs)
        target = query["target_index"]
        losses.append(torch.logsumexp(scores, 0) - scores[target])
        margins.append(scores[target] - scores[1-target])
    return torch.stack(losses), torch.stack(margins)


def inspect(model, corpus, cases, examples):
    model.zero_grad(set_to_none=True)
    losses, margins = losses_for(model, corpus, examples)
    losses.mean().backward()
    gradients = {name: float(param.grad.norm()) for name, param in model.named_parameters() if param.grad is not None}
    with torch.no_grad():
        metric = base.style_metrics(model, corpus, cases, "seen")["summary"]
        result = {"loss": float(losses.mean()), "minimum_margin": float(margins.min()),
                  "score_scale": float(model.score_scale), "gradient_norms": gradients, **metric}
        if hasattr(model, "gate"):
            positions = torch.stack([base.tensors(model, corpus, case, query)[0] for case, query in examples])
            gates = model.gate(positions).softmax(-1)
            result["gate_entropy_mean"] = float(-(gates*gates.clamp_min(1e-12).log()).sum(-1).mean())
            result["gate_max_probability"] = float(gates.max())
            result["gate_query_pair_l1_mean"] = float((gates[::2]-gates[1::2]).abs().sum(-1).mean())
    return result


def run(name, seed, lr, corpus, cases, calibration_cases):
    torch.manual_seed(seed)
    model = base.Probe(name, corpus["documents"].shape[1])
    base.calibrate(model, corpus, calibration_cases)
    examples = examples_for(cases)
    initial = inspect(model, corpus, cases, examples)
    # 保留原评分尺度、权重衰减与裁剪；仅比较学习率与训练时长，不使用验证选模。
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history, sustained, first_success = [], 0, None
    for step in range(1, 801):
        optimizer.zero_grad(set_to_none=True)
        losses, _ = losses_for(model, corpus, examples)
        losses.mean().backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            updated_loss, updated_margin = losses_for(model, corpus, examples)
            success = bool((updated_margin > 0.1).all())
        sustained = sustained+1 if success else 0
        if success and first_success is None:
            first_success = step
        if step == 1 or step % 50 == 0 or sustained == 10:
            history.append({"step": step, "loss": float(updated_loss.mean()), "minimum_margin": float(updated_margin.min()),
                            "gradient_norm_before_clip": float(norm)})
        if sustained == 10:
            break
    final = inspect(model, corpus, cases, examples)
    row = {"name": name, "seed": seed, "lr": lr, "steps": step, "first_success_step": first_success,
           "sustained_success": sustained == 10, "initial": initial, "final": final, "history": history}
    if name == "path_metric":
        with torch.no_grad():
            _, five = losses_for(model, corpus, examples)
            nodes, weights = np.polynomial.legendre.leggauss(17)
            model.nodes = torch.tensor((nodes+1)/2, dtype=torch.float32)
            model.quadrature_weights = torch.tensor(weights/2, dtype=torch.float32)
            _, seventeen = losses_for(model, corpus, examples)
            row["quadrature_margin_max_absolute_difference"] = float((five-seventeen).abs().max())
    base.save_json(OUTPUT / f"{name}_{seed}_{lr}.json", row)
    print(f"{name}，种子{seed}，学习率{lr}：成对正确率{final['paired_accuracy']:.1%}，最小边际{final['minimum_margin']:.3f}，持续通过{row['sustained_success']}，{step}步", flush=True)
    return row


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    all_cases, corpus, _, _ = base.load_dataset()
    train = sorted([case for case in all_cases if case["split"] == "train"], key=lambda case: case["id"])
    cases = train[:4]
    protocol = {"cases": [case["id"] for case in cases], "selection": "按训练链ID排序取前4条，不根据指标选择",
                "models": ["bilinear", "fixed_metric", "local_metric", "path_metric"], "seeds": [11,23,47],
                "learning_rates": [0.0005,0.005], "maximum_steps": 800,
                "success": "8个训练查询的标准化正确边际均大于0.1，连续10步；不是泛化检验",
                "calibration": "沿用全部22条训练链seen查询的初始分差RMS；本轮只改变学习率和允许训练时长",
                "script_sha256": base.old.sha256_file(Path(__file__)),
                "frozen_experiment_sha256": base.old.sha256_file(base.OUTPUT / "results.json")}
    base.save_json(OUTPUT / "protocol.json", protocol)
    rows = [run(name, seed, lr, corpus, cases, train) for name in protocol["models"]
            for lr in protocol["learning_rates"] for seed in protocol["seeds"]]
    assert protocol["frozen_experiment_sha256"] == base.old.sha256_file(base.OUTPUT / "results.json")
    base.save_json(OUTPUT / "results.json", {"protocol": protocol, "runs": rows})
    lines = ["# 时间度量优化诊断", "", "只取训练集按ID排序的前4条链（8问），不计算验证、留出或改写成绩。沿用冻结向量、Whitening、模型结构和初始评分尺度；不调用编码API。", "",
             "成功标准：全部正确边际超过0.1并连续保持10步。两档学习率均预先登记，最多800次全批量更新。该结果衡量小样本可拟合性，不衡量泛化。", "",
             "| 模型 | 学习率 | 持续成功 | 最终成对正确率均值 | 更新步数 |", "|---|---:|---:|---:|---|"]
    for name in protocol["models"]:
        for lr in protocol["learning_rates"]:
            group = [row for row in rows if row["name"] == name and row["lr"] == lr]
            lines.append(f"| {name} | {lr} | {sum(row['sustained_success'] for row in group)}/3 | {np.mean([row['final']['paired_accuracy'] for row in group]):.1%} | " + ", ".join(str(row["steps"]) for row in group)+" |")
    lines += ["", "## 梯度、门控与积分", "", "| 模型 | 种子 | 学习率 | 初始梯度总范数 | 最终最小边际 | 门控时间两问L1差 |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        grad = math.sqrt(sum(v*v for v in row["initial"]["gradient_norms"].values()))
        gate = row["final"].get("gate_query_pair_l1_mean")
        lines.append(f"| {row['name']} | {row['seed']} | {row['lr']} | {grad:.6g} | {row['final']['minimum_margin']:.4f} | {gate if gate is not None else '—'} |")
    error = max(row["quadrature_margin_max_absolute_difference"] for row in rows if row["name"] == "path_metric")
    lines += ["", f"路径模型将积分从5点提升到17点，训练查询标准化边际的最大绝对变化为 {error:.6g}。该检查仅比较固定直线路径的数值积分，不验证最短测地距离。", "",
              "训练链ID：" + ", ".join(protocol["cases"]), "",
              "初始与最终损失、逐参数梯度范数、评分尺度、门控熵及训练过程保存在 results.json。本轮保留了所有失败运行，未改动上一轮实验结果。", "",
              "即使全部拟合成功，也只排除这些具体训练实例上的绝对表达障碍；仍需多问法训练、普通事实联合目标及新的独立测试集。若原学习率失败而更大学习率成功，只能说明该诊断设置对优化预算敏感，不能直接推广为全数据最优超参数。"]
    (OUTPUT / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
