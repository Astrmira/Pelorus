"""汇总读出诊断的数值证据，生成可追溯的中文报告。"""

import json
from collections import defaultdict

import numpy as np

from diagnose_readout import DEFAULT_OUTPUT, input_hashes, save_json
import run_low_capacity_experiment as experiment


def avg(values):
    return float(np.mean(list(values)))


def mean_std(values):
    values = list(values)
    return {"mean": avg(values), "std": float(np.std(values)), "min": min(values), "max": max(values)}


def formatted(values):
    values = list(values)
    return f"{avg(values):.4f}" + (f" ± {np.std(values):.4f}" if len(values) > 1 else "")


def main():
    root = DEFAULT_OUTPUT
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["input_sha256"] != input_hashes():
        raise ValueError("诊断输入或实现发生变化，不能汇总为同一次实验")
    index = json.loads((root / "runs.json").read_text(encoding="utf-8"))
    runs = [json.loads((root / row["file"]).read_text(encoding="utf-8")) for row in index]
    projection = json.loads((root / "projection_controls.json").read_text(encoding="utf-8"))
    saved = json.loads((experiment.OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))
    checks = []
    for run in runs:
        checkpoint = run["checkpoint_reproduction"]
        if not checkpoint["bitwise_equal"]:
            raise ValueError("训练复现与原检查点不一致")
        if experiment.sha256_file(experiment.ROOT / checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("原检查点在诊断后发生变化")
        original = next(item for item in saved["models"][run["model"]]["runs"] if item["seed"] == run["seed"])
        actual = run["stages"]["saved_selected"]["splits"]["development"]["ranking"]
        for key, expected in (("mrr", original["test"]["official"]["mrr"]),
                              ("recall@10", original["test"]["official"]["recall@10"]),
                              ("pilot_ndcg@10", original["test"]["pilot"]["ndcg@10"])):
            if abs(actual[key] - expected) > 1e-12:
                raise ValueError(f"诊断评估与原指标不一致：{run['model']}，{key}")
        if run["training"]["history"] != original["training"]["history"]:
            raise ValueError("复现训练轨迹与原结果不一致")
        checks.append({"model": run["model"], "seed": run["seed"], "checkpoint_and_history_exact": True,
                       "saved_metrics_match": True})
    groups = defaultdict(list)
    for run in runs:
        groups[run["model"]].append(run)
    summary = {"verification": checks, "branch_diagnostics": {}, "projection": []}
    lines = [
        "# QuIVer 读出诊断：步骤 1 与步骤 2", "",
        "本轮确认了两个独立问题：B1 的有界残差存在严重数值饱和；32 维压缩损失与独立双塔的初始对齐破坏也已被对照实验分别观察到。H1/H2 的参数有梯度、训练损失下降，因此不能把它们的 Epoch 0 选择解释成分支完全不可学习，也不能据此否定向量中的关系信息或黎曼几何。", "",
        "只复用已有 Qwen3-Embedding-8B 1536 维缓存。训练 80 题、验证 16 题仍各用 32 候选，14 题开发诊断集使用完整 Haystack。所有 Query 均使用已缓存的指令版本。PCA 和 Whitening 只拟合训练文档。", "",
        "## 复现与边界", "",
        f"- 12/12 组模型 × Seed 的选中 Checkpoint **逐位一致**，历史训练损失、验证 MRR 与选择轮次完全一致。",
        "- 原 Checkpoint 的开发集 MRR、Recall@10、Pilot nDCG@10 与诊断评估一致（误差小于 1e-12）。",
        "- 初始化、原选中状态、复现选中状态、恢复前末轮均单独记录；训练每轮另记录梯度、分数分布和参数变化。",
        "- 本轮没有新增 API 调用、没有修改原训练模型或旧实验产物、没有进行小子集拟合与数据审计。",
        "- 三 Seed 为 11/23/47。表中的 ± 是 Seed 间总体标准差，不是置信区间。",
        "- 14 题已参与历次开发，只用于故障定位；现有 Pilot 标签不完备，不能从这些指标推断最终泛化收益。", "",
        "## 步骤 1：数值饱和与梯度", "",
        "公式保持原 V3 实现：`(base_cosine + sigmoid(gate) * tanh(geometric_score)) / 0.1`。B1/B2 使用负平方欧氏距离，H1 使用负双曲距离，H2 使用负混合平方距离；以下 raw 分数不能直接当作同一种距离比较。", "",
        "精确饱和指 float32 下 `1 - tanh(s)^2 == 0`；近饱和指该导数小于 1e-3。统计对象为全部 2,560 个训练 Query–候选对，先按 Seed 统计，再对三个 Seed 取均值。", "",
        "| 模型 | 初始 raw 分数中位数 | 初始精确饱和 | 初始近饱和 | 末轮精确饱和 | 初始→末轮训练损失 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, group in groups.items():
        short = name.split("_")[0]
        initial = [run["stages"]["initial"]["splits"]["train"] for run in group]
        last = [run["stages"]["last_before_restore"]["splits"]["train"] for run in group]
        summary["branch_diagnostics"][short] = {
            "initial_exact_saturation": mean_std(row["tanh_exact_saturation"] for row in initial),
            "initial_near_saturation": mean_std(row["tanh_derivative_lt_1e_3"] for row in initial),
            "last_exact_saturation": mean_std(row["tanh_exact_saturation"] for row in last),
            "initial_train_loss": mean_std(row["loss_mean"] for row in initial),
            "last_train_loss": mean_std(row["loss_mean"] for row in last),
        }
        lines.append(f"| {short} | {avg(row['raw_geometric_score']['median'] for row in initial):.3f} | "
                     f"{avg(row['tanh_exact_saturation'] for row in initial):.2%} | "
                     f"{avg(row['tanh_derivative_lt_1e_3'] for row in initial):.2%} | "
                     f"{avg(row['tanh_exact_saturation'] for row in last):.2%} | "
                     f"{avg(row['loss_mean'] for row in initial):.4f} → {avg(row['loss_mean'] for row in last):.4f} |")
    lines += ["", "B1 初始化已有约 96%–97% 的训练对精确饱和，但不能说从第一步起所有梯度都是零：剩余少量未饱和对仍提供梯度，参数确实大幅更新，随后距离进一步增大、末轮几乎全饱和。三个 Seed 的投影相对 L2 变化分别为 " +
              " / ".join(f"{run['stages']['last_before_restore']['parameter_changes_from_initial']['projection.weight']['relative_delta_l2']:.3f}" for run in groups["B1_shared_mahalanobis"]) +
              "。参数变化本身不等于有效学习；该模型训练排名仍与 B0 一致。", "",
              "当残差在同一 Query 的候选间为常数时，多正例 logsumexp 损失对这个共同平移不敏感。应关注残差的候选间差异与梯度，而不是仅看门控权重是否非零。", "",
              "| 模型 | 初始→末轮几何分数梯度 L2 | 末轮平均损失的投影参数梯度 L2 | 初始→末轮训练 MRR | 末轮候选间残差跨度 |",
              "|---|---:|---:|---:|---:|"]
    for name, group in groups.items():
        initial = [run["stages"]["initial"]["splits"]["train"] for run in group]
        last = [run["stages"]["last_before_restore"]["splits"]["train"] for run in group]
        gradient = [sum(row["mean_loss_gradient_l2"] ** 2 for key, row in item["gradients"].items() if "projection" in key) ** 0.5 for item in last]
        lines.append(f"| {name.split('_')[0]} | {avg(row['loss_to_geometry_gradient_l2']['mean'] for row in initial):.3g} → "
                     f"{avg(row['loss_to_geometry_gradient_l2']['mean'] for row in last):.3g} | {avg(gradient):.3g} | "
                     f"{avg(row['ranking']['mrr'] for row in initial):.4f} → {avg(row['ranking']['mrr'] for row in last):.4f} | "
                     f"{avg(row['residual_span']['mean'] for row in last):.3g} |")
    lines += ["", "几何分数梯度 L2 为逐题损失对几何分数向量的梯度范数均值；参数梯度先对全部训练题的损失取均值，再按投影参数拼接求 L2。均在梯度裁剪前计算，不包含 AdamW 的权重衰减。残差跨度在除温度之前计算。", "",
              "H1 没有近饱和，B2/H2 仅有少量近饱和。它们训练损失和训练 MRR 均改善，说明当前路径确实可优化；这不等于已学到时间、否定或层级关系。", "",
              "## 验证选择和实际排序变化", "",
              "| 模型 | 三 Seed 最佳 Epoch | 原选中开发 MRR | 末轮开发 MRR | 原选中开发 nDCG@10 | 末轮开发 nDCG@10 |",
              "|---|---|---:|---:|---:|---:|"]
    for name, group in groups.items():
        selected = [run["stages"]["saved_selected"]["splits"]["development"]["ranking"] for run in group]
        last = [run["stages"]["last_before_restore"]["splits"]["development"]["ranking"] for run in group]
        lines.append(f"| {name.split('_')[0]} | " + "/".join(str(run["training"]["best_epoch"]) for run in group) +
                     f" | {formatted(row['mrr'] for row in selected)} | {formatted(row['mrr'] for row in last)} | "
                     f"{formatted(row['pilot_ndcg@10'] for row in selected)} | {formatted(row['pilot_ndcg@10'] for row in last)} |")
    lines += ["", "上表末轮结果只解释训练轨迹，不用它重新选择模型。H1 三个 Seed 的验证 MRR 末轮均与初始化相同（0.8229）；H2 两个相同、一个下降。原选择器只接受严格提高，所以保留 Epoch 0。准确表述是“验证 MRR 未改善”，而非验证器证明没有可学习信号。", "",
              "| 模型 | 初始→末轮开发完整排名变化题数 | 初始→末轮 Top-1 变化题数 | 初始→末轮 Top-10 与 B0 的交集比例 |",
              "|---|---:|---:|---:|"]
    for name, group in groups.items():
        initial = [run["stages"]["initial"]["splits"]["development"]["ranking"] for run in group]
        last = [run["stages"]["last_before_restore"]["splits"]["development"]["ranking"] for run in group]
        lines.append(f"| {name.split('_')[0]} | {avg(row['ranking_changed'] for row in initial)*14:.2f} → {avg(row['ranking_changed'] for row in last)*14:.2f} | "
                     f"{avg(row['top1_changed'] for row in initial)*14:.2f} → {avg(row['top1_changed'] for row in last)*14:.2f} | "
                     f"{avg(row['top10_overlap'] for row in initial):.2%} → {avg(row['top10_overlap'] for row in last):.2%} |")
    lines += ["", "题数为三个 Seed 的均值。完整排名变化可能只是低位候选交换，不能替代证据召回指标。初始门控 sigmoid(-4)≈0.018，Epoch 0 并非严格等于 B0。", "",
              "## 步骤 2：压缩与对齐的无训练对照", "",
              "先分别在原始缓存向量和 Train-only Whitening 后向量上比较。共享随机投影与独立随机双塔使用相同 Query 投影；独立条件只换为实际双塔初始化中的 Document 投影。PCA 使用完整 SVD，仅拟合 2,560 个训练文档，没有读取验证或开发向量来拟合。", "",
              "PCA 自动中心化，因此额外报告训练集中心化后的 1536 维 Cosine，避免把中心化收益算成降维收益。这里的投影没有半径头；实际 V3 的带半径几何独立评分已保存在每个模型的 `geometry_only_ranking` 中。", ""]
    labels = {"full_1536": "完整 1536 维", "train_centered_1536": "训练集中心化 1536 维", "train_pca_32": "Train-only PCA 32 维", "shared_random_32": "共享随机投影 32 维", "independent_random_32": "独立随机双塔 32 维"}
    for space in ("raw", "whitened"):
        lines += ["### " + ("原始缓存向量" if space == "raw" else "Train-only Whitening 后向量"), "",
                  "| Cosine 表示 | 训练 MRR | 验证 MRR | 开发 MRR | 开发 Recall@10 | 开发 nDCG@10 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for method, label in labels.items():
            selected = [row for row in projection if row["space"] == space and row["method"] == method and row["score"] == "cosine"]
            cells = []
            stats = {}
            for split, key in (("train", "mrr"), ("validation", "mrr"), ("development", "mrr"), ("development", "recall@10"), ("development", "pilot_ndcg@10")):
                values = [row["splits"][split]["summary"][key] for row in selected]
                cells.append(formatted(values))
                stats[split + "/" + key] = mean_std(values)
            summary["projection"].append({"space": space, "method": method, "score": "cosine", "metrics": stats})
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines += [""]
    lines += [
        "共享随机 32 维即使保持同一坐标系，开发集指标也明显下降且 Seed 波动很大；再改为独立双塔，初始化 MRR 降到约 0.02。压缩和对齐破坏在尚未引入曲率时就已经存在。V1 的失败因此不能直接归因于双曲距离。",
        "", "PCA 32 维仅保留原始训练文档约 33.45%、Whitening 后约 30.39% 的方差。方差保留率不是语义信息保留率，但检索对照也显示该 32 维配置有明显损失。PCA 验证 MRR 接近完整维度、开发集却明显下降，提示不能依赖当前 16 题/32 候选验证结果判定完整检索；本轮不能区分候选规模与题目组成各自的贡献。", "",
        "### 同坐标负平方 L2 对照", "",
        "负平方 L2 不再对降维后的点做单位归一化；它用于检查范数参与评分的影响，不属于与投影后 Cosine 等价的距离。原始缓存单位向量的完整维度 L2 与 Cosine 的三个核心开发指标已核对一致。", "",
        "| 空间 | 表示 | 开发 MRR | 开发 Recall@10 | 开发 nDCG@10 |", "|---|---|---:|---:|---:|",
    ]
    for space in ("raw", "whitened"):
        for method, label in labels.items():
            selected = [row for row in projection if row["space"] == space and row["method"] == method and row["score"] == "negative_squared_l2"]
            lines.append(f"| {space} | {label} | " + " | ".join(formatted(row["splits"]["development"]["summary"][key] for row in selected)
                          for key in ("mrr", "recall@10", "pilot_ndcg@10")) + " |")
    original_cosine = next(row for row in projection if row["space"] == "raw" and row["method"] == "full_1536" and row["score"] == "cosine")
    original_l2 = next(row for row in projection if row["space"] == "raw" and row["method"] == "full_1536" and row["score"] == "negative_squared_l2")
    for key in ("mrr", "recall@10", "pilot_ndcg@10"):
        if abs(original_cosine["splits"]["development"]["summary"][key] - original_l2["splits"]["development"]["summary"][key]) > 1e-12:
            raise ValueError("单位向量完整维度的 L2 与 Cosine 指标不一致")
    lines += ["", "## 本轮之后的判断", "",
              "1. B1 需要修复输入距离尺度与饱和问题；原 V3 中该基线不适合用来判定 Mahalanobis 是否有效。",
              "2. B2/H1/H2 确实可优化；其验证 MRR 无稳定提升，尚不能定位为信息缺失、监督不足或几何无效。",
              "3. 32 维随机/PCA 投影损失已实测，独立双塔又引入更强的初始化对齐破坏。后续应保留原始高维基线，设计共享或对齐初始化，不能把重学空间的困难算作几何效果。",
              "4. 下一步仍是原计划的小子集拟合与数据协议审计；新的尺度校准、初始化或训练目标应作为独立实验记录。没有根据这 14 题自动调参或发布几何收益结论。", "",
              "## 产物与复现", "",
              "- `manifest.json`：协议、运行环境、输入和实现 SHA-256。",
              "- `projection_controls.json`：全部 36 个投影/评分条件，包含逐题指标和 Seed。",
              "- `pca_raw.npz`、`pca_whitened.npz`：训练拟合的 PCA 参数。",
              "- `B1/B2/H1/H2...seed*.json`：12 组诊断，含每轮梯度、四种状态、逐题排名指标与原 Checkpoint 哈希。",
              "- `summary.json`：机器可读汇总和复现核验。", "",
              "在 research 目录运行：", "", "```powershell",
              "$env:PYTHONIOENCODING = 'utf-8'",
              ".\\.venv\\Scripts\\python.exe -u scripts/diagnose_readout.py",
              ".\\.venv\\Scripts\\python.exe scripts/summarize_readout_diagnostics.py", "```", ""]
    save_json(root / "summary.json", summary)
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("已生成中文报告；12组检查点、训练历史和旧指标均通过一致性核验。")


if __name__ == "__main__":
    main()
