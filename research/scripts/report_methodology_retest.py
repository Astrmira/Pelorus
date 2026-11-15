"""核验固定协议下的全部结果，并报告逐项消融与主几何对照。"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_low_capacity_experiment as old
from audit_readout_splits import OUTPUT
from diagnose_readout import save_json
from run_methodology_retest import better


def average(values):
    return float(np.mean(list(values)))


def cell(values):
    values = list(values)
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"


def paired(group_a, group_b, metric):
    def by_query(group):
        rows = defaultdict(list)
        for run in group:
            for row in run["development"]["per_query"]:
                rows[row["query_id"]].append(row[metric])
        return {qid: average(values) for qid, values in rows.items()}
    left, right = by_query(group_a), by_query(group_b)
    delta = {qid: left[qid] - right[qid] for qid in left}
    return {"mean_delta": average(delta.values()), "wins": sum(value > 1e-10 for value in delta.values()),
            "ties": sum(abs(value) <= 1e-10 for value in delta.values()),
            "losses": sum(value < -1e-10 for value in delta.values()), "per_query_delta": delta}


def main():
    result = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    audit = json.loads((OUTPUT / "split_audit.json").read_text(encoding="utf-8"))
    registry_path = OUTPUT / "selection_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if old.sha256_file(registry_path) != result["selection_registry_sha256"]:
        raise ValueError("模型选择清单发生变化")
    for relative, expected in result["protocol"]["input_sha256"].items():
        if old.sha256_file(old.ROOT / relative) != expected:
            raise ValueError(f"输入或实现发生变化：{relative}")
    groups = defaultdict(list)
    for run, frozen in zip(result["runs"], registry, strict=True):
        if any(run[key] != value for key, value in frozen.items()):
            raise ValueError("最终结果与冻结的选择清单不一致")
        if old.sha256_file(OUTPUT / run["checkpoint"]) != run["checkpoint_sha256"]:
            raise ValueError("检查点与训练结束时不一致")
        selected = run["selected_validation"]["summary"]
        baseline = run["baseline_validation"]["summary"]
        if better(baseline, selected):
            raise ValueError("显式B0回退未生效")
        if run["selected_kind"] == "B0":
            for key in ("map", "mrr", "recall@10", "pilot_ndcg@10"):
                if run["development"]["summary"][key] != result["baseline"]["development"]["summary"][key]:
                    raise ValueError("B0回退与基线不一致")
        groups[(run["variant"], run["model"])].append(run)
    main_groups = {"B1": groups[("R3_all_positive", "B1")], "B2": groups[("R4_shared", "B2")],
                   "H1": groups[("R5_squared_h1", "H1")], "H2": groups[("R4_shared", "H2")]}
    capacities = {name: rows[0]["parameters"] for name, rows in main_groups.items()}
    if len({capacities[name] for name in ("B2", "H1", "H2")}) != 1:
        raise ValueError("主几何对照容量不一致")
    comparisons = {name: {metric: paired(main_groups[name], main_groups["B2"], metric)
                           for metric in ("map", "mrr", "recall@10", "pilot_ndcg@10")}
                   for name in ("H1", "H2")}
    summary = {"verified_runs": len(result["runs"]), "main_parameters": capacities,
               "main_comparisons_to_euclidean": comparisons,
               "selection_registry_sha256": result["selection_registry_sha256"],
               "report_script_sha256": old.sha256_file(Path(__file__)),
               "groups": []}
    base = result["baseline"]["development"]["summary"]
    lines = ["# QuIVer 方法论修复与重测", "",
             "本轮在同一份清洗数据、同一选模规则下完成了分数尺度、初始化对齐、正例覆盖损失及共享几何映射的逐项消融。B1的精确饱和得到修复，四类几何分支均能拟合四题的正例排序；但预先指定的共享几何主对照全部选择B0。欧氏独立双塔消融出现开发均值改善，不能归因为曲率收益。所有版本均报告，没有按开发集择优隐藏结果。", "",
             "## 数据与协议修复", "",
             f"原始训练/验证/开发集分别为80/16/14题。清洗后保留训练 {audit['train_queries_after']} 题、{audit['train_candidates_after']} 个候选，验证 {audit['validation_queries_after']} 题、{audit['validation_candidates_after']} 个候选，开发集仍为14题完整Haystack。", "",
             "| 跨集合 | 原共享 Session 内容数 | 原共享 Pair 内容数 | 清洗后两类共享数 |",
             "|---|---:|---:|---|"]
    for key, row in audit["overlaps_before"].items():
        lines.append(f"| {key} | {row['session_hash']['shared_hash_count']} | {row['pair_hash']['shared_hash_count']} | 0 / 0 |")
    lines += ["", "以去除标注、日期与ID后的角色/原文内容识别重复，空白标准化后计算哈希。共享负候选移除；涉及正证据的训练或验证题整题排除。原始文件与向量缓存保持不变。Whitening 只由最终训练候选重新拟合。", "",
              "这只是缓存范围内的精确内容隔离，不是跨主题/事实链隔离。验证仍缺少完整 Haystack 向量，语义等价的假负例尚未穷举标注。开发集已参与多轮实验，继续称为开发集，不能当作新的最终测试集。", "",
              "所有版本统一使用验证 MAP 为主指标、MRR 为次指标，完全并列时保留更早状态。纯 B0 首先入选，残差模型只有严格改善才能替换；B0回退直接计算基线，不以非零的小门控冒充零残差。", "",
              "57组模型选择全部完成并写入 `selection_registry.json` 后，才执行一次性开发集评估。数据清洗读取留出集文本身份，但未依据留出集相关性调整训练标签。由于数据和选模协议变化，旧V3数值与本轮R0不属于单因素对照。", "",
              "## 固定的消融顺序", "",
              "| 版本 | 相对前一步的变化 |",
              "|---|---|",
              "| R0_control | 原V3模型定义；采用本轮共同的清洗数据和MAP/B0选模 |",
              "| R1_scale | 训练距离中位数校准，改用平滑有界变换 |",
              "| R2_aligned | Query/Document投影与半径头初始化数值相同，训练仍独立 |",
              "| R3_all_positive | 每个正例分别承担交叉熵损失，避免一个正例掩盖其余正例 |",
              "| R4_shared | 两塔使用同一组投影和半径参数，成为对称评分 |",
              "| R5_squared_h1 | 仅对H1追加平方距离，与R4的B2/H2统一距离幂次 |", "",
              "B1原本就是共享映射，因此R2的B1复用R1、R4的B1复用R3，不重复训练。共享B2/H1/H2参数量严格相同；B1缺少半径头，作为较小基线单列。所有方法保留原1536维Cosine，32维只计算残差。", "",
              "R3到R4的参数共享也把B2/H1/H2的参数量从101,379降为50,690。因此不能把这一步的表现变化全部归因于对称性；严格同容量几何比较限于R4/R5的B2/H1/H2之间。", "",
              "分数修复公式为 `s = (cos(q,d) - sigmoid(a) * D/(tau+D)) / T`，其中非负距离D沿用各模型定义，tau为初始化模型在训练候选上的距离中位数，拟合后冻结，T=0.1。R5将H1的D改为双曲距离平方。校准不读取标签，评分不使用在线候选池统计，增减候选不会改变已有Pair的分数。", "",
              "该映射的无量纲导数为 `1/(1+D/tau)^2`，没有tanh的指数尾部，但距离过大仍可能压缩梯度，所以本轮继续记录梯度和近饱和比例。它不保证所有模型梯度分布相同。共享评分也不自动等于一般黎曼测地度量，本轮没有开发ANN。", "",
              "## 全部开发结果", "",
              f"清洗训练统计下，B0开发 MRR={base['mrr']:.4f}，MAP={base['map']:.4f}，Recall@10={base['recall@10']:.4f}，证据完整覆盖@10={base['evidence_set_recall@10']:.4f}，Pilot nDCG@10={base['pilot_ndcg@10']:.4f}。", "",
              "±为三个Seed之间的总体标准差，不是置信区间。选中状态列依次对应Seed 11/23/47，B0为精确回退，E0为未训练残差。", "",
              "| 版本 | 模型 | 选中状态 | 验证MAP | 开发MRR | 开发Recall@10 | 开发nDCG@10 |",
              "|---|---|---|---:|---:|---:|---:|"]
    for (variant, name), group in groups.items():
        states = ["B0" if run["selected_kind"] == "B0" else f"E{run['selected_epoch']}" for run in group]
        lines.append(f"| {variant} | {name} | {'/'.join(states)} | " +
                     cell(run["selected_validation"]["summary"]["map"] for run in group) + " | " +
                     " | ".join(cell(run["development"]["summary"][metric] for run in group)
                                for metric in ("mrr", "recall@10", "pilot_ndcg@10")) + " |")
        summary["groups"].append({"variant": variant, "model": name, "selection": states,
                                  "validation_map": average(run["selected_validation"]["summary"]["map"] for run in group),
                                  "development": {metric: {"mean": average(run["development"]["summary"][metric] for run in group),
                                                           "std": float(np.std([run["development"]["summary"][metric] for run in group]))}
                                                  for metric in ("map", "mrr", "recall@10", "evidence_set_recall@10", "pilot_ndcg@10")}})
    lines += ["", "## 预先指定的主几何对照", "",
              "| 模型 | 参数量 | 开发MAP | 开发MRR | 开发Recall@10 | 证据完整覆盖@10 | 开发nDCG@10 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, group in main_groups.items():
        lines.append(f"| {name} | {capacities[name]:,} | " + " | ".join(cell(run["development"]["summary"][metric] for run in group)
                      for metric in ("map", "mrr", "recall@10", "evidence_set_recall@10", "pilot_ndcg@10")) + " |")
    lines += ["", "按同一问题先平均三个Seed，再对H1/H2与B2做配对差值，避免把42次观测误当作42个独立问题。下表是开发诊断的描述统计，不做显著性或泛化声明。", "",
              "| 对照 | 指标 | 平均差值 | 胜/平/负（题） |", "|---|---|---:|---|"]
    for name, metrics in comparisons.items():
        for metric, row in metrics.items():
            lines.append(f"| {name}−B2 | {metric} | {row['mean_delta']:+.5f} | {row['wins']}/{row['ties']}/{row['losses']} |")
    lines += ["", "## 数值与优化检查", "",
              "以下统计为训练候选上无量纲局部导数；R0对应tanh原分数，修复版本对应D/tau。零导数比例按三个Seed取均值。", "",
              "| 版本 | 模型 | 初始→末轮零导数比例 | 末轮导数<1e-3 | 末轮逐题参数梯度L2均值 | 初始→末轮训练损失 |",
              "|---|---|---:|---:|---:|---:|"]
    for (variant, name), group in groups.items():
        lines.append(f"| {variant} | {name} | {average(run['initial_monitor']['derivative_exact_zero'] for run in group):.2%} → "
                     f"{average(run['last_monitor']['derivative_exact_zero'] for run in group):.2%} | "
                     f"{average(run['last_monitor']['derivative_lt_1e_3'] for run in group):.2%} | "
                     f"{average(run['last_monitor']['per_query_parameter_gradient_l2']['mean'] for run in group):.3g} | "
                     f"{average(run['history'][0]['train']['loss'] for run in group):.4f} → {average(run['history'][-1]['train']['loss'] for run in group):.4f} |")
    lines += ["", "不同损失版本的绝对损失不可直接比较。末轮用于检查优化路径，所有开发指标始终使用验证选中的状态。半径头的sigmoid导数压缩比例另保存在每组的monitor中；平滑距离残差不能替代对所有内部非线性的检查。", "",
              "## 小训练集拟合检查", "",
              "固定按ID排序的前4个训练问题，Seed=11，200步。使用主对照的几何分支独立评分，不加入B0和门控；保持同样温度和全正例损失。此检查用于确认分支能否在有限数据上调整正例排序，不代表完整残差训练的拟合上限，更不代表泛化。", "",
              "| 模型 | 初始→最终损失 | 初始→最终MAP | 最终全部正例排在负例前的题目比例 |",
              "|---|---:|---:|---:|"]
    for row in result["small_fit"]:
        first, last = row["history"][0], row["history"][-1]
        lines.append(f"| {row['model']} | {first['loss']:.4f} → {last['loss']:.4f} | {first['map']:.4f} → {last['map']:.4f} | {last['perfect_positive_order']:.0%} |")
    lines += ["", "## 如何解释这轮重测", "",
              "R3的欧氏双塔开发MRR为0.6471±0.0178、nDCG@10为0.6751±0.0154，相对B0的0.6220/0.6533出现改善，但Recall@10仍为0.9048。三个Seed中两个接受训练残差、一个回退B0。这是值得复验的读出候选，尚不是跨主题泛化证据。",
              "", "共享几何主对照的全部Seed选择纯B0，因此其相同指标来自相同的回退评分，不能把零差值解释成已经证明欧氏和双曲表达能力相同。",
              "", "B1在共同清洗数据上的原方法末轮精确饱和约99.97%，尺度修复后为0；其梯度和训练损失恢复响应。四题分支拟合最终MAP均为1.0，说明存在可优化路径，但不能从训练记忆能力推断预训练向量保留了全部关系信息。", "",
              "方法是否修复，要分别看尺度、梯度、对齐与选模是否按协议工作；语义收益则看相同协议下与B0及同容量欧氏模型的配对结果。某个版本单项改善不能用于断言曲率必要。若主对照回退B0，只能说当前验证协议未选择该残差。",
              "", "当前仍不能把失败归因于Qwen信息缺失：没有新模型/维度对照，也没有独立人工核验的信息可读性探针。下一阶段应优先补齐验证的完整Haystack、审计语义假负例，再用新的内容隔离Holdout验证；本轮不为改善开发分数继续调整超参数。", "",
              "## 产物与核验", "",
              f"- 已核验 {len(result['runs'])} 组冻结选择、Checkpoint哈希、输入哈希和B0回退一致性。",
              "- `split_audit.json`：重叠来源统计、排除规则与实际训练/验证清单。",
              "- `protocol.json`：运行前固定的全部版本、参数、选择规则与输入哈希。",
              "- `selection_registry.json`：开发评分前冻结的模型选择。",
              "- `results.json`：全部逐题评估、训练轨迹、梯度监控和4题拟合结果。",
              "- `summary.json`：主对照和逐题配对差值。",
              "- 每组`.pt`保留初始化、验证选中、末轮状态，`selected_kind`决定是否显式回退B0。", "",
              "复现命令（research目录）：", "", "```powershell", "$env:PYTHONIOENCODING='utf-8'",
              ".\\.venv\\Scripts\\python.exe scripts/audit_readout_splits.py",
              ".\\.venv\\Scripts\\python.exe -u scripts/run_methodology_retest.py",
              ".\\.venv\\Scripts\\python.exe scripts/report_methodology_retest.py", "```", ""]
    save_json(OUTPUT / "summary.json", summary)
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成报告，{len(result['runs'])}组方法论重测均通过产物一致性核验。")


if __name__ == "__main__":
    main()
