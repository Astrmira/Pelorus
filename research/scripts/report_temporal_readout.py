"""核验冻结记录并汇总时间条件读出实验；不使用留出结果重新选模。"""

import json
import numpy as np
import run_temporal_readout as experiment
import run_low_capacity_experiment as old


def main():
    output = experiment.OUTPUT
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    registry = json.loads((output / "selection_registry.json").read_text(encoding="utf-8"))
    assert old.sha256_file(output / "selection_registry.json") == result["selection_registry_sha256"]
    for name, digest in result["protocol"]["input_sha256"].items():
        assert old.sha256_file(old.ROOT / name) == digest, name
    for frozen, run in zip(registry, result["runs"], strict=True):
        assert all(run[key] == value for key, value in frozen.items())
        assert old.sha256_file(output / run["checkpoint"]) == run["checkpoint_sha256"]
    assert len(registry) == 21
    lines = ["# 冻结向量的时间条件读出：第一轮结果", "",
             "本轮是两候选微型诊断：给出同一属性的两条记录，分别问较早与较晚的状态。成对正确率要求两问同时选对。它不等于全库检索，更不能直接证明一般黎曼几何的优越性。", "",
             "本轮结论：留出集上，双线性 / 非线性欧氏读出的 seen 成对正确率分别为 37.5% / 45.8%，高于 cosine 的 0%；但未见改写仅为 8.3% / 0%，普通事实单问正确率均约 50%，低于 Whitening cosine 的 62.5%。固定、局部和路径度量的成对正确率均为 0%。因此只能观察到依赖模板的有限读出改善，未达到稳定泛化、普通事实保持或黎曼优势的要求。", "",
             "## 数据与冻结协议", "",
             "36 条状态链，72 条证据、252 条查询；按主题与共享证据分为训练 22 链、验证 6 链、留出 8 链。训练只用每链两个 seen 时间查询，paraphrase 改写不参与训练或模型选择。普通事实查询用于验证集保护条件及最终对照。留出 8 链仅约 6 个主题组。", "",
             "证据来自公开 LongMemEval 构造对话，保留相关用户话语及日期；查询与标签由单一 AI 构造、核验，尚无独立复核者。日期排序规则能够直接解决这一任务，因此该实验首先用于检测冻结向量能否支持时间条件区分。", "",
             "Qwen3-Embedding-8B，1536 维；只用训练证据拟合 Whitening。三个随机种子为 11、23、47。每种模型使用固定超参数、训练至多 100 轮、15 轮耐心；全部 21 次选择记录先冻结，再计算留出和改写指标。模型选择先比较验证成对正确率，再比较单问正确率、普通事实正确率、负交叉熵；普通事实不得低于回退基线。", "",
             f"验证集选定的回退基线：`{result['baseline_selected']}`。输入、向量、训练脚本、选择登记及 21 个检查点的 SHA-256 已核验。", "",
             "## 最终留出指标", "",
             "以下为三个种子的均值 ± 标准差（百分点），不是置信区间。种子不增加独立链的数量；每个种子只有 8 条留出链，成对正确率每条相差 12.5 个百分点。", "",
             "| 模型 | 参数量 | 时间单问正确率 | 时间成对正确率 | 改写成对正确率 | 普通事实单问正确率 | 回退次数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    specs = [("seen", "single_accuracy"), ("seen", "paired_accuracy"), ("paraphrase", "paired_accuracy"), ("ordinary", "single_accuracy")]
    for name, evaluation in result["baselines"].items():
        cells = [f"{100*evaluation['holdout'][style]['summary'][metric]:.1f}%" for style, metric in specs]
        lines.append(f"| {name} | 0 | " + " | ".join(cells) + " | — |")
    for name in experiment.MODEL_NAMES:
        runs = [row for row in result["runs"] if row["name"] == name]
        cells = []
        for style, metric in specs:
            values = np.array([row["evaluation"]["holdout"][style]["summary"][metric] for row in runs])*100
            cells.append(f"{values.mean():.1f} ± {values.std():.1f}%")
        fallbacks = sum(row["selected_name"] != name for row in runs)
        lines.append(f"| {name} | {runs[0]['parameters']} | " + " | ".join(cells) + f" | {fallbacks}/3 |")
    lines += ["", "## 每次运行与泛化差距", "",
              "| 模型 | 种子 | 选中轮次/回退 | 训练成对 | 验证成对 | 留出成对 | 留出改写成对 |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for row in result["runs"]:
        selected = str(row["selected_epoch"]) if row["selected_name"] == row["name"] else row["selected_name"]
        values = [row["evaluation"][split]["seen"]["summary"]["paired_accuracy"] for split in ("train", "validation", "holdout")]
        values += [row["evaluation"]["holdout"]["paraphrase"]["summary"]["paired_accuracy"]]
        lines.append(f"| {row['name']} | {row['seed']} | {selected} | " + " | ".join(f"{100*v:.1f}%" for v in values) + " |")
    lines += ["", "## 同链对照：路径度量减去其他读出", "",
              "先在每条链内平均种子，再计算差值；仅描述当前固定留出集，不做小样本显著性宣称。正值表示路径版本更好。", "",
              "| 对照模型 | seen 成对差值（百分点） | paraphrase 成对差值（百分点） |",
              "|---|---:|---:|"]
    def chain_scores(name, style):
        runs = [row for row in result["runs"] if row["name"] == name]
        return {case["case_id"]: np.mean([next(item["paired_accuracy"] for item in row["evaluation"]["holdout"][style]["per_chain"] if item["case_id"] == case["case_id"]) for row in runs])
                for case in runs[0]["evaluation"]["holdout"][style]["per_chain"]}
    for name in ("bilinear", "shared_nonlinear", "pair_mlp_matched", "pair_mlp_stronger", "fixed_metric", "local_metric"):
        differences = []
        for style in ("seen", "paraphrase"):
            path, control = chain_scores("path_metric", style), chain_scores(name, style)
            differences.append(100*np.mean([path[key]-control[key] for key in path]))
        lines.append(f"| {name} | {differences[0]:+.1f} | {differences[1]:+.1f} |")
    date = result["date_rule"]
    blind = [row["condition_blind_holdout"]["summary"]["paired_accuracy"] for row in result["runs"]]
    assert max(blind) == 0
    lines += ["", "## 解释边界与下一步", "",
              f"日期规则在 seen / paraphrase 的成对正确率为 {100*date['seen']['summary']['paired_accuracy']:.1f}% / {100*date['paraphrase']['summary']['paired_accuracy']:.1f}%。它显式读取日期元数据，输入权限不同，只是任务可解性对照。去掉查询时间条件后，所有模型成对正确率均为 0%；相同查询不可能同时指向两个互斥答案，这是必要健全性检查。", "",
              "local_metric 的评分严格等于门控加权的欧氏投影距离，不能把两种写法作为两次独立几何证据。path_metric 使用正定位置相关度量场，在直线路径上做 5 点 Gauss-Legendre 长度积分；它不是最短测地距离，也尚未检验该度量场的曲率是否非零。固定、局部和路径版本的差异只能支持相应读出机制的比较。", "",
              "普通事实对照也只有两候选且查询含明确属性值；即使保持准确率，也不证明全库语义检索无损。模型失败不证明向量缺失信息；模型胜过 cosine 只说明该训练与样本条件下的信息可读出性更强。", "",
              "局部与路径模型在选中检查点的训练成对正确率也很低，当前结果混合了优化、模型容量和泛化因素，不能用于否定黎曼方法。接下来先只在训练集做少量链过拟合与梯度/尺度诊断，核实新模型是否学得动；再加入多种训练问法及普通事实联合目标。不得以本轮留出成绩选择这些修改。", "",
              "下一步应冻结本轮作为探索结果，另建独立、更大且经复核的测试集：加入无关候选、多个时间点、有效时间与叙述时间不同的案例，并把解析日期规则作为强基线。只有几何模型持续胜过同容量双线性、非线性欧氏及强 MLP 对照，才值得进一步做测地线求解和曲率消融。本轮留出不能再当作后续调参的最终盲测。", "",
              "## 复现", "", "```powershell", ".\\.venv\\Scripts\\python.exe scripts/run_temporal_readout.py", ".\\.venv\\Scripts\\python.exe scripts/report_temporal_readout.py", ".\\.venv\\Scripts\\python.exe -m pytest -q", "```", "",
              "详细逐链边际、训练梯度和选择历史见同目录 results.json、各模型 JSON；标注与排除原因见 data/processed/temporal_readout。已有向量缓存足够复现，无需再次请求编码 API。"]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("报告已生成，输入与冻结检查点校验通过。")


if __name__ == "__main__":
    main()
