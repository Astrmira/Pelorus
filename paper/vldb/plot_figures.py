"""从随论文保存的实测数据重新生成矢量 PDF 图表。"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator, PercentFormatter


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
DATA = FIGURES / "data" / "figure_points.csv"
STYLES = {
    "A1B1C1": ("Pelorus", "#0072B2", "o", "-"),
    "A1B1C1-M32": ("Pelorus (32/256)", "#0072B2", "o", "-"),
    "A8/symmetric": ("Pelorus-Core", "#D55E00", "s", "--"),
    "A7/main": ("Pelorus-Split", "#009E73", "^", "-."),
    "SQ8/main": ("Polar-SQ8", "#CC79A7", "D", ":"),
    "FP32/main": ("Polar-FP32", "#555555", "v", "--"),
    "A8/colocated": ("w/o SymSketch256", "#E69F00", "^", "--"),
    "A8/main": ("w/o HotSmPage (SoA)", "#009E73", "D", "-."),
    "A8/no_asymmetric_gate": ("SoA, Stage-0.5 off", "#CC79A7", "v", ":"),
    "A8构图关闭-connectivity_repair": ("SoA, Chord-Swap off", "#56B4E9", "P", "--"),
    "A8构图关闭-compass": ("Two-pass, SoA", "#555555", "X", "-."),
    "A8/samegraph_sq8": ("SQ8 (same graph)", "#E69F00", "^", "--"),
    "A8/samegraph_fp32": ("FP32 (same graph)", "#0072B2", "s", "-."),
    "HDB-FP32-dynamic": ("HDB FP32 dynamic", "#D55E00", "s", "--"),
    "HDB-FP32-flat": ("HDB FP32 flat", "#E69F00", "^", ":"),
    "HDB-FP64-dynamic": ("HDB FP64 dynamic", "#882255", "v", "-."),
    "HDB-FP64-flat": ("HDB FP64 flat", "#CC6677", "D", ":"),
    "Faiss-centered_ip": ("Faiss centered IP", "#009E73", "P", "--"),
    "Faiss-direction_rerank": ("Faiss direction", "#44AA99", "X", "-."),
    "Faiss-raw_ip": ("Faiss raw Lorentz-IP", "#999933", "<", ":"),
    "Milvus-Standalone-centered-IP": ("Milvus centered IP", "#332288", ">", "--"),
    "QuIVer框架精确RobustPrune": ("Exact-RobustPrune", "#666666", "h", "-."),
}
MAIN_METHODS = [
    "A1B1C1", "Faiss-centered_ip", "Faiss-direction_rerank", "Faiss-raw_ip",
    "Milvus-Standalone-centered-IP", "HDB-FP32-dynamic", "HDB-FP32-flat",
    "HDB-FP64-dynamic", "HDB-FP64-flat", "QuIVer框架精确RobustPrune",
    "SQ8/main", "FP32/main",
]


def format_qps(value, _):
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def frontier(rows):
    """保留召回和吞吐均未被其他实测点支配的配置。"""
    ordered = sorted(rows, key=lambda row: (row["recall"], row["qps"]), reverse=True)
    result = []
    best_qps = -1.0
    for row in ordered:
        if row["qps"] > best_qps:
            result.append(row)
            best_qps = row["qps"]
    return list(reversed(result))


def draw_series(axis, rows, method, scaling=False, label=None):
    name, color, marker, linestyle = STYLES[method]
    xkey = "threads" if scaling else "recall"
    rows = sorted(rows, key=lambda row: row[xkey])
    errors = {
        "yerr": [
            [max(0.0, row["qps"] - row["qps_min"]) for row in rows],
            [max(0.0, row["qps_max"] - row["qps"]) for row in rows],
        ]
    }
    if not scaling:
        errors["xerr"] = [
            [max(0.0, row["recall"] - row["recall_min"]) for row in rows],
            [max(0.0, row["recall_max"] - row["recall"]) for row in rows],
        ]
    # 单图结果不绘制跨图误差条。
    if rows[0]["graphs"] == 1:
        errors = {}
    axis.errorbar(
        [row[xkey] for row in rows], [row["qps"] for row in rows],
        label=label or name, color=color, marker=marker, linestyle=linestyle,
        linewidth=1.25, markersize=3.2, markeredgewidth=0.55,
        capsize=1.5, elinewidth=0.45, **errors,
    )


def new_axes(width, height, bottom=0.36):
    figure, axis = plt.subplots(figsize=(width, height))
    figure.subplots_adjust(left=0.24 if width < 3 else 0.19, right=0.94, top=0.98,
                           bottom=bottom)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="both", color="#DDDDDD", linewidth=0.45)
    axis.set_axisbelow(True)
    axis.set_ylabel("QPS")
    axis.yaxis.set_major_formatter(FuncFormatter(format_qps))
    axis.tick_params(length=2.5, width=0.6)
    return figure, axis


def save_figure(figure, axis, name, columns=2, legend_y=0.20):
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, legend_y),
                  frameon=False, ncol=columns, fontsize=7.2,
                  handlelength=1.7, handletextpad=0.45, columnspacing=0.9,
                  labelspacing=0.45, borderaxespad=0)
    figure.savefig(FIGURES / f"{name}.pdf", metadata={
        "Title": name,
        "Author": "Astrmira Tech.",
        "Subject": "实测帕累托前沿与查询扩展，数据来源见 data/provenance.json",
        "CreationDate": None,
        "ModDate": None,
    })
    plt.close(figure)
    print(f"已生成矢量图 {name}.pdf")


def pareto_figure(points, dataset, threads, methods, name, wide=False, representation=False):
    rows = [row for row in points if row["dataset"] == dataset
            and row["threads"] == threads and row["kind"] == "point"]
    methods = [method for method in methods if any(row["method"] == method for row in rows)]
    figure, axis = new_axes(3.32 if wide else 2.36, 2.75)
    for method in methods:
        selected = frontier([row for row in rows if row["method"] == method])
        label = "2-bit SM (same graph)" if representation and method == "A8/main" else None
        if method == "A8/symmetric" and name.startswith("ablation"):
            label = "Pelorus-Core"
        draw_series(axis, selected, method, label=label)
    axis.set_xlabel("Recall@10")
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.yaxis.set_major_formatter(FuncFormatter(format_qps))
    axis.yaxis.set_minor_locator(NullLocator())
    minimum = min(row["recall_min"] for row in rows if row["method"] in methods)
    axis.set_xlim(max(0.0, minimum - 0.005), 1.001)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xticks([0.94, 0.96, 0.98, 1.0] if minimum >= 0.93 else [0.92, 0.96, 1.0])
    if minimum > 0.96:
        axis.set_xticks([0.97, 0.98, 0.99, 1.0])
    axis.axvline(0.99, color="#AAAAAA", linewidth=0.6, linestyle=":", zorder=0)
    save_figure(figure, axis, name)


def scaling_figure(points, dataset):
    rows = [row for row in points if row["dataset"] == dataset and row["kind"] == "target"
            and row["target"] == 0.99 and row["method"] in MAIN_METHODS]
    methods = [method for method in MAIN_METHODS if any(row["method"] == method for row in rows)]
    # 预留四行图例所需的底部高度，同时保持两侧并发扩展图坐标框对齐
    figure, axis = new_axes(3.32, 2.82, bottom=0.41)
    for method in methods:
        draw_series(axis, [row for row in rows if row["method"] == method], method, scaling=True)
    axis.set_xlabel("Query threads")
    axis.set_xlim(0, 25)
    axis.set_ylim(bottom=0)
    axis.set_xticks([1, 4, 8, 16, 24])
    axis.set_xticks([2], minor=True)
    save_figure(figure, axis, f"scaling_{dataset}_r99", legend_y=0.24)


def comparison_figure(points, dataset, threads):
    """在同一数据集的完整召回视图和高召回细节中展示全部已测基线。"""
    rows = [row for row in points if row["dataset"] == dataset
            and row["threads"] == threads and row["kind"] == "point"]
    methods = [method for method in MAIN_METHODS if any(row["method"] == method for row in rows)]
    figure = plt.figure(figsize=(3.32, 4.7))
    full_axis = figure.add_axes((0.19, 0.65, 0.75, 0.32))
    detail_axis = figure.add_axes((0.19, 0.28, 0.75, 0.29))
    for method in methods:
        selected = frontier([row for row in rows if row["method"] == method])
        draw_series(full_axis, selected, method)
        detail = [row for row in selected if row["recall"] >= 0.9]
        if detail:
            draw_series(detail_axis, detail, method)
    for axis in [full_axis, detail_axis]:
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axis.yaxis.set_major_formatter(FuncFormatter(format_qps))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.set_ylabel("QPS")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#DDDDDD", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.tick_params(length=2.5, width=0.6)
    full_axis.set_xlim(-0.01, 1.01)
    full_axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
    full_axis.set_title("Full measured range", fontsize=8.5, pad=3)
    detail_axis.set_xlim(0.90, 1.001)
    detail_axis.set_xticks([0.9, 0.95, 1])
    detail_axis.set_xlabel("Recall@10")
    detail_axis.set_title("High-recall detail", fontsize=8.5, pad=3)
    detail_axis.axvline(0.99, color="#AAAAAA", linewidth=0.6, linestyle=":", zorder=0)
    handles, labels = full_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.195),
                  frameon=False, ncol=2, fontsize=7.0, borderaxespad=0,
                  handlelength=1.6, handletextpad=0.4, columnspacing=0.8,
                  labelspacing=0.45)
    name = f"pareto_{dataset}_{threads}t"
    figure.savefig(FIGURES / f"{name}.pdf", metadata={
        "Title": name, "Author": "Astrmira Tech.", "CreationDate": None, "ModDate": None,
    })
    plt.close(figure)
    print(f"已生成完整基线对比矢量图 {name}.pdf")


def comparison_table(source_name="baseline_summary.csv", output_name="baseline_comparison.tex",
                     datasets=("cohere_1m", "meru_1m"), methods=MAIN_METHODS):
    """从核验后的摘要生成主对比表，保持表格与图中方法一致。"""
    source = FIGURES / "data" / source_name
    dataset_labels = {
        "cohere_1m": "Cohere-1M", "meru_1m": "MERU-1M",
        "uniform_narrow_1m": "Uniform-Narrow", "uniform_wide_1m": "Uniform-Wide",
        "clustered_narrow_1m": "Clustered-Narrow", "clustered_wide_1m": "Clustered-Wide",
    }
    with source.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    lines = []
    records = {(row["dataset"], row["method"]): row for row in rows}
    for threads in [1, 24]:
        if threads == 1:
            lines.extend([
                r"\begin{tabular}{@{}llrrrrrrrrr@{}}",
                r"\multicolumn{11}{@{}l}{\textbf{(a) Single-thread QPS}} \\",
                r"\toprule",
                r"\textbf{Dataset} & \textbf{Method} & \textbf{$M/L_c$} & \textbf{\shortstack{Build\\32T (s)}} & \textbf{\shortstack{Peak\\Rec@10}} & \textbf{$\ge90\%$} & \textbf{$\ge95\%$} & \textbf{$\ge98\%$} & \textbf{$\ge99\%$} & \textbf{$\ge99.5\%$} & \textbf{\shortstack{p99 (ms)\\at $99\%$}} \\",
                r"\midrule",
            ])
        else:
            lines.extend([
                r"\par\vspace{5pt}",
                r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrrr@{}}",
                r"\multicolumn{7}{@{}l}{\textbf{(b) 24-thread QPS (queries/s)}} \\",
                r"\toprule",
                r"\textbf{Dataset} & \textbf{Method} & \textbf{$\ge90\%$} & \textbf{$\ge95\%$} & \textbf{$\ge98\%$} & \textbf{$\ge99\%$} & \textbf{$\ge99.5\%$} \\",
                r"\midrule",
            ])
        for dataset in datasets:
            if dataset != datasets[0]:
                lines.append(r"\midrule")
            for method in methods:
                row = records.get((dataset, method))
                if row is None:
                    continue
                label = dataset_labels[dataset] if method == methods[0] else ""
                name = STYLES[method][0].replace("\n", " ")
                values = [label, name]
                if threads == 1:
                    values.extend([
                        f'{row["M"]}/{row["efConstruction"]}',
                        f'{float(row["build_seconds"]):.1f}',
                        f'{100 * float(row["peak_recall"]):.3f}' + r"\%",
                    ])
                keys = [f"qps_{target}_{threads}t" for target in ["90", "95", "98"]]
                keys.extend([f"qps_{threads}t", f"qps_995_{threads}t"])
                for key in keys:
                    values.append(f'{float(row[key]):,.0f}' if row[key] else r"\missing")
                if threads == 1:
                    values.append(f'{float(row["p99_1t"]):.2f}' if row["p99_1t"] else r"\missing")
                lines.append(" & ".join(values) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}" if threads == 1 else r"\end{tabular*}"])
    folder = ROOT / "tables"
    folder.mkdir(exist_ok=True)
    (folder / output_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成实测比较表 {output_name}")


def main():
    plt.rcParams.update({
        "font.size": 8, "font.family": "DejaVu Sans",
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
    })
    with DATA.open(encoding="utf-8", newline="") as source:
        points = list(csv.DictReader(source))
    for row in points:
        for key in ["recall", "recall_min", "recall_max", "qps", "qps_min", "qps_max", "target"]:
            row[key] = float(row[key]) if row[key] else None
        row["threads"] = int(row["threads"])
        row["graphs"] = int(row["graphs"])
    for dataset in ["cohere_1m", "meru_1m"]:
        comparison_figure(points, dataset, 1)
    pareto_figure(points, "openai_1536", 1, ["A1B1C1"], "pareto_openai_1536_1t")
    comparison_figure(points, "cohere_1m", 24)
    for dataset in ["cohere_1m", "meru_1m"]:
        scaling_figure(points, dataset)
    pareto_figure(points, "cohere_1m", 1, [
        "A8/symmetric", "A8/colocated", "A8/main", "A8/no_asymmetric_gate",
        "A8构图关闭-connectivity_repair",
    ], "ablation_geometry_cohere_1m", wide=True)
    pareto_figure(points, "cohere_1m", 1, [
        "A8/main", "A8/samegraph_sq8", "A8/samegraph_fp32", "SQ8/main", "FP32/main",
    ], "ablation_representation_cohere_1m", wide=True, representation=True)
    comparison_table()
    comparison_table(
        source_name="synthetic_summary.csv", output_name="synthetic_comparison.tex",
        datasets=("uniform_narrow_1m", "uniform_wide_1m", "clustered_narrow_1m", "clustered_wide_1m"),
        methods=("A1B1C1", "A1B1C1-M32", "HDB-FP32-flat", "Faiss-centered_ip", "Milvus-Standalone-centered-IP"),
    )


if __name__ == "__main__":
    main()
