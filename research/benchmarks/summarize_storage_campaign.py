"""汇总重排存储实验的实测帕累托点、等Recall表格及线程扩展。"""

import argparse
import csv
import hashlib
import json
import struct
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "fp16_cartesian": "FP16 Cartesian",
    "fp16_polar": "FP16 direction + FP64 radius",
    "centered_sq8": "Mean-centered SQ8",
    "centered_polar_sq4": "Centered SQ4 direction + FP64 radius",
    "memory": "RAM",
    "ssd128": "SSD / 128 MiB cache",
    "ssd512": "SSD / 512 MiB cache",
}
PRECISIONS = list(LABELS)[:4]
BACKENDS = list(LABELS)[4:]
COLORS = ["#2563a6", "#c65f18", "#16846b"]


def verify_requests(root, rows):
    references = {}
    audit = []
    sizes = dict(zip(PRECISIONS, [1024, 1032, 516, 268]))
    for row in rows:
        if row["record_bytes"] != sizes[row["precision"]]:
            raise RuntimeError("重排记录长度与精度标签不符")
        pattern = (f"curves_{row['precision']}_{row['backend']}_{row['threads']}t*."
                   f"{row['condition']}.requests.csv")
        files = list((root / "curves").glob(pattern))
        if len(files) != 1:
            raise RuntimeError(f"逐请求文件数量不符：{pattern}")
        digest = hashlib.sha256()
        crc = 0
        hits = 0
        count = 0
        with files[0].open(newline="", encoding="utf-8") as source:
            for record in csv.DictReader(source):
                query_id = int(record["query_id"])
                ids = [int(value) for value in record["ids"].split("|")]
                scores = [int(value) for value in record["score_bits"].split("|")]
                if query_id != count or len(set(ids)) != 10 or len(scores) != 10:
                    raise RuntimeError("逐请求标识、数量或顺序无效")
                payload = struct.pack("<Q", query_id) + b"".join(
                    struct.pack("<QI", identifier, score) for identifier, score in zip(ids, scores))
                digest.update(payload)
                crc = zlib.crc32(payload, crc)
                hits += int(record["hits"])
                count += 1
        if count != row["queries"] or crc != row["result_crc32"]:
            raise RuntimeError("逐请求记录与结果指纹不符")
        if abs(hits / (count * 10) - row["recall_at_10"]) > 1e-12:
            raise RuntimeError("逐请求命中数与汇总Recall不符")
        key = (row["precision"], row["ef"], row["budget"], count)
        signature = digest.hexdigest()
        if key in references and references[key] != signature:
            raise RuntimeError("跨存储模式的逐请求标识或分数不一致")
        references[key] = signature
        audit.append({"file": files[0].relative_to(root).as_posix(), "queries": count,
                      "result_sha256": signature, "crc32": crc})
    (root / "request_validation.json").write_text(
        json.dumps({"conditions": len(audit), "agreement": True, "files": audit}, indent=2),
        encoding="utf-8")


def pareto(rows):
    return sorted([row for row in rows if not any(
        other["recall_at_10"] >= row["recall_at_10"] and other["qps"] >= row["qps"]
        and (other["recall_at_10"] > row["recall_at_10"] or other["qps"] > row["qps"])
        for other in rows)], key=lambda row: row["recall_at_10"])


def export_csv(path, rows):
    fields = ["precision", "backend", "threads", "ef", "budget", "recall_at_10", "qps",
              "p50_ms", "p95_ms", "p99_ms", "p999_ms", "cpu_seconds", "read_mib_per_1000",
              "ready_hit_percent", "rss_mib", "anonymous_rss_mib", "cache_mib", "pool_peak_mib",
              "record_bytes", "navigation_bytes", "result_crc32"]
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = {name: row.get(name) for name in fields}
            mem = row["process_memory_kib"]
            item["rss_mib"] = mem.get("VmRSS", 0) / 1024
            item["anonymous_rss_mib"] = mem.get("RssAnon", 0) / 1024
            item["read_mib_per_1000"] = row["process_io_delta"].get("read_bytes", 0) / 2**20 / row["queries"] * 1000
            delta = row["ssd_io_delta"]
            if delta:
                demand = sum(delta[name] for name in
                             ["cache_hits", "cache_misses", "pending_hits", "query_page_hits"])
                item["ready_hit_percent"] = (delta["cache_hits"] + delta["query_page_hits"]) / max(demand, 1) * 100
                item["read_mib_per_1000"] = delta["read_bytes"] / 2**20 / row["queries"] * 1000
                item["cache_mib"] = row["ssd_after"]["cache_current_bytes"] / 2**20
                item["pool_peak_mib"] = row["ssd_after"]["pool_peak_live_bytes"] / 2**20
            writer.writerow(item)


def main():
    parser = argparse.ArgumentParser(description="生成存储实验汇总")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    rows = []
    for phase in ["curves", "scaling"]:
        for path in sorted((root / phase).glob("*.jsonl")):
            if not path.with_suffix(".done.json").exists():
                continue
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
    if not rows:
        raise RuntimeError("缺少完成测量")
    verify_requests(root, rows)
    export_csv(root / "measurements.csv", rows)
    fronts = []
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    for precision, ax in zip(PRECISIONS, axes.flat):
        for backend, color in zip(BACKENDS, COLORS):
            candidates = [row for row in rows if row["precision"] == precision
                          and row["backend"] == backend and row["threads"] == 1]
            frontier = pareto(candidates)
            fronts += frontier
            ax.plot([row["recall_at_10"] * 100 for row in frontier],
                    [row["qps"] for row in frontier], "o-", color=color, label=LABELS[backend],
                    linewidth=1.5, markersize=4)
        ax.set_title(LABELS[precision], fontsize=11)
        ax.set_xlabel("Recall@10 (%)")
        ax.set_ylabel("QPS (1 query thread)")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(root / "pareto.pdf")
    fig.savefig(root / "pareto.png", dpi=180)
    plt.close(fig)
    export_csv(root / "pareto_points.csv", fronts)
    table = ["# MERU–ImageNet–1M 重排存储测量", "",
             "每点执行一次计时，使用10,000条测试查询和2,000条独立预热查询。"
             "下表按达到目标Recall的最快已测点取值。", "",
             "| 精度 | 存储 | 目标Recall | 实测Recall | 1T QPS | P99 ms | ef / B |", 
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for precision in PRECISIONS:
        for target in ([0.85, 0.90] if precision == "centered_polar_sq4" else [0.95, 0.99, 0.995]):
            for backend in BACKENDS:
                eligible = [row for row in rows if row["precision"] == precision
                            and row["backend"] == backend and row["threads"] == 1
                            and row["recall_at_10"] >= target]
                if eligible:
                    best = max(eligible, key=lambda row: row["qps"])
                    table.append(f"| {LABELS[precision]} | {LABELS[backend]} | {target:.1%} | "
                                 f"{best['recall_at_10']:.3%} | {best['qps']:,.1f} | "
                                 f"{best['p99_ms']:.3f} | {best['ef']} / {best['budget']} |")
    if any(row["threads"] == 24 for row in rows):
        table += ["", "## 高召回工作点的24线程结果", "",
                  "| 精度 | 存储 | 目标Recall | 实测Recall | 24T QPS | P99 ms | ef / B |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
    for precision in PRECISIONS:
        target = 0.90 if precision == "centered_polar_sq4" else 0.99
        for backend in BACKENDS:
            eligible = [row for row in rows if row["precision"] == precision
                        and row["backend"] == backend and row["threads"] == 24
                        and row["recall_at_10"] >= target]
            if eligible:
                best = max(eligible, key=lambda row: row["qps"])
                table.append(f"| {LABELS[precision]} | {LABELS[backend]} | {target:.0%} | "
                             f"{best['recall_at_10']:.3%} | {best['qps']:,.1f} | "
                             f"{best['p99_ms']:.3f} | {best['ef']} / {best['budget']} |")
    table += ["", "需求读取到达时已经就绪的页记为缓存命中，包含已完成的预读。"
              "真实读盘开销以read_bytes及进程read_bytes增量记录。"
              "缓存页载荷、缓冲池峰值、进程RSS与原始向量映射分别统计。", ""]
    (root / "report.md").write_text("\n".join(table), encoding="utf-8")
    print(f"已汇总 {len(rows)} 个完成条件，生成帕累托图和实测点表")


if __name__ == "__main__":
    main()
