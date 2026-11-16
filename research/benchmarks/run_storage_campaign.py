"""顺序测量同一图上的重排编码、SSD 缓存及查询线程，保留逐请求结果。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


PRECISIONS = ["fp16_cartesian", "fp16_polar", "centered_sq8", "centered_polar_sq4"]
BACKENDS = ["memory", "ssd128", "ssd512"]
GRID = [(32, 32), (64, 64), (128, 128), (256, 128), (512, 256), (1024, 256)]
THREADS = [2, 4, 8, 16, 24]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conditions(points):
    return [{"id": f"ef{ef}_b{budget}", "ef": ef, "budget": budget} for ef, budget in points]


def execute(root, binary, data, precision, backend, threads, points, phase):
    name = f"{phase}_{precision}_{backend}_{threads}t"
    directory = root / phase
    directory.mkdir(exist_ok=True)
    output = directory / (name + ".jsonl")
    done = directory / (name + ".done.json")
    plan = {
        "index": str(root / "snapshots" / (precision + ".pelorus")),
        "precision": precision, "backend": backend, "dimensions": 512,
        "queries": 100 if phase == "pilot" else 10000,
        "warmup_queries": 100 if phase == "pilot" else 2000,
        "threads": threads, "conditions": conditions(points),
    }
    plan_path = root / "plans" / (name + ".json")
    if done.exists():
        saved = json.loads(done.read_text())
        if saved["sha256"] != fingerprint(output):
            raise RuntimeError(f"完成记录与结果指纹不一致：{name}")
        previous = dict(saved["plan"])
        previous["conditions"] = plan["conditions"]
        if previous != plan:
            raise RuntimeError(f"数据、存储或线程协议发生变化：{name}")
    # 中断后复用已同步的完整JSONL行，只补缺少的条件。
    rows = []
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            if (row["precision"], row["backend"], row["threads"], row["queries"]) != (
                    precision, backend, threads, plan["queries"]):
                raise RuntimeError(f"已有结果的协议不符：{name}")
            rows.append(row)
        validate(rows)
    # 控制器自身被终止时，子进程已同步的批次结果也可继续使用。
    known = {(row["ef"], row["budget"]): row for row in rows}
    for partial in sorted(directory.glob(name + ".remaining*.partial")):
        for line in partial.read_text().splitlines(keepends=True):
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            if (row["precision"], row["backend"], row["threads"], row["queries"]) != (
                    precision, backend, threads, plan["queries"]):
                raise RuntimeError(f"中断批次的协议不符：{name}")
            key = (row["ef"], row["budget"])
            if key in known:
                validate([known[key], row])
            else:
                rows.append(row)
                known[key] = row
    if rows:
        temporary = output.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        temporary.replace(output)
    completed = {(row["ef"], row["budget"]) for row in rows}
    missing = [point for point in points if tuple(point) not in completed]
    write_json(plan_path, plan)
    if not missing:
        write_json(done, {"plan": plan, "sha256": fingerprint(output), "completed": time.time()})
        return rows
    batch_plan = dict(plan, conditions=conditions(missing))
    batch_path = root / "plans" / (name + f".remaining{len(rows)}.json")
    batch_output = directory / (name + f".remaining{len(rows)}.partial")
    write_json(batch_path, batch_plan)
    cpus = ",".join(str(cpu) for cpu in range(160, 160 + threads))
    command = ["taskset", "-c", cpus, str(binary), "--phase", "run", "--data", str(data),
               "--plan", str(batch_path), "--output", str(batch_output)]
    write_json(root / "state.json", {"status": "运行中", "job": name, "started": time.time()})
    print(f"开始 {name}，补测 {len(missing)} 个条件，已完成 {len(rows)} 个", flush=True)
    with (root / "logs" / (name + ".log")).open("a", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        # 完整批次行留在.partial中，下次启动按条件恢复。
        raise RuntimeError(f"测量进程退出：{name}，状态码{process.returncode}")
    additions = [json.loads(line) for line in batch_output.read_text().splitlines()]
    if len(additions) != len(missing):
        raise RuntimeError(f"结果数量不符：{name}")
    rows += additions
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(output)
    write_json(done, {"plan": plan, "sha256": fingerprint(output), "completed": time.time()})
    for row in additions:
        print(f"完成 {name} {row['condition']}，Recall {row['recall_at_10']:.5f}，"
              f"QPS {row['qps']:.1f}，P99 {row['p99_ms']:.3f} 毫秒", flush=True)
    return rows


def validate(rows):
    reference = {}
    for row in rows:
        key = (row["precision"], row["ef"], row["budget"], row["queries"])
        value = (row["result_crc32"], row["recall_at_10"])
        if key in reference and reference[key] != value:
            raise RuntimeError(f"跨存储或线程结果不一致：{key}")
        reference[key] = value
        if row["backend"] != "memory":
            after = row["ssd_after"]
            if after["read_errors"] or after["cache_peak_bytes"] > after["cache_capacity_bytes"]:
                raise RuntimeError("SSD 读取错误或缓存超预算")
            if after["pool_peak_live_bytes"] > after["pool_limit_bytes"]:
                raise RuntimeError("SSD 缓冲池超预算")


def select_points(rows, precision):
    targets = [0.85, 0.90] if precision == "centered_polar_sq4" else [0.95, 0.99]
    selected = []
    attained = []
    for target in targets:
        eligible = [row for row in rows if row["recall_at_10"] >= target]
        if eligible:
            best = max(eligible, key=lambda row: row["qps"])
            attained.append({"target": target, "attained": True, "condition": best["condition"]})
        else:
            best = max(rows, key=lambda row: row["recall_at_10"])
            attained.append({"target": target, "attained": False, "condition": best["condition"]})
        point = (best["ef"], best["budget"])
        if point not in selected:
            selected.append(point)
    return selected, attained


def main():
    parser = argparse.ArgumentParser(description="MERU SSD 缓存实验控制器")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--curves-only", action="store_true", help="完成单线程曲线后结束")
    args = parser.parse_args()
    root, data = args.root.resolve(), args.data.resolve()
    binary = root / "target/release/examples/benchmark_storage_sweep"
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {191})
    try:
        # 准备阶段独占32个构图核心，后续作业按线程数固定CPU预算。
        while not (root / "prepare.done.json").exists():
            time.sleep(10)
        if json.loads((root / "prepare.done.json").read_text())["exit_code"] != 0:
            raise RuntimeError("Linux 核心验证或快照准备失败")
        hashes = {name: fingerprint(data / name) for name in
                  ["documents.f32", "queries.f32", "truth.u64", "tuning_queries.f32"]}
        write_json(root / "protocol.json", {
            "dataset": "MERU–ImageNet–1M", "data_sha256": hashes,
            "binary_sha256": fingerprint(binary), "grid": GRID,
            "query_threads": [1] if args.curves_only else [1] + THREADS,
            "queries": 10000, "disjoint_warmup_queries": 2000, "timing_passes": 1,
            "cache_mib": [128, 512], "inflight_mib": 32, "io_workers": 4,
            "cpu_start": 160, "cpu_budget": "查询及I/O线程共享所列T个CPU",
            "prefetch_window": 32, "build_threads": 32,
            "selection": "每精度以1线程内存曲线选择两个目标Recall下最快已测点，跨存储沿用相同ef与B",
        })
        pilot = []
        for backend in BACKENDS:
            pilot += execute(root, binary, data, "fp16_polar", backend, 1,
                             [(128, 128), (512, 256)], "pilot")
        validate(pilot)
        write_json(root / "pilot_summary.json", pilot)
        all_rows = []
        selected = {}
        for precision in PRECISIONS:
            memory_rows = execute(root, binary, data, precision, "memory", 1, GRID, "curves")
            all_rows += memory_rows
            points, attained = select_points(memory_rows, precision)
            selected[precision] = {"points": points, "targets": attained}
            for backend in BACKENDS[1:]:
                all_rows += execute(root, binary, data, precision, backend, 1, GRID, "curves")
                validate(all_rows)
            write_json(root / "working_points.json", selected)
        if not args.curves_only:
            for precision in PRECISIONS:
                for threads in THREADS:
                    for backend in BACKENDS:
                        all_rows += execute(root, binary, data, precision, backend, threads,
                                            selected[precision]["points"], "scaling")
                        validate(all_rows)
                        write_json(root / "summary.json", all_rows)
        write_json(root / "summary.json", all_rows)
        write_json(root / "state.json", {"status": "完成", "conditions": len(all_rows),
                                        "completed": time.time(), "result_agreement": True})
        print(f"全部完成，共 {len(all_rows)} 个条件，跨存储和线程结果指纹一致", flush=True)
    except Exception as error:
        write_json(root / "state.json", {"status": "失败", "error": str(error), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
