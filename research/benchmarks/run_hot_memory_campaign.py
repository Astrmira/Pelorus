"""中心化重排编码与内存、SSD矩阵，保存计划、逐请求结果与可校验断点。"""

import argparse
import ctypes
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

EFS = [
    32,
    64,
    96,
    112,
    120,
    128,
    144,
    160,
    164,
    168,
    176,
    192,
    200,
    208,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
]
MARGINS = [0.08, 0.12, 0.20]
TARGETS = [0.50, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
THREADS = [1, 2, 4, 8, 16, 24]
PRECISIONS = ["fp32_reference", "centered_sq8"]


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def child_setup():
    # 单个测量进程禁用透明大页，配置由子进程继承。
    if ctypes.CDLL(None, use_errno=True).prctl(41, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "禁用进程透明大页失败")


def frontier(rows):
    kept = []
    best = -1
    for row in sorted(rows, key=lambda r: (-r["recall_at_10"], -r["qps"])):
        if row["qps"] > best:
            kept.append(row)
            best = row["qps"]
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "matrix.json").read_text())
    precisions = config.get("precisions", PRECISIONS)
    backends = config.get("backends", ["memory"])
    binary = root / "benchmark"
    profile = root / "profile"
    available_cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {191})
    for folder in ["plans", "logs", "results", "snapshots", "profiles"]:
        (root / folder).mkdir(exist_ok=True)

    def execute(command, name, cpus):
        command = list(map(str, command))
        save(
            root / "state.json",
            {
                "status": "运行中",
                "job": name,
                "started": time.time(),
                "command": command,
            },
        )
        started = time.monotonic()
        with (root / "logs" / (name + ".log")).open("a") as log:
            process = subprocess.Popen(
                ["taskset", "-c", cpus, *command],
                stdout=log,
                stderr=subprocess.STDOUT,
                preexec_fn=child_setup,
            )
            _, status, usage = os.wait4(process.pid, 0)
            code = os.waitstatus_to_exitcode(status)
            process.returncode = code
        save(
            root / "logs" / (name + ".resources.json"),
            {
                "wall_seconds": time.monotonic() - started,
                "user_seconds": usage.ru_utime,
                "system_seconds": usage.ru_stime,
                "peak_rss_kib": usage.ru_maxrss,
                "major_faults": usage.ru_majflt,
                "minor_faults": usage.ru_minflt,
                "input_blocks": usage.ru_inblock,
                "output_blocks": usage.ru_oublock,
                "voluntary_switches": usage.ru_nvcsw,
                "involuntary_switches": usage.ru_nivcsw,
                "exit_code": code,
            },
        )
        if code:
            raise RuntimeError(f"作业{name}退出，状态码{code}")

    def measure(dataset, precision, split, threads, margin, points, backend="memory"):
        name = f'{dataset["name"]}_{precision}_{split}_{threads}t_m{int(margin*100):02}'
        if len(backends) > 1:
            name += "_" + backend
        target = root / "results" / (name + ".json")
        if target.exists():
            return json.loads(target.read_text())["rows"]
        rows = {}
        for path in sorted((root / "results").glob(name + ".part*.jsonl")):
            for line in path.read_text().splitlines(keepends=True):
                if not line.endswith("\n"):
                    break
                row = json.loads(line)
                key = (row["ef"], row["budget"])
                request = path.with_name(
                    path.stem + "." + row["condition"] + ".requests.csv"
                )
                if (
                    request.exists()
                    or request.with_suffix(request.suffix + ".gz").exists()
                ):
                    rows[key] = row
        missing = [p for p in points if tuple(p) not in rows]
        plan = {
            "index": str(
                root / "snapshots" / dataset["name"] / (precision + ".pelorus")
            ),
            "precision": precision,
            "backend": backend,
            "simd_regime": "native",
            "io_driver": "uring",
            "io_workers": 1,
            "dimensions": dataset["dim"],
            "queries": (
                dataset["tuning_queries"] if split == "tuning" else dataset["queries"]
            ),
            "warmup_queries": min(200, dataset["tuning_queries"] or dataset["queries"]),
            "warmup_file": (
                "tuning_queries.f32" if dataset["tuning_queries"] else "queries.f32"
            ),
            "query_split": split,
            "threads": threads,
            "conditions": [
                {"id": f"ef{ef}_b{budget}", "ef": ef, "budget": budget}
                for ef, budget in missing
            ],
        }
        if missing:
            attempt = len(list((root / "results").glob(name + ".part*.jsonl")))
            path = root / "results" / (name + f".part{attempt}.jsonl")
            plan_path = root / "plans" / (name + f".part{attempt}.json")
            save(plan_path, plan)
            execute(
                [
                    binary,
                    "--phase",
                    "run",
                    "--margin",
                    str(margin),
                    "--data",
                    dataset["path"],
                    "--plan",
                    plan_path,
                    "--output",
                    path,
                ],
                name,
                "160" if threads == 1 else f"160-{159+threads}",
            )
            for line in path.read_text().splitlines():
                row = json.loads(line)
                rows[row["ef"], row["budget"]] = row
            for request in (root / "results").glob(path.stem + ".*.requests.csv"):
                with request.open("rb") as source, gzip.open(
                    str(request) + ".gz", "wb", compresslevel=1
                ) as out:
                    import shutil

                    shutil.copyfileobj(source, out)
                request.unlink()
        result = [rows[tuple(point)] for point in points]
        for row in result:
            if backend != "memory":
                after = row["ssd_after"]
                if after["read_errors"] or after["ring_errors"]:
                    raise RuntimeError("SSD读取或队列出现错误")
                if (
                    after["cache_peak_bytes"] > after["cache_capacity_bytes"]
                    or after["pool_peak_live_bytes"] > after["pool_limit_bytes"]
                ):
                    raise RuntimeError("SSD缓存或在途缓冲超过预算")
        if split == "test":
            fingerprint_path = root / "query_fingerprints.json"
            saved = (
                json.loads(fingerprint_path.read_text())
                if fingerprint_path.exists()
                else {}
            )
            for row in result:
                key = f'{dataset["name"]}/{precision}/{margin:.2f}/{row["ef"]}/{row["budget"]}'
                value = [row["result_crc32"], row["recall_at_10"]]
                if key in saved and saved[key] != value:
                    raise RuntimeError(f"跨线程查询结果不一致{key}")
                saved[key] = value
            save(fingerprint_path, saved)
        if len(result) != len(points):
            raise RuntimeError("作业条件记录不全")
        save(target, {"plan": plan, "rows": result})
        print(f"完成{name}，共{len(result)}个条件", flush=True)
        return result

    environment = {
        "uname": list(os.uname()),
        "available_cpus": available_cpus,
        "controller_affinity": sorted(os.sched_getaffinity(0)),
        "THP_child_disabled": True,
    }
    for label, path in {
        "cpu_quota": "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "cpu_period": "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        "memory_limit": "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "THP_global": "/sys/kernel/mm/transparent_hugepage/enabled",
    }.items():
        if Path(path).exists():
            environment[label] = Path(path).read_text().strip()
    save(root / "environment.json", environment)
    all_rows = []
    for dataset in config["datasets"]:
        data = Path(dataset["path"])
        if not (data / "documents.f32").exists():
            raise FileNotFoundError(f"数据底库尚未恢复{data}")
        verification = root / (dataset["name"] + ".data.json")
        if not verification.exists():
            metadata = json.loads((data / "dataset.json").read_text())
            expected = metadata.get("sha256", {})
            files = {
                "documents.f32": dataset["n"] * dataset["dim"] * 4,
                "queries.f32": dataset["queries"] * dataset["dim"] * 4,
                "truth.u64": dataset["queries"] * 80,
            }
            if dataset["tuning_queries"]:
                files.update(
                    {
                        "tuning_queries.f32": dataset["tuning_queries"]
                        * dataset["dim"]
                        * 4,
                        "tuning_truth.u64": dataset["tuning_queries"] * 80,
                    }
                )
            hashes = {}
            for name, size in files.items():
                file = data / name
                if file.stat().st_size != size:
                    raise RuntimeError(f"数据形状不符{file}")
                hashes[name] = sha(file)
                if name in expected and hashes[name] != expected[name]:
                    raise RuntimeError(f"归档数据哈希不符{file}")
            save(verification, {"dataset": dataset, "sha256": hashes})
        prepared = root / "snapshots" / dataset["name"] / "preparation.json"
        if not prepared.exists():
            source = config.get("reference_root")
            preparation_args = (
                [
                    "--phase",
                    "encode",
                    "--source-index",
                    str(
                        Path(source)
                        / "snapshots"
                        / dataset["name"]
                        / "fp32_reference.pelorus"
                    ),
                ]
                if source
                else ["--phase", "prepare"]
            )
            execute(
                [
                    binary,
                    *preparation_args,
                    "--precisions",
                    ",".join(precisions),
                    "--data",
                    data,
                    "--output",
                    prepared.parent,
                    "--n",
                    str(dataset["n"]),
                    "--dim",
                    str(dataset["dim"]),
                ],
                dataset["name"] + "_prepare",
                "160-191",
            )
        grid = [(ef, b) for ef in EFS for b in [32, 64, 128] if b <= ef]
        split = "tuning" if dataset["tuning_queries"] else "test"
        tuning = []
        for precision in precisions:
            for backend in backends:
                for margin in MARGINS:
                    tuning += measure(
                        dataset, precision, split, 1, margin, grid, backend
                    )
        chosen = set()
        for precision, backend in ((p, b) for p in precisions for b in backends):
            group = [
                r
                for r in tuning
                if r["precision"] == precision and r["backend"] == backend
            ]
            keep = frontier(group)
            for threshold in TARGETS:
                eligible = sorted(
                    [r for r in group if r["recall_at_10"] >= threshold],
                    key=lambda r: -r["qps"],
                )
                keep += eligible[:2]
            for row in keep:
                chosen.add((row["sketch_margin"], row["ef"], row["budget"]))
        # 各精度与存储模式共同复测候选前沿，支持同参数和同召回比较。
        tests = []
        for threads in THREADS:
            for precision, backend in ((p, b) for p in precisions for b in backends):
                for margin in MARGINS:
                    points = sorted(
                        (ef, b) for m, ef, b in chosen if abs(m - margin) < 1e-6
                    )
                    if points:
                        tests += measure(
                            dataset, precision, "test", threads, margin, points, backend
                        )
        all_rows += tests
        for precision in precisions:
            candidates = [
                r
                for r in tests
                if r["threads"] == 1
                and r["precision"] == precision
                and r["backend"] == "memory"
            ]
            if not candidates:
                continue
            eligible = [r for r in candidates if r["recall_at_10"] >= 0.99]
            best = (
                max(eligible, key=lambda r: r["qps"])
                if eligible
                else max(candidates, key=lambda r: r["recall_at_10"])
            )
            name = dataset["name"] + "_" + precision
            done = root / "profiles" / (name + ".json")
            if not done.exists():
                plan = {
                    "index": str(prepared.parent / (precision + ".pelorus")),
                    "precision": precision,
                    "simd_regime": "native",
                    "dimensions": dataset["dim"],
                    "queries": min(256, dataset["queries"]),
                    "warmup_queries": min(200, dataset["tuning_queries"]),
                    "margin": best["sketch_margin"],
                    "expected_topology_crc32": best["topology_crc32"],
                }
                # 阶段入口的重排预算与选中配置一致。
                plan["budget"] = best["budget"]
                plan["loss_audit"] = True
                plan["pure_queries"] = 32
                plan_path = root / "plans" / (name + "_profile.json")
                save(plan_path, plan)
                execute(
                    [
                        profile,
                        "--data",
                        data,
                        "--plan",
                        plan_path,
                        "--ef",
                        str(best["ef"]),
                        "--output",
                        done,
                    ],
                    name + "_profile",
                    "160",
                )
        save(
            root / (dataset["name"] + ".complete.json"),
            {
                "completed": time.time(),
                "test_conditions": len(tests),
                "tuning_conditions": len(tuning),
                "selection": "独立调参前沿与各门槛最快两个点的精度和存储模式并集",
            },
        )
        save(root / "frontiers.json", all_rows)
    save(
        root / "state.json",
        {
            "status": "全部完成",
            "test_conditions": len(all_rows),
            "completed": time.time(),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"实验控制器中断，已同步条件可续跑，原因{error}", flush=True)
        raise
