"""在固定候选集合上测量冷向量表示的纯重排精度损失。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits


MODES = (
    ("cartesian", "fp32"),
    ("cartesian", "fp16"),
    ("cartesian", "bf16"),
    ("cartesian", "sq8"),
    ("cartesian", "sq4"),
    ("polar", "fp32"),
    ("polar", "fp16"),
    ("polar", "bf16"),
    ("polar", "sq8"),
    ("polar", "sq4"),
) + tuple((geometry, precision) for geometry in ("cartesian", "polar")
          for precision in ("centered_sq8", "centered_sq4", "affine8_g128", "affine4_g128",
                            "affine8_g64", "affine4_g64")) + tuple(
    ("cartesian_norm", precision)
    for precision in ("affine8_g128", "affine4_g128", "affine8_g64", "affine4_g64")
)


def fingerprint(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def decode_precision(values: np.ndarray, precision: str) -> tuple[np.ndarray, int]:
    """实际进行舍入与解码，区分零中心、集合均值中心和逐组仿射编码。"""
    source = np.asarray(values, dtype=np.float32)
    dimension = source.shape[1]
    if precision == "fp32":
        return source.astype(np.float64), dimension * 4
    if precision == "fp16":
        return source.astype(np.float16).astype(np.float64), dimension * 2
    if precision == "bf16":
        bits = source.view(np.uint32)
        rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
        decoded = (rounded & np.uint32(0xFFFF0000)).view(np.float32)
        return decoded.astype(np.float64), dimension * 2
    if precision.startswith("centered_"):
        center = source.mean(axis=0, dtype=np.float64).astype(np.float32)
        decoded, payload = decode_precision(source - center, precision.removeprefix("centered_"))
        return (decoded.astype(np.float32) + center).astype(np.float64), payload
    affine = precision.startswith("affine")
    four_bit = precision == "sq4" or precision.startswith("affine4_")
    limit = 7 if four_bit else 127
    if affine:
        group = int(precision.split("_g")[1])
        assert dimension % group == 0
        source = source.reshape(len(source), dimension // group, group)
        low, high = source.min(axis=2), source.max(axis=2)
        bias = ((high + low) * np.float32(0.5)).astype(np.float32)
        scales = ((high - low) / (2 * limit)).astype(np.float32)
    else:
        bias = np.zeros(len(source), dtype=np.float32)
        scales = (np.max(np.abs(source), axis=1) / limit).astype(np.float32)
    scales = np.where(scales > 0, scales, np.float32(1))
    codes = np.clip(np.rint((source - bias[..., None]) / scales[..., None]), -limit, limit).astype(np.int8)
    codes = codes.reshape(len(source), dimension)
    if four_bit:
        # 四位码使用偏置七，保留一个未使用码字；解码前真实打包成半字节。
        unsigned = (codes.astype(np.int16) + 7).astype(np.uint8)
        if dimension % 2:
            unsigned = np.pad(unsigned, ((0, 0), (0, 1)), constant_values=7)
        packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
        unpacked = np.empty_like(unsigned)
        unpacked[:, 0::2] = packed & 15
        unpacked[:, 1::2] = packed >> 4
        codes = (unpacked[:, :dimension].astype(np.int16) - 7).astype(np.int8)
    decoded = codes.reshape(source.shape).astype(np.float32) * scales[..., None] + bias[..., None]
    decoded = decoded.reshape(len(source), dimension).astype(np.float64)
    code_bytes = (dimension + 1) // 2 if four_bit else dimension
    parameter_bytes = dimension // group * 8 if affine else 4
    payload = code_bytes + parameter_bytes
    return decoded, payload


def representation(vectors: np.ndarray, geometry: str, precision: str) -> dict:
    radii = np.linalg.norm(vectors, axis=1)
    if geometry == "polar":
        units = np.divide(vectors, radii[:, None], out=np.zeros_like(vectors), where=radii[:, None] > 0)
        decoded, payload = decode_precision(units, precision)
        norms = np.linalg.norm(decoded, axis=1)
        decoded = np.divide(decoded, norms[:, None], out=np.zeros_like(decoded), where=norms[:, None] > 0)
        payload += 8
        factors = 1.0 / (1.0 - radii * radii)
        valid = np.isfinite(norms) & ((norms > 0) | (radii == 0))
    else:
        decoded, payload = decode_precision(vectors, precision)
        if geometry == "cartesian_norm":
            norms = np.linalg.norm(decoded, axis=1)
            correction = np.divide(radii, norms, out=np.ones_like(radii), where=norms > 0).astype(np.float32)
            decoded *= correction[:, None]
            payload += 4
        squared = np.einsum("ij,ij->i", decoded, decoded)
        radii = np.sqrt(squared)
        valid = np.isfinite(squared) & (squared < 1.0)
        factors = np.divide(1.0, 1.0 - squared, out=np.zeros_like(squared), where=valid)
    return {
        "vectors": np.ascontiguousarray(decoded),
        "radii": radii,
        "factors": factors,
        "valid": valid,
        "bytes_per_vector": payload,
        "shared_metadata_bytes": vectors.shape[1] * 4 if precision.startswith("centered_") else 0,
        "geometry": geometry,
    }


def score_block(queries: np.ndarray, encoded: dict) -> np.ndarray:
    """所有模式使用原始查询，统一在 FP64 中计算单调双曲代理距离。"""
    q_squared = np.einsum("ij,ij->i", queries, queries)
    q_factor = 1.0 / (1.0 - q_squared)
    data = encoded["vectors"]
    if encoded["geometry"] == "polar":
        q_radius = np.sqrt(q_squared)
        units = np.divide(queries, q_radius[:, None], out=np.zeros_like(queries), where=q_radius[:, None] > 0)
        unit_squared = np.einsum("ij,ij->i", data, data)
        query_unit_squared = np.einsum("ij,ij->i", units, units)
        squared = query_unit_squared[:, None] + unit_squared[None, :] - 2 * (units @ data.T)
        squared = np.maximum(squared, 0.0)
        scores = (q_radius[:, None] - encoded["radii"][None, :]) ** 2
        scores += q_radius[:, None] * encoded["radii"][None, :] * squared
    else:
        squared = np.einsum("ij,ij->i", data, data)
        scores = q_squared[:, None] + squared[None, :] - 2 * (queries @ data.T)
        np.maximum(scores, 0.0, out=scores)
    scores *= q_factor[:, None]
    scores *= encoded["factors"][None, :]
    scores[:, ~encoded["valid"]] = np.inf
    return scores


def ordered_topk(scores: np.ndarray, ids: np.ndarray, k: int) -> np.ndarray:
    partial = np.argpartition(scores, k - 1, axis=1)[:, :k]
    result = np.empty_like(partial)
    for row, selected in enumerate(partial):
        boundary = scores[row, selected].max()
        eligible = np.flatnonzero(scores[row] <= boundary)
        result[row] = eligible[np.lexsort((ids[eligible], scores[row, eligible]))[:k]]
    return result


def direct_proxy(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    squared = np.sum((vectors - query) ** 2, axis=1, dtype=np.float64)
    return squared / ((1.0 - np.sum(query * query)) * (1.0 - np.sum(vectors * vectors, axis=1)))


def evaluate(queries: np.ndarray, documents: np.ndarray, document_ids: np.ndarray,
             geometry: str, precision: str, batch: int, k: int) -> tuple[np.ndarray, dict]:
    start = time.perf_counter()
    encoded = representation(documents, geometry, precision)
    prepare_seconds = time.perf_counter() - start
    assert np.count_nonzero(encoded["valid"]) >= k
    output = np.empty((len(queries), k), dtype=np.int64)
    max_direct_relative_error = 0.0
    scan_start = time.perf_counter()
    for begin in range(0, len(queries), batch):
        end = min(begin + batch, len(queries))
        scores = score_block(queries[begin:end], encoded)
        output[begin:end] = ordered_topk(scores, document_ids, k)
        if geometry == "cartesian" and precision == "fp32":
            # 用坐标差直接累加核对代数展开，并核对每条查询的前十名。
            for local, selected in enumerate(output[begin:end]):
                direct = direct_proxy(queries[begin + local], documents[selected])
                expanded = scores[local, selected]
                relative = np.max(np.abs(direct - expanded) / np.maximum(direct, 1e-30))
                max_direct_relative_error = max(max_direct_relative_error, float(relative))
                assert np.allclose(direct, expanded, rtol=1e-10, atol=1e-12)
                assert np.array_equal(np.lexsort((document_ids[selected], direct)), np.arange(k))
    result = {
        "mode": f"{geometry}_{precision}",
        "geometry": geometry,
        "storage_precision": precision,
        "distance_accumulation": "fp64",
        "bytes_per_vector": encoded["bytes_per_vector"],
        "shared_metadata_bytes": encoded["shared_metadata_bytes"],
        "invalid_vectors": int(np.count_nonzero(~encoded["valid"])),
        "prepare_seconds": prepare_seconds,
        "scan_seconds": time.perf_counter() - scan_start,
        "baseline_direct_relative_error_max": max_direct_relative_error,
    }
    return output, result


def comparison(reference: np.ndarray, actual: np.ndarray) -> tuple[dict, np.ndarray]:
    k = reference.shape[1]
    hits = np.array([np.intersect1d(left, right, assume_unique=True).size
                     for left, right in zip(reference, actual)], dtype=np.int64)
    return {
        "recall_at_k": float(hits.mean() / k),
        "loss_percentage_points": float((k - hits).mean() * 100 / k),
        "lost_neighbors": int(np.sum(k - hits)),
        "affected_queries": int(np.count_nonzero(hits < k)),
        "top1_agreement": float(np.mean(reference[:, 0] == actual[:, 0])),
        "ordered_topk_agreement": float(np.mean(np.all(reference == actual, axis=1))),
        "per_query_recall_min": float(hits.min() / k),
    }, hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=100000)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    mapping_path = args.data / "mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert mapping["canonical_curvature"] == -1.0
    dimension = mapping["dimensions"]
    assert 10 <= args.documents <= mapping["base_count"]
    assert 1 <= args.queries <= mapping["query_count"]
    assert args.batch > 0 and args.threads > 0
    raw_docs = np.memmap(args.data / "documents.f32", dtype="<f4", mode="r",
                         shape=(mapping["base_count"], dimension))
    raw_queries = np.memmap(args.data / "queries.f32", dtype="<f4", mode="r",
                            shape=(mapping["query_count"], dimension))
    rng = np.random.default_rng(args.seed)
    document_ids = np.sort(rng.choice(len(raw_docs), args.documents, replace=False))
    query_ids = np.sort(rng.choice(len(raw_queries), args.queries, replace=False))
    documents = np.asarray(raw_docs[document_ids], dtype=np.float64)
    queries = np.asarray(raw_queries[query_ids], dtype=np.float64)
    for values in (documents, queries):
        assert np.isfinite(values).all()
        assert np.all(np.sum(values * values, axis=1) < 1)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "selection.npz", document_ids=document_ids, query_ids=query_ids)
    results = []
    per_query = []
    reference = None
    metadata = {
        "protocol": "固定文档子集全扫描，原始 FP32 查询，FP64 距离累加，按代理距离与原始行号稳定排序",
        "dataset": str(args.data.resolve()),
        "documents": args.documents, "queries": args.queries, "dimensions": dimension,
        "k": 10, "seed": args.seed, "threads": args.threads, "batch": args.batch,
        "mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        "documents_sha256": fingerprint(documents.astype(np.float32)),
        "queries_sha256": fingerprint(queries.astype(np.float32)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "numpy": np.__version__, "python": platform.python_version(), "platform": platform.platform(),
        "polar_radius_storage": "每条向量额外保存原始 FP64 欧氏半径，方向解码后重新归一化",
        "scalar_quantization": "逐向量最大绝对值对称量化，SQ8 使用 -127..127，SQ4 使用 -7..7，FP32 尺度",
        "centered_quantization": "仅用文档子集训练逐维全局均值并存储为 FP32，逐向量量化残差，另计共享均值空间",
        "affine_quantization": "每条向量每 128 或 64 维独立 minmax，FP32 中点偏置及尺度，码范围 -127..127 或 -7..7",
        "norm_correction": "cartesian_norm 额外保存逐向量 FP32 系数，将解码向量范数恢复到原始范数附近",
        "timing_scope": "离线 NumPy 全扫描耗时，包含候选选择，不用于 ANN 或 SSD QPS 比较",
        "results": results,
    }
    with threadpool_limits(limits=args.threads, user_api="blas"):
        metadata["threadpools"] = threadpool_info()
        for geometry, precision in MODES:
            print(f"开始测量 {geometry}_{precision}", flush=True)
            actual, result = evaluate(queries, documents, document_ids, geometry, precision, args.batch, 10)
            if reference is None:
                reference = actual.copy()
                np.save(args.output / "truth_subset.u64.npy", document_ids[reference].astype("<u8"))
            metrics, hits = comparison(reference, actual)
            result.update(metrics)
            results.append(result)
            for index, query_id in enumerate(query_ids):
                per_query.append({"mode": result["mode"], "query_id": int(query_id),
                                  "hits": int(hits[index]),
                                  "top1_agreement": bool(reference[index, 0] == actual[index, 0]),
                                  "ordered_topk_agreement": bool(np.array_equal(reference[index], actual[index]))})
            np.save(args.output / f"{result['mode']}_top10.npy", document_ids[actual].astype("<u8"))
            (args.output / "results.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"完成 {result['mode']}，Recall@10={metrics['recall_at_k']:.6%}，"
                  f"损失={metrics['loss_percentage_points']:.4f} 个百分点，"
                  f"漏失近邻={metrics['lost_neighbors']}，耗时={result['scan_seconds']:.2f} 秒", flush=True)
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (args.output / "per_query.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)
    assert len(results) == len(MODES)
    print(f"全部测量完成，结果位于 {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
