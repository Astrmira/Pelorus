import argparse
import hashlib
import json
import math
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "longmemeval_encoding.yml"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class ApiResult:
    vectors: list[list[float]]
    model: str | None
    usage: dict[str, Any]


def _load_embedding_api_setting(name: str) -> str:
    """优先读取环境变量，再读取本地忽略文件中的服务配置。"""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    local_config = ROOT / ".env"
    if local_config.exists():
        for line in local_config.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key == name:
                value = value.strip().strip("\"'")
                if value:
                    return value
    raise ValueError(f"请设置 {name}，或在 research/.env 中配置同名字段")


def load_embedding_api_key() -> str:
    """读取编码服务密钥。"""
    return _load_embedding_api_setting("QUIVER_EMBEDDING_API_KEY")


def load_embedding_api_base_url() -> str:
    """读取编码服务地址。"""
    return _load_embedding_api_setting("QUIVER_EMBEDDING_API_BASE_URL")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def make_batches(
    rows: list[dict[str, Any]], batch_size: int, max_batch_tokens: int
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    batch: list[dict[str, Any]] = []
    token_total = 0
    for row in rows:
        tokens = estimate_tokens(str(row["text"]))
        if batch and (len(batch) >= batch_size or token_total + tokens > max_batch_tokens):
            batches.append(batch)
            batch = []
            token_total = 0
        batch.append(row)
        token_total += tokens
    if batch:
        batches.append(batch)
    return batches


def load_cache(path: Path, expected_dimensions: int) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        vector = row.get("embedding")
        if not isinstance(vector, list) or len(vector) != expected_dimensions:
            raise ValueError(f"缓存第 {line_number} 行向量维度无效：{path}")
        records[str(row["id"])] = row
    return records


def retry_delay(error: urllib.error.HTTPError, attempt: int, base: float, maximum: float) -> float:
    retry_after = error.headers.get("Retry-After")
    if retry_after:
        try:
            return min(maximum, max(0.0, float(retry_after)))
        except ValueError:
            pass
    return min(maximum, base * (2 ** (attempt - 1)) + random.uniform(0.0, base))


def request_embeddings(
    url: str,
    api_key: str,
    user_agent: str,
    model: str,
    dimensions: int,
    texts: list[str],
    max_attempts: int,
    backoff_base: float,
    backoff_max: float,
) -> ApiResult:
    payload = json.dumps(
        {
            "model": model,
            "input": texts,
            "encoding_format": "float",
            "dimensions": dimensions,
        }
    ).encode("utf-8")
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/plain, */*",
                "User-Agent": user_agent,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
            data = sorted(body["data"], key=lambda row: int(row["index"]))
            vectors = [row["embedding"] for row in data]
            if len(vectors) != len(texts):
                raise ValueError(f"API 返回 {len(vectors)} 个向量，但请求了 {len(texts)} 条文本")
            if any(
                len(vector) != dimensions
                or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector)
                for vector in vectors
            ):
                raise ValueError("API 返回了维度错误或包含非有限数值的向量")
            return ApiResult(vectors=vectors, model=body.get("model"), usage=body.get("usage", {}))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")[:500]
            if error.code not in RETRYABLE_STATUS or attempt == max_attempts:
                raise RuntimeError(f"Embedding API HTTP {error.code}：{body}") from error
            delay = retry_delay(error, attempt, backoff_base, backoff_max)
            print(f"请求失败 HTTP {error.code}，{delay:.1f} 秒后进行第 {attempt + 1} 次尝试。")
            time.sleep(delay)
        except (TimeoutError, urllib.error.URLError) as error:
            if attempt == max_attempts:
                raise RuntimeError(f"Embedding API 网络错误：{error}") from error
            delay = min(backoff_max, backoff_base * (2 ** (attempt - 1)) + random.uniform(0.0, backoff_base))
            print(f"网络请求失败，{delay:.1f} 秒后进行第 {attempt + 1} 次尝试：{error}")
            time.sleep(delay)
    raise AssertionError("不可达分支")


def append_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def encode_rows(
    name: str,
    rows: list[dict[str, Any]],
    cache_path: Path,
    config: dict[str, Any],
    api_url: str,
    api_key: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int], set[str]]:
    embedding = config["embedding"]
    retry = config["retry"]
    dimensions = int(embedding["dimensions"])
    cache = load_cache(cache_path, dimensions)
    pending = [row for row in rows if str(row["id"]) not in cache]
    batches = make_batches(pending, int(embedding["batch_size"]), int(embedding["max_batch_tokens"]))
    interval = 60.0 / float(embedding["requests_per_minute"])
    usage = {"prompt_tokens": 0, "total_tokens": 0, "requests": 0}
    response_models: set[str] = set()
    last_request_at: float | None = None

    print(f"{name}：共 {len(rows)} 条，缓存已有 {len(cache)} 条，待编码 {len(pending)} 条，共 {len(batches)} 批。")
    for batch_index, batch in enumerate(batches, 1):
        if last_request_at is not None:
            time.sleep(max(0.0, interval - (time.monotonic() - last_request_at)))
        last_request_at = time.monotonic()
        result = request_embeddings(
            api_url,
            api_key,
            str(embedding["user_agent"]),
            str(embedding["model"]),
            dimensions,
            [str(row["text"]) for row in batch],
            int(retry["max_attempts"]),
            float(retry["backoff_base_seconds"]),
            float(retry["backoff_max_seconds"]),
        )
        records = [
            {"id": str(row["id"]), "embedding": vector}
            for row, vector in zip(batch, result.vectors, strict=True)
        ]
        append_cache(cache_path, records)
        cache.update({record["id"]: record for record in records})
        usage["requests"] += 1
        usage["prompt_tokens"] += int(result.usage.get("prompt_tokens", 0))
        usage["total_tokens"] += int(result.usage.get("total_tokens", 0))
        if result.model:
            response_models.add(result.model)
        print(f"{name}：完成第 {batch_index}/{len(batches)} 批，累计 {len(cache)}/{len(rows)} 条。")
    return cache, usage, response_models


def materialize(
    rows: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    vectors_path: Path,
    ids_path: Path,
    normalize: bool = True,
) -> dict[str, float | int]:
    ids = [str(row["id"]) for row in rows]
    missing = [item_id for item_id in ids if item_id not in cache]
    if missing:
        raise ValueError(f"尚缺少 {len(missing)} 个向量，不能生成最终产物")
    vectors = np.asarray([cache[item_id]["embedding"] for item_id in ids], dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("最终向量含非有限数值")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("最终向量中存在零向量")
    if normalize:
        vectors /= norms[:, None]
        norms = np.linalg.norm(vectors, axis=1)
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors)
    ids_path.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "count": int(vectors.shape[0]),
        "dimensions": int(vectors.shape[1]),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
    }


def resolve_path(value: str) -> Path:
    return ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser(description="通过 OpenAI 兼容 API 生成 Embedding，支持限速和断点续传。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--only", choices=("documents", "queries", "all"), default="all")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    embedding = config["embedding"]
    api_key = load_embedding_api_key()
    base_url = load_embedding_api_base_url()
    api_url = base_url.rstrip("/") + "/" + str(embedding["endpoint"]).lstrip("/")

    inputs = config["inputs"]
    caches = config["cache"]
    selected = ("documents", "queries") if args.only == "all" else (args.only,)
    all_usage = {"prompt_tokens": 0, "total_tokens": 0, "requests": 0}
    response_models: set[str] = set()
    stats: dict[str, dict[str, float | int]] = {}

    for name in selected:
        input_path = resolve_path(str(inputs[name]))
        rows = read_jsonl(input_path)
        ids = [str(row["id"]) for row in rows]
        if len(ids) != len(set(ids)) or any(not str(row["text"]).strip() for row in rows):
            raise ValueError(f"{name} 输入存在重复 ID 或空文本")
        cache_key = "document_cache" if name == "documents" else "query_cache"
        cache, usage, models = encode_rows(
            name,
            rows,
            resolve_path(str(caches[cache_key])),
            config,
            api_url,
            api_key,
        )
        for key in all_usage:
            all_usage[key] += usage[key]
        response_models.update(models)
        vector_key = "document_vectors" if name == "documents" else "query_vectors"
        id_key = "document_ids" if name == "documents" else "query_ids"
        stats[name] = materialize(
            rows,
            cache,
            resolve_path(str(caches[vector_key])),
            resolve_path(str(caches[id_key])),
            normalize=bool(embedding["normalize_embeddings"]),
        )

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provider": embedding["provider"],
        "api_origin": urllib.parse.urlsplit(base_url).netloc,
        "request_model": embedding["model"],
        "response_models": sorted(response_models),
        "dimensions": int(embedding["dimensions"]),
        "requests_per_minute": float(embedding["requests_per_minute"]),
        "usage_this_run": all_usage,
        "outputs": stats,
        "input_sha256": {
            name: sha256_file(resolve_path(str(inputs[name]))) for name in selected
        },
    }
    manifest_path = resolve_path(str(caches["runtime_manifest"]))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
