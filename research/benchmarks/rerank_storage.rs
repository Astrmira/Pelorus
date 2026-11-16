//! 在实际公开查询与持久化入口验证 MERU 子集的重排精度及映射存储。

use quiver_ann::{
    BuildConfig, Index, PoincareNavigation, RerankConfig, RerankPrecision, RerankRepresentation,
    ScalarQuantization, SearchConfig, StorageMode,
};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{fs, path::PathBuf, time::Instant};

fn configurations() -> Vec<(String, RerankConfig)> {
    let mut result = Vec::new();
    for (geometry, representation) in [
        ("cartesian", RerankRepresentation::Cartesian),
        ("polar", RerankRepresentation::Polar),
    ] {
        for (name, precision) in [
            ("fp32", RerankPrecision::Fp32),
            ("fp16", RerankPrecision::Fp16),
            ("bf16", RerankPrecision::Bf16),
            ("sq8", RerankPrecision::Sq8),
            ("sq4", RerankPrecision::Sq4),
        ] {
            result.push((
                format!("{geometry}_{name}"),
                RerankConfig {
                    precision,
                    representation,
                    ..RerankConfig::default()
                },
            ));
        }
        for (name, precision) in [("sq8", RerankPrecision::Sq8), ("sq4", RerankPrecision::Sq4)] {
            for (encoding, scalar_quantization) in [
                ("centered".to_owned(), ScalarQuantization::Centered),
                (
                    "affine_g128".to_owned(),
                    ScalarQuantization::Affine { group_size: 128 },
                ),
                (
                    "affine_g64".to_owned(),
                    ScalarQuantization::Affine { group_size: 64 },
                ),
            ] {
                result.push((
                    format!("{geometry}_{encoding}_{name}"),
                    RerankConfig {
                        precision,
                        representation,
                        scalar_quantization,
                        norm_correction: false,
                    },
                ));
            }
        }
    }
    for (name, precision) in [("sq8", RerankPrecision::Sq8), ("sq4", RerankPrecision::Sq4)] {
        for group_size in [128, 64] {
            result.push((
                format!("cartesian_norm_affine_g{group_size}_{name}"),
                RerankConfig {
                    precision,
                    scalar_quantization: ScalarQuantization::Affine { group_size },
                    norm_correction: true,
                    ..RerankConfig::default()
                },
            ));
        }
    }
    result
}

fn hits(result: &[(u64, f32)], truth: &[u64]) -> usize {
    result.iter().filter(|(id, _)| truth.contains(id)).count()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let value = |name: &str| -> Result<&str, Box<dyn std::error::Error>> {
        let position = args
            .iter()
            .position(|arg| arg == name)
            .ok_or_else(|| format!("缺少参数 {name}"))?;
        Ok(args
            .get(position + 1)
            .ok_or_else(|| format!("缺少参数值 {name}"))?)
    };
    let data = PathBuf::from(value("--data")?);
    let output = PathBuf::from(value("--output")?);
    let dim = value("--dim")?.parse::<usize>()?;
    let query_count = value("--queries")?.parse::<usize>()?;
    let threads = value("--threads")?.parse::<usize>()?;
    if dim == 0 || query_count == 0 || threads == 0 {
        return Err("维度、查询数和线程数必须为正".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let raw = fs::read(data.join("documents.f32"))?;
    if raw.len() % 4 != 0 {
        return Err("库向量字节数错误".into());
    }
    let vectors = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    drop(raw);
    if vectors.len() % dim != 0 {
        return Err("库向量维度不匹配".into());
    }
    let raw = fs::read(data.join("ids.u64"))?;
    if raw.len() % 8 != 0 {
        return Err("标识字节数错误".into());
    }
    let ids = raw
        .as_chunks::<8>()
        .0
        .iter()
        .map(|bytes| u64::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    let raw = fs::read(data.join("queries.f32"))?;
    if raw.len() % 4 != 0 {
        return Err("查询字节数错误".into());
    }
    let queries = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    if query_count > queries.len() / dim {
        return Err("查询数超出输入".into());
    }
    let queries = &queries[..query_count * dim];
    let build_start = Instant::now();
    let mut index = Index::build_poincare(
        dim,
        &ids,
        &vectors,
        &BuildConfig::default(),
        1.0,
        PoincareNavigation::GeometryBq2,
    )?;
    let build_seconds = build_start.elapsed().as_secs_f64();
    drop(vectors);
    println!(
        "构图完成，节点 {}，耗时 {:.3} 秒",
        index.len(),
        build_seconds
    );
    let pool_start = Instant::now();
    let pools = queries
        .par_chunks_exact(dim)
        .map(|query| index.search_exact(query, 128))
        .collect::<quiver_ann::Result<Vec<_>>>()?;
    let truth = pools
        .iter()
        .map(|pool| pool.iter().take(10).map(|&(id, _)| id).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let reference_bytes = fs::read(data.join("truth.u64"))?;
    if reference_bytes.len() % 8 != 0 || reference_bytes.len() / 8 < query_count * 10 {
        return Err("独立真值长度不足".into());
    }
    let reference = reference_bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|bytes| u64::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    for (number, row) in truth.iter().enumerate() {
        assert_eq!(
            row.as_slice(),
            &reference[number * 10..(number + 1) * 10],
            "查询 {number} 的原始真值与独立归档不一致"
        );
    }
    let candidates = pools
        .iter()
        .map(|pool| pool.iter().map(|&(id, _)| id).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let pool_seconds = pool_start.elapsed().as_secs_f64();
    let config = SearchConfig {
        top_k: 10,
        ef_search: 256,
        rerank_limit: Some(128),
    };
    let mut rows = Vec::<Value>::new();
    for (name, codec) in configurations() {
        index.set_storage_mode(StorageMode::Memory)?;
        let encode_start = Instant::now();
        index.set_rerank_config(codec)?;
        let encode_seconds = encode_start.elapsed().as_secs_f64();
        let mut pure = Vec::with_capacity(query_count);
        let mut ann = Vec::with_capacity(query_count);
        let start = Instant::now();
        for (number, query) in queries.chunks_exact(dim).enumerate() {
            pure.push(index.rerank_candidates(query, &candidates[number], 10)?);
            ann.push(index.search(query, &config)?);
        }
        let memory_validation_seconds = start.elapsed().as_secs_f64();
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("index.pelorus");
        index.save_snapshot(&path)?;
        let start = Instant::now();
        let mapped = Index::load_mmap(&path)?;
        let load_seconds = start.elapsed().as_secs_f64();
        let bytes = mapped.vector_storage_stats();
        assert_eq!(bytes.canonical_heap_bytes, 0);
        assert_eq!(bytes.rerank_heap_bytes, 0);
        let mut pure_hits = 0;
        let mut ann_hits = 0;
        let start = Instant::now();
        for (number, query) in queries.chunks_exact(dim).enumerate() {
            let actual_pure = mapped.rerank_candidates(query, &candidates[number], 10)?;
            let actual_ann = mapped.search(query, &config)?;
            assert_eq!(
                actual_pure, pure[number],
                "配置 {name}，查询 {number} 的纯重排不一致"
            );
            assert_eq!(
                actual_ann, ann[number],
                "配置 {name}，查询 {number} 的普通查询不一致"
            );
            pure_hits += hits(&actual_pure, &truth[number]);
            ann_hits += hits(&actual_ann, &truth[number]);
        }
        let mapped_validation_seconds = start.elapsed().as_secs_f64();
        rows.push(json!({"mode":name,"pure_recall_at_10":pure_hits as f64/(query_count*10) as f64,
            "ann_recall_at_10":ann_hits as f64/(query_count*10) as f64,
            "record_bytes":bytes.rerank_record_bytes,"canonical_mapped_bytes":bytes.canonical_mapped_bytes,
            "rerank_mapped_bytes":bytes.rerank_mapped_bytes,"shared_metadata_bytes":bytes.shared_metadata_bytes,
            "rerank_metadata_bytes":bytes.rerank_metadata_bytes,
            "encode_seconds":encode_seconds,"load_seconds":load_seconds,
            "memory_validation_seconds":memory_validation_seconds,"mapped_validation_seconds":mapped_validation_seconds,
            "identical_pure_queries":query_count,"identical_ann_queries":query_count}));
        let report = json!({"documents":ids.len(),"dimensions":dim,"queries":query_count,"build_threads":threads,
            "query_threads":1,"candidate_pool":"原始 FP32 子集全扫描前128名，固定候选仅测重排损失",
            "timing_scope":"正确性测量，包含内存与映射结果对账，文件页缓存未清空",
            "build_seconds":build_seconds,"pool_seconds":pool_seconds,"results":rows});
        fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
        println!(
            "完成 {}，纯重排 Recall@10 {:.3}%，普通查询 {:.3}%，堆与映射逐查询一致",
            name,
            pure_hits as f64 / (query_count * 10) as f64 * 100.0,
            ann_hits as f64 / (query_count * 10) as f64 * 100.0
        );
    }
    println!("全部完成，结果保存到 {}", output.display());
    Ok(())
}
