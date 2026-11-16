//! 同一 MERU 图上的重排精度、热内存和有界 SSD 缓存测量。

use quiver_ann::research::{
    ExecutionConfig, GeometryComponents, HyperbolicStrategy, RecallConfig, SketchFilter,
    StorageConfig, set_execution_config, set_geometry_components, set_hyperbolic_strategy,
    set_recall_config, set_storage_config,
};
use quiver_ann::{
    BuildConfig, Index, PoincareNavigation, RerankConfig, RerankPrecision, RerankRepresentation,
    ScalarQuantization, SearchConfig, SsdCacheConfig, SsdIoDriver, SsdIoStats,
};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

fn number(plan: &Value, name: &str, default: usize) -> usize {
    plan.get(name)
        .and_then(Value::as_u64)
        .map_or(default, |n| n as usize)
}

fn configure(margin: f32) {
    set_hyperbolic_strategy(HyperbolicStrategy::Full);
    set_geometry_components(GeometryComponents::default());
    set_execution_config(ExecutionConfig::optimized(3));
    set_recall_config(RecallConfig::adaptive());
    set_storage_config(StorageConfig {
        colocated_store: true,
        early_gate: SketchFilter::SymSketch256 { margin },
        header_prefetch_window: 8,
        eager_prefetch_overlap: true,
    });
}

fn codecs() -> Vec<(&'static str, RerankConfig)> {
    vec![
        (
            "fp16_cartesian",
            RerankConfig {
                precision: RerankPrecision::Fp16,
                ..RerankConfig::default()
            },
        ),
        (
            "fp16_polar",
            RerankConfig {
                precision: RerankPrecision::Fp16,
                representation: RerankRepresentation::Polar,
                ..RerankConfig::default()
            },
        ),
        (
            "centered_sq8",
            RerankConfig {
                precision: RerankPrecision::Sq8,
                scalar_quantization: ScalarQuantization::Centered,
                ..RerankConfig::default()
            },
        ),
        (
            "centered_polar_sq4",
            RerankConfig {
                precision: RerankPrecision::Sq4,
                representation: RerankRepresentation::Polar,
                scalar_quantization: ScalarQuantization::Centered,
                norm_correction: false,
            },
        ),
    ]
}

fn read_floats(
    path: &Path,
    rows: usize,
    dim: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let bytes = fs::read(path)?;
    let count = rows.checked_mul(dim).ok_or("输入大小溢出")?;
    if bytes.len() < count * 4 || !bytes.len().is_multiple_of(4) {
        return Err("浮点输入长度不足".into());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .take(count)
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

fn proc_fields(path: &str) -> BTreeMap<String, u64> {
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            Some((
                fields.next()?.trim_end_matches(':').to_owned(),
                fields.next()?.parse().ok()?,
            ))
        })
        .collect()
}

fn cpu_ticks() -> u64 {
    let stat = fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let Some(end) = stat.rfind(')') else {
        return 0;
    };
    let parts = stat[end + 1..].split_whitespace().collect::<Vec<_>>();
    parts
        .get(11)
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
        + parts
            .get(12)
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
}

fn stats(value: Option<SsdIoStats>) -> Value {
    match value {
        None => Value::Null,
        Some(s) => {
            json!({"driver":match s.driver { SsdIoDriver::IoUring=>"uring", SsdIoDriver::Pread=>"pread" },
            "io_threads":s.io_threads,"queue_depth":s.queue_depth,"pending_peak_bytes":s.pending_peak_bytes,
            "prefetch_skipped":s.prefetch_skipped,"uring_submitted":s.uring_submitted,"uring_completed":s.uring_completed,
            "uring_max_inflight":s.uring_max_inflight,"uring_submit_batches":s.uring_submit_batches,
            "uring_completion_batches":s.uring_completion_batches,"uring_completion_batch_max":s.uring_completion_batch_max,
            "uring_enter_calls":s.uring_enter_calls,"uring_enter_eintr":s.uring_enter_eintr,"ring_errors":s.ring_errors,
            "registered_buffer_bytes":s.registered_buffer_bytes,"ring_mapping_bytes":s.ring_mapping_bytes,
            "io_wq_affinity_registered":s.io_wq_affinity_registered,"io_wq_workers_registered":s.io_wq_workers_registered,
            "read_calls":s.read_calls,"read_bytes":s.read_bytes,"read_errors":s.read_errors,
            "cache_hits":s.cache_hits,"cache_misses":s.cache_misses,"pending_hits":s.pending_hits,
            "query_page_hits":s.query_page_hits,"prefetch_requests":s.prefetch_requests,"prefetch_used":s.prefetch_used,
            "prefetch_wasted":s.prefetch_wasted,"evicted_pages":s.evicted_pages,
            "cache_current_bytes":s.cache_current_bytes,"cache_peak_bytes":s.cache_peak_bytes,
            "cache_capacity_bytes":s.cache_capacity_bytes,"pool_allocated_bytes":s.pool_allocated_bytes,
            "pool_live_bytes":s.pool_live_bytes,"pool_peak_live_bytes":s.pool_peak_live_bytes,"pool_limit_bytes":s.pool_limit_bytes})
        }
    }
}

fn io_delta(before: Option<SsdIoStats>, after: Option<SsdIoStats>) -> Value {
    let (Some(a), Some(b)) = (before, after) else {
        return Value::Null;
    };
    json!({"read_calls":b.read_calls-a.read_calls,"read_bytes":b.read_bytes-a.read_bytes,
        "uring_submitted":b.uring_submitted-a.uring_submitted,"uring_completed":b.uring_completed-a.uring_completed,
        "uring_submit_batches":b.uring_submit_batches-a.uring_submit_batches,"uring_completion_batches":b.uring_completion_batches-a.uring_completion_batches,
        "uring_enter_calls":b.uring_enter_calls-a.uring_enter_calls,"ring_errors":b.ring_errors-a.ring_errors,
        "prefetch_skipped":b.prefetch_skipped-a.prefetch_skipped,
        "read_errors":b.read_errors-a.read_errors,"cache_hits":b.cache_hits-a.cache_hits,
        "cache_misses":b.cache_misses-a.cache_misses,"pending_hits":b.pending_hits-a.pending_hits,
        "query_page_hits":b.query_page_hits-a.query_page_hits,"prefetch_requests":b.prefetch_requests-a.prefetch_requests,
        "prefetch_used":b.prefetch_used-a.prefetch_used,"prefetch_wasted":b.prefetch_wasted-a.prefetch_wasted,
        "evicted_pages":b.evicted_pages-a.evicted_pages})
}

fn prepare(
    data: &Path,
    output: &Path,
    n: usize,
    dim: usize,
    threads: usize,
    precisions: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if threads != 32 {
        return Err("本轮构图使用32线程".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let file = File::open(data.join("documents.f32"))?;
    // SAFETY：复用已冻结的只读公开数据，文件长度和维度在借用前核对，构图期间不修改文件。
    let mapping = unsafe { memmap2::Mmap::map(&file)? };
    let values = n.checked_mul(dim).ok_or("输入大小溢出")?;
    let vectors =
        bytemuck::try_cast_slice::<u8, f32>(mapping.get(..values * 4).ok_or("库向量长度不足")?)
            .map_err(|error| format!("库向量对齐或长度错误：{error}"))?;
    let ids = (0..n as u64).collect::<Vec<_>>();
    let timer = Instant::now();
    let mut index = Index::build_poincare(
        dim,
        &ids,
        vectors,
        &BuildConfig::default(),
        1.0,
        PoincareNavigation::GeometryBq2,
    )?;
    let build_seconds = timer.elapsed().as_secs_f64();
    let hot_seconds = index.prepare_colocated_hot_store(true)?;
    let topology = index.graph_diagnostics().topology_crc32;
    fs::create_dir_all(output)?;
    let mut rows = Vec::new();
    let available = std::iter::once(("fp32_reference", RerankConfig::default())).chain(codecs());
    for (name, codec) in
        available.filter(|(name, _)| precisions.split(',').any(|value| value == *name))
    {
        let timer = Instant::now();
        if index.rerank_config() != codec {
            index.set_rerank_config(codec)?;
        }
        let encode_seconds = timer.elapsed().as_secs_f64();
        let path = output.join(format!("{name}.pelorus"));
        let timer = Instant::now();
        index.save_snapshot(&path)?;
        let save_seconds = timer.elapsed().as_secs_f64();
        assert_eq!(index.graph_diagnostics().topology_crc32, topology);
        rows.push(json!({"precision":name,"index":path,"encode_seconds":encode_seconds,"save_seconds":save_seconds,
            "record_bytes":index.vector_storage_stats().rerank_record_bytes,"navigation_bytes":index.navigation_storage_bytes()}));
        println!("快照准备完成 {}，编码 {:.3} 秒", name, encode_seconds);
        fs::write(
            output.join("preparation.json"),
            serde_json::to_vec_pretty(&json!({"documents":n,"dimensions":dim,
            "build_threads":threads,"build_seconds":build_seconds,"hot_preparation_seconds":hot_seconds,
            "process_memory_kib":proc_fields("/proc/self/status"),
            "topology_crc32":topology,"residual_cones":index.residual_cone_count(),"indices":rows}))?,
        )?;
    }
    Ok(())
}

fn run(
    data: &Path,
    plan: &Value,
    output: &Path,
    margin: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let dim = number(plan, "dimensions", 512);
    let nq = number(plan, "queries", 10000);
    let warm = number(plan, "warmup_queries", 2000);
    let threads = number(plan, "threads", 1);
    if ![1, 2, 4, 8, 16, 24].contains(&threads) {
        return Err("查询线程仅使用1/2/4/8/16/24".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let tuning = plan["query_split"].as_str() == Some("tuning");
    let query_file = if tuning {
        "tuning_queries.f32"
    } else {
        "queries.f32"
    };
    let truth_file = if tuning {
        "tuning_truth.u64"
    } else {
        "truth.u64"
    };
    let queries = read_floats(&data.join(query_file), nq, dim)?;
    let warm_file = plan["warmup_file"].as_str().unwrap_or("tuning_queries.f32");
    let warm_queries = read_floats(&data.join(warm_file), warm, dim)?;
    let bytes = fs::read(data.join(truth_file))?;
    if bytes.len() < nq * 10 * 8 {
        return Err("查询真值长度不足".into());
    }
    let truth = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .take(nq * 10)
        .map(|v| u64::from_le_bytes(*v))
        .collect::<Vec<_>>();
    let index_path = PathBuf::from(plan["index"].as_str().ok_or("缺少快照路径")?);
    let backend = plan["backend"].as_str().ok_or("缺少后端")?;
    let timer = Instant::now();
    let mut index = match backend {
        "memory" => Index::load(&index_path)?,
        "ssd128" | "ssd512" => Index::open_ssd(
            &index_path,
            SsdCacheConfig {
                cache_bytes: if backend == "ssd128" {
                    128 * 1024 * 1024
                } else {
                    512 * 1024 * 1024
                },
                io_workers: number(plan, "io_workers", 1),
                driver: match plan["io_driver"].as_str() {
                    None | Some("uring") => SsdIoDriver::IoUring,
                    Some("pread") => SsdIoDriver::Pread,
                    _ => return Err("未知SSD读取驱动".into()),
                },
                ..SsdCacheConfig::default()
            },
        )?,
        _ => return Err("未知后端".into()),
    };
    if plan["rerank_override"].as_str() == Some("fp32") {
        index.set_rerank_config(RerankConfig::default())?;
    }
    let topology_crc32 = index.graph_diagnostics().topology_crc32;
    if plan["expected_topology_crc32"]
        .as_u64()
        .is_some_and(|expected| expected != u64::from(topology_crc32))
    {
        return Err("参考测量的图拓扑指纹不符".into());
    }
    let load_seconds = timer.elapsed().as_secs_f64();
    let tick_rate = std::process::Command::new("getconf")
        .arg("CLK_TCK")
        .output()?;
    let tick_rate = String::from_utf8(tick_rate.stdout)?.trim().parse::<f64>()?;
    let mut report = File::create(output)?;
    let mut order = (0..nq).collect::<Vec<_>>();
    let mut seed = 20261116u64;
    for position in (1..order.len()).rev() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        order.swap(position, seed as usize % (position + 1));
    }
    let conditions = plan["conditions"].as_array().ok_or("缺少测量条件")?;
    for condition in conditions {
        let name = condition["id"].as_str().ok_or("缺少条件标识")?;
        let config = SearchConfig {
            top_k: 10,
            ef_search: number(condition, "ef", 128),
            rerank_limit: Some(number(condition, "budget", 128)),
        };
        if backend != "memory" {
            index.clear_ssd_cache();
        }
        warm_queries
            .par_chunks_exact(dim)
            .try_for_each(|query| index.search(query, &config).map(|_| ()))?;
        index.drain_ssd_reads();
        let cache_before = index.ssd_io_stats();
        let proc_before = proc_fields("/proc/self/io");
        let cpu_before = cpu_ticks();
        let timer = Instant::now();
        let mut rows = order
            .par_iter()
            .map(|&query_id| {
                let wall = Instant::now();
                let result =
                    index.search(&queries[query_id * dim..(query_id + 1) * dim], &config)?;
                Ok((query_id, wall.elapsed().as_nanos() as u64, result))
            })
            .collect::<quiver_ann::Result<Vec<_>>>()?;
        index.drain_ssd_reads();
        let seconds = timer.elapsed().as_secs_f64();
        let cpu_after = cpu_ticks();
        let proc_after = proc_fields("/proc/self/io");
        let cache_after = index.ssd_io_stats();
        rows.sort_unstable_by_key(|row| row.0);
        let mut result_crc = crc32fast::Hasher::new();
        let mut total_hits = 0;
        let mut latency = Vec::with_capacity(nq);
        let mut requests = File::create(output.with_file_name(format!(
            "{}.{}.requests.csv",
            output.file_stem().unwrap().to_string_lossy(),
            name
        )))?;
        writeln!(requests, "query_id,latency_ns,hits,ids,score_bits")?;
        for (query_id, ns, result) in &rows {
            if result.len() != 10
                || result.iter().any(|row| !row.1.is_finite())
                || result.windows(2).any(|pair| pair[0].1 < pair[1].1)
            {
                return Err("查询结果数量或分数无效".into());
            }
            let mut unique = result.iter().map(|row| row.0).collect::<Vec<_>>();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() != 10 {
                return Err("查询返回重复标识".into());
            }
            let hits = result
                .iter()
                .filter(|(id, _)| truth[query_id * 10..(query_id + 1) * 10].contains(id))
                .count();
            total_hits += hits;
            latency.push(*ns);
            result_crc.update(&(*query_id as u64).to_le_bytes());
            for &(id, score) in result {
                result_crc.update(&id.to_le_bytes());
                result_crc.update(&score.to_bits().to_le_bytes());
            }
            writeln!(
                requests,
                "{query_id},{ns},{hits},{},{}",
                result
                    .iter()
                    .map(|row| row.0.to_string())
                    .collect::<Vec<_>>()
                    .join("|"),
                result
                    .iter()
                    .map(|row| row.1.to_bits().to_string())
                    .collect::<Vec<_>>()
                    .join("|")
            )?;
        }
        requests.sync_all()?;
        latency.sort_unstable();
        let percentile = |p: f64| {
            latency[((latency.len() as f64 * p).ceil() as usize)
                .saturating_sub(1)
                .min(latency.len() - 1)] as f64
                / 1e6
        };
        let delta = io_delta(cache_before, cache_after);
        if delta["read_errors"].as_u64().unwrap_or(0) > 0
            || delta["ring_errors"].as_u64().unwrap_or(0) > 0
        {
            return Err("SSD 测量出现读取错误".into());
        }
        let storage = index.vector_storage_stats();
        let io = proc_after
            .iter()
            .map(|(key, value)| {
                (
                    key.clone(),
                    value.saturating_sub(*proc_before.get(key).unwrap_or(&0)),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let row = json!({"precision":plan["precision"],"backend":backend,"threads":threads,"condition":name,
            "sketch_margin":margin,"query_split":if tuning {"tuning"} else {"test"},"dimensions":dim,
            "simd_regime":plan["simd_regime"],
            "io_workers":index.ssd_cache_config().map(|config| config.io_workers),
            "topology_crc32":topology_crc32,
            "ef":config.ef_search,"budget":config.rerank_limit,"queries":nq,"warmup_queries":warm,
            "recall_at_10":total_hits as f64/(nq*10) as f64,"qps":nq as f64/seconds,"wall_seconds":seconds,
            "p50_ms":percentile(0.5),"p95_ms":percentile(0.95),"p99_ms":percentile(0.99),"p999_ms":percentile(0.999),
            "cpu_seconds":(cpu_after-cpu_before) as f64/tick_rate,"result_crc32":result_crc.finalize(),
            "process_io_delta":io,"ssd_io_delta":delta,"ssd_before":stats(cache_before),"ssd_after":stats(cache_after),
            "process_memory_kib":proc_fields("/proc/self/status"),"load_seconds":load_seconds,
            "canonical_heap_bytes":storage.canonical_heap_bytes,"canonical_mapped_bytes":storage.canonical_mapped_bytes,
            "rerank_heap_bytes":storage.rerank_heap_bytes,"rerank_mapped_bytes":storage.rerank_mapped_bytes,
            "shared_metadata_bytes":storage.shared_metadata_bytes,"record_bytes":storage.rerank_record_bytes,
            "rerank_metadata_bytes":storage.rerank_metadata_bytes,
            "navigation_bytes":index.navigation_storage_bytes()});
        writeln!(report, "{row}")?;
        report.sync_all()?;
        println!(
            "完成 {} {} {}线程 {}，Recall {:.4}%，QPS {:.1}",
            plan["precision"].as_str().unwrap(),
            backend,
            threads,
            name,
            total_hits as f64 / (nq * 10) as f64 * 100.0,
            nq as f64 / seconds
        );
    }
    Ok(())
}

fn encode_snapshot(
    source: &Path,
    output: &Path,
    precisions: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(32)
        .build_global()?;
    let timer = Instant::now();
    let mut index = Index::load(source)?;
    let load_seconds = timer.elapsed().as_secs_f64();
    let topology = index.graph_diagnostics().topology_crc32;
    fs::create_dir_all(output)?;
    let mut rows = Vec::new();
    for (name, codec) in codecs()
        .into_iter()
        .filter(|(name, _)| precisions.split(',').any(|value| value == *name))
    {
        let timer = Instant::now();
        index.set_rerank_config(codec)?;
        let encode_seconds = timer.elapsed().as_secs_f64();
        let path = output.join(format!("{name}.pelorus"));
        let timer = Instant::now();
        index.save_snapshot(&path)?;
        let save_seconds = timer.elapsed().as_secs_f64();
        if index.graph_diagnostics().topology_crc32 != topology {
            return Err("重排编码改变了图拓扑".into());
        }
        rows.push(json!({"precision":name,"index":path,"encode_seconds":encode_seconds,"save_seconds":save_seconds,
            "record_bytes":index.vector_storage_stats().rerank_record_bytes,"navigation_bytes":index.navigation_storage_bytes()}));
        fs::write(
            output.join("preparation.json"),
            serde_json::to_vec_pretty(&json!({
                "source_index":source,"load_seconds":load_seconds,"encoding_threads":32,
                "topology_crc32":topology,"residual_cones":index.residual_cone_count(),"indices":rows,
                "process_memory_kib":proc_fields("/proc/self/status")
            }))?,
        )?;
        println!("原图重排编码完成 {}，编码 {:.3} 秒", name, encode_seconds);
    }
    if rows.is_empty() {
        return Err("未选择可用的重排编码".into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let arg = |name: &str, default: &str| {
        args.iter()
            .position(|value| value == name)
            .and_then(|position| args.get(position + 1))
            .map_or_else(|| default.to_owned(), Clone::clone)
    };
    let phase = arg("--phase", "run");
    let margin = arg("--margin", "0.08").parse()?;
    configure(margin);
    let data = PathBuf::from(arg("--data", "."));
    let output = PathBuf::from(arg("--output", "storage.jsonl"));
    if phase == "encode" {
        encode_snapshot(
            &PathBuf::from(arg("--source-index", "")),
            &output,
            &arg("--precisions", "centered_polar_sq4"),
        )
    } else if phase == "prepare" {
        // 全局构图环境在任何线程启动之前设置。
        unsafe {
            std::env::set_var("QUIVER_BUILD_SEED", "12345");
            std::env::set_var("QUIVER_BUILD_PHASES", "1");
        }
        prepare(
            &data,
            &output,
            arg("--n", "1000000").parse()?,
            arg("--dim", "512").parse()?,
            32,
            &arg(
                "--precisions",
                "fp16_cartesian,fp16_polar,centered_sq8,centered_polar_sq4",
            ),
        )
    } else {
        let plan: Value = serde_json::from_slice(&fs::read(arg("--plan", "plan.json"))?)?;
        run(&data, &plan, &output, margin)
    }
}
