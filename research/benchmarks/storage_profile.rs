//! 同一张图上逐查询核对阶段耗时、实际重排数量及分配，计时与分配观测分开执行。

use quiver_ann::research::{
    ExecutionConfig, GeometryComponents, HyperbolicStrategy, RecallConfig, SketchFilter,
    StorageConfig, set_execution_config, set_geometry_components, set_hyperbolic_strategy,
    set_recall_config, set_storage_config,
};
use quiver_ann::{Index, PoincareNavigation, RerankConfig, SearchConfig};
use serde_json::json;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    fs::{self, File},
    io::Write,
    path::PathBuf,
    time::Instant,
};

#[derive(Clone, Copy, Default)]
struct Allocations {
    calls: u64,
    reallocs: u64,
    bytes: u64,
}
thread_local! {
    static ALLOC_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static ALLOC_DATA: Cell<Allocations> = const { Cell::new(Allocations { calls: 0, reallocs: 0, bytes: 0 }) };
}
struct CountingAllocator;
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
fn count_allocation(bytes: usize, realloc: bool) {
    let _ = ALLOC_ACTIVE.try_with(|active| {
        if active.get() {
            ALLOC_DATA.with(|data| {
                let mut value = data.get();
                value.calls += u64::from(!realloc);
                value.reallocs += u64::from(realloc);
                value.bytes += bytes as u64;
                data.set(value);
            });
        }
    });
}
// SAFETY：全部分配仍由System拥有和释放，线程本地计数使用无分配的Cell。
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation(layout.size(), false);
        // SAFETY：将调用方提供的有效布局原样交给系统分配器。
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation(layout.size(), false);
        // SAFETY：布局及零初始化契约原样委托给系统分配器。
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_allocation(size, true);
        // SAFETY：原指针、布局及新长度来自GlobalAlloc调用方。
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY：所有指针均来自本包装器委托的系统分配器。
        unsafe { System.dealloc(pointer, layout) }
    }
}
fn result_crc(result: &[(u64, f32)]) -> u32 {
    let mut crc = crc32fast::Hasher::new();
    for &(id, score) in result {
        crc.update(&id.to_le_bytes());
        crc.update(&score.to_bits().to_le_bytes());
    }
    crc.finalize()
}
fn floats(path: PathBuf) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if !bytes.len().is_multiple_of(4) {
        return Err("浮点文件长度错误".into());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let value = |key: &str| -> Result<String, Box<dyn std::error::Error>> {
        let position = args.iter().position(|v| v == key).ok_or("缺少命令行参数")?;
        Ok(args.get(position + 1).ok_or("参数缺少取值")?.clone())
    };
    let plan: serde_json::Value = serde_json::from_slice(&fs::read(value("--plan")?)?)?;
    let data = PathBuf::from(value("--data")?);
    let output = PathBuf::from(value("--output")?);
    let ef = value("--ef")?.parse::<usize>()?;
    let dim = plan["dimensions"].as_u64().unwrap_or(512) as usize;
    let nq = plan["queries"].as_u64().unwrap_or(10000) as usize;
    let warm_count = plan["warmup_queries"].as_u64().unwrap_or(2000) as usize;
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    set_hyperbolic_strategy(HyperbolicStrategy::Full);
    set_geometry_components(GeometryComponents::default());
    set_execution_config(ExecutionConfig::optimized(3));
    set_recall_config(RecallConfig::adaptive());
    set_storage_config(StorageConfig {
        colocated_store: true,
        early_gate: SketchFilter::SymSketch256 { margin: 0.20 },
        header_prefetch_window: 8,
        eager_prefetch_overlap: true,
    });
    let mut queries = floats(data.join("queries.f32"))?;
    if queries.len() < nq * dim {
        return Err("阶段查询长度不足".into());
    }
    queries.truncate(nq * dim);
    let warm = if warm_count == 0 {
        Vec::new()
    } else {
        let mut values = floats(data.join("tuning_queries.f32"))?;
        if values.len() < warm_count * dim {
            return Err("预热查询长度不足".into());
        }
        values.truncate(warm_count * dim);
        values
    };
    let truth_bytes = fs::read(data.join("truth.u64"))?;
    let truth = truth_bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|v| u64::from_le_bytes(*v))
        .collect::<Vec<_>>();
    if queries.len() != nq * dim || warm.len() != warm_count * dim || truth.len() < nq * 10 {
        return Err("测量输入形状错误".into());
    }
    let mut index = Index::load(plan["index"].as_str().ok_or("缺少快照路径")?)?;
    if plan["rerank_override"].as_str() == Some("fp32") {
        index.set_rerank_config(RerankConfig::default())?;
    }
    let topology = index.graph_diagnostics().topology_crc32;
    if plan["expected_topology_crc32"]
        .as_u64()
        .is_some_and(|expected| expected != u64::from(topology))
    {
        return Err("图拓扑指纹不符".into());
    }
    let config = SearchConfig {
        top_k: 10,
        ef_search: ef,
        rerank_limit: Some(plan["budget"].as_u64().unwrap_or(128) as usize),
    };
    let mut order = (0..nq).collect::<Vec<_>>();
    let mut seed = 20261116u64;
    for position in (1..order.len()).rev() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        order.swap(position, seed as usize % (position + 1));
    }
    for query in warm.chunks_exact(dim) {
        std::hint::black_box(index.search(query, &config)?);
    }
    let mut normal_ns = vec![0u64; nq];
    let mut crc = vec![0u32; nq];
    let mut hits = vec![0usize; nq];
    let mut phase_rows = Vec::with_capacity(nq);
    let mut allocation_rows = vec![Allocations::default(); nq];
    for &id in &order {
        let query = &queries[id * dim..(id + 1) * dim];
        let start = Instant::now();
        let result = index.search(query, &config)?;
        normal_ns[id] = start.elapsed().as_nanos() as u64;
        hits[id] = result
            .iter()
            .filter(|(doc, _)| truth[id * 10..(id + 1) * 10].contains(doc))
            .count();
        crc[id] = result_crc(&result);
    }
    // 阶段计时不启用分配统计，保留正常的高速导航与候选池复用路径。
    for &id in &order {
        let start = Instant::now();
        let profile = index.search_stage_profile(
            &queries[id * dim..(id + 1) * dim],
            &config,
            PoincareNavigation::GeometryBq2,
        )?;
        let total = start.elapsed().as_nanos() as u64;
        if profile.reranked_count + profile.rerank_radial_rejections
            != profile.rerank_selected_count
        {
            return Err("重排计数没有闭合".into());
        }
        phase_rows.push((id, total, profile));
    }
    // 分配计数单独走公开search入口，其耗时不加入阶段计时。
    for &id in &order {
        ALLOC_DATA.with(|data| data.set(Allocations::default()));
        ALLOC_ACTIVE.with(|active| active.set(true));
        let result = index.search(&queries[id * dim..(id + 1) * dim], &config);
        ALLOC_ACTIVE.with(|active| active.set(false));
        allocation_rows[id] = ALLOC_DATA.with(Cell::get);
        let result = result?;
        if result_crc(&result) != crc[id] {
            return Err("分配观测改变查询结果".into());
        }
    }
    // 召回损失集合追踪放在独立诊断轮，避免HashSet记录进入阶段与普通查询计时。
    if plan["loss_audit"].as_bool().unwrap_or(false) {
        let mut loss_file = File::create(output.with_extension("loss.jsonl"))?;
        for &id in &order {
            let stats = index.search_loss_decomposition(
                &queries[id * dim..(id + 1) * dim],
                &config,
                PoincareNavigation::GeometryBq2,
                &truth[id * 10..(id + 1) * 10],
            )?;
            if stats.hits_returned != hits[id] {
                return Err("集合追踪改变返回命中数".into());
            }
            writeln!(
                loss_file,
                "{}",
                json!({"query_id":id,"hits_visited":stats.hits_visited,
                "hits_beam":stats.hits_in_beam,"hits_selected":stats.hits_selected_for_rerank,
                "hits_returned":stats.hits_returned,"visited_nodes":stats.visited_nodes,
                "expanded_nodes":stats.expanded_nodes,"expansion_calls":stats.expansion_calls,
                "approximate_evals":stats.approximate_evals,"cold_score_evals":stats.exact_vector_evals,
                "header_checks":stats.header_checks,"header_rejections":stats.header_rejections,
                "bound_rejections":stats.bound_rejections})
            )?;
        }
        loss_file.sync_all()?;
    }
    // 从原始FP32全库扫描取得精确前128名候选，隔离冷存重排表示本身的精度损失。
    let pure_queries = plan["pure_queries"].as_u64().unwrap_or(0) as usize;
    if pure_queries > 0 {
        let mut pure_file = File::create(output.with_extension("pure.jsonl"))?;
        for &id in order.iter().take(pure_queries) {
            let query = &queries[id * dim..(id + 1) * dim];
            let pool = index.search_exact(query, 128)?;
            let ids = pool.iter().map(|row| row.0).collect::<Vec<_>>();
            let result = index.rerank_candidates(query, &ids, 10)?;
            let pure_hits = result
                .iter()
                .filter(|(doc, _)| truth[id * 10..(id + 1) * 10].contains(doc))
                .count();
            writeln!(
                pure_file,
                "{}",
                json!({"query_id":id,"pool_size":ids.len(),
                "hits":pure_hits,"ids":result.iter().map(|row|row.0).collect::<Vec<_>>()})
            )?;
        }
        pure_file.sync_all()?;
    }
    phase_rows.sort_unstable_by_key(|row| row.0);
    let mut csv = File::create(output.with_extension("queries.csv"))?;
    writeln!(
        csv,
        "query_id,normal_ns,profile_total_ns,upper_greedy_ns,router_seed_ns,beam_nav_ns,stage05_ns,candidate_select_ns,rerank_total_ns,rerank_score_ns,selected_count,reranked_count,radial_rejections,navigation_passes,main_expansions,horizon_expansions,stage05_candidates,ef_nav,router_seeds,alloc_calls,realloc_calls,alloc_bytes,hits,result_crc32"
    )?;
    for &(id, total, p) in &phase_rows {
        let a = allocation_rows[id];
        writeln!(
            csv,
            "{id},{},{total},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            normal_ns[id],
            p.upper_greedy_ns,
            p.router_seed_ns,
            p.beam_nav_ns,
            p.asym_gate_ns,
            p.candidate_select_ns,
            p.exact_rerank_ns,
            p.rerank_score_ns,
            p.rerank_selected_count,
            p.reranked_count,
            p.rerank_radial_rejections,
            p.navigation_passes,
            p.main_expansion_calls,
            p.horizon_expansion_calls,
            p.stage05_candidates,
            p.ef_nav,
            p.router_seeds,
            a.calls,
            a.reallocs,
            a.bytes,
            hits[id],
            crc[id]
        )?;
    }
    csv.sync_all()?;
    let total_normal = normal_ns.iter().sum::<u64>();
    let summary = json!({"precision":plan["precision"],"simd_regime":plan["simd_regime"],"ef":ef,"budget":config.rerank_limit,"queries":nq,"warmup":warm_count,"topology_crc32":topology,"normal_qps":nq as f64*1e9/total_normal as f64,"recall_at_10":hits.iter().sum::<usize>() as f64/(nq*10) as f64,"record_bytes":index.vector_storage_stats().rerank_record_bytes,"canonical_heap_bytes":index.vector_storage_stats().canonical_heap_bytes});
    fs::write(&output, serde_json::to_vec_pretty(&summary)?)?;
    println!(
        "完成 {} {} ef{}，QPS {:.1}，Recall {:.4}%",
        plan["precision"].as_str().unwrap(),
        plan["simd_regime"].as_str().unwrap(),
        ef,
        summary["normal_qps"].as_f64().unwrap(),
        summary["recall_at_10"].as_f64().unwrap() * 100.0
    );
    Ok(())
}
