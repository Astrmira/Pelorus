//! 单页合表消除二次 DTLB（Opt 1）与束内头行两级早筛（Opt 2）同图受控评测入口。
#[cfg(feature = "research")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use quiver_ann::research::{
        ExecutionConfig, GeometryComponents, HyperbolicStrategy as Mode, SketchFilter,
        StorageConfig, TraversalProfile, execution_config, set_asym_sm_lut_override,
        set_execution_config, set_geometry_components, set_hyperbolic_strategy, set_storage_config,
        storage_config,
    };
    use quiver_ann::{BuildConfig, Index, PoincareNavigation, SearchConfig};
    use rayon::prelude::*;
    use serde_json::{Value, json};
    use std::{
        collections::BTreeMap,
        io::{Read, Write},
        path::PathBuf,
        time::Instant,
    };

    // 读取 Linux 进程常驻物理内存与峰值物理内存状态
    fn memory() -> String {
        std::fs::read_to_string("/proc/self/status")
            .unwrap_or_default()
            .lines()
            .filter(|line| line.starts_with("VmRSS:") || line.starts_with("VmHWM:"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    // 读取容器 cgroup CPU 节流计数以监测多线程是否触发配额限流
    fn cpu_stat() -> BTreeMap<String, u64> {
        std::fs::read_to_string("/sys/fs/cgroup/cpu/cpu.stat")
            .unwrap_or_default()
            .lines()
            .filter_map(|line| {
                let mut parts = line.split_whitespace();
                Some((parts.next()?.to_owned(), parts.next()?.parse().ok()?))
            })
            .collect()
    }

    // 确定性 Fisher-Yates 洗牌：确保每一轮连续查询流均为无局部性的独立乱序请求
    fn shuffle<T>(values: &mut [T], mut seed: u64) {
        for i in (1..values.len()).rev() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            values.swap(i, seed as usize % (i + 1));
        }
    }

    fn flag(value: &Value, name: &str, default: bool) -> bool {
        value.get(name).and_then(Value::as_bool).unwrap_or(default)
    }

    fn number(value: &Value, name: &str, default: usize) -> usize {
        value
            .get(name)
            .and_then(Value::as_u64)
            .map_or(default, |v| v as usize)
    }

    fn float32(value: &Value, name: &str, default: f32) -> f32 {
        value
            .get(name)
            .and_then(Value::as_f64)
            .map_or(default, |v| v as f32)
    }

    fn parse_speed(value: &Value) -> ExecutionConfig {
        ExecutionConfig {
            zero_trace_overhead: flag(value, "trace_opt", true),
            two_level_cones: flag(value, "cones", true),
            rolling_prefetch: flag(value, "prefetch", true),
            fast_kernels: flag(value, "reuse", true),
            kernel_variant: number(value, "kernel", 0) as u8,
            traversal: match value["profile"].as_str().unwrap_or("default") {
                "balanced" => TraversalProfile::Balanced,
                "lean" => TraversalProfile::Lean,
                _ => TraversalProfile::Standard,
            },
        }
    }

    fn parse_hot_opt(value: &Value) -> Result<StorageConfig, Box<dyn std::error::Error>> {
        let colocated = flag(value, "colocated", false);
        let eager_overlap = flag(value, "eager_overlap", false);
        let pf_hdr = number(value, "pf_hdr", 8).clamp(1, 32) as u8;
        let margin = float32(value, "margin", 0.10);
        let early_gate = match value["gate"].as_str().unwrap_or("off") {
            "off" => SketchFilter::Off,
            "radial" => SketchFilter::LosslessRadial,
            "sym256" => SketchFilter::SymSketch256 { margin },
            "asym256" => SketchFilter::AsymSketch256 { margin },
            other => return Err(format!("未知头行早筛模式: {other}").into()),
        };
        Ok(StorageConfig {
            colocated_store: colocated,
            early_gate,
            header_prefetch_window: pf_hdr,
            eager_prefetch_overlap: eager_overlap,
        })
    }

    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: &str| -> String {
        args.iter()
            .position(|v| v == name)
            .map(|i| args[i + 1].clone())
            .unwrap_or_else(|| default.to_owned())
    };

    let n: usize = arg("--n", "1000000").parse()?;
    let dim: usize = arg("--dim", "768").parse()?;
    let nq: usize = arg("--queries", "1000").parse()?;
    let bt: usize = arg("--build-threads", "12").parse()?;
    let seed: u64 = arg("--seed", "12345").parse()?;
    let data = PathBuf::from(arg("--data", "."));
    let output = PathBuf::from(arg("--output", "colocated_early_report.jsonl"));
    let plan: Value = serde_json::from_slice(&std::fs::read(arg("--plan", "plan.json"))?)?;
    let conditions = plan["conditions"]
        .as_array()
        .ok_or("实验计划缺少 conditions 数组")?
        .clone();
    if n == 0 || dim != 768 || nq == 0 || bt == 0 || conditions.is_empty() {
        return Err("数据规模、768 维度、查询数、构图线程数和实验条件均须有效".into());
    }

    unsafe {
        std::env::set_var("QUIVER_BUILD_SEED", seed.to_string());
        std::env::set_var("QUIVER_BUILD_PHASES", "1");
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(bt)
        .build_global()?;

    let read_floats = |name: &str, rows: usize| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::open(data.join(name))?;
        if file.metadata()?.len() != (rows * dim * 4) as u64 {
            return Err(format!("向量文件 {name} 大小不匹配").into());
        }
        let mut values = vec![0f32; rows * dim];
        file.read_exact(bytemuck::cast_slice_mut(&mut values))?;
        if !cfg!(target_endian = "little") {
            for value in &mut values {
                *value = f32::from_bits(value.to_bits().swap_bytes());
            }
        }
        Ok(values)
    };

    println!("正在加载数据集: n={n}, dim={dim}, queries={nq}, seed={seed} ...");
    let vectors = read_floats("documents.f32", n)?;
    let queries = read_floats("queries.f32", nq)?;
    let truth_bytes = std::fs::read(data.join("truth.u64"))?;
    if truth_bytes.len() != nq * 80 {
        return Err("真值文件 truth.u64 大小错误".into());
    }
    let truth: Vec<u64> = truth_bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| u64::from_le_bytes(*b))
        .collect();
    if truth.iter().any(|&id| id >= n as u64) {
        return Err("真值标识越界".into());
    }

    let document_crc = crc32fast::hash(bytemuck::cast_slice(&vectors));
    let query_crc = crc32fast::hash(bytemuck::cast_slice(&queries));
    let truth_crc = crc32fast::hash(&truth_bytes);

    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut report = std::io::BufWriter::new(std::fs::File::create(&output)?);

    // 与 followup_v2 完全对齐的 A8 (Mode::Full) 建图配置
    set_hyperbolic_strategy(Mode::Full);
    set_geometry_components(GeometryComponents::default());
    set_execution_config(parse_speed(&plan["build"]));
    set_storage_config(StorageConfig::default());
    set_asym_sm_lut_override(Some(flag(&plan["build"], "asym_lut", false)));

    let ids: Vec<u64> = (0..n as u64).collect();
    let t_build = Instant::now();
    let mut index = Index::build_poincare(
        dim,
        &ids,
        &vectors,
        &BuildConfig {
            m: 16,
            ef_construction: 128,
            alpha: 1.2,
        },
        1.0,
        PoincareNavigation::GeometryBq2,
    )?;
    let build_seconds = t_build.elapsed().as_secs_f64();
    drop(ids);
    drop(vectors);
    let build_memory = memory();

    // 独立真值抽样核验
    let oracle_count: usize = arg("--oracle", "10").parse()?;
    let oracle_indices: Vec<usize> = (0..nq)
        .step_by((nq / oracle_count.max(1)).max(1))
        .take(oracle_count.min(nq))
        .collect();
    oracle_indices
        .par_iter()
        .try_for_each(|&i| -> Result<(), String> {
            let actual: Vec<_> = index
                .search_exact(&queries[i * dim..(i + 1) * dim], 10)
                .map_err(|e| e.to_string())?
                .iter()
                .map(|r| r.0)
                .collect();
            if actual != truth[i * 10..i * 10 + 10] {
                return Err(format!("独立真值核验失败，查询序号 {i}"));
            }
            Ok(())
        })?;

    let graph = index.graph_diagnostics();
    if graph.invalid_edges + graph.self_loops + graph.duplicate_edges > 0 {
        return Err("图包含非法边、自环或重复边".into());
    }
    let topology_crc = graph.topology_crc32;
    let nav_bytes_split = index.navigation_storage_bytes();

    // 构建完成后先保持纯净原始分离内存态（零冗余双存），使得基线条件在原始 485MiB 内存占用下实测；
    // 待首个 colocated=true 条件到达时再按需物化 4KiB 单页对齐 256B 合表并按配置释放旧分离数组。
    let build_record = json!({
        "类型": "构建",
        "模式": "a8",
        "构图种子": seed,
        "数据规模": n,
        "维度": dim,
        "独立查询数": nq,
        "构图线程": bt,
        "可查询就绪秒": build_seconds,
        "构建后内存": build_memory,
        "分离布局导航载荷字节": nav_bytes_split,
        "拓扑CRC32": topology_crc,
        "文档CRC32": document_crc,
        "查询CRC32": query_crc,
        "真值CRC32": truth_crc,
        "入口可达": graph.reachable_nodes,
        "零入度": graph.zero_in_degree,
        "边数": graph.edges,
        "实验条件数": conditions.len(),
    });
    println!("{build_record}");
    writeln!(report, "{build_record}")?;
    report.flush()?;
    drop(graph);

    let mut colocated_ready = false;
    let mut reclaimed = false;
    for condition in &conditions {
        let name = condition["name"].as_str().ok_or("实验条件缺少名称")?;
        let hot_opt = parse_hot_opt(condition)?;
        let want_reclaim = flag(condition, "reclaim_split", false);
        if hot_opt.colocated_store && !colocated_ready {
            let colocated_build_sec = index.prepare_colocated_hot_store(want_reclaim)?;
            colocated_ready = true;
            reclaimed = want_reclaim;
            let state_event = json!({
                "类型": "合表物化",
                "触发条件": name,
                "释放旧分离数组": reclaimed,
                "单页合表物化秒": colocated_build_sec,
                "当前导航有效载荷字节": index.navigation_storage_bytes(),
                "当前内存": memory(),
            });
            println!("{state_event}");
            writeln!(report, "{state_event}")?;
            report.flush()?;
        } else if want_reclaim && !reclaimed {
            let extra_sec = index.prepare_colocated_hot_store(true)?;
            reclaimed = true;
            let state_event = json!({
                "类型": "释放分离数组",
                "触发条件": name,
                "耗时秒": extra_sec,
                "当前导航有效载荷字节": index.navigation_storage_bytes(),
                "当前内存": memory(),
            });
            println!("{state_event}");
            writeln!(report, "{state_event}")?;
            report.flush()?;
        } else if !hot_opt.colocated_store && reclaimed {
            return Err("释放旧分离热数组后不可再回退执行 colocated=false 的基线条件".into());
        }

        let contiguous = flag(condition, "contiguous", true);
        if !reclaimed {
            index.set_bq_contiguous_layout(contiguous);
        }
        set_hyperbolic_strategy(Mode::Full);
        set_geometry_components(GeometryComponents {
            asymmetric_gate: flag(condition, "asym_gate", true),
            radial_bound: flag(condition, "bound", true),
            ..GeometryComponents::default()
        });
        let query_speed = parse_speed(condition);
        set_execution_config(query_speed);
        if execution_config() != query_speed {
            return Err("ExecutionConfig 读写校验不一致".into());
        }
        set_storage_config(hot_opt);
        if storage_config() != hot_opt {
            return Err("StorageConfig 读写校验不一致".into());
        }
        set_asym_sm_lut_override(Some(flag(condition, "asym_lut", false)));

        let ef = number(condition, "ef", 64);
        let budget = number(condition, "B", 0);
        let threads = number(condition, "threads", 1);
        let rounds = number(condition, "rounds", 5);
        if threads == 0 || rounds == 0 {
            return Err("线程数与轮数必须大于零".into());
        }
        let config = SearchConfig {
            top_k: 10,
            ef_search: ef,
            rerank_limit: (budget != 0).then_some(budget),
        };
        let search = |i: usize| {
            let query = &queries[i * dim..(i + 1) * dim];
            index.search_poincare_quantized_navigation(query, &config)
        };

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        // 预热线程池与指令缓存（使用前 128 条查询，正式计时在乱序全集上跑满 rounds 轮）
        pool.install(|| {
            (0..nq.min(128))
                .into_par_iter()
                .try_for_each(|i| search(i).map(|_| ()))
        })?;

        let condition_memory = memory();
        let mut all_latencies = Vec::with_capacity(nq * rounds);
        let mut round_qps = Vec::with_capacity(rounds);
        let mut all_hits = 0usize;
        let mut result_crc = crc32fast::Hasher::new();
        let mut cpu_delta = BTreeMap::<String, u64>::new();

        for round in 0..rounds {
            let mut order: Vec<usize> = (0..nq).collect();
            shuffle(
                &mut order,
                seed.wrapping_add(round as u64 * 7919)
                    .wrapping_add(ef as u64),
            );
            let before = cpu_stat();
            let wall = Instant::now();
            let rows: Vec<_> = pool.install(|| {
                order
                    .par_iter()
                    .map(|&i| {
                        let timer = Instant::now();
                        let result = search(i)?;
                        let nanos = timer.elapsed().as_nanos() as u64;
                        Ok((i, nanos, result))
                    })
                    .collect::<Result<Vec<_>, quiver_ann::Error>>()
            })?;
            let wall_seconds = wall.elapsed().as_secs_f64();
            let after = cpu_stat();
            for (key, value) in &after {
                if let Some(previous) = before.get(key) {
                    *cpu_delta.entry(key.clone()).or_default() += value.saturating_sub(*previous);
                }
            }
            round_qps.push(nq as f64 / wall_seconds);

            let mut canonical: Vec<_> = rows.iter().collect();
            canonical.sort_unstable_by_key(|r| r.0);
            for &(i, _, ref result) in canonical {
                result_crc.update(&(i as u64).to_le_bytes());
                for &(id, score) in result {
                    result_crc.update(&id.to_le_bytes());
                    result_crc.update(&score.to_bits().to_le_bytes());
                }
            }
            for (i, ns, result) in &rows {
                if result.len() != 10
                    || result.windows(2).any(|p| p[0].1 < p[1].1)
                    || result.iter().any(|r| !r.1.is_finite() || r.0 >= n as u64)
                {
                    return Err(format!("返回结果非法，条件={name}，查询序号={i}").into());
                }
                let hits = result
                    .iter()
                    .filter(|r| truth[i * 10..i * 10 + 10].contains(&r.0))
                    .count();
                all_hits += hits;
                all_latencies.push(*ns);
            }
        }

        all_latencies.sort_unstable();
        let recall = all_hits as f64 / (nq * rounds * 10) as f64;
        let qps = round_qps.iter().sum::<f64>() / rounds as f64;
        let mut sorted_qps = round_qps.clone();
        sorted_qps.sort_by(|a, b| a.total_cmp(b));
        let median_qps = sorted_qps[sorted_qps.len() / 2];
        let quantile = |p: f64| {
            all_latencies[((all_latencies.len() as f64 * p).ceil() as usize).saturating_sub(1)]
                as f64
                / 1e6
        };

        // 吞吐计时窗口外独立采集损失统计与零污染四阶段墙钟剖析
        let diag_n = number(condition, "diagnostics", 0).min(nq);
        let mut diag_json = Value::Null;
        if let Some(step) = nq.checked_div(diag_n) {
            let step = step.max(1);
            let mut sum_visited = 0usize;
            let mut sum_expanded = 0usize;
            let mut sum_approx = 0usize;
            let mut sum_exact = 0usize;
            let mut sum_bound_rej = 0usize;
            let mut sum_hdr_chk = 0usize;
            let mut sum_hdr_rej = 0usize;
            let mut sum_router_ns = 0u64;
            let mut sum_beam_ns = 0u64;
            let mut sum_gate_ns = 0u64;
            let mut sum_rerank_ns = 0u64;
            let mut cnt_diag = 0usize;
            for i in (0..nq).step_by(step).take(diag_n) {
                let q = &queries[i * dim..(i + 1) * dim];
                let st = index.search_loss_decomposition(
                    q,
                    &config,
                    PoincareNavigation::GeometryBq2,
                    &truth[i * 10..i * 10 + 10],
                )?;
                let prof =
                    index.search_stage_profile(q, &config, PoincareNavigation::GeometryBq2)?;
                sum_visited += st.visited_nodes;
                sum_expanded += st.expanded_nodes;
                sum_approx += st.approximate_evals;
                sum_exact += st.exact_vector_evals;
                sum_bound_rej += st.bound_rejections;
                sum_hdr_chk += st.header_checks;
                sum_hdr_rej += st.header_rejections;
                sum_router_ns += prof.router_seed_ns;
                sum_beam_ns += prof.beam_nav_ns;
                sum_gate_ns += prof.asym_gate_ns;
                sum_rerank_ns += prof.exact_rerank_ns;
                cnt_diag += 1;
            }
            let d = cnt_diag.max(1) as f64;
            let skip_ratio = if sum_hdr_chk > 0 {
                sum_hdr_rej as f64 / sum_hdr_chk as f64
            } else {
                0.0
            };
            diag_json = json!({
                "样本数": cnt_diag,
                "均访问节点V": sum_visited as f64 / d,
                "均展开节点": sum_expanded as f64 / d,
                "均完整768D近似评分": sum_approx as f64 / d,
                "均头行早筛检查": sum_hdr_chk as f64 / d,
                "均头行早筛拦截": sum_hdr_rej as f64 / d,
                "头行早筛跳过率": skip_ratio,
                "均精确向量读取": sum_exact as f64 / d,
                "均精排径向下界拒绝": sum_bound_rej as f64 / d,
                "阶段耗时微秒_路由选种": sum_router_ns as f64 / d / 1e3,
                "阶段耗时微秒_主导航束": sum_beam_ns as f64 / d / 1e3,
                "阶段耗时微秒_非对称门神": sum_gate_ns as f64 / d / 1e3,
                "阶段耗时微秒_精确精排": sum_rerank_ns as f64 / d / 1e3,
            });
        }

        let record = json!({
            "类型": "查询",
            "条件": name,
            "配置": condition,
            "构图种子": seed,
            "拓扑CRC32": topology_crc,
            "线程数": threads,
            "ef": ef,
            "B": budget,
            "recall10": recall,
            "QPS均值": qps,
            "QPS中位数": median_qps,
            "各轮QPS": round_qps,
            "P50毫秒": quantile(0.50),
            "P95毫秒": quantile(0.95),
            "P99毫秒": quantile(0.99),
            "均值毫秒": all_latencies.iter().map(|&v| v as f64).sum::<f64>() / all_latencies.len() as f64 / 1e6,
            "结果CRC32": result_crc.finalize(),
            "内存": condition_memory,
            "导航有效载荷字节": index.navigation_storage_bytes(),
            "cgroup计数增量": cpu_delta,
            "诊断分解": diag_json,
        });
        println!("{record}");
        writeln!(report, "{record}")?;
        report.flush()?;
    }

    println!("全部评测条件顺利完成，最终内存: {}", memory());
    Ok(())
}

#[cfg(not(feature = "research"))]
fn main() {
    eprintln!("评测入口需要启用 --features research");
    std::process::exit(1);
}
