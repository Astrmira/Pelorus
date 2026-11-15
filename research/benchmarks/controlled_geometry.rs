//! 固定数据、独立真值、分离构建与查询预算的核心 ANN 实验入口。
#[cfg(feature = "research")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use quiver_ann::research::{
        GeometryComponents, HyperbolicStrategy as Mode, set_geometry_components,
        set_hyperbolic_strategy,
    };
    use quiver_ann::{BuildConfig, Index, PoincareNavigation, SearchConfig};
    use rayon::prelude::*;
    use serde_json::json;
    use std::io::{Read, Write};
    use std::path::PathBuf;
    use std::time::Instant;
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: &str| -> String {
        args.iter()
            .position(|v| v == name)
            .map(|i| args.get(i + 1).expect("参数缺少值").clone())
            .unwrap_or(default.to_owned())
    };
    let list = |name: &str, default: &str| -> Vec<usize> {
        arg(name, default)
            .split(',')
            .map(|s| s.parse().expect("列表参数必须为整数"))
            .collect()
    };
    let n: usize = arg("--n", "1000000").parse()?;
    let dim: usize = arg("--dim", "768").parse()?;
    let nq: usize = arg("--queries", "1000").parse()?;
    let bt: usize = arg("--build-threads", "12").parse()?;
    let repeats: usize = arg("--repeats", "1").parse()?;
    let diag_n: usize = arg("--diagnostic-queries", "100").parse()?;
    let seed: u64 = arg("--seed", "12345").parse()?;
    let data = PathBuf::from(arg("--data", "."));
    let output = PathBuf::from(arg("--output", "rigorous.jsonl"));
    let mode_name = arg("--mode", "a8");
    let mode = match mode_name.as_str() {
        "a8" => Mode::Full,
        "a7" => Mode::FixedShellCompass,
        "sq8" => Mode::ScalarQuantizedDirection,
        "fp32" => Mode::FullPrecisionDirection,
        "vamana" => Mode::GeodesicPruning,
        "vamana-fill" => Mode::GeodesicPruningWithFill,
        "vamana-portal" => Mode::GeodesicPortals,
        _ => return Err("未知构图模式".into()),
    };
    let disable = arg("--disable", "none");
    for (flag, default, allowed) in [
        ("--navs", "bq2,exact", &["bq2", "exact", "sq8", "fp32"][..]),
        (
            "--query-strategies",
            "full",
            &["full", "no-asym", "no-bound"][..],
        ),
        (
            "--disable",
            "none",
            &["none", "asym", "shells", "bound", "repair", "compass"][..],
        ),
        ("--tail", "false", &["false", "true", "both"][..]),
    ] {
        if arg(flag, default).split(',').any(|v| !allowed.contains(&v)) {
            return Err(format!("实验参数{flag}包含未知条件").into());
        }
    }
    let components = GeometryComponents {
        asymmetric_gate: !disable.split(',').any(|v| v == "asym"),
        adaptive_shells: !disable.split(',').any(|v| v == "shells"),
        radial_bound: !disable.split(',').any(|v| v == "bound"),
        connectivity_repair: !disable.split(',').any(|v| v == "repair"),
        compass: !disable.split(',').any(|v| v == "compass"),
    };
    set_hyperbolic_strategy(mode);
    set_geometry_components(components);
    if n == 0 || nq == 0 || bt == 0 || repeats == 0 || dim == 0 {
        return Err("规模与线程数必须大于零".into());
    }
    // 环境仅在启动工作线程之前设置，实验进程负责独占自身配置。
    unsafe {
        std::env::set_var("QUIVER_BUILD_SEED", seed.to_string());
        std::env::set_var("QUIVER_BUILD_PHASES", "1");
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(bt)
        .build_global()?;
    let read_f32 = |name: &str, rows: usize| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut f = std::fs::File::open(data.join(name))?;
        if f.metadata()?.len() != (rows * dim * 4) as u64 {
            return Err("向量文件大小错误".into());
        }
        let mut v = vec![0.0f32; rows * dim];
        f.read_exact(bytemuck::cast_slice_mut(&mut v))?;
        if !cfg!(target_endian = "little") {
            for x in &mut v {
                *x = f32::from_bits(x.to_bits().swap_bytes());
            }
        }
        Ok(v)
    };
    let load_started = Instant::now();
    let vectors = read_f32("documents.f32", n)?;
    let queries = read_f32("queries.f32", nq)?;
    let truth_bytes = std::fs::read(data.join("truth.u64"))?;
    if truth_bytes.len() != nq * 10 * 8 {
        return Err("真值文件大小错误".into());
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
    let load_seconds = load_started.elapsed().as_secs_f64();
    let doc_crc = crc32fast::hash(bytemuck::cast_slice(&vectors));
    let query_crc = crc32fast::hash(bytemuck::cast_slice(&queries));
    let truth_crc = crc32fast::hash(&truth_bytes);
    let build_config = BuildConfig {
        m: arg("--m", "16").parse()?,
        ef_construction: arg("--efc", "128").parse()?,
        alpha: arg("--alpha", "1.2").parse()?,
    };
    let navigation = if mode_name.starts_with("vamana") {
        PoincareNavigation::Exact
    } else {
        PoincareNavigation::GeometryBq2
    };
    let started = Instant::now();
    let mut index = Index::build_poincare(
        dim,
        &(0..n as u64).collect::<Vec<_>>(),
        &vectors,
        &build_config,
        1.0,
        navigation,
    )?;
    let build_seconds = started.elapsed().as_secs_f64();
    drop(vectors);
    let status = || {
        std::fs::read_to_string("/proc/self/status")
            .unwrap_or_default()
            .lines()
            .filter(|l| l.starts_with("VmRSS:") || l.starts_with("VmHWM:"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let cpu_stat = || -> std::collections::BTreeMap<String, u64> {
        std::fs::read_to_string("/sys/fs/cgroup/cpu/cpu.stat")
            .unwrap_or_default()
            .lines()
            .filter_map(|line| {
                let mut parts = line.split_whitespace();
                let name = parts.next()?.to_owned();
                let value = parts.next()?.parse().ok()?;
                Some((name, value))
            })
            .collect()
    };
    let build_memory = status();
    let mut report = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)?,
    );
    let mut cases = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.with_extension("requests.csv"))?,
    );
    let mut diagnostics = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.with_extension("diagnostics.jsonl"))?,
    );
    writeln!(
        cases,
        "阶段,导航,模块,ef,B,线程,轮次,请求序号,查询序号,命中数,延迟纳秒"
    )?;
    // 独立坐标差 FP64 核在计时之外交叉核验固定的十条查询。
    let oracle_indices: Vec<_> = if arg("--oracle-all", "false") == "true" {
        (0..nq).collect()
    } else {
        (0..nq).step_by((nq / 10).max(1)).take(10).collect()
    };
    oracle_indices
        .par_iter()
        .try_for_each(|&i| -> Result<(), String> {
            let actual: Vec<_> = index
                .search_exact(&queries[i * dim..(i + 1) * dim], 10)
                .map_err(|e| e.to_string())?
                .iter()
                .map(|h| h.0)
                .collect();
            if actual != truth[i * 10..i * 10 + 10] {
                return Err(format!(
                    "独立真值核验失败，查询{i}，坐标差结果{actual:?}，文件真值{:?}",
                    &truth[i * 10..i * 10 + 10]
                ));
            }
            Ok(())
        })?;
    let graph = index.graph_diagnostics();
    if graph.invalid_edges + graph.self_loops + graph.duplicate_edges != 0 {
        return Err("图结构无效".into());
    }
    // 多门户可以选择固定入口闭包之外的起点，这里仅描述固定入口的图结构。
    let entry_truth_fraction = truth
        .iter()
        .filter(|&&id| graph.nodes[id as usize].reachable)
        .count() as f64
        / truth.len() as f64;
    let build = json!({"类型":"构建", "数据规模":n,"维度":dim,"独立查询数":nq,"模式":mode_name,"禁用模块":disable,
        "种子":seed,"构图线程":bt,"m":build_config.m,"efc":build_config.ef_construction,"alpha":build_config.alpha,"独立真值核验数":oracle_indices.len(),
        "读取秒":load_seconds,"可查询就绪秒":build_seconds,"构建后内存":build_memory,
        "入口可达":graph.reachable_nodes,"零入度":graph.zero_in_degree,"边数":graph.edges,
        "弱连通分量":graph.weak_components,"入口真值比例":entry_truth_fraction,"拓扑CRC32":graph.topology_crc32,
        "文档CRC32":doc_crc,"查询CRC32":query_crc,"真值CRC32":truth_crc,"原始FP32向量字节":n*dim*4,"导航有效载荷字节":index.navigation_storage_bytes()});
    println!("{build}");
    writeln!(report, "{build}")?;
    report.flush()?;
    let topology_crc = graph.topology_crc32;
    drop(graph);
    let navs = arg("--navs", "bq2,exact");
    let query_strategy = arg("--query-strategies", "full");
    let efs = list("--efs", "32,64,128,256,512,1024");
    let budgets = list("--budgets", "0,10,32,64,128");
    let mts = list("--query-threads", "1");
    let tail_efs = list("--tail-efs", "64,256");
    let tail_repeats: usize = arg("--tail-repeats", "10").parse()?;
    let tail_mode = arg("--tail", "false");
    let phases = if tail_mode == "both" {
        vec![false, true]
    } else {
        vec![tail_mode == "true"]
    };
    let conditions: Vec<_> = navs
        .split(',')
        .flat_map(|nav| query_strategy.split(',').map(move |qa| (nav, qa)))
        .collect();
    for (nav, qa) in conditions {
        for &tail in &phases {
            if tail && (qa != "full" || nav == "sq8" || nav == "fp32") {
                continue;
            }
            let query_mode = match nav {
                "sq8" => Mode::ScalarQuantizedDirection,
                "fp32" => Mode::FullPrecisionDirection,
                _ => mode,
            };
            index.prepare_direction_representation(query_mode)?;
            set_hyperbolic_strategy(query_mode);
            let mut qc = components;
            if qa == "no-asym" {
                qc.asymmetric_gate = false;
            }
            if qa == "no-bound" {
                qc.radial_bound = false;
            }
            set_geometry_components(qc);
            for &mt in &mts {
                if !tail && mt != 1 {
                    continue;
                }
                if mt == 0 {
                    return Err("查询线程数必须大于零".into());
                }
                let pool = rayon::ThreadPoolBuilder::new().num_threads(mt).build()?;
                for &ef in &efs {
                    for &budget in &budgets {
                        if budget != 0 && (budget < 10 || budget > ef) {
                            continue;
                        }
                        if nav == "exact" && budget != 0 {
                            continue;
                        }
                        // 多线程尾延迟限定在预先登记的候选工作点，完整曲线单线程测量。
                        if tail && (!tail_efs.contains(&ef) || budget != 0) {
                            continue;
                        }
                        let cfg = SearchConfig {
                            top_k: 10,
                            ef_search: ef,
                            rerank_limit: (budget != 0).then_some(budget),
                        };
                        let search = |i: usize| -> Result<_, quiver_ann::Error> {
                            let q = &queries[i * dim..(i + 1) * dim];
                            if nav == "exact" {
                                index.search_poincare_exact_navigation(q, &cfg)
                            } else {
                                index.search_poincare_quantized_navigation(q, &cfg)
                            }
                        };
                        pool.install(|| {
                            (0..nq.min(100))
                                .into_par_iter()
                                .try_for_each(|i| search(i).map(|_| ()))
                        })?;
                        let rounds = if tail { tail_repeats } else { repeats };
                        for round in 0..rounds {
                            let mut order: Vec<usize> = (0..nq).collect();
                            let mut rng = seed
                                .wrapping_add(round as u64 * 7919)
                                .wrapping_add(ef as u64);
                            for i in (1..nq).rev() {
                                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                                order.swap(i, (rng as usize) % (i + 1));
                            }
                            // 限流计数在计时窗口外读取，记录整个容器的计数增量。
                            let cpu_before = cpu_stat();
                            let wall = Instant::now();
                            let rows: Vec<_> = pool.install(|| {
                                order
                                    .par_iter()
                                    .map(|&i| {
                                        let timer = Instant::now();
                                        let result = search(i)?;
                                        let nanos = timer.elapsed().as_nanos() as u64;
                                        if result.len() != 10
                                            || result.windows(2).any(|p| p[0].1 < p[1].1)
                                        {
                                            return Err(quiver_ann::Error::InvalidSearchConfig(
                                                "结果数量或顺序错误",
                                            ));
                                        }
                                        let hits = result
                                            .iter()
                                            .filter(|h| truth[i * 10..i * 10 + 10].contains(&h.0))
                                            .count();
                                        Ok((i, hits, nanos))
                                    })
                                    .collect::<Result<Vec<_>, _>>()
                            })?;
                            let wall_seconds = wall.elapsed().as_secs_f64();
                            let cpu_after = cpu_stat();
                            let cpu_delta: std::collections::BTreeMap<_, _> = cpu_after
                                .iter()
                                .filter_map(|(name, &after)| {
                                    cpu_before
                                        .get(name)
                                        .map(|&before| (name.clone(), after.saturating_sub(before)))
                                })
                                .collect();
                            let mut latency: Vec<_> = rows.iter().map(|r| r.2).collect();
                            latency.sort_unstable();
                            let quantile = |p: f64| {
                                latency
                                    [((latency.len() as f64 * p).ceil() as usize).saturating_sub(1)]
                                    as f64
                                    / 1e6
                            };
                            let hits: usize = rows.iter().map(|r| r.1).sum();
                            let record = json!({"类型":"查询","阶段":if tail {"并发尾延迟"} else {"曲线"},"模式":mode_name,"禁用模块":disable,"导航":nav,"查询模块":qa,
                            "ef":ef,"B":budget,"线程":mt,"轮次":round,"构图种子":seed,"拓扑CRC32":topology_crc,
                            "独立查询数":nq,"请求数":nq,"recall10":hits as f64/(nq*10) as f64,"QPS":nq as f64/wall_seconds,
                            "P50毫秒":quantile(0.5),"P95毫秒":quantile(0.95),"P99毫秒":quantile(0.99),
                            "墙钟秒":wall_seconds,"cgroup计数增量":cpu_delta,
                            "原始FP32向量字节":n*dim*4,"导航有效载荷字节":index.navigation_storage_bytes()});
                            println!("{record}");
                            writeln!(report, "{record}")?;
                            report.flush()?;
                            for (request, (i, hits, ns)) in rows.iter().enumerate() {
                                writeln!(
                                    cases,
                                    "{},{nav},{qa},{ef},{budget},{mt},{round},{request},{i},{hits},{ns}",
                                    if tail { "并发尾延迟" } else { "曲线" }
                                )?;
                            }
                            cases.flush()?;
                        }
                        if mt == 1 && !tail && diag_n > 0 {
                            let dn = diag_n.min(nq);
                            for i in (0..nq).step_by((nq / dn).max(1)).take(dn) {
                                let stats = index.search_loss_decomposition(
                                    &queries[i * dim..(i + 1) * dim],
                                    &cfg,
                                    if nav == "exact" {
                                        PoincareNavigation::Exact
                                    } else {
                                        PoincareNavigation::GeometryBq2
                                    },
                                    &truth[i * 10..i * 10 + 10],
                                )?;
                                let hits = search(i)?
                                    .iter()
                                    .filter(|h| truth[i * 10..i * 10 + 10].contains(&h.0))
                                    .count();
                                if hits != stats.hits_returned
                                    || stats.hits_selected_for_rerank != stats.hits_returned
                                {
                                    return Err(
                                        format!("诊断与实际搜索或精排真值不一致，查询{i}").into()
                                    );
                                }
                                let record = json!({"导航":nav,"模块":qa,"ef":ef,"B":budget,"查询序号":i,
                                "V命中":stats.hits_visited,"C命中":stats.hits_in_beam,"F命中":stats.hits_selected_for_rerank,"R命中":stats.hits_returned,
                                "V节点":stats.visited_nodes,"C节点":stats.beam_candidates,"F节点":stats.selected_for_rerank,
                                "展开节点":stats.expanded_nodes,"展开次数":stats.expansion_calls,"精确向量读取":stats.exact_vector_evals,
                                "近似评分次数":stats.approximate_evals,"下界拒绝":stats.bound_rejections});
                                writeln!(diagnostics, "{record}")?;
                            }
                            diagnostics.flush()?;
                        }
                    }
                }
            }
        }
    }
    println!("实验完成，进程内存状态 {}", status());
    Ok(())
}

#[cfg(not(feature = "research"))]
fn main() {
    eprintln!("严谨实验入口需要启用 research 特性");
    std::process::exit(1);
}
