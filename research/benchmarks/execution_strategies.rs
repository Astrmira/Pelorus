//! 黎曼 QuIVer（A8 / 2-bit SM）系统与微架构逐项消融、遍历预算前沿及同图表示对照实验入口。
//!
//! 在固定数据、独立真值与完全相同的 A8 邻接拓扑（通过 `topology_crc32` 核验）上，
//! 依次测量：
//!   1. 768D 对称 SM 5-popcount 四种微架构位核（AVX2 VPSHUFB CSA / POPCNT-60 / POPCNT-CSA-40 / Hybrid Port1+Port5）延迟；
//!   2. 正向累积无损加速消融（S0 v3 基线 -> +O0 零追踪 -> +O1 两级方向锥 -> +O2a 64B 连续签名 -> +O2b LFB 滚动预取 -> +O3 硬件位核）；
//!   3. 从 S5 无损全栈出发的留一法（Leave-One-Out）反向消融及非对称门神/径向下界独立增益核验；
//!   4. O4 遍历预算档位（Standard / Balanced / Lean）与精排预算 B 的 Recall-QPS-P99 帕累托前沿；
//!   5. 多线程并发吞吐与合并请求 P50/P95/P99 尾延迟对比（S0 vs S5 vs O4-Balanced）；
//!   6. 同一张 A8 图上的方向表示对照（2-bit SM vs SQ8 vs 共享 FP32 vs Exact 导航）。

#[cfg(feature = "research")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use quiver_ann::research::{
        ExecutionConfig, GeometryComponents, HyperbolicStrategy as Mode, TraversalProfile,
        benchmark_sign_magnitude_kernels, set_execution_config, set_geometry_components,
        set_hyperbolic_strategy,
    };
    use quiver_ann::{BuildConfig, Index, PoincareNavigation, SearchConfig};
    use rayon::prelude::*;
    use serde_json::json;
    use std::io::{Read, Write};
    use std::path::PathBuf;
    use std::time::Instant;

    #[derive(Clone)]
    struct ConditionSpec {
        group: &'static str,
        name: &'static str,
        nav: &'static str, // "bq2", "sq8", "fp32", "exact"
        contiguous: bool,
        speed: ExecutionConfig,
        asym_gate: bool,
        radial_bound: bool,
        efs: Vec<usize>,
        budgets: Vec<usize>,
        threads: Vec<usize>,
        rounds: usize,
        run_diag: bool,
    }

    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: &str| -> String {
        args.iter()
            .position(|v| v == name)
            .map(|i| args.get(i + 1).expect("命令行参数缺少对应取值").clone())
            .unwrap_or_else(|| default.to_owned())
    };

    let n: usize = arg("--n", "1000000").parse()?;
    let dim: usize = arg("--dim", "768").parse()?;
    let nq: usize = arg("--queries", "1000").parse()?;
    let bt: usize = arg("--build-threads", "12").parse()?;
    let repeats: usize = arg("--repeats", "3").parse()?;
    let tail_repeats: usize = arg("--tail-repeats", "10").parse()?;
    let diag_n: usize = arg("--diagnostic-queries", "100").parse()?;
    let seed: u64 = arg("--seed", "12345").parse()?;
    let m_deg: usize = arg("--m", "16").parse()?;
    let efc: usize = arg("--efc", "128").parse()?;
    let alpha: f32 = arg("--alpha", "1.2").parse()?;
    let suite = arg("--suite", "full"); // "full", "core", "codegen", "smoke"
    let tag = arg("--tag", "default_release");
    let build_opt = arg("--build-opt", "fast"); // "fast" 或 "v3"
    let oracle_all = arg("--oracle-all", "false") == "true";
    let data = PathBuf::from(arg("--data", "."));
    let output = PathBuf::from(arg("--output", "execution_strategies.jsonl"));

    if n == 0 || nq == 0 || bt == 0 || repeats == 0 || dim == 0 {
        return Err("数据规模、维度、查询数、线程数与重复轮次均必须大于零".into());
    }
    if !matches!(suite.as_str(), "full" | "core" | "codegen" | "smoke") {
        return Err(format!("未知实验套件类型: {suite}").into());
    }

    // 在启动全局线程池前固定构图随机种子与阶段计时日志
    unsafe {
        std::env::set_var("QUIVER_BUILD_SEED", seed.to_string());
        std::env::set_var("QUIVER_BUILD_PHASES", "1");
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(bt)
        .build_global()?;

    let mut report = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)?,
    );
    let mut diagnostics_out = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.with_extension("diagnostics.jsonl"))?,
    );

    // ─────────────────────────────────────────────────────────────────────────
    // 0. 768D 对称 SM 四种微架构位核基准测量（自动选定当前 CPU 最快硬件位核）
    // ─────────────────────────────────────────────────────────────────────────
    let kernel_rounds = if suite == "smoke" { 20_000 } else { 400_000 };
    let _ = benchmark_sign_magnitude_kernels(10_000); // 预热指令与 L1 数据缓存
    let kernel_bench = benchmark_sign_magnitude_kernels(kernel_rounds);
    let mut best_kvar = 3u8;
    let mut best_ns = f64::INFINITY;
    for &(vid, kname, ns) in &kernel_bench {
        if vid >= 1 && ns < best_ns {
            best_ns = ns;
            best_kvar = vid;
        }
        let krec = json!({
            "类型": "微基准",
            "编译标签": tag,
            "构图种子": seed,
            "变体编号": vid,
            "内核名称": kname,
            "测量轮次": kernel_rounds,
            "单次纳秒": ns,
            "等效每秒亿次评估": 10.0 / ns,
        });
        println!("{krec}");
        writeln!(report, "{krec}")?;
    }
    report.flush()?;

    // ─────────────────────────────────────────────────────────────────────────
    // 1. 加载固定数据集并校验 CRC32
    // ─────────────────────────────────────────────────────────────────────────
    let read_f32 = |name: &str, rows: usize| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut f = std::fs::File::open(data.join(name))?;
        if f.metadata()?.len() != (rows * dim * 4) as u64 {
            return Err(format!("向量文件 {name} 字节大小与预期 ({rows}x{dim}) 不符").into());
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
    let load_seconds = load_started.elapsed().as_secs_f64();
    let doc_crc = crc32fast::hash(bytemuck::cast_slice(&vectors));
    let query_crc = crc32fast::hash(bytemuck::cast_slice(&queries));
    let truth_crc = crc32fast::hash(&truth_bytes);

    // ─────────────────────────────────────────────────────────────────────────
    // 2. 构建 A8 / 2-bit SM 图索引并执行独立 FP64 真值核验与拓扑完整性检查
    // ─────────────────────────────────────────────────────────────────────────
    set_hyperbolic_strategy(Mode::Full);
    set_geometry_components(GeometryComponents::default());
    if build_opt == "v3" {
        set_execution_config(ExecutionConfig::reference());
    } else {
        set_execution_config(ExecutionConfig::optimized(best_kvar));
    }

    let build_config = BuildConfig {
        m: m_deg,
        ef_construction: efc,
        alpha,
    };
    let ids: Vec<u64> = (0..n as u64).collect();
    let started = Instant::now();
    let mut index = Index::build_poincare(
        dim,
        &ids,
        &vectors,
        &build_config,
        1.0,
        PoincareNavigation::GeometryBq2,
    )?;
    let build_seconds = started.elapsed().as_secs_f64();
    drop(ids);
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
    let oracle_indices: Vec<usize> = if oracle_all {
        (0..nq).collect()
    } else {
        (0..nq).step_by((nq / 10).max(1)).take(10).collect()
    };
    oracle_indices
        .par_iter()
        .try_for_each(|&i| -> Result<(), String> {
            let actual: Vec<u64> = index
                .search_exact(&queries[i * dim..(i + 1) * dim], 10)
                .map_err(|e| e.to_string())?
                .iter()
                .map(|h| h.0)
                .collect();
            if actual != truth[i * 10..i * 10 + 10] {
                return Err(format!(
                    "独立 FP64 真值核验失败，查询 {i}，实算 {actual:?} != 真值 {:?}",
                    &truth[i * 10..i * 10 + 10]
                ));
            }
            Ok(())
        })?;

    let graph = index.graph_diagnostics();
    if graph.invalid_edges + graph.self_loops + graph.duplicate_edges != 0 {
        return Err("构建图包含非法边、自环或重复边".into());
    }
    let entry_truth_fraction = truth
        .iter()
        .filter(|&&id| graph.nodes[id as usize].reachable)
        .count() as f64
        / truth.len() as f64;
    let topology_crc = graph.topology_crc32;
    let build_rec = json!({
        "类型": "构建",
        "编译标签": tag,
        "套件": suite,
        "构建加速配置": build_opt,
        "选定最快位核": best_kvar,
        "数据规模": n,
        "维度": dim,
        "独立查询数": nq,
        "构图种子": seed,
        "构图线程": bt,
        "m": build_config.m,
        "efc": build_config.ef_construction,
        "alpha": build_config.alpha,
        "独立真值核验数": oracle_indices.len(),
        "读取秒": load_seconds,
        "可查询就绪秒": build_seconds,
        "构建后内存": build_memory,
        "入口可达": graph.reachable_nodes,
        "零入度": graph.zero_in_degree,
        "边数": graph.edges,
        "弱连通分量": graph.weak_components,
        "入口真值比例": entry_truth_fraction,
        "拓扑CRC32": topology_crc,
        "文档CRC32": doc_crc,
        "查询CRC32": query_crc,
        "真值CRC32": truth_crc,
        "原始FP32向量字节": n * dim * 4,
        "导航有效载荷字节": index.navigation_storage_bytes(),
    });
    println!("{build_rec}");
    writeln!(report, "{build_rec}")?;
    report.flush()?;
    drop(graph);

    // ─────────────────────────────────────────────────────────────────────────
    // 3. 组装同图消融与前沿测量条件列表
    // ─────────────────────────────────────────────────────────────────────────
    let s0 = ExecutionConfig::reference();
    let s1 = ExecutionConfig {
        zero_trace_overhead: true,
        ..s0
    };
    let s2 = ExecutionConfig {
        two_level_cones: true,
        ..s1
    };
    let s3 = s2; // O2a 通过 contiguous = true 切换
    let s4 = ExecutionConfig {
        rolling_prefetch: true,
        ..s3
    };
    let s5 = ExecutionConfig::optimized(best_kvar);
    let s5_bal = ExecutionConfig {
        traversal: TraversalProfile::Balanced,
        ..s5
    };
    let s5_lean = ExecutionConfig {
        traversal: TraversalProfile::Lean,
        ..s5
    };

    let mut specs: Vec<ConditionSpec> = Vec::new();
    let base_efs = if suite == "smoke" {
        vec![32, 64]
    } else {
        vec![32, 64, 128, 256, 512]
    };

    // Group A：正向累积无损加速消融（B=0, Standard, 单线程）
    for (name, contig, spd) in [
        ("S0_v3_baseline", false, s0),
        ("S1_plus_O0_notrace", false, s1),
        ("S2_plus_O1_2lvl_cone", false, s2),
        ("S3_plus_O2a_contig64", true, s3),
        ("S4_plus_O2b_roll_pf", true, s4),
        ("S5_lossless_full", true, s5),
    ] {
        if suite == "codegen" && !matches!(name, "S0_v3_baseline" | "S5_lossless_full") {
            continue;
        }
        specs.push(ConditionSpec {
            group: "正向累积消融",
            name,
            nav: "bq2",
            contiguous: contig,
            speed: spd,
            asym_gate: true,
            radial_bound: true,
            efs: base_efs.clone(),
            budgets: vec![0],
            threads: vec![1],
            rounds: repeats,
            run_diag: matches!(
                name,
                "S0_v3_baseline" | "S2_plus_O1_2lvl_cone" | "S5_lossless_full"
            ),
        });
    }

    if suite == "full" {
        // 附加：在端到端真实搜索下直接横评全部 4 种 768D 对称位核（0, 1, 2, 3）
        for (kname, kvar, fk) in [
            ("K0_avx2_vpshufb_csa", 0u8, false),
            ("K1_popcnt_unrolled60", 1u8, true),
            ("K2_popcnt_csa40", 2u8, true),
            ("K3_hybrid_port1_port5", 3u8, true),
        ] {
            specs.push(ConditionSpec {
                group: "位核端到端横评",
                name: kname,
                nav: "bq2",
                contiguous: true,
                speed: ExecutionConfig {
                    fast_kernels: fk,
                    kernel_variant: kvar,
                    ..s5
                },
                asym_gate: true,
                radial_bound: true,
                efs: vec![64, 256],
                budgets: vec![0],
                threads: vec![1],
                rounds: repeats,
                run_diag: false,
            });
        }

        // Group B：从 S5_lossless_full 出发的留一法（Leave-One-Out）与组件消融
        for (lname, contig, spd, asym, rbound, diag) in [
            (
                "LOO_no_O0_trace",
                true,
                ExecutionConfig {
                    zero_trace_overhead: false,
                    ..s5
                },
                true,
                true,
                false,
            ),
            (
                "LOO_no_O1_2lvl_cone",
                true,
                ExecutionConfig {
                    two_level_cones: false,
                    ..s5
                },
                true,
                true,
                true,
            ),
            ("LOO_no_O2a_contig64", false, s5, true, true, false),
            (
                "LOO_no_O2b_roll_pf",
                true,
                ExecutionConfig {
                    rolling_prefetch: false,
                    ..s5
                },
                true,
                true,
                false,
            ),
            (
                "LOO_no_O3_fast_kern",
                true,
                ExecutionConfig {
                    fast_kernels: false,
                    kernel_variant: 0,
                    ..s5
                },
                true,
                true,
                false,
            ),
            ("S5_no_asym_gate", true, s5, false, true, true),
            ("S5_no_radial_bound", true, s5, true, false, true),
        ] {
            specs.push(ConditionSpec {
                group: "留一法与组件消融",
                name: lname,
                nav: "bq2",
                contiguous: contig,
                speed: spd,
                asym_gate: asym,
                radial_bound: rbound,
                efs: vec![64, 256],
                budgets: vec![0],
                threads: vec![1],
                rounds: repeats,
                run_diag: diag,
            });
        }
    }

    // Group C：O4 遍历预算档位 × 有界精排预算 B 的帕累托前沿
    let pareto_efs = if suite == "smoke" {
        vec![48, 64]
    } else if suite == "full" {
        vec![32, 48, 64, 96, 128, 256, 512]
    } else {
        vec![32, 64, 96, 128, 256, 512]
    };
    let pareto_budgets = if suite == "smoke" {
        vec![0, 48]
    } else {
        vec![0, 32, 48, 64]
    };
    for (pname, pspd) in [
        ("P_V3Default", s5),
        ("P_Balanced", s5_bal),
        ("P_Lean", s5_lean),
    ] {
        specs.push(ConditionSpec {
            group: "遍历与精排帕累托",
            name: pname,
            nav: "bq2",
            contiguous: true,
            speed: pspd,
            asym_gate: true,
            radial_bound: true,
            efs: pareto_efs.clone(),
            budgets: pareto_budgets.clone(),
            threads: vec![1],
            rounds: repeats,
            run_diag: suite != "codegen",
        });
    }

    // Group D（先于 SQ8 缓存分配执行）：多线程吞吐扩展与合并 10,000 请求尾延迟
    if suite == "full" {
        for (mname, contig, mspd, mbudgets) in [
            ("MT_S0_v3_baseline", false, s0, vec![0]),
            ("MT_S5_lossless_full", true, s5, vec![0]),
            ("MT_O4_Balanced_B48", true, s5_bal, vec![48]),
        ] {
            specs.push(ConditionSpec {
                group: "并发吞吐与尾延迟",
                name: mname,
                nav: "bq2",
                contiguous: contig,
                speed: mspd,
                asym_gate: true,
                radial_bound: true,
                efs: vec![64, 256],
                budgets: mbudgets,
                threads: vec![1, 2, 4, 8, 12],
                rounds: tail_repeats,
                run_diag: false,
            });
        }
    }

    // Group E：同图方向表示对照（SQ8 / 共享 FP32 / Exact 精确导航）
    if suite == "full" || suite == "core" {
        let rep_efs = vec![64, 128, 256];
        for (rname, rnav, rspd, rbudgets) in [
            ("SameGraph_SQ8_v3", "sq8", s0, vec![0, 48]),
            ("SameGraph_SQ8_opt", "sq8", s5, vec![0, 48]),
            ("SameGraph_SQ8_bal", "sq8", s5_bal, vec![0, 48]),
            ("SameGraph_FP32_v3", "fp32", s0, vec![0, 48]),
            ("SameGraph_FP32_opt", "fp32", s5, vec![0, 48]),
            ("SameGraph_FP32_bal", "fp32", s5_bal, vec![0, 48]),
            ("SameGraph_Exact_v3", "exact", s0, vec![0]),
            ("SameGraph_Exact_opt", "exact", s5, vec![0]),
        ] {
            if suite == "core"
                && !matches!(
                    rname,
                    "SameGraph_SQ8_v3"
                        | "SameGraph_SQ8_opt"
                        | "SameGraph_FP32_v3"
                        | "SameGraph_FP32_opt"
                        | "SameGraph_Exact_opt"
                )
            {
                continue;
            }
            specs.push(ConditionSpec {
                group: "同图表示对照",
                name: rname,
                nav: rnav,
                contiguous: true,
                speed: rspd,
                asym_gate: true,
                radial_bound: true,
                efs: rep_efs.clone(),
                budgets: rbudgets,
                threads: vec![1],
                rounds: repeats,
                run_diag: matches!(
                    rname,
                    "SameGraph_SQ8_opt" | "SameGraph_FP32_opt" | "SameGraph_Exact_opt"
                ),
            });
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 4. 顺序执行全部测量条件，严密校验排序、真值一致性与统计指标
    // ─────────────────────────────────────────────────────────────────────────
    let mut current_contig: Option<bool> = None;
    let mut current_dir_mode: Option<Mode> = Some(Mode::Full);

    for spec in &specs {
        let qmode = match spec.nav {
            "sq8" => Mode::ScalarQuantizedDirection,
            "fp32" => Mode::FullPrecisionDirection,
            _ => Mode::Full,
        };
        if current_dir_mode != Some(qmode) {
            index.prepare_direction_representation(qmode)?;
            current_dir_mode = Some(qmode);
        }
        if current_contig != Some(spec.contiguous) {
            index.set_bq_contiguous_layout(spec.contiguous);
            current_contig = Some(spec.contiguous);
        }
        set_hyperbolic_strategy(qmode);
        set_geometry_components(GeometryComponents {
            asymmetric_gate: spec.asym_gate,
            radial_bound: spec.radial_bound,
            ..GeometryComponents::default()
        });
        set_execution_config(spec.speed);

        let nav_bytes = index.navigation_storage_bytes();
        let mem_now = status();

        for &mt in &spec.threads {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(mt).build()?;
            for &ef in &spec.efs {
                for &budget in &spec.budgets {
                    if budget != 0 && (budget < 10 || budget > ef) {
                        continue;
                    }
                    if spec.nav == "exact" && budget != 0 {
                        continue;
                    }
                    // 避免与正向消融完全重复测量 (P_V3Default, B=0, ef in base_efs)
                    if spec.name == "P_V3Default" && budget == 0 && base_efs.contains(&ef) {
                        continue;
                    }

                    let cfg = SearchConfig {
                        top_k: 10,
                        ef_search: ef,
                        rerank_limit: (budget != 0).then_some(budget),
                    };
                    let search = |i: usize| -> Result<Vec<(u64, f32)>, quiver_ann::Error> {
                        let q = &queries[i * dim..(i + 1) * dim];
                        if spec.nav == "exact" {
                            index.search_poincare_exact_navigation(q, &cfg)
                        } else {
                            index.search_poincare_quantized_navigation(q, &cfg)
                        }
                    };

                    // 预热当前工作点（最多 100 条查询）
                    pool.install(|| {
                        (0..nq.min(100))
                            .into_par_iter()
                            .try_for_each(|i| search(i).map(|_| ()))
                    })?;

                    let mut all_latencies: Vec<u64> = Vec::with_capacity(nq * spec.rounds);
                    let mut round_qps: Vec<f64> = Vec::with_capacity(spec.rounds);
                    let mut total_hits = 0usize;
                    let cpu_before = cpu_stat();

                    for round in 0..spec.rounds {
                        let mut order: Vec<usize> = (0..nq).collect();
                        let mut rng = seed
                            .wrapping_add(round as u64 * 7919)
                            .wrapping_add(ef as u64);
                        for i in (1..nq).rev() {
                            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                            order.swap(i, (rng as usize) % (i + 1));
                        }
                        let wall = Instant::now();
                        let rows: Vec<(usize, u64)> = pool.install(|| {
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
                                            "返回结果数量不足10或分数未按降序排列",
                                        ));
                                    }
                                    let hits = result
                                        .iter()
                                        .filter(|h| truth[i * 10..i * 10 + 10].contains(&h.0))
                                        .count();
                                    Ok((hits, nanos))
                                })
                                .collect::<Result<Vec<_>, _>>()
                        })?;
                        let wall_secs = wall.elapsed().as_secs_f64();
                        round_qps.push(nq as f64 / wall_secs);
                        for (h, ns) in rows {
                            total_hits += h;
                            all_latencies.push(ns);
                        }
                    }

                    let cpu_after = cpu_stat();
                    let cpu_delta: std::collections::BTreeMap<_, _> = cpu_after
                        .iter()
                        .filter_map(|(k, &after)| {
                            cpu_before
                                .get(k)
                                .map(|&before| (k.clone(), after.saturating_sub(before)))
                        })
                        .collect();

                    all_latencies.sort_unstable();
                    let quantile_ms = |p: f64| -> f64 {
                        let idx = ((all_latencies.len() as f64 * p).ceil() as usize)
                            .saturating_sub(1)
                            .min(all_latencies.len() - 1);
                        all_latencies[idx] as f64 / 1e6
                    };
                    let mean_qps = round_qps.iter().sum::<f64>() / (round_qps.len() as f64);
                    let mut sorted_qps = round_qps.clone();
                    sorted_qps.sort_by(|a, b| a.total_cmp(b));
                    let median_qps = sorted_qps[sorted_qps.len() / 2];
                    let min_qps = *sorted_qps.first().unwrap();
                    let max_qps = *sorted_qps.last().unwrap();
                    let recall10 = total_hits as f64 / ((nq * 10 * spec.rounds) as f64);

                    // 计时窗口之外执行无偏候选损失与评估次数诊断
                    let mut diag_summary = serde_json::Value::Null;
                    if mt == 1 && spec.run_diag && diag_n > 0 && matches!(ef, 64 | 256) {
                        let dn = diag_n.min(nq);
                        let mut sum_v = 0usize;
                        let mut sum_c = 0usize;
                        let mut sum_f = 0usize;
                        let mut sum_exp = 0usize;
                        let mut sum_exp_calls = 0usize;
                        let mut sum_approx = 0usize;
                        let mut sum_exact = 0usize;
                        let mut sum_rej = 0usize;
                        let mut sum_hv = 0usize;
                        let mut sum_hc = 0usize;
                        let mut sum_hf = 0usize;
                        let mut sum_hr = 0usize;
                        let nav_enum = if spec.nav == "exact" {
                            PoincareNavigation::Exact
                        } else {
                            PoincareNavigation::GeometryBq2
                        };
                        let mut actual_cnt = 0usize;
                        for i in (0..nq).step_by((nq / dn).max(1)).take(dn) {
                            let stats = index.search_loss_decomposition(
                                &queries[i * dim..(i + 1) * dim],
                                &cfg,
                                nav_enum,
                                &truth[i * 10..i * 10 + 10],
                            )?;
                            let hits = search(i)?
                                .iter()
                                .filter(|h| truth[i * 10..i * 10 + 10].contains(&h.0))
                                .count();
                            if hits != stats.hits_returned
                                || stats.hits_selected_for_rerank != stats.hits_returned
                            {
                                return Err(format!(
                                    "诊断与实际搜索不一致: 条件={}, ef={ef}, B={budget}, 查询={i}, hits={hits}, diag_r={}, diag_f={}",
                                    spec.name, stats.hits_returned, stats.hits_selected_for_rerank
                                )
                                .into());
                            }
                            sum_v += stats.visited_nodes;
                            sum_c += stats.beam_candidates;
                            sum_f += stats.selected_for_rerank;
                            sum_exp += stats.expanded_nodes;
                            sum_exp_calls += stats.expansion_calls;
                            sum_approx += stats.approximate_evals;
                            sum_exact += stats.exact_vector_evals;
                            sum_rej += stats.bound_rejections;
                            sum_hv += stats.hits_visited;
                            sum_hc += stats.hits_in_beam;
                            sum_hf += stats.hits_selected_for_rerank;
                            sum_hr += stats.hits_returned;
                            actual_cnt += 1;
                        }
                        let inv = 1.0 / (actual_cnt as f64);
                        diag_summary = json!({
                            "诊断查询数": actual_cnt,
                            "均值V节点": sum_v as f64 * inv,
                            "均值C节点": sum_c as f64 * inv,
                            "均值F节点": sum_f as f64 * inv,
                            "均值展开节点": sum_exp as f64 * inv,
                            "均值展开调用": sum_exp_calls as f64 * inv,
                            "均值近似评估": sum_approx as f64 * inv,
                            "均值精确评估": sum_exact as f64 * inv,
                            "均值下界拒绝": sum_rej as f64 * inv,
                            "触达召回V": sum_hv as f64 / ((actual_cnt * 10) as f64),
                            "束保留召回C": sum_hc as f64 / ((actual_cnt * 10) as f64),
                            "预算选择召回F": sum_hf as f64 / ((actual_cnt * 10) as f64),
                            "最终召回R": sum_hr as f64 / ((actual_cnt * 10) as f64),
                        });
                        let drec = json!({
                            "编译标签": tag,
                            "构图种子": seed,
                            "拓扑CRC32": topology_crc,
                            "分组": spec.group,
                            "条件": spec.name,
                            "导航": spec.nav,
                            "ef": ef,
                            "B": budget,
                            "诊断汇总": diag_summary,
                        });
                        writeln!(diagnostics_out, "{drec}")?;
                        diagnostics_out.flush()?;
                    }

                    let rec = json!({
                        "类型": "查询",
                        "编译标签": tag,
                        "套件": suite,
                        "分组": spec.group,
                        "条件": spec.name,
                        "导航": spec.nav,
                        "O0_零追踪": spec.speed.zero_trace_overhead,
                        "O1_两级锥": spec.speed.two_level_cones,
                        "O2a_连续64B": spec.contiguous,
                        "O2b_滚动预取": spec.speed.rolling_prefetch,
                        "O3_快速位核": spec.speed.fast_kernels,
                        "位核变体": if spec.speed.fast_kernels { spec.speed.kernel_variant } else { 0 },
                        "遍历档位": format!("{:?}", spec.speed.traversal),
                        "非对称门神": spec.asym_gate,
                        "径向下界": spec.radial_bound,
                        "ef": ef,
                        "B": budget,
                        "线程": mt,
                        "重复轮数": spec.rounds,
                        "构图种子": seed,
                        "拓扑CRC32": topology_crc,
                        "独立查询数": nq,
                        "总请求数": nq * spec.rounds,
                        "recall10": recall10,
                        "QPS均值": mean_qps,
                        "QPS中位": median_qps,
                        "QPS最小": min_qps,
                        "QPS最大": max_qps,
                        "单查微秒均值": 1e6 / mean_qps,
                        "P50毫秒": quantile_ms(0.50),
                        "P95毫秒": quantile_ms(0.95),
                        "P99毫秒": quantile_ms(0.99),
                        "各轮QPS": round_qps,
                        "导航有效载荷字节": nav_bytes,
                        "进程内存": mem_now,
                        "cgroup计数增量": cpu_delta,
                        "诊断": diag_summary,
                    });
                    println!("{rec}");
                    writeln!(report, "{rec}")?;
                    report.flush()?;
                }
            }
        }
    }

    println!("全部速度消融条件执行完毕，最终内存状态: {}", status());
    Ok(())
}

#[cfg(not(feature = "research"))]
fn main() {
    eprintln!("速度消融基准需要启用 --features research");
    std::process::exit(1);
}
