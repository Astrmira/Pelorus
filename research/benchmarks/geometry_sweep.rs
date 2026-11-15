//! 固定源码、保存逐请求结果并独立控制内核的双曲 ANN 复测入口。
#[cfg(feature = "research")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use quiver_ann::research::{
        ExecutionConfig, GeometryComponents, HyperbolicStrategy as Mode, RecallConfig,
        SketchFilter, StorageConfig, TraversalProfile, execution_config, set_asym_sm_lut_override,
        set_execution_config, set_geometry_components, set_hyperbolic_strategy, set_recall_config,
        set_storage_config,
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

    fn mode(name: &str) -> Result<Mode, Box<dyn std::error::Error>> {
        Ok(match name {
            "a8" => Mode::Full,
            "a7" => Mode::FixedShellCompass,
            "sq8" => Mode::ScalarQuantizedDirection,
            "fp32" => Mode::FullPrecisionDirection,
            "vamana" => Mode::GeodesicPruning,
            _ => return Err(format!("未知模式 {name}").into()),
        })
    }
    fn memory() -> String {
        std::fs::read_to_string("/proc/self/status")
            .unwrap_or_default()
            .lines()
            .filter(|line| line.starts_with("VmRSS:") || line.starts_with("VmHWM:"))
            .collect::<Vec<_>>()
            .join(" ")
    }
    fn cpu_stat() -> BTreeMap<String, u64> {
        std::fs::read_to_string("/sys/fs/cgroup/cpu.stat")
            .or_else(|_| std::fs::read_to_string("/sys/fs/cgroup/cpu/cpu.stat"))
            .unwrap_or_default()
            .lines()
            .filter_map(|line| {
                let mut parts = line.split_whitespace();
                Some((parts.next()?.to_owned(), parts.next()?.parse().ok()?))
            })
            .collect()
    }
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
    fn rescue(value: &Value) -> RecallConfig {
        RecallConfig {
            tangent_residual_sm: flag(value, "tangent_residual_sm", false),
            adaptive_rerank: flag(value, "adaptive_rerank", false),
            variance_stabilized_sketch: flag(value, "variance_stabilized_sketch", false),
            plateau_rel_gap: value["plateau_rel_gap"].as_f64().unwrap_or(0.22) as f32,
            adaptive_rerank_cap: number(value, "adaptive_rerank_cap", 768)
                .clamp(16, u16::MAX as usize) as u16,
        }
    }
    fn speed(value: &Value) -> ExecutionConfig {
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
    fn hot(value: &Value) -> Result<StorageConfig, Box<dyn std::error::Error>> {
        let margin = value["margin"].as_f64().unwrap_or(0.12) as f32;
        let early_gate = match value["gate"].as_str().unwrap_or("off") {
            "off" => SketchFilter::Off,
            "radial" => SketchFilter::LosslessRadial,
            "sym256" => SketchFilter::SymSketch256 { margin },
            "asym256" => SketchFilter::AsymSketch256 { margin },
            _ => return Err("未知草图过滤模式".into()),
        };
        Ok(StorageConfig {
            colocated_store: flag(value, "colocated", false),
            early_gate,
            header_prefetch_window: number(value, "pf_hdr", 8).clamp(1, 32) as u8,
            eager_prefetch_overlap: flag(value, "eager_overlap", true),
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
    let test_nq: usize = arg("--test-queries", "0").parse()?;
    let tuning = arg("--query-split", "test") == "tuning";
    let bt: usize = arg("--build-threads", "12").parse()?;
    let seed: u64 = arg("--seed", "12345").parse()?;
    let build_mode_name = arg("--mode", "a8");
    let build_mode = mode(&build_mode_name)?;
    let data = PathBuf::from(arg("--data", "."));
    let output = PathBuf::from(arg("--output", "geometry_sweep.jsonl"));
    let plan: Value = serde_json::from_slice(&std::fs::read(arg("--plan", "plan.json"))?)?;
    let mut conditions = plan["conditions"]
        .as_array()
        .ok_or("实验计划缺少条件列表")?
        .clone();
    if n == 0 || dim == 0 || nq == 0 || bt == 0 || conditions.is_empty() {
        return Err("数据规模、维度、查询数、线程数和实验条件均须有效".into());
    }
    if flag(&plan, "zen4_protocol", false) && bt != 32 {
        return Err("本轮构图统一使用32线程".into());
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
            return Err("向量文件大小错误".into());
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
    let vectors = read_floats("documents.f32", n)?;
    let queries = read_floats(
        if tuning {
            "tuning_queries.f32"
        } else {
            "queries.f32"
        },
        nq,
    )?;
    let truth_bytes = std::fs::read(data.join(if tuning {
        "tuning_truth.u64"
    } else {
        "truth.u64"
    }))?;
    if truth_bytes.len() != nq * 80 {
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
    let test_queries = if test_nq > 0 {
        read_floats("queries.f32", test_nq)?
    } else {
        Vec::new()
    };
    let test_truth_bytes = if test_nq > 0 {
        std::fs::read(data.join("truth.u64"))?
    } else {
        Vec::new()
    };
    if test_truth_bytes.len() != test_nq * 80 {
        return Err("测试真值文件大小错误".into());
    }
    let test_truth: Vec<u64> = test_truth_bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| u64::from_le_bytes(*b))
        .collect();
    if test_truth.iter().any(|&id| id >= n as u64) {
        return Err("测试真值标识越界".into());
    }
    let document_crc = crc32fast::hash(bytemuck::cast_slice(&vectors));
    let query_crc = crc32fast::hash(bytemuck::cast_slice(&queries));
    let truth_crc = crc32fast::hash(&truth_bytes);
    let mut report = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)?,
    );
    let mut requests = std::io::BufWriter::new(
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
        requests,
        "条件,轮次,请求序号,查询序号,命中数,延迟纳秒,返回标识,返回分数位模式"
    )?;
    set_hyperbolic_strategy(build_mode);
    let build_components = GeometryComponents {
        adaptive_shells: flag(&plan["build"], "adaptive_shells", true),
        connectivity_repair: flag(&plan["build"], "connectivity_repair", true),
        compass: flag(&plan["build"], "compass", true),
        ..GeometryComponents::default()
    };
    set_geometry_components(build_components);
    set_storage_config(StorageConfig::default());
    set_execution_config(speed(&plan["build"]));
    set_asym_sm_lut_override(Some(flag(&plan["build"], "asym_lut", false)));
    set_recall_config(rescue(&plan["build"]["rescue"]));
    let ids: Vec<u64> = (0..n as u64).collect();
    let start = Instant::now();
    let mut index = Index::build_poincare(
        dim,
        &ids,
        &vectors,
        &BuildConfig {
            m: number(&plan["build"], "m", 16),
            ef_construction: number(&plan["build"], "ef_construction", 128),
            alpha: plan["build"]["alpha"].as_f64().unwrap_or(1.2) as f32,
        },
        1.0,
        if build_mode_name == "vamana" {
            PoincareNavigation::Exact
        } else {
            PoincareNavigation::GeometryBq2
        },
    )?;
    // 独立方向表示先完成缓存准备，将预编码及数据库范数计算计入就绪时间。
    index.prepare_direction_representation(build_mode)?;
    let build_seconds = start.elapsed().as_secs_f64();
    drop(ids);
    drop(vectors);
    let build_memory = memory();
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
                return Err(format!("独立真值核验失败，查询 {i}"));
            }
            Ok(())
        })?;
    let graph = index.graph_diagnostics();
    if graph.invalid_edges + graph.self_loops + graph.duplicate_edges > 0 {
        return Err("图包含非法边、自环或重复边".into());
    }
    let topology_crc = graph.topology_crc32;
    let build_record = json!({"类型":"构建", "模式":build_mode_name, "构图种子":seed,
        "数据规模":n,"维度":dim,"独立查询数":nq,"构图线程":bt,"可查询就绪秒":build_seconds,
        "构建后内存":build_memory,"导航有效载荷字节":index.navigation_storage_bytes(),
        "拓扑CRC32":topology_crc,"文档CRC32":document_crc,"查询CRC32":query_crc,"真值CRC32":truth_crc,
        "独立真值核验数":oracle_indices.len(),"入口可达":graph.reachable_nodes,"零入度":graph.zero_in_degree,
        "边数":graph.edges,"构建配置":plan["build"],"实验条件数":conditions.len()});
    println!("{build_record}");
    writeln!(report, "{build_record}")?;
    report.flush()?;
    drop(graph);
    // 打乱条件顺序，三个构图种子分别使用独立顺序。
    shuffle(&mut conditions, seed ^ 0x73b91a);
    for phase in [
        "curve_conditions",
        "selection_conditions",
        "tail_conditions",
    ] {
        if let Some(items) = plan[phase].as_array() {
            let mut items = items.clone();
            shuffle(&mut items, seed ^ 0x10c7);
            conditions.extend(items);
        }
    }
    let targets = [0.95f64, 0.98, 0.99, 0.995];
    let mut best: BTreeMap<(String, usize), (f64, Value)> = BTreeMap::new();
    let mut current_mode = build_mode_name.clone();
    let mut current_contiguous = true;
    let mut current_hot = false;
    for original_condition in &conditions {
        let mut resolved = original_condition.clone();
        if let Some(target) = original_condition["auto_target"].as_f64() {
            let key = (target * 10000.0).round() as usize;
            let family = original_condition["auto_family"]
                .as_str()
                .unwrap_or("main")
                .to_owned();
            let Some(selected) = best.get(&(family.clone(), key)) else {
                let row = json!({"类型":"未达标", "条件":original_condition["name"],
                    "构图种子":seed,"模式":build_mode_name,"目标":target,"配置族":family,
                    "说明":"调参集上已测范围未达到该目标"});
                writeln!(report, "{row}")?;
                report.flush()?;
                continue;
            };
            resolved = selected.1.clone();
            resolved["selected_name"] = resolved["name"].clone();
            for field in [
                "name",
                "threads",
                "rounds",
                "auto_target",
                "auto_family",
                "split",
                "group",
            ] {
                resolved[field] = original_condition[field].clone();
            }
            resolved["diagnostics"] = json!(0);
        }
        let condition = &resolved;
        let query_rescue = condition.get("rescue").unwrap_or(&plan["build"]["rescue"]);
        set_recall_config(rescue(query_rescue));
        let test_split = condition["split"].as_str() == Some("test") && test_nq > 0;
        let (queries, truth, nq) = if test_split {
            (&test_queries, &test_truth, test_nq)
        } else {
            (&queries, &truth, nq)
        };
        let name = condition["name"].as_str().ok_or("实验条件缺少名称")?;
        if name.contains([',', '\n', '\r']) {
            return Err("条件名称包含 CSV 分隔符".into());
        }
        let query_mode_name = condition["mode"].as_str().unwrap_or(&build_mode_name);
        let query_mode = mode(query_mode_name)?;
        if query_mode_name != current_mode {
            let timer = Instant::now();
            index.prepare_direction_representation(query_mode)?;
            current_mode = query_mode_name.to_owned();
            writeln!(
                report,
                "{}",
                json!({"类型":"方向表示准备", "条件":name,
                "查询模式":query_mode_name, "耗时秒":timer.elapsed().as_secs_f64(),
                "内存":memory(), "导航有效载荷字节":index.navigation_storage_bytes()})
            )?;
        }
        let contiguous = flag(condition, "contiguous", true);
        if contiguous != current_contiguous {
            index.set_bq_contiguous_layout(contiguous);
            current_contiguous = contiguous;
        }
        let hot_config = hot(condition)?;
        if hot_config.colocated_store != current_hot {
            let timer = Instant::now();
            if hot_config.colocated_store {
                index.prepare_colocated_hot_store(true)?;
            } else {
                index.restore_split_hot_store();
                index.set_bq_contiguous_layout(contiguous);
            }
            current_hot = hot_config.colocated_store;
            let row = json!({"类型":"布局准备", "条件":name, "合表":current_hot,
                "耗时秒":timer.elapsed().as_secs_f64(), "内存":memory(),
                "导航有效载荷字节":index.navigation_storage_bytes()});
            writeln!(report, "{row}")?;
        }
        set_storage_config(hot_config);
        set_hyperbolic_strategy(query_mode);
        set_geometry_components(GeometryComponents {
            asymmetric_gate: flag(condition, "asym_gate", true),
            radial_bound: flag(condition, "bound", true),
            ..build_components
        });
        let query_speed = speed(condition);
        set_execution_config(query_speed);
        if execution_config() != query_speed {
            return Err("实验开关读写不一致".into());
        }
        set_asym_sm_lut_override(Some(flag(condition, "asym_lut", false)));
        let ef = number(condition, "ef", 64);
        let budget = number(condition, "B", 0);
        let threads = number(condition, "threads", 1);
        let rounds = number(condition, "rounds", 3);
        if threads == 0 || rounds == 0 {
            return Err("线程数与轮数必须大于零".into());
        }
        if flag(&plan, "zen4_protocol", false) && ![1, 2, 4, 8, 16, 24].contains(&threads) {
            return Err("本轮查询线程仅允许 1/2/4/8/16/24".into());
        }
        let config = SearchConfig {
            top_k: 10,
            ef_search: ef,
            rerank_limit: (budget != 0).then_some(budget),
        };
        let exact = flag(condition, "exact", false);
        let search = |i: usize| {
            let query = &queries[i * dim..(i + 1) * dim];
            if exact {
                index.search_poincare_exact_navigation(query, &config)
            } else {
                index.search_poincare_quantized_navigation(query, &config)
            }
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        pool.install(|| {
            (0..nq.min(100))
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
            // 搜索调用与收集组成吞吐窗口，校验和归档在窗口结束后执行。
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
            for (request, (i, ns, result)) in rows.iter().enumerate() {
                if result.len() != 10
                    || result.windows(2).any(|p| p[0].1 < p[1].1)
                    || result.iter().any(|r| !r.1.is_finite() || r.0 >= n as u64)
                {
                    return Err(
                        format!("结果数量、排序、数值或标识错误，条件 {name}，查询 {i}").into(),
                    );
                }
                let mut ids: Vec<_> = result.iter().map(|r| r.0).collect();
                ids.sort_unstable();
                if ids.windows(2).any(|p| p[0] == p[1]) {
                    return Err("返回结果含重复标识".into());
                }
                let hits = result
                    .iter()
                    .filter(|r| truth[i * 10..i * 10 + 10].contains(&r.0))
                    .count();
                let returned_ids = result
                    .iter()
                    .map(|r| r.0.to_string())
                    .collect::<Vec<_>>()
                    .join("|");
                let returned_scores = result
                    .iter()
                    .map(|r| r.1.to_bits().to_string())
                    .collect::<Vec<_>>()
                    .join("|");
                writeln!(
                    requests,
                    "{name},{round},{request},{i},{hits},{ns},{returned_ids},{returned_scores}"
                )?;
                all_hits += hits;
                all_latencies.push(*ns);
            }
            requests.flush()?;
        }
        all_latencies.sort_unstable();
        let recall = all_hits as f64 / (nq * rounds * 10) as f64;
        let qps = round_qps.iter().sum::<f64>() / rounds as f64;
        let tuning_frontier = condition["group"].as_str() == Some("frontier")
            && original_condition["auto_target"].is_null();
        let test_frontier = matches!(
            original_condition["group"].as_str(),
            Some("test_curve" | "selected_test")
        );
        if threads == 1 && (tuning_frontier || test_frontier) {
            for target in targets {
                let key = (target * 10000.0).round() as usize;
                let family = condition["family"].as_str().unwrap_or("main");
                let family = if test_frontier {
                    format!("test:{family}")
                } else {
                    family.to_owned()
                };
                if recall >= target
                    && best
                        .get(&(family.clone(), key))
                        .is_none_or(|value| qps > value.0)
                {
                    best.insert((family, key), (qps, condition.clone()));
                }
            }
        }
        let quantile = |p: f64| {
            all_latencies[((all_latencies.len() as f64 * p).ceil() as usize).saturating_sub(1)]
                as f64
                / 1e6
        };
        let record = json!({"类型":"查询","条件":name,"配置":condition,"模式":build_mode_name,"查询模式":query_mode_name,
            "构图种子":seed,"拓扑CRC32":topology_crc,"独立查询数":nq,"总请求数":nq*rounds,
            "查询划分":if test_split {"独立测试集"} else if tuning {"调参集"} else {"历史复用查询集"},
            "recall10":recall,
            "QPS均值":qps,"各轮QPS":round_qps,
            "P50毫秒":quantile(0.5),"P95毫秒":quantile(0.95),"P99毫秒":quantile(0.99),
            "调用延迟均值毫秒":all_latencies.iter().map(|&v|v as f64).sum::<f64>()/all_latencies.len() as f64/1e6,
            "结果CRC32":result_crc.finalize(),"内存":condition_memory,"导航有效载荷字节":index.navigation_storage_bytes(),
            "cgroup计数增量":cpu_delta,"延迟口径":"搜索调用耗时，排队时间另计","吞吐口径":"搜索调用及并行收集，校验与归档在窗口外"});
        println!("{record}");
        writeln!(report, "{record}")?;
        report.flush()?;
        let diagnostic_n = number(condition, "diagnostics", 0).min(nq);
        for i in (0..nq)
            .step_by((nq / diagnostic_n.max(1)).max(1))
            .take(diagnostic_n)
        {
            let stats = index.search_loss_decomposition(
                &queries[i * dim..(i + 1) * dim],
                &config,
                if exact {
                    PoincareNavigation::Exact
                } else {
                    PoincareNavigation::GeometryBq2
                },
                &truth[i * 10..i * 10 + 10],
            )?;
            let actual = search(i)?
                .iter()
                .filter(|r| truth[i * 10..i * 10 + 10].contains(&r.0))
                .count();
            if actual != stats.hits_returned {
                return Err("诊断路径与公共查询结果不一致".into());
            }
            if stats.hits_visited < stats.hits_in_beam
                || stats.hits_in_beam < stats.hits_selected_for_rerank
                || stats.hits_selected_for_rerank != stats.hits_returned
            {
                return Err("候选损失分解或精确精排与真值不一致".into());
            }
            let row = json!({"条件":name,"查询序号":i,"V节点":stats.visited_nodes,"C节点":stats.beam_candidates,
                "F节点":stats.selected_for_rerank,
                "展开节点":stats.expanded_nodes,"展开次数":stats.expansion_calls,"近似评分次数":stats.approximate_evals,
                "精确向量读取":stats.exact_vector_evals,"下界拒绝":stats.bound_rejections,"R命中":stats.hits_returned,
                "V命中":stats.hits_visited,"C命中":stats.hits_in_beam,"F命中":stats.hits_selected_for_rerank,
                "头行检查":stats.header_checks,"头行拒绝":stats.header_rejections});
            writeln!(diagnostics, "{row}")?;
            let stages = index.search_stage_profile(
                &queries[i * dim..(i + 1) * dim],
                &config,
                if exact {
                    PoincareNavigation::Exact
                } else {
                    PoincareNavigation::GeometryBq2
                },
            )?;
            writeln!(
                diagnostics,
                "{}",
                json!({"条件":name,"查询序号":i,
                "路由纳秒":stages.router_seed_ns,"导航纳秒":stages.beam_nav_ns,
                "候选门控纳秒":stages.asym_gate_ns,"精排纳秒":stages.exact_rerank_ns})
            )?;
        }
        diagnostics.flush()?;
    }
    println!("全部实验条件完成，进程内存 {}", memory());
    Ok(())
}

#[cfg(not(feature = "research"))]
fn main() {
    eprintln!("复测入口需要启用 research 特性");
    std::process::exit(1);
}
