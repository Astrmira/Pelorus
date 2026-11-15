//! 双曲 QuIVer（A8 + 单页合表 + 对称草图早筛）合成分布召回坍塌诊断与救援本地基准。
//!
//! 严格复刻 `zen4_full_matrix_v1/prepare_data.py` 与 `campaign.py` 的分布公式、构图配置和扫描网格，
//! 隔离测试方案 A（平台感知自适应精排）、方案 B（球面切锥正交残差 2-bit SM）与方案 C（方差稳定草图与各向同性保护），
//! 并在 200K 全消融矩阵与 1M（百万级）实测尺度上输出完整 V->C->F->R 漏斗、Recall@10、吞吐与内存报告。

use quiver_ann::research::{
    ExecutionConfig, GeometryComponents, HyperbolicStrategy, RecallConfig, SketchFilter,
    StorageConfig, set_execution_config, set_geometry_components, set_hyperbolic_strategy,
    set_recall_config, set_storage_config,
};
use quiver_ann::{BuildConfig, Index, PoincareNavigation, SearchConfig};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::fs;
use std::path::Path;
use std::time::Instant;

/// 确定性 SplitMix64 + Box-Muller 高斯随机数生成器（零外部依赖、跨平台完全可复现）。
#[derive(Clone)]
struct FastRng {
    state: u64,
}

impl FastRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
    }

    #[inline]
    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % (bound as u64)) as usize
    }

    #[inline]
    fn fill_unit_sphere(&mut self, dst: &mut [f32]) {
        let mut sq = 0.0f64;
        let mut i = 0;
        while i + 1 < dst.len() {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f64::consts::TAU * u2;
            let g0 = r * theta.cos();
            let g1 = r * theta.sin();
            dst[i] = g0 as f32;
            dst[i + 1] = g1 as f32;
            sq += g0 * g0 + g1 * g1;
            i += 2;
        }
        if i < dst.len() {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            let g0 = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            dst[i] = g0 as f32;
            sq += g0 * g0;
        }
        let inv = (1.0 / sq.sqrt().max(1e-12)) as f32;
        for v in dst.iter_mut() {
            *v *= inv;
        }
    }
}

#[derive(Clone, Copy)]
enum DistKind {
    UniformNarrow,
    UniformWide,
    ClusteredNarrow,
    ClusteredWide,
    HierarchicalSemantic,
}

impl DistKind {
    fn name(self) -> &'static str {
        match self {
            Self::UniformNarrow => "uniform_narrow",
            Self::UniformWide => "uniform_wide",
            Self::ClusteredNarrow => "clustered_narrow",
            Self::ClusteredWide => "clustered_wide",
            Self::HierarchicalSemantic => "hierarchical_semantic",
        }
    }

    fn cn_label(self) -> &'static str {
        match self {
            Self::UniformNarrow => "均匀方向+窄半径 [3.9,4.1]",
            Self::UniformWide => "均匀方向+宽半径 [0.4,6.0]",
            Self::ClusteredNarrow => "簇状方向+窄半径 [3.9,4.1]",
            Self::ClusteredWide => "簇状方向+宽半径 [0.4,6.0]",
            Self::HierarchicalSemantic => "多尺度层次语义流形(类Cohere/MERU对照)",
        }
    }
}

struct SyntheticDataset {
    docs: Vec<f32>,
    queries: Vec<f32>,
    truth: Vec<[u64; 10]>,
    n: usize,
    nq: usize,
    dim: usize,
}

/// 严格按 `prepare_data.py` 公式并行合成双曲数据集并计算 FP64 精确 Top-10 真值。
fn generate_dataset(
    kind: DistKind,
    n: usize,
    nq: usize,
    dim: usize,
    seed: u64,
) -> SyntheticDataset {
    let t0 = Instant::now();
    let total = n + nq;

    // 预生成簇中心（256 个单位方向；层次流形额外生成 32 个一级超类中心）
    let mut center_rng = FastRng::new(seed);
    let mut centres = vec![0.0f32; 256 * dim];
    for c in 0..256 {
        center_rng.fill_unit_sphere(&mut centres[c * dim..(c + 1) * dim]);
    }
    let mut super_centres = vec![0.0f32; 32 * dim];
    if matches!(kind, DistKind::HierarchicalSemantic) {
        for s in 0..32 {
            center_rng.fill_unit_sphere(&mut super_centres[s * dim..(s + 1) * dim]);
        }
        for c in 0..256 {
            let s = c / 8;
            let mut sq = 0.0f64;
            for j in 0..dim {
                let mixed = super_centres[s * dim + j] + 0.42 * centres[c * dim + j];
                centres[c * dim + j] = mixed;
                sq += f64::from(mixed) * f64::from(mixed);
            }
            let inv = (1.0 / sq.sqrt().max(1e-12)) as f32;
            for j in 0..dim {
                centres[c * dim + j] *= inv;
            }
        }
    }

    let mut all_vecs = vec![0.0f32; total * dim];
    let chunk_rows = 4096usize;
    all_vecs
        .par_chunks_mut(chunk_rows * dim)
        .enumerate()
        .for_each(|(chunk_id, chunk_slice)| {
            let rows = chunk_slice.len() / dim;
            let base_row = chunk_id * chunk_rows;
            let mut dir_rng = FastRng::new(seed.wrapping_add(100_003 * (chunk_id as u64 + 1)));
            let mut rad_rng =
                FastRng::new((seed + 1).wrapping_add(100_003 * (chunk_id as u64 + 1)));
            let mut unit_buf = vec![0.0f32; dim];

            for r_i in 0..rows {
                let _global_row = base_row + r_i;
                let dst = &mut chunk_slice[r_i * dim..(r_i + 1) * dim];
                dir_rng.fill_unit_sphere(&mut unit_buf);
                let u_rad = rad_rng.next_f64();

                match kind {
                    DistKind::UniformNarrow => {
                        let depth = 3.9 + 0.2 * u_rad;
                        let radius = (depth * 0.5).tanh().min(1.0 - 1e-6) as f32;
                        for j in 0..dim {
                            dst[j] = unit_buf[j] * radius;
                        }
                    }
                    DistKind::UniformWide => {
                        let depth = 0.4 + 5.6 * u_rad;
                        let radius = (depth * 0.5).tanh().min(1.0 - 1e-6) as f32;
                        for j in 0..dim {
                            dst[j] = unit_buf[j] * radius;
                        }
                    }
                    DistKind::ClusteredNarrow => {
                        let cid = dir_rng.next_usize(256);
                        let c_slice = &centres[cid * dim..(cid + 1) * dim];
                        let mut sq = 0.0f64;
                        for j in 0..dim {
                            let v = c_slice[j] + 0.16 * unit_buf[j];
                            dst[j] = v;
                            sq += f64::from(v) * f64::from(v);
                        }
                        let depth = 3.9 + 0.2 * u_rad;
                        let radius = (depth * 0.5).tanh().min(1.0 - 1e-6);
                        let scale = (radius / sq.sqrt().max(1e-12)) as f32;
                        for value in dst.iter_mut() {
                            *value *= scale;
                        }
                    }
                    DistKind::ClusteredWide => {
                        let cid = dir_rng.next_usize(256);
                        let c_slice = &centres[cid * dim..(cid + 1) * dim];
                        let mut sq = 0.0f64;
                        for j in 0..dim {
                            let v = c_slice[j] + 0.16 * unit_buf[j];
                            dst[j] = v;
                            sq += f64::from(v) * f64::from(v);
                        }
                        let depth = 0.4 + 5.6 * u_rad;
                        let radius = (depth * 0.5).tanh().min(1.0 - 1e-6);
                        let scale = (radius / sq.sqrt().max(1e-12)) as f32;
                        for value in dst.iter_mut() {
                            *value *= scale;
                        }
                    }
                    DistKind::HierarchicalSemantic => {
                        // 真实双曲树嵌入特征：深度中位数 ~2.1，越靠近边界语义越细但仍具中等局部散度 (noise ~ 0.38..0.65)
                        let depth = 0.55 + 3.20 * u_rad.powf(0.85);
                        let cid = dir_rng.next_usize(256);
                        let c_slice = &centres[cid * dim..(cid + 1) * dim];
                        let noise = (0.68 - 0.08 * depth) as f32;
                        let mut sq = 0.0f64;
                        for j in 0..dim {
                            let v = c_slice[j] + noise * unit_buf[j];
                            dst[j] = v;
                            sq += f64::from(v) * f64::from(v);
                        }
                        let radius = (depth * 0.5).tanh().min(1.0 - 1e-6);
                        let scale = (radius / sq.sqrt().max(1e-12)) as f32;
                        for value in dst.iter_mut() {
                            *value *= scale;
                        }
                    }
                }
            }
        });

    let queries = all_vecs.split_off(n * dim);
    let docs = all_vecs;

    // 预计算底库 FP64 (1 - ||x||^2) 边界分母，加速精确真值扫描
    let doc_denom: Vec<f64> = docs
        .par_chunks_exact(dim)
        .map(|v| {
            let mut n2 = 0.0f64;
            for &x in v {
                let xf = f64::from(x);
                n2 += xf * xf;
            }
            (1.0 - n2).max(1e-15)
        })
        .collect();

    // 并行计算每条查询的 FP64 坐标差精确 Poincaré Top-10 真值（行号破同分，与 prepare_data.py 一致）
    let truth: Vec<[u64; 10]> = queries
        .par_chunks_exact(dim)
        .map(|q| {
            let mut qn2 = 0.0f64;
            for &x in q {
                let xf = f64::from(x);
                qn2 += xf * xf;
            }
            let q_denom = (1.0 - qn2).max(1e-15);
            let mut best = [(f64::INFINITY, u64::MAX); 10];
            for (doc_id, d_vec) in docs.chunks_exact(dim).enumerate() {
                let mut diff2 = 0.0f64;
                for j in 0..dim {
                    let df = f64::from(q[j]) - f64::from(d_vec[j]);
                    diff2 += df * df;
                }
                let proxy = diff2 / (q_denom * doc_denom[doc_id]);
                let cand = (proxy, doc_id as u64);
                if cand < best[9] {
                    best[9] = cand;
                    let mut k = 9;
                    while k > 0 && best[k] < best[k - 1] {
                        best.swap(k, k - 1);
                        k -= 1;
                    }
                }
            }
            best.map(|(_, id)| id)
        })
        .collect();

    println!(
        "[数据就绪] {} ({}) | N={}, Q={}, D={} | 耗时 {:.2}s",
        kind.name(),
        kind.cn_label(),
        n,
        nq,
        dim,
        t0.elapsed().as_secs_f64()
    );
    SyntheticDataset {
        docs,
        queries,
        truth,
        n,
        nq,
        dim,
    }
}

#[derive(Clone, Copy)]
struct ModeSpec {
    id: &'static str,
    cn_name: &'static str,
    rescue: RecallConfig,
    navigation: PoincareNavigation,
}

fn evaluate_mode(
    ds: &SyntheticDataset,
    idx: &Index,
    spec: ModeSpec,
    build_sec: f64,
    ef_list: &[usize],
    diag_queries: usize,
) -> Vec<Value> {
    set_hyperbolic_strategy(HyperbolicStrategy::Full);
    set_geometry_components(GeometryComponents::default());
    set_execution_config(ExecutionConfig::default());
    set_storage_config(StorageConfig {
        colocated_store: true,
        early_gate: SketchFilter::SymSketch256 { margin: 0.12 },
        header_prefetch_window: 8,
        eager_prefetch_overlap: true,
    });
    set_recall_config(spec.rescue);

    let hot_bytes = idx.navigation_storage_bytes();
    let num_axes = idx.residual_cone_count();
    let mut rows = Vec::new();

    for &ef in ef_list {
        let budget = ef.min(128);
        let scfg = SearchConfig {
            ef_search: ef,
            top_k: 10,
            rerank_limit: if spec.navigation == PoincareNavigation::Exact {
                None
            } else {
                Some(budget)
            },
        };

        let run_search = |q: &[f32]| {
            if spec.navigation == PoincareNavigation::Exact {
                idx.search_poincare_exact_navigation(q, &scfg).unwrap()
            } else {
                idx.search(q, &scfg).unwrap()
            }
        };

        // 预热 16 条查询
        for q_i in 0..16.min(ds.nq) {
            let q = &ds.queries[q_i * ds.dim..(q_i + 1) * ds.dim];
            let _ = run_search(q);
        }

        // 单线程全量测试集计时与 Recall@10 统计
        let t_1t = Instant::now();
        let mut total_hits = 0usize;
        for q_i in 0..ds.nq {
            let q = &ds.queries[q_i * ds.dim..(q_i + 1) * ds.dim];
            let res = run_search(q);
            let t_row = &ds.truth[q_i];
            for &(id, _) in &res {
                if t_row.contains(&id) {
                    total_hits += 1;
                }
            }
        }
        let sec_1t = t_1t.elapsed().as_secs_f64();
        let recall = total_hits as f64 / (ds.nq * 10) as f64;
        let qps_1t = ds.nq as f64 / sec_1t;

        // 多线程吞吐（仅在关键档位 ef=128, 256, 512 测量以节省总时长）
        let qps_mt = if matches!(ef, 128 | 256 | 512) {
            let t_mt = Instant::now();
            ds.queries.par_chunks_exact(ds.dim).for_each(|q| {
                set_hyperbolic_strategy(HyperbolicStrategy::Full);
                set_execution_config(ExecutionConfig::default());
                set_storage_config(StorageConfig {
                    colocated_store: true,
                    early_gate: SketchFilter::SymSketch256 { margin: 0.12 },
                    header_prefetch_window: 8,
                    eager_prefetch_overlap: true,
                });
                set_recall_config(spec.rescue);
                let _ = run_search(q);
            });
            Some(ds.nq as f64 / t_mt.elapsed().as_secs_f64())
        } else {
            None
        };

        // 在 ef=256（及 ef=2048）采集 V->C->F->R 四阶段损失分解
        let (v_rate, c_rate, f_rate, r_rate, avg_selected, avg_exact_evals) =
            if matches!(ef, 256 | 2048) && diag_queries > 0 {
                let dq = diag_queries.min(ds.nq);
                let mut v_sum = 0usize;
                let mut c_sum = 0usize;
                let mut f_sum = 0usize;
                let mut r_sum = 0usize;
                let mut sel_sum = 0usize;
                let mut exact_sum = 0usize;
                for q_i in 0..dq {
                    let q = &ds.queries[q_i * ds.dim..(q_i + 1) * ds.dim];
                    let st = idx
                        .search_loss_decomposition(q, &scfg, spec.navigation, &ds.truth[q_i])
                        .unwrap();
                    v_sum += st.hits_visited;
                    c_sum += st.hits_in_beam;
                    f_sum += st.hits_selected_for_rerank;
                    r_sum += st.hits_returned;
                    sel_sum += st.selected_for_rerank;
                    exact_sum += st.exact_vector_evals;
                }
                let denom = (dq * 10) as f64;
                (
                    Some(v_sum as f64 / denom),
                    Some(c_sum as f64 / denom),
                    Some(f_sum as f64 / denom),
                    Some(r_sum as f64 / denom),
                    Some(sel_sum as f64 / dq as f64),
                    Some(exact_sum as f64 / dq as f64),
                )
            } else {
                (None, None, None, None, None, None)
            };

        if let (Some(vr), Some(cr), Some(fr), Some(rr), Some(asel), Some(aex)) = (
            v_rate,
            c_rate,
            f_rate,
            r_rate,
            avg_selected,
            avg_exact_evals,
        ) {
            println!(
                "  [{:<18}] ef={:<4} B={:<3} | Recall@10={:>6.2}% | 1T={:>7.0} QPS{} | V->C->F->R: {:>5.1}% -> {:>5.1}% -> {:>5.1}% -> {:>5.1}% (精排入选={:.1}, 精确读={:.1})",
                spec.id,
                ef,
                budget,
                recall * 100.0,
                qps_1t,
                qps_mt.map_or(String::new(), |m| format!(" | MT={m:>7.0} QPS")),
                vr * 100.0,
                cr * 100.0,
                fr * 100.0,
                rr * 100.0,
                asel,
                aex,
            );
        } else {
            println!(
                "  [{:<18}] ef={:<4} B={:<3} | Recall@10={:>6.2}% | 1T={:>7.0} QPS{}",
                spec.id,
                ef,
                budget,
                recall * 100.0,
                qps_1t,
                qps_mt.map_or(String::new(), |m| format!(" | MT={m:>7.0} QPS")),
            );
        }

        rows.push(json!({
            "scale_n": ds.n,
            "dim": ds.dim,
            "mode": spec.id,
            "mode_cn": spec.cn_name,
            "ef": ef,
            "base_budget": budget,
            "recall_at_10": recall,
            "qps_1t": qps_1t,
            "qps_mt": qps_mt,
            "build_sec": build_sec,
            "hot_bytes": hot_bytes,
            "hot_bytes_per_node": hot_bytes as f64 / ds.n as f64,
            "residual_axes": num_axes,
            "diag_v": v_rate,
            "diag_c": c_rate,
            "diag_f": f_rate,
            "diag_r": r_rate,
            "avg_selected_rerank": avg_selected,
            "avg_exact_evals": avg_exact_evals,
        }));
    }

    rows
}

fn build_index_for_rescue(ds: &SyntheticDataset, rescue: RecallConfig) -> (Index, f64) {
    // 确保所有 Rayon 工作线程同步感知当前构图消融配置
    rayon::broadcast(|_| {
        set_hyperbolic_strategy(HyperbolicStrategy::Full);
        set_geometry_components(GeometryComponents::default());
        set_execution_config(ExecutionConfig::default());
        set_storage_config(StorageConfig::default());
        set_recall_config(rescue);
    });
    set_hyperbolic_strategy(HyperbolicStrategy::Full);
    set_geometry_components(GeometryComponents::default());
    set_execution_config(ExecutionConfig::default());
    set_storage_config(StorageConfig::default());
    set_recall_config(rescue);

    let ids: Vec<u64> = (0..ds.n as u64).collect();
    let cfg = BuildConfig {
        m: 16,
        ef_construction: 128,
        alpha: 1.2,
    };
    let t0 = Instant::now();
    let mut idx = Index::build_poincare(
        ds.dim,
        &ids,
        &ds.docs,
        &cfg,
        1.0,
        PoincareNavigation::GeometryBq2,
    )
    .expect("构建双曲 QuIVer 索引失败");
    let build_sec = t0.elapsed().as_secs_f64();
    let coloc_sec = idx
        .prepare_colocated_hot_store(true)
        .expect("构建单页合表失败");
    println!(
        "  [构图完成] 耗时 {:.2}s (合表物化 {:.2}s) | 切锥残差轴数={} | 常驻热存储={:.2} MiB ({:.1} B/node)",
        build_sec,
        coloc_sec,
        idx.residual_cone_count(),
        idx.navigation_storage_bytes() as f64 / (1024.0 * 1024.0),
        idx.navigation_storage_bytes() as f64 / ds.n as f64,
    );
    (idx, build_sec)
}

fn main() {
    let out_dir = Path::new("research/artifacts/experiments/local_synthetic_rescue_v1");
    fs::create_dir_all(out_dir).expect("创建实验输出目录失败");

    println!(
        "=========================================================================================="
    );
    println!(
        "  双曲 QuIVer (A8) 合成极端分布召回救援本地验证基准 (Zen 4: {} 逻辑线程)",
        rayon::current_num_threads()
    );
    println!(
        "=========================================================================================="
    );

    let mut all_records: Vec<Value> = Vec::new();
    let ef_grid = [64usize, 128, 256, 512, 1024, 2048];

    // ─────────────────────────────────────────────────────────────────────────
    // 阶段 1：N = 200,000 × 512D 五分布全消融对照（隔离 A+C、B、Full A+B+C 及真实层次对照）
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n>>> 阶段 1：N = 200,000 × 512D 五种流形分布全消融对照矩阵");
    let pilot_distributions = [
        DistKind::ClusteredNarrow,
        DistKind::ClusteredWide,
        DistKind::UniformNarrow,
        DistKind::UniformWide,
        DistKind::HierarchicalSemantic,
    ];

    for kind in pilot_distributions {
        println!(
            "\n------------------------------------------------------------------------------------------"
        );
        let ds = generate_dataset(kind, 200_000, 200, 512, 20261114);

        // 1) 构建 Baseline A8 图（关闭所有救援开关）
        println!("-> 构建基线图 (Baseline A8: 全局 2-bit SM + 固定 B + 常规草图)");
        let (base_idx, base_build_sec) = build_index_for_rescue(&ds, RecallConfig::default());

        for spec in [
            ModeSpec {
                id: "Baseline_A8",
                cn_name: "基线 A8 (合表+对称草图)",
                rescue: RecallConfig::default(),
                navigation: PoincareNavigation::GeometryBq2,
            },
            ModeSpec {
                id: "QueryOnly_AC",
                cn_name: "仅查询期救援 A+C (复用基线图)",
                rescue: RecallConfig {
                    tangent_residual_sm: false,
                    adaptive_rerank: true,
                    plateau_rel_gap: 0.22,
                    adaptive_rerank_cap: 768,
                    variance_stabilized_sketch: true,
                },
                navigation: PoincareNavigation::GeometryBq2,
            },
        ] {
            for mut r in evaluate_mode(&ds, &base_idx, spec, base_build_sec, &ef_grid, 40) {
                r["dataset"] = json!(kind.name());
                r["phase"] = json!("component_comparison_200k");
                all_records.push(r);
            }
        }
        drop(base_idx);

        // 2) 构建救援图（开启切锥正交残差编码 B + 各向同性路由保护）
        println!("-> 构建救援图 (Rescue A8: 球面切锥正交残差 2-bit SM)");
        let (rescue_idx, rescue_build_sec) = build_index_for_rescue(&ds, RecallConfig::adaptive());

        for spec in [
            ModeSpec {
                id: "ResidualOnly_B",
                cn_name: "仅切锥正交残差 B (固定精排B)",
                rescue: RecallConfig {
                    tangent_residual_sm: true,
                    adaptive_rerank: false,
                    plateau_rel_gap: 0.22,
                    adaptive_rerank_cap: 768,
                    variance_stabilized_sketch: false,
                },
                navigation: PoincareNavigation::GeometryBq2,
            },
            ModeSpec {
                id: "Full_Rescue_ABC",
                cn_name: "完整救援 A+B+C",
                rescue: RecallConfig::adaptive(),
                navigation: PoincareNavigation::GeometryBq2,
            },
        ] {
            for mut r in evaluate_mode(&ds, &rescue_idx, spec, rescue_build_sec, &ef_grid, 40) {
                r["dataset"] = json!(kind.name());
                r["phase"] = json!("component_comparison_200k");
                all_records.push(r);
            }
        }

        // 在 uniform_narrow 上额外测试同图 Exact FP32 导航物理上限对照
        if matches!(kind, DistKind::UniformNarrow) {
            let exact_spec = ModeSpec {
                id: "Exact_FP32_Nav",
                cn_name: "同图 FP32 精确导航上限对照",
                rescue: RecallConfig::adaptive(),
                navigation: PoincareNavigation::Exact,
            };
            for mut r in evaluate_mode(
                &ds,
                &rescue_idx,
                exact_spec,
                rescue_build_sec,
                &[256, 1024, 2048],
                40,
            ) {
                r["dataset"] = json!(kind.name());
                r["phase"] = json!("component_comparison_200k");
                all_records.push(r);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 阶段 2：Zen 4 主矩阵同尺度 N = 1,000,000 × 512D 百万级硬核验证
    // ─────────────────────────────────────────────────────────────────────────
    println!(
        "\n=========================================================================================="
    );
    println!(">>> 阶段 2：N = 1,000,000 × 512D 百万级原尺寸复现与召回救援验证");
    println!(
        "=========================================================================================="
    );

    let full_distributions = [
        DistKind::ClusteredNarrow,
        DistKind::ClusteredWide,
        DistKind::UniformNarrow,
    ];

    for kind in full_distributions {
        println!(
            "\n------------------------------------------------------------------------------------------"
        );
        let ds = generate_dataset(kind, 1_000_000, 200, 512, 20261114);

        println!("-> [1M] 构建基线图 (Baseline A8)");
        let (base_idx, base_build_sec) = build_index_for_rescue(&ds, RecallConfig::default());
        let base_spec = ModeSpec {
            id: "Baseline_A8",
            cn_name: "基线 A8 (Zen4线上同款)",
            rescue: RecallConfig::default(),
            navigation: PoincareNavigation::GeometryBq2,
        };
        for mut r in evaluate_mode(&ds, &base_idx, base_spec, base_build_sec, &ef_grid, 50) {
            r["dataset"] = json!(format!("{}_1m", kind.name()));
            r["phase"] = json!("scale_1m");
            all_records.push(r);
        }
        drop(base_idx);

        println!("-> [1M] 构建完整救援图 (Full Rescue A+B+C)");
        let (rescue_idx, rescue_build_sec) = build_index_for_rescue(&ds, RecallConfig::adaptive());
        let rescue_spec = ModeSpec {
            id: "Full_Rescue_ABC",
            cn_name: "完整救援 A+B+C",
            rescue: RecallConfig::adaptive(),
            navigation: PoincareNavigation::GeometryBq2,
        };
        for mut r in evaluate_mode(
            &ds,
            &rescue_idx,
            rescue_spec,
            rescue_build_sec,
            &ef_grid,
            50,
        ) {
            r["dataset"] = json!(format!("{}_1m", kind.name()));
            r["phase"] = json!("scale_1m");
            all_records.push(r);
        }
        if matches!(kind, DistKind::UniformNarrow) {
            let exact_spec = ModeSpec {
                id: "Exact_FP32_Nav",
                cn_name: "同图 FP32 精确导航上限对照",
                rescue: RecallConfig::adaptive(),
                navigation: PoincareNavigation::Exact,
            };
            for mut r in evaluate_mode(
                &ds,
                &rescue_idx,
                exact_spec,
                rescue_build_sec,
                &[256, 2048],
                50,
            ) {
                r["dataset"] = json!(format!("{}_1m", kind.name()));
                r["phase"] = json!("scale_1m");
                all_records.push(r);
            }
        }
        drop(rescue_idx);
    }

    // 重置全局开关
    set_storage_config(StorageConfig::default());
    set_recall_config(RecallConfig::default());
    set_execution_config(ExecutionConfig::default());

    // 写出 JSON 与 CSV 汇总文件
    let json_path = out_dir.join("summary.json");
    fs::write(
        &json_path,
        serde_json::to_string_pretty(&all_records).expect("序列化 JSON 失败"),
    )
    .expect("写入 summary.json 失败");

    let mut csv = String::from(
        "phase,dataset,scale_n,dim,mode,ef,base_budget,recall_at_10,qps_1t,qps_mt,build_sec,hot_bytes,hot_bytes_per_node,residual_axes,diag_v,diag_c,diag_f,diag_r,avg_selected_rerank,avg_exact_evals\n",
    );
    for r in &all_records {
        let fmt_opt = |k: &str| -> String {
            r.get(k)
                .and_then(Value::as_f64)
                .map_or(String::new(), |v| format!("{v:.6}"))
        };
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{:.6},{:.2},{},{:.3},{},{:.2},{},{},{},{},{},{},{}\n",
            r["phase"].as_str().unwrap_or(""),
            r["dataset"].as_str().unwrap_or(""),
            r["scale_n"].as_u64().unwrap_or(0),
            r["dim"].as_u64().unwrap_or(0),
            r["mode"].as_str().unwrap_or(""),
            r["ef"].as_u64().unwrap_or(0),
            r["base_budget"].as_u64().unwrap_or(0),
            r["recall_at_10"].as_f64().unwrap_or(0.0),
            r["qps_1t"].as_f64().unwrap_or(0.0),
            fmt_opt("qps_mt"),
            r["build_sec"].as_f64().unwrap_or(0.0),
            r["hot_bytes"].as_u64().unwrap_or(0),
            r["hot_bytes_per_node"].as_f64().unwrap_or(0.0),
            r["residual_axes"].as_u64().unwrap_or(0),
            fmt_opt("diag_v"),
            fmt_opt("diag_c"),
            fmt_opt("diag_f"),
            fmt_opt("diag_r"),
            fmt_opt("avg_selected_rerank"),
            fmt_opt("avg_exact_evals"),
        ));
    }
    let csv_path = out_dir.join("summary.csv");
    fs::write(&csv_path, csv).expect("写入 summary.csv 失败");

    println!(
        "\n[全部实验完成] 结果已保存至 {} 与 {}",
        json_path.display(),
        csv_path.display()
    );
}
