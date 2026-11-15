//! 黎曼 QuIVer 构图与搜索诊断实验：量化分析双曲空间中的真值分布、剪枝坍缩与导航势垒。

use quiver_ann::{BuildConfig, Index, Metric, PoincareNavigation, SearchConfig};
use rayon::prelude::*;
use std::collections::HashSet;
use std::error::Error;
use std::time::Instant;

struct Random(u64);
impl Random {
    // 生成均匀随机数
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    // 生成与 hyperbolic_bench 完全一致的采样点
    fn points(&mut self, count: usize, dimensions: usize) -> (Vec<f32>, Vec<usize>, Vec<f64>) {
        let mut vectors = Vec::with_capacity(count * dimensions);
        let mut strata = Vec::with_capacity(count);
        let mut depths = Vec::with_capacity(count);
        for _ in 0..count {
            let start = vectors.len();
            while vectors.len() < start + dimensions {
                let magnitude = (-2.0 * self.uniform().ln()).sqrt();
                let angle = std::f64::consts::TAU * self.uniform();
                let (sin, cos) = angle.sin_cos();
                vectors.push((magnitude * cos) as f32);
                if vectors.len() < start + dimensions {
                    vectors.push((magnitude * sin) as f32);
                }
            }
            let depth = 0.05 + 5.95 * self.uniform();
            let radius = (depth / 2.0).tanh();
            let norm = vectors[start..]
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt();
            for value in &mut vectors[start..] {
                *value = (f64::from(*value) * radius / norm) as f32;
            }
            strata.push((depth / 2.0).floor().min(2.0) as usize);
            depths.push(depth);
        }
        (vectors, strata, depths)
    }

    // 生成模拟真实语义层级的聚簇双曲点（根概念在内层，叶子簇在外层锥体）
    fn clustered_points(
        &mut self,
        count: usize,
        query_count: usize,
        dimensions: usize,
        num_clusters: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
        let mut centers = Vec::with_capacity(num_clusters * dimensions);
        let mut center_depths = Vec::with_capacity(num_clusters);
        for _ in 0..num_clusters {
            let start = centers.len();
            while centers.len() < start + dimensions {
                let magnitude = (-2.0 * self.uniform().ln()).sqrt();
                let angle = std::f64::consts::TAU * self.uniform();
                let (sin, cos) = angle.sin_cos();
                centers.push(magnitude * cos);
                if centers.len() < start + dimensions {
                    centers.push(magnitude * sin);
                }
            }
            let norm = centers[start..].iter().map(|v| v * v).sum::<f64>().sqrt();
            for v in &mut centers[start..] {
                *v /= norm;
            }
            center_depths.push(1.0 + 4.0 * self.uniform());
        }

        let mut sample_set = |n: usize| -> (Vec<f32>, Vec<usize>) {
            let mut vecs = Vec::with_capacity(n * dimensions);
            let mut strata = Vec::with_capacity(n);
            for _ in 0..n {
                let cid = (self.uniform() * num_clusters as f64) as usize % num_clusters;
                let center = &centers[cid * dimensions..(cid + 1) * dimensions];
                let start = vecs.len();
                let noise_scale = 0.08 + 0.12 * self.uniform();
                let mut k = 0;
                while k < dimensions {
                    let magnitude = (-2.0 * self.uniform().ln()).sqrt();
                    let angle = std::f64::consts::TAU * self.uniform();
                    let (sin, cos) = angle.sin_cos();
                    vecs.push((center[k] + noise_scale * magnitude * cos) as f32);
                    k += 1;
                    if k < dimensions {
                        vecs.push((center[k] + noise_scale * magnitude * sin) as f32);
                        k += 1;
                    }
                }
                let depth = (center_depths[cid] + (self.uniform() - 0.5) * 1.2).clamp(0.1, 5.8);
                let radius = (depth / 2.0).tanh();
                let norm = vecs[start..]
                    .iter()
                    .map(|&v| f64::from(v).powi(2))
                    .sum::<f64>()
                    .sqrt();
                for v in &mut vecs[start..] {
                    *v = (f64::from(*v) * radius / norm) as f32;
                }
                strata.push((depth / 2.0).floor().min(2.0) as usize);
            }
            (vecs, strata)
        };

        let (docs, _) = sample_set(count);
        let (queries, q_strata) = sample_set(query_count);
        (docs, queries, q_strata)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = f64::from(x);
        let yf = f64::from(y);
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    if na <= 0.0 || nb <= 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn parse_arg(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    match args.iter().position(|v| v == name) {
        Some(i) => Ok(args.get(i + 1).ok_or("参数缺少数值")?.parse()?),
        None => Ok(default),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let count = parse_arg("--count", 20_000)?;
    let dim = parse_arg("--dim", 64)?;
    let query_count = parse_arg("--queries", 500)?;
    let threads = parse_arg("--threads", 12)?;
    let thread_matrix = parse_arg("--thread-matrix", 0)?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    println!("配置参数: 规模 N={count}, 维度 D={dim}, 查询数 Q={query_count}, 并行线程={threads}");
    let (vectors, doc_strata, doc_depths) = Random(20261112).points(count, dim);
    let (queries, query_strata, query_depths) = Random(20261113).points(query_count, dim);
    let ids = (0..count as u64).collect::<Vec<_>>();

    let t0 = Instant::now();
    let index = Index::build_with_metric(
        dim,
        &ids,
        &vectors,
        &BuildConfig {
            m: 16,
            ef_construction: 128,
            alpha: 1.2,
        },
        Metric::Poincare { curvature: 1.0 },
    )?;
    println!(
        "=== 1. 各向同性基准 (N={count}, D={dim}) 构图耗时: {:.3}秒 ===",
        t0.elapsed().as_secs_f64()
    );
    #[cfg(feature = "research")]
    println!(
        "常驻热导航存储: {:.2} MiB ({:.1} B/vec)",
        index.navigation_storage_bytes() as f64 / (1024.0 * 1024.0),
        index.navigation_storage_bytes() as f64 / count as f64
    );

    // 统计真值近邻的径向深度与余弦分布
    let truth: Vec<Vec<u64>> = queries
        .par_chunks_exact(dim)
        .map(|q| {
            index
                .search_exact(q, 10)
                .unwrap()
                .into_iter()
                .map(|r| r.0)
                .collect()
        })
        .collect();

    let mut truth_depth_sum = [0.0; 3];
    let mut truth_cos_sum = [0.0; 3];
    let mut truth_in_strata = [[0usize; 3]; 3];
    let mut q_counts = [0usize; 3];

    for (qi, t_ids) in truth.iter().enumerate() {
        let qs = query_strata[qi];
        let q_vec = &queries[qi * dim..(qi + 1) * dim];
        q_counts[qs] += 1;
        for &tid in t_ids {
            let ti = tid as usize;
            let d_vec = &vectors[ti * dim..(ti + 1) * dim];
            truth_depth_sum[qs] += doc_depths[ti];
            truth_cos_sum[qs] += cosine(q_vec, d_vec);
            truth_in_strata[qs][doc_strata[ti]] += 1;
        }
    }

    for s in 0..3 {
        let n_pairs = (q_counts[s] * 10) as f64;
        println!(
            "查询分组 {s} (查询数={}, 平均深度={:.2}): 真值Top-10平均深度={:.3}, 平均方向余弦={:.4}, 真值落入各层比例=[{:.1}%, {:.1}%, {:.1}%]",
            q_counts[s],
            query_depths
                .iter()
                .enumerate()
                .filter(|(i, _)| query_strata[*i] == s)
                .map(|(_, &d)| d)
                .sum::<f64>()
                / q_counts[s] as f64,
            truth_depth_sum[s] / n_pairs,
            truth_cos_sum[s] / n_pairs,
            truth_in_strata[s][0] as f64 * 100.0 / n_pairs,
            truth_in_strata[s][1] as f64 * 100.0 / n_pairs,
            truth_in_strata[s][2] as f64 * 100.0 / n_pairs,
        );
    }

    // 细分 6 个径向区间的出入度统计
    #[cfg(feature = "research")]
    {
        let diag = index.graph_diagnostics();
        let mut bins = [[0usize; 4]; 6]; // [节点数, 出度和, 入度和, 零入度数]
        for node in &diag.nodes {
            let i = node.id as usize;
            let b = (doc_depths[i].floor() as usize).min(5);
            bins[b][0] += 1;
            bins[b][1] += node.out_degree;
            bins[b][2] += node.in_degree;
            bins[b][3] += usize::from(node.in_degree == 0);
        }
        println!("--- 6个测地深度区间 [R, R+1) 的度数分布 ---");
        for (b, row) in bins.iter().enumerate() {
            println!(
                "深度区间 [{b}, {}): 节点数={}, 平均出度={:.2}, 平均入度={:.2}, 零入度数={}",
                b + 1,
                row[0],
                row[1] as f64 / row[0] as f64,
                row[2] as f64 / row[0] as f64,
                row[3]
            );
        }

        // 进一步细分内层 [0, 1.2) 的 6 个子区间 [0,0.2), [0.2,0.4), ..., [1.0,1.2)
        let mut inner_bins = [[0usize; 4]; 6]; // [节点数, 出度和, 入度和, 真值命中次数]
        let mut truth_freq = vec![0usize; count];
        for t_ids in &truth {
            for &tid in t_ids {
                truth_freq[tid as usize] += 1;
            }
        }
        for node in &diag.nodes {
            let i = node.id as usize;
            if doc_depths[i] < 1.2 {
                let b = ((doc_depths[i] / 0.2).floor() as usize).min(5);
                inner_bins[b][0] += 1;
                inner_bins[b][1] += node.out_degree;
                inner_bins[b][2] += node.in_degree;
                inner_bins[b][3] += truth_freq[i];
            }
        }
        println!("--- 内层 [0, 1.2) 细分深度区间的度数与真值分布 ---");
        for (b, row) in inner_bins.iter().enumerate() {
            println!(
                "内层子区间 [{:.1}, {:.1}): 节点数={:3}, 平均出度={:5.2}, 总平均入度={:6.2}, 真值总出现次数={:4}",
                b as f64 * 0.2,
                (b + 1) as f64 * 0.2,
                row[0],
                row[1] as f64 / row[0].max(1) as f64,
                row[2] as f64 / row[0].max(1) as f64,
                row[3]
            );
        }
    }

    // 测试聚簇层级双曲分布 (模拟真实语义空间)
    let (c_docs, c_queries, c_qstrata) =
        Random(20261114).clustered_points(count, query_count, dim, 200);
    let c_index_exact = Index::build_poincare(
        dim,
        &ids,
        &c_docs,
        &BuildConfig {
            m: 16,
            ef_construction: 128,
            alpha: 1.2,
        },
        1.0,
        PoincareNavigation::Exact,
    )?;
    let c_index_bq2 = Index::build_poincare(
        dim,
        &ids,
        &c_docs,
        &BuildConfig {
            m: 16,
            ef_construction: 128,
            alpha: 1.2,
        },
        1.0,
        PoincareNavigation::GeometryBq2,
    )?;
    let c_truth: Vec<HashSet<u64>> = c_queries
        .par_chunks_exact(dim)
        .map(|q| {
            c_index_exact
                .search_exact(q, 10)
                .unwrap()
                .into_iter()
                .map(|r| r.0)
                .collect()
        })
        .collect();

    println!("=== 2. 聚簇语义双曲基准 (N={count}, D={dim}, 200簇) ===");
    #[cfg(feature = "research")]
    {
        let d1 = c_index_exact.graph_diagnostics();
        let d2 = c_index_bq2.graph_diagnostics();
        println!(
            "聚簇图诊断 -> 精确构图平均出度={:.2} (零入度={}), BQ2构图平均出度={:.2} (零入度={})",
            d1.edges as f64 / count as f64,
            d1.zero_in_degree,
            d2.edges as f64 / count as f64,
            d2.zero_in_degree
        );
    }
    for ef in [32, 64, 128, 256] {
        let cfg = SearchConfig {
            top_k: 10,
            ef_search: ef,
            rerank_limit: None,
        };
        let mut r_exact = [0.0; 3];
        let mut r_bq2 = [0.0; 3];
        let mut cnt = [0usize; 3];
        for (i, q) in c_queries.chunks_exact(dim).enumerate() {
            let s = c_qstrata[i];
            cnt[s] += 1;
            let res_e = c_index_exact.search(q, &cfg)?;
            let res_b = c_index_bq2.search(q, &cfg)?;
            r_exact[s] += res_e.iter().filter(|r| c_truth[i].contains(&r.0)).count() as f64 / 10.0;
            r_bq2[s] += res_b.iter().filter(|r| c_truth[i].contains(&r.0)).count() as f64 / 10.0;
        }
        let tot_e = r_exact.iter().sum::<f64>() / query_count as f64;
        let tot_b = r_bq2.iter().sum::<f64>() / query_count as f64;
        println!(
            "ef={ef:3}: 精确导航 Recall@10={:.4} [{:.3}, {:.3}, {:.3}] | GeometryBq2 Recall@10={:.4} [{:.3}, {:.3}, {:.3}]",
            tot_e,
            r_exact[0] / cnt[0] as f64,
            r_exact[1] / cnt[1] as f64,
            r_exact[2] / cnt[2] as f64,
            tot_b,
            r_bq2[0] / cnt[0] as f64,
            r_bq2[1] / cnt[1] as f64,
            r_bq2[2] / cnt[2] as f64,
        );
    }

    #[cfg(feature = "research")]
    {
        use quiver_ann::research::{HyperbolicStrategy, set_hyperbolic_strategy};
        let truth_sets: Vec<HashSet<u64>> =
            truth.iter().map(|v| v.iter().copied().collect()).collect();

        println!(
            "\n=== 3. 算法组件正交消融实验 (N={count}, D={dim}, Q={query_count}, T={threads}) ==="
        );
        let strategy_cases = [
            (
                "A0: 原版双曲 Vamana (无补边 + 单阶段搜索)",
                HyperbolicStrategy::GeodesicPruning,
            ),
            (
                "A1: 原版双曲剪枝 + 仅保底补边 (单阶段搜索)",
                HyperbolicStrategy::GeodesicPruningWithFill,
            ),
            (
                "A1+P: 原版剪枝保底补边 + 两阶段门户一跳搜索",
                HyperbolicStrategy::GeodesicPortals,
            ),
            (
                "A2: 第一代双纤维构图 + 纯单阶段双曲搜索 (禁用BQ2门户)",
                HyperbolicStrategy::DualFiber,
            ),
            (
                "A3: 第一代双纤维构图 + Stage-0 BQ2门户 (禁用门户一跳展开)",
                HyperbolicStrategy::DualFiberPortals,
            ),
            (
                "A4: 第一代完整版 (双纤维构图 + 未校准BQ2门户 + 一跳测地展开)",
                HyperbolicStrategy::DualFiberExpansion,
            ),
            (
                "A5: + 极坐标锥-壳路由图谱 PolarRouter 与内向外测地主干构图",
                HyperbolicStrategy::PolarRouting,
            ),
            (
                "A6: 第二代黎曼 QuIVer (+ 固定Hermite校准 + 双遍Stage-0 + 四向罗盘剪枝)",
                HyperbolicStrategy::CalibratedCompass,
            ),
            (
                "A7: 第三代黎曼 QuIVer (+ 无超参SM球面重构 + 闭式d*聚焦选种 + 单遍双保持 + 无损径向预滤)",
                HyperbolicStrategy::FixedShellCompass,
            ),
            (
                "A8: 终极版黎曼 QuIVer (+ 凸支撑分位极坐标 + Stage-0.5正交去偏非对称守门 + 生成树零孤岛接线 + 零冗余热存)",
                HyperbolicStrategy::Full,
            ),
        ];

        for (label, mode) in strategy_cases {
            set_hyperbolic_strategy(mode);
            let b_start = Instant::now();
            let ab_index = Index::build_with_metric(
                dim,
                &ids,
                &vectors,
                &BuildConfig {
                    m: 16,
                    ef_construction: 128,
                    alpha: 1.2,
                },
                Metric::Poincare { curvature: 1.0 },
            )?;
            let b_sec = b_start.elapsed().as_secs_f64();
            let diag = ab_index.graph_diagnostics();
            let mut shell_out = [0usize; 3];
            let mut shell_in = [0usize; 3];
            let mut shell_n = [0usize; 3];
            for node in &diag.nodes {
                let s = doc_strata[node.id as usize];
                shell_n[s] += 1;
                shell_out[s] += node.out_degree;
                shell_in[s] += node.in_degree;
            }
            println!(
                "\n[{label}]\n  构图耗时={b_sec:.3}s | 热导航字节={:.1} B/vec | 总平均出度={:.2} | 三层出度=[{:.2}, {:.2}, {:.2}] | 三层入度=[{:.2}, {:.2}, {:.2}] | 零入度={}",
                ab_index.navigation_storage_bytes() as f64 / count as f64,
                diag.edges as f64 / count as f64,
                shell_out[0] as f64 / shell_n[0] as f64,
                shell_out[1] as f64 / shell_n[1] as f64,
                shell_out[2] as f64 / shell_n[2] as f64,
                shell_in[0] as f64 / shell_n[0] as f64,
                shell_in[1] as f64 / shell_n[1] as f64,
                shell_in[2] as f64 / shell_n[2] as f64,
                diag.zero_in_degree
            );
            for ef in [32, 64, 128, 256] {
                let cfg = SearchConfig {
                    top_k: 10,
                    ef_search: ef,
                    rerank_limit: None,
                };
                for q in queries.chunks_exact(dim).take(16) {
                    ab_index.search(q, &cfg)?;
                }
                let mut r_strata = [0.0; 3];
                let mut s_cnt = [0usize; 3];
                let q_start = Instant::now();
                for (i, q) in queries.chunks_exact(dim).enumerate() {
                    let res = ab_index.search(q, &cfg)?;
                    let hit =
                        res.iter().filter(|r| truth_sets[i].contains(&r.0)).count() as f64 / 10.0;
                    r_strata[query_strata[i]] += hit;
                    s_cnt[query_strata[i]] += 1;
                }
                let qps = query_count as f64 / q_start.elapsed().as_secs_f64();
                let total_recall = r_strata.iter().sum::<f64>() / query_count as f64;
                println!(
                    "  ef={ef:3}: Recall@10={:.4} (内={:.3}, 中={:.3}, 外={:.3}) | 单核QPS={qps:7.1}",
                    total_recall,
                    r_strata[0] / s_cnt[0] as f64,
                    r_strata[1] / s_cnt[1] as f64,
                    r_strata[2] / s_cnt[2] as f64,
                );
            }
        }

        println!(
            "\n=== 4. 同算法 (A8 Full) 下三种极坐标方向表示对照: Polar-SM-BQ2 (2-bit) vs Polar-SQ8 (8-bit) vs Polar-FP32 (32-bit) ==="
        );
        let polar_cases = [
            (
                "Polar-SM-BQ2 (2-bit SM 自校准球面重构 + 精确径向元数据, 方向码 16 B/vec @64D)",
                HyperbolicStrategy::Full,
            ),
            (
                "Polar-SQ8    (8-bit 归一化方向码 + 精确径向元数据, 方向码 64 B/vec @64D)",
                HyperbolicStrategy::ScalarQuantizedDirection,
            ),
            (
                "Polar-FP32   (32-bit 精确单位方向 + 精确径向元数据, 方向码 256 B/vec @64D)",
                HyperbolicStrategy::FullPrecisionDirection,
            ),
        ];

        for (label, mode) in polar_cases {
            set_hyperbolic_strategy(mode);
            let b_start = Instant::now();
            let p_index = Index::build_with_metric(
                dim,
                &ids,
                &vectors,
                &BuildConfig {
                    m: 16,
                    ef_construction: 128,
                    alpha: 1.2,
                },
                Metric::Poincare { curvature: 1.0 },
            )?;
            let b_sec = b_start.elapsed().as_secs_f64();
            println!("\n[{label}] 构图耗时={b_sec:.3}s");
            for ef in [32, 64, 128, 256] {
                let cfg = SearchConfig {
                    top_k: 10,
                    ef_search: ef,
                    rerank_limit: None,
                };
                for q in queries.chunks_exact(dim).take(16) {
                    p_index.search(q, &cfg)?;
                }
                let mut r_strata = [0.0; 3];
                let mut s_cnt = [0usize; 3];
                let q_start = Instant::now();
                for (i, q) in queries.chunks_exact(dim).enumerate() {
                    let res = p_index.search(q, &cfg)?;
                    let hit =
                        res.iter().filter(|r| truth_sets[i].contains(&r.0)).count() as f64 / 10.0;
                    r_strata[query_strata[i]] += hit;
                    s_cnt[query_strata[i]] += 1;
                }
                let qps = query_count as f64 / q_start.elapsed().as_secs_f64();
                let total_recall = r_strata.iter().sum::<f64>() / query_count as f64;
                println!(
                    "  ef={ef:3}: Recall@10={:.4} (内={:.3}, 中={:.3}, 外={:.3}) | 单核QPS={qps:7.1}",
                    total_recall,
                    r_strata[0] / s_cnt[0] as f64,
                    r_strata[1] / s_cnt[1] as f64,
                    r_strata[2] / s_cnt[2] as f64,
                );
            }
        }
        set_hyperbolic_strategy(HyperbolicStrategy::Full);

        println!(
            "\n=== 5. 2×2 构图×导航精度矩阵 & ICDE Eq.(60) 三阶段候选损失精确分解 (N={count}, D={dim}) ==="
        );
        let t_eb = Instant::now();
        let idx_exact_build = Index::build_poincare(
            dim,
            &ids,
            &vectors,
            &BuildConfig {
                m: 16,
                ef_construction: 128,
                alpha: 1.2,
            },
            1.0,
            PoincareNavigation::Exact,
        )?;
        let eb_sec = t_eb.elapsed().as_secs_f64();

        let t_qb = Instant::now();
        let idx_quant_build = Index::build_poincare(
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
        let qb_sec = t_qb.elapsed().as_secs_f64();
        println!(
            "  构图耗时对照 -> Exact 构图={eb_sec:.3}s | SM-Poincaré (GeometryBq2) 构图={qb_sec:.3}s (加速 {:.2}x)",
            eb_sec / qb_sec.max(1e-9)
        );

        let matrix_cases = [
            (
                "Exact Build × Exact Nav",
                &idx_exact_build,
                PoincareNavigation::Exact,
            ),
            (
                "Exact Build × SM-Poincaré Nav",
                &idx_exact_build,
                PoincareNavigation::GeometryBq2,
            ),
            (
                "SM-Poincaré Build × Exact Nav",
                &idx_quant_build,
                PoincareNavigation::Exact,
            ),
            (
                "SM-Poincaré Build × SM-Poincaré Nav",
                &idx_quant_build,
                PoincareNavigation::GeometryBq2,
            ),
        ];
        for (m_label, m_idx, nav_mode) in matrix_cases {
            println!("\n  [{m_label}]");
            for ef in [64, 128, 256] {
                let cfg = SearchConfig {
                    top_k: 10,
                    ef_search: ef,
                    rerank_limit: None,
                };
                for q in queries.chunks_exact(dim).take(16) {
                    match nav_mode {
                        PoincareNavigation::Exact => {
                            m_idx.search_poincare_exact_navigation(q, &cfg)?
                        }
                        PoincareNavigation::GeometryBq2 => {
                            m_idx.search_poincare_quantized_navigation(q, &cfg)?
                        }
                    };
                }
                let q_start = Instant::now();
                let mut hits = 0usize;
                for (i, q) in queries.chunks_exact(dim).enumerate() {
                    let res = match nav_mode {
                        PoincareNavigation::Exact => {
                            m_idx.search_poincare_exact_navigation(q, &cfg)?
                        }
                        PoincareNavigation::GeometryBq2 => {
                            m_idx.search_poincare_quantized_navigation(q, &cfg)?
                        }
                    };
                    hits += res.iter().filter(|r| truth_sets[i].contains(&r.0)).count();
                }
                let qps = query_count as f64 / q_start.elapsed().as_secs_f64();
                let rec = hits as f64 / (query_count * 10) as f64;

                let mut sum_reach_loss = 0.0;
                let mut sum_beam_loss = 0.0;
                let mut sum_rerank_loss = 0.0;
                let mut sum_visited = 0usize;
                let mut sum_cold = 0usize;
                for (i, q) in queries.chunks_exact(dim).enumerate() {
                    let st = m_idx.search_loss_decomposition(q, &cfg, nav_mode, &truth[i])?;
                    sum_reach_loss += st.reachability_loss(10);
                    sum_beam_loss += st.beam_retention_loss(10);
                    sum_rerank_loss += st.rerank_truncation_loss(10);
                    sum_visited += st.visited_nodes;
                    sum_cold += st.exact_vector_evals;
                }
                let inv_q = 1.0 / query_count as f64;
                println!(
                    "    ef={ef:3}: Recall@10={rec:.4} | 单核QPS={qps:7.1} | Δ_reach={:.4}, Δ_beam={:.4}, Δ_rerank={:.4} | 平均访问|V|={:6.1}, 冷存FP32读取={:5.1}",
                    sum_reach_loss * inv_q,
                    sum_beam_loss * inv_q,
                    sum_rerank_loss * inv_q,
                    sum_visited as f64 * inv_q,
                    sum_cold as f64 * inv_q,
                );
            }
        }
    }

    if thread_matrix != 0 {
        println!(
            "\n=== 5. 1/2/4/8/12 多线程并行构图与并发检索扩展性矩阵 (N={count}, D={dim}, Q={query_count}) ==="
        );
        let truth_sets_iso: Vec<HashSet<u64>> =
            truth.iter().map(|v| v.iter().copied().collect()).collect();
        let mut base_build_exact = 0.0f64;
        let mut base_build_bq2 = 0.0f64;
        for t in [1usize, 2, 4, 8, 12] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build()?;
            let (b_exact, idx_exact) = pool.install(|| -> quiver_ann::Result<(f64, Index)> {
                let ts = Instant::now();
                let idx = Index::build_poincare(
                    dim,
                    &ids,
                    &vectors,
                    &BuildConfig {
                        m: 16,
                        ef_construction: 128,
                        alpha: 1.2,
                    },
                    1.0,
                    PoincareNavigation::Exact,
                )?;
                Ok((ts.elapsed().as_secs_f64(), idx))
            })?;
            let (b_bq2, idx_bq2) = pool.install(|| -> quiver_ann::Result<(f64, Index)> {
                let ts = Instant::now();
                let idx = Index::build_poincare(
                    dim,
                    &ids,
                    &c_docs,
                    &BuildConfig {
                        m: 16,
                        ef_construction: 128,
                        alpha: 1.2,
                    },
                    1.0,
                    PoincareNavigation::GeometryBq2,
                )?;
                Ok((ts.elapsed().as_secs_f64(), idx))
            })?;
            if t == 1 {
                base_build_exact = b_exact;
                base_build_bq2 = b_bq2;
            }
            let cfg64 = SearchConfig {
                top_k: 10,
                ef_search: 64,
                rerank_limit: None,
            };
            // 预热并发检索工作线程
            pool.install(|| {
                queries.par_chunks_exact(dim).for_each(|q| {
                    let _ = idx_exact.search(q, &cfg64);
                });
                c_queries.par_chunks_exact(dim).for_each(|q| {
                    let _ = idx_bq2.search(q, &cfg64);
                });
            });
            let rounds = 8usize;
            let q_exact_start = Instant::now();
            let hits_exact: usize = pool.install(|| {
                (0..rounds * query_count)
                    .into_par_iter()
                    .map(|k| {
                        let qi = k % query_count;
                        let q = &queries[qi * dim..(qi + 1) * dim];
                        let res = idx_exact.search(q, &cfg64).unwrap();
                        res.iter()
                            .filter(|r| truth_sets_iso[qi].contains(&r.0))
                            .count()
                    })
                    .sum()
            });
            let qps_exact = (rounds * query_count) as f64 / q_exact_start.elapsed().as_secs_f64();
            let rec_exact = hits_exact as f64 / (rounds * query_count * 10) as f64;

            let q_bq2_start = Instant::now();
            let hits_bq2: usize = pool.install(|| {
                (0..rounds * query_count)
                    .into_par_iter()
                    .map(|k| {
                        let qi = k % query_count;
                        let q = &c_queries[qi * dim..(qi + 1) * dim];
                        let res = idx_bq2.search(q, &cfg64).unwrap();
                        res.iter().filter(|r| c_truth[qi].contains(&r.0)).count()
                    })
                    .sum()
            });
            let qps_bq2 = (rounds * query_count) as f64 / q_bq2_start.elapsed().as_secs_f64();
            let rec_bq2 = hits_bq2 as f64 / (rounds * query_count * 10) as f64;

            println!(
                "  线程={t:2} | 各向同性Exact: 构图={b_exact:6.3}s ({:5.2}x), ef=64 Recall@10={rec_exact:.4}, 并发QPS={qps_exact:8.1} | 聚簇GeometryBq2: 构图={b_bq2:6.3}s ({:5.2}x), ef=64 Recall@10={rec_bq2:.4}, 并发QPS={qps_bq2:8.1}",
                base_build_exact / b_exact,
                base_build_bq2 / b_bq2,
            );
        }
    }
    Ok(())
}
