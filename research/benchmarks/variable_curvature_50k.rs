//! 变曲率几何原型 50k 纯合成对比实验（Tier 2 混合/乘积流形 + Tier 3 径向变曲率翘曲乘积流形）。
//!
//! 运行命令：
//! `cargo run --release --features research --example benchmark_variable_curvature_50k`

use quiver_ann::research::var_curvature::{
    GraphBuildConfig, ProductBaselineIndex, ProductBaselineMode, ProductBlockSmIndex,
    ProductFactor, ProductSingleBallIndex, SearchOutcome, Tier3Method, WarpedManifoldIndex,
    WarpedProfile, generate_product_synthetic, generate_warped_synthetic, product_ground_truth,
    warped_ground_truth,
};
use std::time::Instant;

const TOTAL_DIM: usize = 256;
const N_DOCS: usize = 50_000;
const N_QUERIES: usize = 500;
const TOP_K: usize = 10;
const M_MAX: usize = 16;
const EF_CONSTRUCT: usize = 64;
const TIER3_M_MAX: usize = 32;
const TIER3_EF_CONSTRUCT: usize = 128;
const ALPHA: f64 = 1.2;

const SWEEP: [(usize, usize); 5] = [(32, 32), (64, 64), (128, 64), (128, 128), (256, 128)];

#[derive(Default)]
struct AggStats {
    recall_pct: f64,
    qps_1t: f64,
    avg_ms: f64,
    avg_rerank: f64,
    avg_radial_rej: f64,
    avg_expansions: f64,
}

fn evaluate_queries<F>(queries: &[f32], truth: &[Vec<usize>], mut run: F) -> AggStats
where
    F: FnMut(&[f32]) -> SearchOutcome,
{
    let q_chunks = queries.as_chunks::<TOTAL_DIM>().0;
    let n_q = q_chunks.len();
    // 预热少量查询，避开冷缓存首跳毛刺
    for q in q_chunks.iter().take(8.min(n_q)) {
        let _ = run(q);
    }

    let mut hits = 0usize;
    let mut sum_rerank = 0usize;
    let mut sum_rej = 0usize;
    let mut sum_exp = 0usize;

    let t0 = Instant::now();
    for (qi, q) in q_chunks.iter().enumerate() {
        let out = run(q);
        hits += out
            .results
            .iter()
            .filter(|(id, _)| truth[qi].contains(id))
            .count();
        sum_rerank += out.exact_reranks;
        sum_rej += out.radial_rejections;
        sum_exp += out.expansions;
    }
    let elapsed = t0.elapsed().as_secs_f64();

    AggStats {
        recall_pct: 100.0 * (hits as f64) / ((n_q * TOP_K) as f64),
        qps_1t: (n_q as f64) / elapsed.max(1e-9),
        avg_ms: 1000.0 * elapsed / (n_q as f64),
        avg_rerank: (sum_rerank as f64) / (n_q as f64),
        avg_radial_rej: (sum_rej as f64) / (n_q as f64),
        avg_expansions: (sum_exp as f64) / (n_q as f64),
    }
}

fn print_header(title: &str) {
    println!(
        "\n================================================================================================================"
    );
    println!("{title}");
    println!(
        "================================================================================================================"
    );
    println!(
        "{:<34} | {:>5} | {:>6} | {:>9} | {:>9} | {:>8} | {:>8} | {:>8} | {:>8}",
        "方案名称",
        "ef",
        "budget",
        "Recall@10",
        "QPS(1T)",
        "延迟(ms)",
        "平均精排",
        "径向剪枝",
        "平均扩展"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------"
    );
}

fn print_row(name: &str, ef: usize, budget: usize, mem_mb: f64, st: &AggStats) {
    println!(
        "{:<34} | {:>5} | {:>6} | {:>8.2}% | {:>9.1} | {:>8.3} | {:>8.1} | {:>8.1} | {:>8.1} | 热索引={:.2} MiB",
        name,
        ef,
        budget,
        st.recall_pct,
        st.qps_1t,
        st.avg_ms,
        st.avg_rerank,
        st.avg_radial_rej,
        st.avg_expansions,
        mem_mb
    );
}

fn run_tier2_experiment() {
    println!(
        "\n>>> [实验一：层级 2 混合/乘积常曲率流形] N={N_DOCS}, D={TOTAL_DIM}, H_{{-4.0}}^64 x H_{{-0.25}}^64 x R^64 x S_{{1.0}}^64"
    );
    let factors = vec![
        ProductFactor::Hyperbolic {
            dim: 64,
            curvature: 4.0,
        },
        ProductFactor::Hyperbolic {
            dim: 64,
            curvature: 0.25,
        },
        ProductFactor::Euclidean { dim: 64 },
        ProductFactor::Spherical {
            dim: 64,
            curvature: 1.0,
        },
    ];

    let t_gen = Instant::now();
    let (docs, queries) = generate_product_synthetic(&factors, N_DOCS, N_QUERIES, 202_602);
    let truth = product_ground_truth(&factors, &docs, &queries, TOP_K);
    println!(
        "数据生成与精确真值计算完成，耗时 {:.2}s",
        t_gen.elapsed().as_secs_f64()
    );

    let fp32_vec_mb = (N_DOCS * TOTAL_DIM * 4) as f64 / (1024.0 * 1024.0);

    let t = Instant::now();
    let idx_l2 = ProductBaselineIndex::build(
        factors.clone(),
        &docs,
        ProductBaselineMode::EuclideanL2,
        M_MAX,
        EF_CONSTRUCT,
        ALPHA,
    );
    println!(
        "构建完成: EuclideanL2 (图 {:.2} MiB + FP32原向量 {:.2} MiB), 耗时 {:.2}s",
        idx_l2.index_memory_bytes() as f64 / (1024.0 * 1024.0),
        fp32_vec_mb,
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let idx_single =
        ProductSingleBallIndex::build(factors.clone(), &docs, 1.0, M_MAX, EF_CONSTRUCT, ALPHA);
    println!(
        "构建完成: SingleBallSm(c=1.0) (热索引 {:.2} MiB), 耗时 {:.2}s",
        idx_single.index_memory_bytes() as f64 / (1024.0 * 1024.0),
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let idx_blk_noc =
        ProductBlockSmIndex::build(factors.clone(), &docs, M_MAX, EF_CONSTRUCT, ALPHA, false);
    println!(
        "构建完成: ProductBlockSm(无Compass) (热索引 {:.2} MiB), 耗时 {:.2}s",
        idx_blk_noc.index_memory_bytes() as f64 / (1024.0 * 1024.0),
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let idx_blk_comp =
        ProductBlockSmIndex::build(factors.clone(), &docs, M_MAX, EF_CONSTRUCT, ALPHA, true);
    println!(
        "构建完成: ProductBlockSm(含Compass) (热索引 {:.2} MiB), 耗时 {:.2}s",
        idx_blk_comp.index_memory_bytes() as f64 / (1024.0 * 1024.0),
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let idx_exact = ProductBaselineIndex::build(
        factors,
        &docs,
        ProductBaselineMode::ExactProduct,
        M_MAX,
        EF_CONSTRUCT,
        ALPHA,
    );
    println!(
        "构建完成: ExactProduct(Oracle) (图 {:.2} MiB + FP32原向量 {:.2} MiB), 耗时 {:.2}s",
        idx_exact.index_memory_bytes() as f64 / (1024.0 * 1024.0),
        fp32_vec_mb,
        t.elapsed().as_secs_f64()
    );

    print_header("层级 2 (50k x 256D 混合流形) 全消融与基线对比结果");

    let l2_mem = (idx_l2.index_memory_bytes() as f64) / (1024.0 * 1024.0) + fp32_vec_mb;
    let single_mem = (idx_single.index_memory_bytes() as f64) / (1024.0 * 1024.0);
    let noc_mem = (idx_blk_noc.index_memory_bytes() as f64) / (1024.0 * 1024.0);
    let comp_mem = (idx_blk_comp.index_memory_bytes() as f64) / (1024.0 * 1024.0);
    let exact_mem = (idx_exact.index_memory_bytes() as f64) / (1024.0 * 1024.0) + fp32_vec_mb;

    for &(ef, budget) in &SWEEP {
        let st_l2 = evaluate_queries(&queries, &truth, |q| idx_l2.search(q, TOP_K, ef, budget));
        print_row("EuclideanL2 + 精确重排", ef, budget, l2_mem, &st_l2);

        let st_single = evaluate_queries(&queries, &truth, |q| {
            idx_single.search(q, TOP_K, ef, budget)
        });
        print_row(
            "SingleBallSm(强行单曲率)",
            ef,
            budget,
            single_mem,
            &st_single,
        );

        let st_noc = evaluate_queries(&queries, &truth, |q| {
            idx_blk_noc.search(q, TOP_K, ef, budget, false, false)
        });
        print_row(
            "BlockSM(无Compass/无径向/精块)",
            ef,
            budget,
            noc_mem,
            &st_noc,
        );

        let st_c_fast_norad = evaluate_queries(&queries, &truth, |q| {
            idx_blk_comp.search(q, TOP_K, ef, budget, false, true)
        });
        print_row(
            "BlockSM(含Compass/无径向/快代理)",
            ef,
            budget,
            comp_mem,
            &st_c_fast_norad,
        );

        let st_c_fast_rad = evaluate_queries(&queries, &truth, |q| {
            idx_blk_comp.search(q, TOP_K, ef, budget, true, true)
        });
        print_row(
            "BlockSM(含Compass/多块径向/快代理)",
            ef,
            budget,
            comp_mem,
            &st_c_fast_rad,
        );

        let st_c_exact_rad = evaluate_queries(&queries, &truth, |q| {
            idx_blk_comp.search(q, TOP_K, ef, budget, true, false)
        });
        print_row(
            "BlockSM(含Compass/多块径向/精块)",
            ef,
            budget,
            comp_mem,
            &st_c_exact_rad,
        );

        let st_exact =
            evaluate_queries(&queries, &truth, |q| idx_exact.search(q, TOP_K, ef, budget));
        print_row(
            "ExactProduct(FP32真乘积Oracle)",
            ef,
            budget,
            exact_mem,
            &st_exact,
        );
        println!(
            "----------------------------------------------------------------------------------------------------------------"
        );
    }
}

fn run_tier3_case(label: &str, profile: WarpedProfile, clustered: bool, seed: u64) {
    println!(
        "\n>>> [实验二：{label}] N={N_DOCS}, D={TOTAL_DIM}, M={TIER3_M_MAX}, efC={TIER3_EF_CONSTRUCT}, profile={profile:?}, clustered={clustered}"
    );
    let t_gen = Instant::now();
    let (docs, queries) =
        generate_warped_synthetic(TOTAL_DIM, N_DOCS, N_QUERIES, 0.25, 3.2, clustered, seed);
    let truth = warped_ground_truth(profile, TOTAL_DIM, &docs, &queries, TOP_K);
    println!(
        "数据生成与精确 Clairaut 真值计算完成，耗时 {:.2}s",
        t_gen.elapsed().as_secs_f64()
    );

    let fp32_vec_mb = (N_DOCS * TOTAL_DIM * 4) as f64 / (1024.0 * 1024.0);
    let cfg_default = GraphBuildConfig {
        m_max: TIER3_M_MAX,
        ef_construction: TIER3_EF_CONSTRUCT,
        alpha: ALPHA,
        use_compass: false,
        use_cones: true,
    };

    let t = Instant::now();
    let idx_l2 = WarpedManifoldIndex::build_with_config(
        profile,
        Tier3Method::EuclideanL2,
        TOTAL_DIM,
        &docs,
        cfg_default,
    );
    println!(
        "构建完成: EuclideanL2, 耗时 {:.2}s",
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let idx_fixed = WarpedManifoldIndex::build_with_config(
        profile,
        Tier3Method::FixedConstantHyperbolicSm { curvature: 1.0 },
        TOTAL_DIM,
        &docs,
        cfg_default,
    );
    println!(
        "构建完成: FixedConstantHyperbolicSm(c=1.0+Cone), 耗时 {:.2}s",
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let idx_local = WarpedManifoldIndex::build_with_config(
        profile,
        Tier3Method::LocalWarpedFormulaSm,
        TOTAL_DIM,
        &docs,
        cfg_default,
    );
    println!(
        "构建完成: LocalWarpedFormulaSm(+Cone), 耗时 {:.2}s",
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let idx_tbl_nocone = WarpedManifoldIndex::build_with_config(
        profile,
        Tier3Method::WarpedTableSm,
        TOTAL_DIM,
        &docs,
        GraphBuildConfig {
            use_cones: false,
            ..cfg_default
        },
    );
    println!(
        "构建完成: WarpedTableSm(无Cone裸2bit) (热索引 {:.2} MiB), 耗时 {:.2}s",
        idx_tbl_nocone.index_memory_bytes() as f64 / (1024.0 * 1024.0),
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let idx_tbl = WarpedManifoldIndex::build_with_config(
        profile,
        Tier3Method::WarpedTableSm,
        TOTAL_DIM,
        &docs,
        GraphBuildConfig {
            use_compass: true,
            ..cfg_default
        },
    );
    let tbl_tag = if clustered {
        "Cone+2bit"
    } else {
        "QuadriPrune+2bit"
    };
    println!(
        "构建完成: WarpedTableSm({tbl_tag}) (热索引 {:.2} MiB), 耗时 {:.2}s",
        idx_tbl.index_memory_bytes() as f64 / (1024.0 * 1024.0),
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let idx_exact = WarpedManifoldIndex::build_with_config(
        profile,
        Tier3Method::ExactClairaut,
        TOTAL_DIM,
        &docs,
        cfg_default,
    );
    println!(
        "构建完成: ExactClairaut(Oracle), 耗时 {:.2}s",
        t.elapsed().as_secs_f64()
    );

    print_header(&format!(
        "层级 3 (50k x 256D 径向变曲率 - {label}) 对比结果"
    ));

    let l2_mem = (idx_l2.index_memory_bytes() as f64) / (1024.0 * 1024.0) + fp32_vec_mb;
    let fixed_mem = (idx_fixed.index_memory_bytes() as f64) / (1024.0 * 1024.0);
    let local_mem = (idx_local.index_memory_bytes() as f64) / (1024.0 * 1024.0);
    let nocone_mem = (idx_tbl_nocone.index_memory_bytes() as f64) / (1024.0 * 1024.0);
    let tbl_mem = (idx_tbl.index_memory_bytes() as f64) / (1024.0 * 1024.0);
    let exact_mem = (idx_exact.index_memory_bytes() as f64) / (1024.0 * 1024.0) + fp32_vec_mb;

    for &(ef, budget) in &SWEEP {
        let st_l2 = evaluate_queries(&queries, &truth, |q| {
            idx_l2.search(q, TOP_K, ef, budget, false)
        });
        print_row("EuclideanL2 + Clairaut重排", ef, budget, l2_mem, &st_l2);

        let st_fixed = evaluate_queries(&queries, &truth, |q| {
            idx_fixed.search(q, TOP_K, ef, budget, false)
        });
        print_row(
            "FixedConstHypSM(c=1.0+Cone)",
            ef,
            budget,
            fixed_mem,
            &st_fixed,
        );

        let st_local = evaluate_queries(&queries, &truth, |q| {
            idx_local.search(q, TOP_K, ef, budget, false)
        });
        print_row(
            "LocalWarpedFormulaSM(中点φ+Cone)",
            ef,
            budget,
            local_mem,
            &st_local,
        );

        let st_nocone = evaluate_queries(&queries, &truth, |q| {
            idx_tbl_nocone.search(q, TOP_K, ef, budget, false)
        });
        print_row(
            "WarpedTableSM(无Cone裸2bit)",
            ef,
            budget,
            nocone_mem,
            &st_nocone,
        );

        let tbl_mode = if clustered { "Cone" } else { "Quadri" };
        let label_norad = format!("WarpedTableSM({tbl_mode}+无几何剪枝)");
        let label_rad = format!("WarpedTableSM({tbl_mode}+几何剪枝)");

        let st_tbl_norad = evaluate_queries(&queries, &truth, |q| {
            idx_tbl.search(q, TOP_K, ef, budget, false)
        });
        print_row(&label_norad, ef, budget, tbl_mem, &st_tbl_norad);

        let st_tbl_rad = evaluate_queries(&queries, &truth, |q| {
            idx_tbl.search(q, TOP_K, ef, budget, true)
        });
        print_row(&label_rad, ef, budget, tbl_mem, &st_tbl_rad);

        let st_exact = evaluate_queries(&queries, &truth, |q| {
            idx_exact.search(q, TOP_K, ef, budget, false)
        });
        print_row(
            "ExactClairaut(FP32精确Oracle)",
            ef,
            budget,
            exact_mem,
            &st_exact,
        );
        println!(
            "----------------------------------------------------------------------------------------------------------------"
        );
    }
}

fn main() {
    println!("=== QuIVer 变曲率流形原型 50k 纯合成基准测试（仅 research 探索，暂不转正）===");
    println!(
        "公共参数: N={N_DOCS}, D={TOTAL_DIM}, Q={N_QUERIES}, TopK={TOP_K}, alpha={ALPHA} (Tier2: M={M_MAX}, efC={EF_CONSTRUCT}; Tier3: M={TIER3_M_MAX}, efC={TIER3_EF_CONSTRUCT})"
    );

    run_tier2_experiment();

    run_tier3_case(
        "Softening 外围渐平 (聚类)",
        WarpedProfile::Softening { alpha: 0.5 },
        true,
        2_026_031,
    );
    run_tier3_case(
        "Steepening 外围陡增 (聚类)",
        WarpedProfile::Steepening { alpha: 0.22 },
        true,
        2_026_032,
    );
    run_tier3_case(
        "Steepening 外围陡增 (宽壳均匀)",
        WarpedProfile::Steepening { alpha: 0.22 },
        false,
        2_026_033,
    );
}
