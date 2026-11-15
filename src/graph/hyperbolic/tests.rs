//! 双曲几何、残差与数值内核回归。

use super::*;
#[cfg(feature = "research")]
#[test]
fn sq8增量方向回退兼容极小坐标与大维度() {
    for dim in [3, 200001] {
        let tiny = f32::from_bits(1);
        let a: Vec<_> = (0..dim).map(|i| tiny * (i % 3 + 1) as f32).collect();
        let b: Vec<_> = (0..dim).map(|i| tiny * (3 - i % 3) as f32).collect();
        let raw: Vec<_> = a.iter().chain(&b).copied().collect();
        let store = crate::direction::DirectionStore::build(&raw, dim, true);
        let actual =
            directional_cosine_slices(&a, &b, HyperbolicStrategy::ScalarQuantizedDirection);
        assert!((actual - store.pair_cosine(0, 1, &a, &b)).abs() < 1e-12);
    }
}
#[test]
fn 径向舍入下界在原点边界与不同维度保持保守() {
    for dim in [1, 3, 64, 768, 4097] {
        for qr in [0.0, 1e-20, 1e-10, 0.3, 0.9, 0.9999999] {
            for vr in [0.0, 1e-20, 1e-10, 0.2, 0.9, 0.9999999] {
                let q: Vec<f32> = vec![(qr / (dim as f64).sqrt()) as f32; dim];
                let v: Vec<f32> = vec![(vr / (dim as f64).sqrt()) as f32; dim];
                let qn = norm_squared_f64(&q);
                let vn = norm_squared_f64(&v);
                let qf = 1.0 / (1.0 - qn);
                let vf = 1.0 / (1.0 - vn);
                let meta = PolarNodeMeta {
                    radius: vn.sqrt() as f32,
                    boundary_factor: vf as f32,
                    ..Default::default()
                };
                let bound = meta.radial_lower_bound(qn.sqrt(), qf);
                let exact = poincare_proxy(&q, &v, qf * vf);
                assert!(
                    bound <= exact,
                    "径向下界越界，维度{dim}，查询半径{qr}，文档半径{vr}"
                );
            }
        }
    }
}

#[cfg(feature = "research")]
static CONFIGURATION_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(feature = "research")]
#[test]
fn 速度消融开关与连续布局保持结果及损失统计严格一致() {
    let _guard = CONFIGURATION_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    set_storage_config(StorageConfig::default());
    let benchmark = benchmark_sign_magnitude_kernels(256);
    assert_eq!(benchmark.len(), 4);
    for &(_, _, ns) in &benchmark {
        assert!(ns.is_finite() && ns > 0.0);
    }

    let dim = 768usize;
    let n = 384usize;
    let mut raw = vec![0.0f32; n * dim];
    let mut state = 0x5EED_CAFE_1234_5678u64;
    for row in 0..n {
        let mut sq = 0.0f64;
        for col in 0..dim {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = (((state >> 32) as i32) as f64) / (i32::MAX as f64);
            raw[row * dim + col] = v as f32;
            sq += v * v;
        }
        let target_r = 0.25 + 0.65 * ((row % 19) as f64 / 19.0);
        let scale = (target_r / sq.sqrt().max(1e-9)) as f32;
        for col in 0..dim {
            raw[row * dim + col] *= scale;
        }
    }
    let ids: Vec<u64> = (0..n as u64).collect();
    let cfg = crate::BuildConfig {
        m: 12,
        ef_construction: 64,
        alpha: 1.2,
    };
    set_hyperbolic_strategy(HyperbolicStrategy::Full);
    set_geometry_components(GeometryComponents::default());
    set_execution_config(ExecutionConfig::reference());
    let mut idx = crate::Index::build_poincare(
        dim,
        &ids,
        &raw,
        &cfg,
        1.0,
        crate::PoincareNavigation::GeometryBq2,
    )
    .unwrap();

    let query = &raw[0..dim];
    let scfg = SearchConfig {
        ef_search: 48,
        top_k: 10,
        rerank_limit: None,
    };

    idx.set_bq_contiguous_layout(false);
    set_execution_config(ExecutionConfig {
        zero_trace_overhead: false,
        two_level_cones: false,
        rolling_prefetch: false,
        fast_kernels: false,
        kernel_variant: 0,
        traversal: TraversalProfile::Standard,
    });
    let r_base = idx.search(query, &scfg).unwrap();
    let loss_base = idx
        .search_loss_decomposition(query, &scfg, crate::PoincareNavigation::GeometryBq2, &[0])
        .unwrap();

    idx.set_bq_contiguous_layout(true);
    set_execution_config(ExecutionConfig {
        zero_trace_overhead: true,
        two_level_cones: false,
        rolling_prefetch: true,
        fast_kernels: true,
        kernel_variant: 3,
        traversal: TraversalProfile::Standard,
    });
    let r_fast = idx.search(query, &scfg).unwrap();
    let loss_fast = idx
        .search_loss_decomposition(query, &scfg, crate::PoincareNavigation::GeometryBq2, &[0])
        .unwrap();

    assert_eq!(r_base.len(), r_fast.len());
    for (a, b) in r_base.iter().zip(r_fast.iter()) {
        assert_eq!(a.0, b.0);
        assert!((a.1 - b.1).abs() < 1e-6);
    }
    assert_eq!(loss_base, loss_fast);
    set_execution_config(ExecutionConfig::default());
}

#[cfg(feature = "research")]
#[test]
fn 单页合表与束内头行两级早筛正确性验证() {
    let _guard = CONFIGURATION_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    assert_eq!(std::mem::size_of::<HotSmHeader>(), 64);
    assert_eq!(std::mem::align_of::<HotSmHeader>(), 64);
    assert_eq!(std::mem::size_of::<HotSmPage>(), 4096);
    assert_eq!(std::mem::align_of::<HotSmPage>(), 4096);

    for dim in [512usize, 768, 1536] {
        let n = 512usize;
        let mut raw = vec![0.0f32; n * dim];
        let mut state = 0x77AA_CC33_1998_2026u64;
        for row in 0..n {
            let mut sq = 0.0f64;
            for col in 0..dim {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = (((state >> 32) as i32) as f64) / (i32::MAX as f64);
                raw[row * dim + col] = v as f32;
                sq += v * v;
            }
            let target_r = 0.72 + 0.24 * ((row % 23) as f64 / 23.0);
            let scale = (target_r / sq.sqrt().max(1e-9)) as f32;
            for col in 0..dim {
                raw[row * dim + col] *= scale;
            }
        }
        let ids: Vec<u64> = (0..n as u64).collect();
        let cfg = crate::BuildConfig {
            m: 16,
            ef_construction: 96,
            alpha: 1.2,
        };
        set_hyperbolic_strategy(HyperbolicStrategy::Full);
        set_geometry_components(GeometryComponents::default());
        set_execution_config(ExecutionConfig::default());
        set_storage_config(StorageConfig::default());

        let mut idx = crate::Index::build_poincare(
            dim,
            &ids,
            &raw,
            &cfg,
            1.0,
            crate::PoincareNavigation::GeometryBq2,
        )
        .unwrap();

        let scfg = SearchConfig {
            ef_search: 64,
            top_k: 10,
            rerank_limit: Some(48),
        };

        let mut baseline_res = Vec::new();
        for q_row in [0usize, 17, 99, 255, 411] {
            let q = &raw[q_row * dim..(q_row + 1) * dim];
            baseline_res.push((q_row, idx.search(q, &scfg).unwrap()));
        }
        set_execution_config(ExecutionConfig::reference());
        let baseline_slow: Vec<_> = baseline_res
            .iter()
            .map(|(row, _)| {
                let q = &raw[*row * dim..(*row + 1) * dim];
                (*row, idx.search(q, &scfg).unwrap())
            })
            .collect();
        set_execution_config(ExecutionConfig::default());

        // 构建 4KiB 单页对齐合表并验证不释放分离数组时无损模式 100% 逐项等价
        let prep_sec = idx.prepare_colocated_hot_store(false).unwrap();
        assert!(prep_sec >= 0.0);

        for cfg_lossless in [
            StorageConfig {
                colocated_store: true,
                early_gate: SketchFilter::Off,
                header_prefetch_window: 8,
                eager_prefetch_overlap: false,
            },
            StorageConfig {
                colocated_store: true,
                early_gate: SketchFilter::Off,
                header_prefetch_window: 8,
                eager_prefetch_overlap: true,
            },
            StorageConfig {
                colocated_store: true,
                early_gate: SketchFilter::LosslessRadial,
                header_prefetch_window: 8,
                eager_prefetch_overlap: true,
            },
        ] {
            set_storage_config(cfg_lossless);
            assert_eq!(storage_config(), cfg_lossless);
            for (q_row, expected) in &baseline_res {
                let q = &raw[*q_row * dim..(*q_row + 1) * dim];
                let actual = idx.search(q, &scfg).unwrap();
                assert_eq!(
                    actual, *expected,
                    "无损热路径配置 {:?} 结果漂移",
                    cfg_lossless
                );
            }
        }

        // 验证 SymSketch256 与 AsymSketch256 两级头行早筛记录拦截计数且保持查询自身命中
        for gate in [
            SketchFilter::SymSketch256 { margin: 0.12 },
            SketchFilter::AsymSketch256 { margin: 0.10 },
        ] {
            set_storage_config(StorageConfig {
                colocated_store: true,
                early_gate: gate,
                header_prefetch_window: 8,
                eager_prefetch_overlap: true,
            });
            let q = &raw[17 * dim..18 * dim];
            let res = idx.search(q, &scfg).unwrap();
            assert_eq!(res[0].0, 17, "头行草图早筛 {:?} 丢失精确自身 Top-1", gate);
            let loss = idx
                .search_loss_decomposition(q, &scfg, crate::PoincareNavigation::GeometryBq2, &[17])
                .unwrap();
            assert!(loss.header_checks > 0, "头行草图早筛 {:?} 未触发检查", gate);
            assert!(
                loss.header_rejections > 0,
                "头行草图早筛 {:?} 未产生任何候选拦截",
                gate
            );
            assert_eq!(loss.hits_returned, 1);
        }

        // 验证 reclaim_split = true 释放旧分离数组后无损查询依旧 100% 逐项一致
        let _ = idx.prepare_colocated_hot_store(true).unwrap();
        set_storage_config(StorageConfig {
            colocated_store: true,
            early_gate: SketchFilter::Off,
            header_prefetch_window: 8,
            eager_prefetch_overlap: true,
        });
        for (q_row, expected) in &baseline_res {
            let q = &raw[*q_row * dim..(*q_row + 1) * dim];
            let actual = idx.search(q, &scfg).unwrap();
            assert_eq!(
                actual, *expected,
                "释放旧分离数组后查询 {} 结果不一致",
                q_row
            );
        }

        // 回收操作可重复执行，关闭合表热循环或快速内核后仍保持原路径结果。
        let hot_bytes = idx.navigation_storage_bytes();
        idx.prepare_colocated_hot_store(true).unwrap();
        assert_eq!(idx.navigation_storage_bytes(), hot_bytes);
        set_storage_config(StorageConfig::default());
        for (q_row, expected) in &baseline_res {
            let q = &raw[*q_row * dim..(*q_row + 1) * dim];
            assert_eq!(idx.search(q, &scfg).unwrap(), *expected);
        }
        set_execution_config(ExecutionConfig::reference());
        for (q_row, expected) in &baseline_slow {
            let q = &raw[*q_row * dim..(*q_row + 1) * dim];
            assert_eq!(idx.search(q, &scfg).unwrap(), *expected);
        }
        set_execution_config(ExecutionConfig::default());

        // 布局与方向表示切换必须恢复完整可变后端，再次合表也保持查询等价。
        for contiguous in [false, true] {
            idx.set_bq_contiguous_layout(contiguous);
            for (q_row, expected) in &baseline_res {
                let q = &raw[*q_row * dim..(*q_row + 1) * dim];
                assert_eq!(idx.search(q, &scfg).unwrap(), *expected);
            }
            idx.prepare_colocated_hot_store(true).unwrap();
        }
        let query = &raw[17 * dim..18 * dim];
        for mode in [
            HyperbolicStrategy::ScalarQuantizedDirection,
            HyperbolicStrategy::FullPrecisionDirection,
        ] {
            idx.prepare_direction_representation(mode).unwrap();
            set_hyperbolic_strategy(mode);
            let expected = idx.search(query, &scfg).unwrap();
            idx.prepare_colocated_hot_store(true).unwrap();
            assert_eq!(idx.search(query, &scfg).unwrap(), expected);
        }
        set_hyperbolic_strategy(HyperbolicStrategy::Full);
        idx.prepare_direction_representation(HyperbolicStrategy::Full)
            .unwrap();
        idx.prepare_colocated_hot_store(true).unwrap();

        // 拥有型持久化、软删除、增量追加、重新合表和压实均走公共入口。
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("合表回收.quiver");
        idx.save(&path).unwrap();
        let loaded = crate::Index::load(&path).unwrap();
        assert_eq!(
            loaded.search_exact(query, 10).unwrap(),
            idx.search_exact(query, 10).unwrap(),
        );
        assert!(idx.remove(17));
        assert!(
            idx.search(query, &scfg)
                .unwrap()
                .iter()
                .all(|result| result.0 != 17)
        );
        let appended: Vec<f32> = query.iter().map(|value| value * 0.99).collect();
        idx.insert(10_001, &appended).unwrap();
        assert_eq!(idx.len(), n);
        assert_eq!(idx.search(&appended, &scfg).unwrap()[0].0, 10_001);
        idx.prepare_colocated_hot_store(true).unwrap();
        idx.insert(17, query).unwrap();
        assert_eq!(idx.search(query, &scfg).unwrap()[0].0, 17);
        assert!(idx.remove(10_001));
        idx.prepare_colocated_hot_store(true).unwrap();
        assert_eq!(idx.compact().unwrap(), 2);
        assert_eq!(idx.len(), n);
        idx.save(&path).unwrap();
        let loaded = crate::Index::load(&path).unwrap();
        assert!(loaded.contains(17));
        assert!(!loaded.contains(10_001));
        assert_eq!(loaded.search(query, &scfg).unwrap()[0].0, 17);

        // 底层图文件必须写出合表接管的全部签名，保持原磁盘格式。
        let count = 32;
        let factors: Vec<f64> = raw[..count * dim]
            .chunks_exact(dim)
            .map(|v| 1.0 / (1.0 - norm_squared_f64(v)))
            .collect();
        let space = NativeSpace {
            vectors: &raw[..count * dim],
            boundary_factors: &factors,
            dimensions: dim,
            curvature: 1.0,
        };
        let mut graph = QuIVer::new(dim, &cfg);
        let slots: Vec<_> = (0..count).collect();
        graph.batch_build_native(&space, &ids[..count], &slots, true);
        let signatures = (
            graph.bq_store.collect_pos_data(),
            graph.bq_store.collect_strong_data(),
        );
        graph.prepare_colocated_hot_store(&space, false);
        graph.set_bq_contiguous_layout(false);
        assert!(
            graph
                .polar_router
                .as_ref()
                .unwrap()
                .colocated_store
                .is_some()
        );
        graph.prepare_colocated_hot_store(&space, true);
        assert_eq!(graph.bq_store.len(), count);
        let graph_path = directory.path().join("合表底层图.quiver");
        graph.save_to_file(&graph_path).unwrap();
        let restored = QuIVer::load_from_file(&graph_path).unwrap();
        assert_eq!(restored.total_count(), count);
        assert_eq!(restored.bq_store.collect_pos_data(), signatures.0);
        assert_eq!(restored.bq_store.collect_strong_data(), signatures.1);

        set_storage_config(StorageConfig::default());
        set_execution_config(ExecutionConfig::default());
    }
}

#[test]
fn 非对称草图评分器标量与simd逐比特等价验证() {
    let mut rng = 0x517C_C1B7_2722_0A95u64;
    let mut next_u64 = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut q = [0.0f32; 768];
    let mut scorer = AsymSketch256Query::default();
    for _ in 0..64 {
        for val in &mut q {
            *val = ((next_u64() >> 40) as f32 / 8_388_608.0) - 1.0;
        }
        scorer.encode_into(&q);
        assert!(scorer.valid);
        for _ in 0..8 {
            let sketch = [next_u64(), next_u64(), next_u64(), next_u64()];
            let expected = scorer.agree_sum_scalar(&sketch);
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    let got_avx2 = unsafe { scorer.agree_sum_avx2(&sketch) };
                    assert_eq!(got_avx2, expected, "AVX2 归约与标量不一致");
                }
                #[cfg(not(coverage))]
                if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                    let got_avx512 = unsafe { scorer.agree_sum_avx512bw(&sketch) };
                    assert_eq!(got_avx512, expected, "AVX-512BW 归约与标量不一致");
                }
            }
        }
    }
}

#[cfg(feature = "research")]
#[test]
fn 切锥正交残差与自适应精排及草图方差稳定验证() {
    let _guard = CONFIGURATION_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    assert_eq!(std::mem::size_of::<PolarNodeMeta>(), 32);
    assert_eq!(std::mem::size_of::<HotSmHeader>(), 64);
    assert_eq!(std::mem::size_of::<HotSmPage>(), 4096);

    let dim = 512usize;
    let n = 1536usize;
    let ids: Vec<u64> = (0..n as u64).collect();
    let cfg = crate::BuildConfig {
        m: 16,
        ef_construction: 96,
        alpha: 1.2,
    };

    // 1. 均匀窄壳分布：验证无监督 Leader-Means 自动抑制伪轴（0 残差轴）且正常检索
    let mut uniform_raw = vec![0.0f32; n * dim];
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    let next_gauss = |st: &mut u64| -> f64 {
        *st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u1 = (((*st >> 11) as f64) + 0.5) / ((1u64 << 53) as f64);
        *st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = (((*st >> 11) as f64) + 0.5) / ((1u64 << 53) as f64);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    for row in 0..n {
        let mut sq = 0.0f64;
        for col in 0..dim {
            let g = next_gauss(&mut state);
            uniform_raw[row * dim + col] = g as f32;
            sq += g * g;
        }
        let depth = 3.90 + 0.20 * ((row % 31) as f64 / 31.0);
        let r = (depth * 0.5).tanh();
        let scale = (r / sq.sqrt().max(1e-12)) as f32;
        for col in 0..dim {
            uniform_raw[row * dim + col] *= scale;
        }
    }
    set_hyperbolic_strategy(HyperbolicStrategy::Full);
    set_geometry_components(GeometryComponents::default());
    set_execution_config(ExecutionConfig::default());
    set_storage_config(StorageConfig::default());
    set_recall_config(RecallConfig::adaptive());

    let mut uni_idx = crate::Index::build_poincare(
        dim,
        &ids,
        &uniform_raw,
        &cfg,
        1.0,
        crate::PoincareNavigation::GeometryBq2,
    )
    .unwrap();
    assert_eq!(
        uni_idx.residual_cone_count(),
        0,
        "均匀各向同性分布不应生成伪残差主轴"
    );
    uni_idx.prepare_colocated_hot_store(true).unwrap();
    set_storage_config(StorageConfig {
        colocated_store: true,
        early_gate: SketchFilter::SymSketch256 { margin: 0.12 },
        header_prefetch_window: 8,
        eager_prefetch_overlap: true,
    });
    let scfg = SearchConfig {
        ef_search: 128,
        top_k: 10,
        rerank_limit: Some(64),
    };
    let uni_res = uni_idx.search(&uniform_raw[0..dim], &scfg).unwrap();
    assert_eq!(uni_res[0].0, 0);

    // 2. 高深窄壳聚簇分布（复现 clustered_narrow 高深共形坍塌机理）：验证残差锥捕获与召回跃升
    let num_clusters = 16usize;
    let mut centers = vec![0.0f64; num_clusters * dim];
    for c in 0..num_clusters {
        let mut sq = 0.0f64;
        for col in 0..dim {
            let g = next_gauss(&mut state);
            centers[c * dim + col] = g;
            sq += g * g;
        }
        let inv = 1.0 / sq.sqrt();
        for col in 0..dim {
            centers[c * dim + col] *= inv;
        }
    }
    let mut clus_raw = vec![0.0f32; n * dim];
    for row in 0..n {
        let cid = row % num_clusters;
        let mut sq = 0.0f64;
        for col in 0..dim {
            let v =
                centers[cid * dim + col] + (0.16 / (dim as f64).sqrt()) * next_gauss(&mut state);
            clus_raw[row * dim + col] = v as f32;
            sq += v * v;
        }
        let depth = 3.90 + 0.20 * (((row * 7) % 29) as f64 / 29.0);
        let r = (depth * 0.5).tanh();
        let scale = (r / sq.sqrt().max(1e-12)) as f32;
        for col in 0..dim {
            clus_raw[row * dim + col] *= scale;
        }
    }
    let mut queries = vec![0.0f32; 24 * dim];
    for q_i in 0..24 {
        let cid = q_i % num_clusters;
        let mut sq = 0.0f64;
        for col in 0..dim {
            let v =
                centers[cid * dim + col] + (0.16 / (dim as f64).sqrt()) * next_gauss(&mut state);
            queries[q_i * dim + col] = v as f32;
            sq += v * v;
        }
        let depth = 3.92 + 0.16 * ((q_i % 11) as f64 / 11.0);
        let r = (depth * 0.5).tanh();
        let scale = (r / sq.sqrt().max(1e-12)) as f32;
        for col in 0..dim {
            queries[q_i * dim + col] *= scale;
        }
    }

    // 基线 A8
    set_storage_config(StorageConfig::default());
    set_recall_config(RecallConfig::default());
    let mut base_idx = crate::Index::build_poincare(
        dim,
        &ids,
        &clus_raw,
        &cfg,
        1.0,
        crate::PoincareNavigation::GeometryBq2,
    )
    .unwrap();
    base_idx.prepare_colocated_hot_store(true).unwrap();

    // 救援模式（A + B + C）
    set_recall_config(RecallConfig::adaptive());
    let mut rescue_idx = crate::Index::build_poincare(
        dim,
        &ids,
        &clus_raw,
        &cfg,
        1.0,
        crate::PoincareNavigation::GeometryBq2,
    )
    .unwrap();
    let num_axes = rescue_idx.residual_cone_count();
    assert!(num_axes >= num_clusters / 2, "残差主轴数量异常: {num_axes}");

    let split_sample = rescue_idx.search(&queries[0..dim], &scfg).unwrap();
    rescue_idx.prepare_colocated_hot_store(true).unwrap();
    set_storage_config(StorageConfig {
        colocated_store: true,
        early_gate: SketchFilter::Off,
        header_prefetch_window: 8,
        eager_prefetch_overlap: true,
    });
    let coloc_sample = rescue_idx.search(&queries[0..dim], &scfg).unwrap();
    assert_eq!(
        split_sample, coloc_sample,
        "残差模式下合表与分离布局结果必须严格一致"
    );

    set_storage_config(StorageConfig {
        colocated_store: true,
        early_gate: SketchFilter::SymSketch256 { margin: 0.12 },
        header_prefetch_window: 8,
        eager_prefetch_overlap: true,
    });
    let eval_scfg = SearchConfig {
        ef_search: 64,
        top_k: 10,
        rerank_limit: Some(32),
    };
    let mut base_hits = 0usize;
    let mut rescue_hits = 0usize;
    for q_i in 0..24 {
        let q = &queries[q_i * dim..(q_i + 1) * dim];
        let truth = base_idx.search_exact(q, 10).unwrap();
        let truth_set: std::collections::HashSet<u64> = truth.iter().map(|x| x.0).collect();

        set_recall_config(RecallConfig::default());
        let rb = base_idx.search(q, &eval_scfg).unwrap();
        base_hits += rb.iter().filter(|x| truth_set.contains(&x.0)).count();

        set_recall_config(RecallConfig::adaptive());
        let rr = rescue_idx.search(q, &eval_scfg).unwrap();
        rescue_hits += rr.iter().filter(|x| truth_set.contains(&x.0)).count();
    }
    assert!(
        rescue_hits > base_hits + 20 && rescue_hits >= 210,
        "切锥正交残差救援应显著提升高深窄壳聚簇召回: base_hits={base_hits}, rescue_hits={rescue_hits}"
    );

    // 验证增量插入兼容性
    rescue_idx.insert(99_999, &queries[0..dim]).unwrap();
    assert_eq!(
        rescue_idx.search(&queries[0..dim], &scfg).unwrap()[0].0,
        99_999
    );

    // 3. 浅层连续模态锥对照（复刻 MERU 类洛伦兹投影特征：mean_norm ≈ 0.78，浅测地深度）：必须零伪轴触发
    let mut modal_raw = vec![0.0f32; n * dim];
    for row in 0..n {
        let mut sq = 0.0f64;
        for col in 0..dim {
            let base_dir = if col == 0 { 1.25 } else { 0.0 };
            let v = base_dir + (0.80 / (dim as f64).sqrt()) * next_gauss(&mut state);
            modal_raw[row * dim + col] = v as f32;
            sq += v * v;
        }
        let depth = 0.16 + 0.35 * ((row % 37) as f64 / 37.0);
        let r = (depth * 0.5).tanh();
        let scale = (r / sq.sqrt().max(1e-12)) as f32;
        for col in 0..dim {
            modal_raw[row * dim + col] *= scale;
        }
    }
    let modal_idx = crate::Index::build_poincare(
        dim,
        &ids,
        &modal_raw,
        &cfg,
        1.0,
        crate::PoincareNavigation::GeometryBq2,
    )
    .unwrap();
    assert_eq!(
        modal_idx.residual_cone_count(),
        0,
        "浅层连续模态锥流形不应生成伪残差主轴"
    );

    // 4. 验证 CJ-Gate 离轴查询一阶误差传播系数严格等于 β_q * β_v（消除 β_v^2 近似的 5.49 倍低估）
    let alpha_q_off = 0.50f64;
    let alpha_v_tight = 0.9875f64;
    let expected_bq_bv =
        ((1.0 - alpha_q_off * alpha_q_off) * (1.0 - alpha_v_tight * alpha_v_tight)).sqrt();
    let exact_off = residual_cosine_std_scale(alpha_q_off, alpha_v_tight, true);
    let legacy_bv2 = 1.0 - alpha_v_tight * alpha_v_tight;
    assert!((exact_off - expected_bq_bv).abs() < 1e-15);
    assert!(((exact_off / legacy_bv2) - 5.4944225575).abs() < 1e-7);

    let mut test_axis = vec![0.0f32; dim];
    test_axis[0] = 1.0;
    let test_dict = ResidualConeDict {
        num_axes: 1,
        dim,
        axes: test_axis,
    };
    let mut test_q = vec![0.0f32; dim];
    test_q[0] = 0.50 * 0.90;
    test_q[1] = (1.0f64 - 0.25f64).sqrt() as f32 * 0.90;
    let mut q_cache = QueryConeCache::new();
    q_cache.begin_query(&test_q, 0.90, true);
    let (_, _, slot_mat) = q_cache.ensure_cone(0, &test_dict);
    assert_ne!(slot_mat, u16::MAX);
    let mut test_meta = PolarNodeMeta {
        radius: 0.96,
        boundary_factor: 12.7,
        depth: 4.0,
        sm_weak: 0.02,
        sm_delta: 0.04,
        sm_sigma: 0.85,
        packed_strong_cone: 0,
        cone_alpha: 0.9875,
    };
    test_meta.set_residual_cone(
        0,
        0.9875,
        SmCodeParams {
            weak: 0.02,
            delta: 0.04,
            sigma: 0.85,
            strong_ones: 200,
        },
    );
    let cached_atten = q_cache.query_residual_atten(Some(&test_dict), &test_q, 0.90, test_meta);
    assert!((cached_atten - exact_off).abs() < 1e-5);

    set_storage_config(StorageConfig::default());
    set_recall_config(RecallConfig::default());
    set_execution_config(ExecutionConfig::default());
}
