use quiver_ann::{
    BuildConfig, Error, Index, Metric, PoincareNavigation, RerankConfig, RerankPrecision,
    RerankRepresentation, ScalarQuantization, SearchConfig, StorageMode,
};

fn configs() -> Vec<RerankConfig> {
    let mut result = Vec::new();
    for representation in [RerankRepresentation::Cartesian, RerankRepresentation::Polar] {
        for precision in [
            RerankPrecision::Fp32,
            RerankPrecision::Fp16,
            RerankPrecision::Bf16,
            RerankPrecision::Sq8,
            RerankPrecision::Sq4,
        ] {
            result.push(RerankConfig {
                precision,
                representation,
                ..RerankConfig::default()
            });
        }
        for precision in [RerankPrecision::Sq8, RerankPrecision::Sq4] {
            for scalar_quantization in [
                ScalarQuantization::Centered,
                // 17 维下 group_size=8 真正覆盖多组含残余尾块（8+8+1），64 覆盖单组路径
                ScalarQuantization::Affine { group_size: 8 },
                ScalarQuantization::Affine { group_size: 64 },
            ] {
                result.push(RerankConfig {
                    precision,
                    representation,
                    scalar_quantization,
                    norm_correction: false,
                });
            }
        }
    }
    for precision in [RerankPrecision::Sq8, RerankPrecision::Sq4] {
        for group_size in [8, 64] {
            result.push(RerankConfig {
                precision,
                scalar_quantization: ScalarQuantization::Affine { group_size },
                norm_correction: true,
                ..RerankConfig::default()
            });
        }
    }
    result
}

fn data(count: usize, dimension: usize) -> Vec<f32> {
    let mut state = 7091u64;
    (0..count * dimension)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((state >> 32) as u32) as f64 / f64::from(u32::MAX) * 0.12 - 0.06) as f32
        })
        .collect()
}

fn build(vectors: &[f32], dimension: usize) -> Index {
    Index::build_poincare(
        dimension,
        &(0..(vectors.len() / dimension) as u64).collect::<Vec<_>>(),
        vectors,
        &BuildConfig::default(),
        1.0,
        PoincareNavigation::GeometryBq2,
    )
    .unwrap()
}

fn search_config() -> SearchConfig {
    SearchConfig {
        top_k: 10,
        ef_search: 256,
        rerank_limit: None,
    }
}

#[test]
fn 所有精度在堆和映射快照重开后保持原向量及实际评分() {
    let vectors = data(80, 17);
    let mut index = build(&vectors, 17);
    index.prepare_colocated_hot_store(true).unwrap();
    let all_ids: Vec<_> = index.ids().collect();
    let query = &vectors[51..68];
    let exact = index.search_exact(query, 10).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("index.pelorus");

    for config in configs() {
        index.set_rerank_config(config).unwrap();
        if config != RerankConfig::default() {
            assert_eq!(index.vector_storage_stats().canonical_heap_bytes, 0);
            assert_eq!(
                index.vector_storage_stats().canonical_mapped_bytes,
                vectors.len() * 4
            );
        }
        let baseline = index.search(query, &search_config()).unwrap();
        let reranked = index.rerank_candidates(query, &all_ids, 10).unwrap();
        assert_eq!(
            baseline, reranked,
            "普通查询与固定候选精排必须使用相同表示，配置 {config:?}"
        );
        index.save_snapshot(&path).unwrap();
        let verify_exact = config == RerankConfig::default()
            || matches!(
                config,
                RerankConfig {
                    precision: RerankPrecision::Sq8,
                    scalar_quantization: ScalarQuantization::Centered,
                    ..
                } | RerankConfig {
                    precision: RerankPrecision::Sq4,
                    scalar_quantization: ScalarQuantization::Affine { group_size: 8 },
                    norm_correction: true,
                    ..
                }
            );
        for mode in [StorageMode::Memory, StorageMode::Mmap] {
            let loaded = Index::open(&path, mode).unwrap();
            assert_eq!(loaded.rerank_config(), config);
            assert_eq!(loaded.storage_mode(), mode);
            assert_eq!(loaded.len(), 80);
            let bytes = loaded.vector_storage_stats();
            if mode == StorageMode::Memory && config != RerankConfig::default() {
                assert_eq!(bytes.canonical_heap_bytes, 0);
                assert_eq!(bytes.canonical_mapped_bytes, vectors.len() * 4);
                assert!(bytes.rerank_heap_bytes > 0);
            } else if mode == StorageMode::Mmap {
                assert_eq!(bytes.canonical_heap_bytes, 0);
                assert_eq!(bytes.canonical_mapped_bytes, vectors.len() * 4);
                assert_eq!(bytes.rerank_heap_bytes, 0);
            }
            assert_eq!(
                loaded.search(query, &search_config()).unwrap(),
                baseline,
                "配置 {config:?}"
            );
            assert_eq!(
                loaded.rerank_candidates(query, &all_ids, 10).unwrap(),
                reranked
            );
            if verify_exact {
                for id in 0..80 {
                    assert_eq!(
                        loaded.vector(id),
                        Some(&vectors[id as usize * 17..(id as usize + 1) * 17])
                    );
                }
                assert_eq!(loaded.search_exact(query, 10).unwrap(), exact);
            }
        }
    }
}

#[test]
fn 映射增量删除标识复用压缩和重复发布保留旧读者() {
    let vectors = data(96, 16);
    let mut index = build(&vectors, 16);
    index.prepare_colocated_hot_store(true).unwrap();
    let codec = RerankConfig {
        precision: RerankPrecision::Sq8,
        scalar_quantization: ScalarQuantization::Centered,
        ..RerankConfig::default()
    };
    index.set_rerank_config(codec).unwrap();
    index.set_storage_mode(StorageMode::Mmap).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("index.pelorus");
    index.save(&path).unwrap();
    let reader = Index::load_mmap(&path).unwrap();
    let query = vectors[..16].to_vec();
    let previous = reader.search(&query, &search_config()).unwrap();
    assert!(index.remove(0));
    let replacement = [0.01; 16];
    index.insert(0, &replacement).unwrap();
    index.insert(1000, &[0.02; 16]).unwrap();
    assert_eq!(index.vector(0), Some(&replacement[..]));
    assert_eq!(
        index.vector_storage_stats().canonical_heap_bytes,
        2 * 16 * 4
    );
    assert_eq!(reader.vector(0), Some(&vectors[..16]));
    index.save(&path).unwrap();
    let newest = Index::load_mmap(&path).unwrap();
    assert_eq!(newest.vector(0), Some(&replacement[..]));
    assert!(newest.contains(1000));
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let reader = &reader;
            let previous = &previous;
            let query = &query;
            scope.spawn(move || {
                for _ in 0..16 {
                    assert_eq!(reader.search(query, &search_config()).unwrap(), *previous);
                }
            });
        }
        index.save(&path).unwrap();
    });
    assert_eq!(index.compact().unwrap(), 1);
    assert_eq!(index.storage_mode(), StorageMode::Mmap);
    assert_eq!(index.rerank_config(), codec);
    assert_eq!(index.vector_storage_stats().canonical_heap_bytes, 0);
    index.save(&path).unwrap();
    let latest = Index::load_mmap(&path).unwrap();
    assert_eq!(latest.len(), 97);
    assert_eq!(latest.vector(0), Some(&replacement[..]));
    assert_eq!(reader.search(&query, &search_config()).unwrap(), previous);
    index.set_storage_mode(StorageMode::Memory).unwrap();
    assert_eq!(index.vector_storage_stats().canonical_heap_bytes, 0);
    assert_eq!(
        index.vector_storage_stats().canonical_mapped_bytes,
        97 * 16 * 4
    );
    assert_eq!(index.vector(0), Some(&replacement[..]));
    index.set_rerank_config(RerankConfig::default()).unwrap();
    assert_eq!(index.vector_storage_stats().canonical_mapped_bytes, 0);
    assert_eq!(
        index.vector_storage_stats().canonical_heap_bytes,
        97 * 16 * 4
    );
}

fn distance(query: &[f32], values: &[f64], curvature: f64) -> f32 {
    let qnorm = query.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
    let norm = values.iter().map(|&v| v * v).sum::<f64>();
    let squared = query
        .iter()
        .zip(values)
        .map(|(&q, &v)| (f64::from(q) - v).powi(2))
        .sum::<f64>();
    -(1.0 + 2.0 * curvature * squared / ((1.0 - curvature * qnorm) * (1.0 - curvature * norm)))
        .acosh() as f32
        / curvature.sqrt() as f32
}

#[test]
fn 实际编码评分与手工标量量化和测地公式一致() {
    let mut index = Index::build_with_metric(
        2,
        &[7, 8],
        &[0.25, 0.125, 0.5, 0.25],
        &BuildConfig::default(),
        Metric::Poincare { curvature: 0.5 },
    )
    .unwrap();
    let query = [0.1, -0.03];
    // SQ4 的 FP32 尺度使 0.0625/scale 略低于 3.5，因此按偶数舍入规则得到码字 3。
    for (precision, limit, code) in [
        (RerankPrecision::Sq8, 127.0f32, 64.0f32),
        (RerankPrecision::Sq4, 7.0, 3.0),
    ] {
        index
            .set_rerank_config(RerankConfig {
                precision,
                scalar_quantization: ScalarQuantization::Centered,
                ..RerankConfig::default()
            })
            .unwrap();
        let scale = 0.125f32 / limit;
        let first = [
            f64::from(-limit * scale + 0.375),
            f64::from(-code * scale + 0.1875),
        ];
        let actual = index.rerank_candidates(&query, &[7, 7, 999], 1).unwrap();
        assert_eq!(actual[0].0, 7);
        assert!((actual[0].1 - distance(&query, &first, 0.5)).abs() < 2e-6);
    }
    let mut single = Index::build_with_metric(
        3,
        &[1],
        &[0.125, 0.25, 0.5],
        &BuildConfig::default(),
        Metric::Poincare { curvature: 1.0 },
    )
    .unwrap();
    single
        .set_rerank_config(RerankConfig {
            precision: RerankPrecision::Sq4,
            scalar_quantization: ScalarQuantization::Affine { group_size: 2 },
            ..RerankConfig::default()
        })
        .unwrap();
    let actual = single.rerank_candidates(&[0.0; 3], &[1], 1).unwrap();
    assert!((actual[0].1 - distance(&[0.0; 3], &[0.125, 0.25, 0.5], 1.0)).abs() < 2e-6);
    assert_eq!(single.vector_storage_stats().rerank_record_bytes, 18);
}

#[test]
fn 编码失败与非法配置不改变索引且近边界极坐标可用() {
    let mut index = Index::build_with_metric(
        1,
        &[1],
        &[0.9999],
        &BuildConfig::default(),
        Metric::Poincare { curvature: 1.0 },
    )
    .unwrap();
    let before = index.search(&[0.5], &SearchConfig::default()).unwrap();
    assert!(matches!(
        index.set_rerank_config(RerankConfig {
            precision: RerankPrecision::Fp16,
            ..RerankConfig::default()
        }),
        Err(Error::OutsideManifold { .. })
    ));
    assert_eq!(index.rerank_config(), RerankConfig::default());
    assert_eq!(
        index.search(&[0.5], &SearchConfig::default()).unwrap(),
        before
    );
    index
        .set_rerank_config(RerankConfig {
            precision: RerankPrecision::Fp16,
            representation: RerankRepresentation::Polar,
            ..RerankConfig::default()
        })
        .unwrap();
    assert_eq!(
        index.search(&[0.5], &SearchConfig::default()).unwrap(),
        before
    );
    let invalid = RerankConfig {
        precision: RerankPrecision::Sq8,
        scalar_quantization: ScalarQuantization::Affine { group_size: 0 },
        ..RerankConfig::default()
    };
    assert!(matches!(
        index.set_rerank_config(invalid),
        Err(Error::InvalidRerankConfig(_))
    ));
    let previous = index.rerank_config();
    assert!(index.insert(2, &[1.0]).is_err());
    assert_eq!(index.len(), 1);
    assert_eq!(index.rerank_config(), previous);
}

#[test]
fn 精确导航也使用所选重排表示且笛卡尔舍入不会被原径向下界误删() {
    let vectors = (0..80)
        .map(|index| 0.705 + index as f32 * 0.0001)
        .collect::<Vec<_>>();
    let ids = (0..80).collect::<Vec<u64>>();
    let mut index = Index::build_with_metric(
        1,
        &ids,
        &vectors,
        &BuildConfig::default(),
        Metric::Poincare { curvature: 1.0 },
    )
    .unwrap();
    index
        .set_rerank_config(RerankConfig {
            precision: RerankPrecision::Fp16,
            ..RerankConfig::default()
        })
        .unwrap();
    let expected = index.rerank_candidates(&[0.7091], &ids, 10).unwrap();
    assert_eq!(index.search(&[0.7091], &search_config()).unwrap(), expected);
}

#[test]
fn 墓碑不阻止更换精度且插入沿用冻结中心() {
    let mut index = Index::build_with_metric(
        1,
        &[1, 2],
        &[0.9999, 0.5],
        &BuildConfig::default(),
        Metric::Poincare { curvature: 1.0 },
    )
    .unwrap();
    assert!(index.remove(1));
    index
        .set_rerank_config(RerankConfig {
            precision: RerankPrecision::Fp16,
            ..RerankConfig::default()
        })
        .unwrap();
    assert_eq!(index.rerank_candidates(&[0.5], &[1, 2], 1).unwrap()[0].0, 2);
    let mut index = Index::build_with_metric(
        2,
        &[7, 8],
        &[0.25, 0.125, 0.5, 0.25],
        &BuildConfig::default(),
        Metric::Poincare { curvature: 1.0 },
    )
    .unwrap();
    index
        .set_rerank_config(RerankConfig {
            precision: RerankPrecision::Sq8,
            scalar_quantization: ScalarQuantization::Centered,
            ..RerankConfig::default()
        })
        .unwrap();
    let previous = index.rerank_candidates(&[0.2, 0.1], &[7, 8], 2).unwrap();
    index.insert(9, &[0.7, 0.1]).unwrap();
    assert_eq!(
        index.rerank_candidates(&[0.2, 0.1], &[7, 8], 2).unwrap(),
        previous
    );
}

#[test]
fn 空映射索引可保存重开插入且余弦精度入口可用() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty.pelorus");
    let mut index = Index::build(3, &[], &[], &BuildConfig::default()).unwrap();
    index
        .set_rerank_config(RerankConfig {
            precision: RerankPrecision::Sq8,
            scalar_quantization: ScalarQuantization::Centered,
            ..RerankConfig::default()
        })
        .unwrap();
    index.set_storage_mode(StorageMode::Mmap).unwrap();
    index.save(&path).unwrap();
    let mut loaded = Index::load_mmap(&path).unwrap();
    loaded.insert(9, &[0.1, 0.2, 0.3]).unwrap();
    assert_eq!(
        loaded
            .search(
                &[0.1, 0.2, 0.3],
                &SearchConfig {
                    top_k: 1,
                    ..SearchConfig::default()
                }
            )
            .unwrap()[0]
            .0,
        9
    );
    loaded.save(&path).unwrap();
    assert_eq!(
        Index::load(&path).unwrap().vector(9),
        Some(&[0.1, 0.2, 0.3][..])
    );
}

#[test]
fn 快照拒绝损坏和截断且失败保存后仍可读取() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("index.pelorus");
    let index = build(&data(32, 7), 7);
    index.save_snapshot(&path).unwrap();
    let before = index.search(&[0.0; 7], &search_config()).unwrap();
    assert!(index.save_snapshot(directory.path()).is_err());
    assert_eq!(index.search(&[0.0; 7], &search_config()).unwrap(), before);
    let entry = std::fs::read(&path).unwrap();
    let name_len = u64::from_le_bytes(entry[12..20].try_into().unwrap()) as usize;
    let name = std::str::from_utf8(&entry[20..20 + name_len]).unwrap();
    let image_path = directory.path().join(name);
    let original = std::fs::read(&image_path).unwrap();
    let mut corrupted = original.clone();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 1;
    std::fs::write(&image_path, &corrupted).unwrap();
    assert!(matches!(
        Index::load_mmap(&path),
        Err(Error::CorruptIndex(_))
    ));
    std::fs::write(&image_path, &original[..original.len() / 2]).unwrap();
    assert!(matches!(
        Index::load_mmap(&path),
        Err(Error::CorruptIndex(_))
    ));
    std::fs::write(&image_path, &original).unwrap();
    assert_eq!(
        Index::load_mmap(&path)
            .unwrap()
            .search(&[0.0; 7], &search_config())
            .unwrap(),
        before
    );
}

#[test]
fn 历史单文件可映射迁移并在追加后发布完整快照() {
    let directory = tempfile::tempdir().unwrap();
    for metric in [Metric::Cosine, Metric::Poincare { curvature: 0.5 }] {
        let path = directory.path().join("legacy.quiver");
        let mut source = Index::build_with_metric(
            3,
            &[10, 20],
            &[0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
            &BuildConfig::default(),
            metric,
        )
        .unwrap();
        source.save(&path).unwrap();
        let mut mapped = Index::load_mmap(&path).unwrap();
        assert_eq!(mapped.storage_mode(), StorageMode::Mmap);
        assert_eq!(mapped.vector_storage_stats().canonical_heap_bytes, 0);
        assert_eq!(
            mapped
                .search(&[0.1, 0.2, 0.3], &SearchConfig::default())
                .unwrap(),
            source
                .search(&[0.1, 0.2, 0.3], &SearchConfig::default())
                .unwrap()
        );
        mapped.insert(30, &[0.2, 0.2, 0.2]).unwrap();
        source.insert(30, &[0.2, 0.2, 0.2]).unwrap();
        mapped.save(&path).unwrap();
        let restored = Index::load_mmap(&path).unwrap();
        assert_eq!(
            restored.search_exact(&[0.2, 0.2, 0.2], 3).unwrap(),
            source.search_exact(&[0.2, 0.2, 0.2], 3).unwrap()
        );
    }
}

#[cfg(target_os = "linux")]
fn linux_ssd_test_config() -> quiver_ann::SsdCacheConfig {
    // CI 等非特权 Linux 环境默认 RLIMIT_MEMLOCK 通常为 8 MiB，测试使用页级小预算覆盖真实 io_uring 路径
    quiver_ann::SsdCacheConfig {
        cache_bytes: 16 * 4096,
        inflight_bytes: 4 * 4096,
        io_workers: 2,
        prefetch_window: 16,
        ..quiver_ann::SsdCacheConfig::default()
    }
}

#[cfg(target_os = "linux")]
#[test]
fn 原始精度直接读取的增量保存直接复用原始段() {
    let vectors = data(80, 17);
    let mut original = build(&vectors, 17);
    original.prepare_colocated_hot_store(true).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fp32.pelorus");
    original.save_snapshot(&path).unwrap();
    let config = linux_ssd_test_config();
    let mut ssd = Index::open_ssd(&path, config).unwrap();
    let query = &vectors[..17];
    assert_eq!(
        ssd.search(query, &search_config()).unwrap(),
        original.search(query, &search_config()).unwrap()
    );
    original.insert(1000, &[0.01; 17]).unwrap();
    ssd.insert(1000, &[0.01; 17]).unwrap();
    ssd.save_snapshot(&path).unwrap();
    let mapped = Index::load_mmap(&path).unwrap();
    assert_eq!(mapped.vector_storage_stats().rerank_mapped_bytes, 0);
    assert_eq!(mapped.rerank_config(), RerankConfig::default());
    assert_eq!(
        mapped.search(query, &search_config()).unwrap(),
        original.search(query, &search_config()).unwrap()
    );
    let restored = Index::open_ssd(&path, config).unwrap();
    assert_eq!(
        restored.search(query, &search_config()).unwrap(),
        original.search(query, &search_config()).unwrap()
    );
}

#[cfg(target_os = "linux")]
#[test]
fn 直接读取缓存的普通查询命中扩容与增量保存均保留结果() {
    use quiver_ann::SsdCacheConfig;
    let vectors = data(80, 17);
    let mut source = build(&vectors, 17);
    source.prepare_colocated_hot_store(true).unwrap();
    source
        .set_rerank_config(RerankConfig {
            precision: RerankPrecision::Fp16,
            ..RerankConfig::default()
        })
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cached.pelorus");
    source.save_snapshot(&path).unwrap();
    let config = linux_ssd_test_config();
    let mut ssd = Index::open_ssd(&path, config).unwrap();
    let expected = source.search(&vectors[..17], &search_config()).unwrap();
    for _ in 0..2 {
        assert_eq!(
            ssd.search(&vectors[..17], &search_config()).unwrap(),
            expected
        );
    }
    assert!(ssd.ssd_io_stats().unwrap().read_bytes > 0);
    assert!(ssd.ssd_io_stats().unwrap().cache_hits > 0);
    assert!(ssd.ssd_io_stats().unwrap().cache_peak_bytes <= config.cache_bytes);
    ssd.clear_ssd_cache();
    assert_eq!(ssd.ssd_io_stats().unwrap().cache_current_bytes, 0);
    let expanded_bytes = 64 * 4096;
    ssd.set_ssd_cache_config(SsdCacheConfig {
        cache_bytes: expanded_bytes,
        ..config
    })
    .unwrap();
    assert_eq!(
        ssd.ssd_io_stats().unwrap().cache_capacity_bytes,
        expanded_bytes
    );
    assert_eq!(
        ssd.search(&vectors[..17], &search_config()).unwrap(),
        expected
    );
    ssd.set_ssd_cache_config(SsdCacheConfig {
        prefetch_window: 0,
        ..config
    })
    .unwrap();
    assert_eq!(
        ssd.search(&vectors[..17], &search_config()).unwrap(),
        expected
    );
    assert_eq!(ssd.ssd_io_stats().unwrap().prefetch_requests, 0);
    ssd.insert(1000, &[0.01; 17]).unwrap();
    ssd.save(&path).unwrap();
    assert_eq!(
        Index::open_ssd(&path, config)
            .unwrap()
            .search(&[0.01; 17], &search_config())
            .unwrap(),
        ssd.search(&[0.01; 17], &search_config()).unwrap()
    );
    ssd.set_storage_mode(StorageMode::Memory).unwrap();
    assert_eq!(ssd.ssd_cache_config(), None);
    assert_eq!(ssd.ssd_selective_gating(), None);
}

#[cfg(target_os = "linux")]
#[test]
fn 选择性门控参数校验并在多页直读上削减尾页读取且保持召回() {
    use quiver_ann::SsdCacheConfig;
    let dim = 512;
    let n = 400;
    let vectors = data(n, dim);
    let mut index = build(&vectors, dim);
    index.prepare_colocated_hot_store(true).unwrap();
    // 使用 Mmap 状态设置门控须报错
    assert!(index.set_ssd_selective_gating(Some(2.5)).is_err());

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gated.pelorus");
    index.save_snapshot(&path).unwrap();

    // 768 KiB 远低于非特权 Linux 8 MiB RLIMIT_MEMLOCK 上限，同时提供 >= 64 页批量映射空间
    let cfg = SsdCacheConfig {
        cache_bytes: 128 * 4096,
        inflight_bytes: 64 * 4096,
        io_workers: 1,
        prefetch_window: 32,
        ..SsdCacheConfig::default()
    };
    let mut ssd = Index::open_ssd(&path, cfg).unwrap();
    assert_eq!(ssd.ssd_selective_gating(), None);
    assert!(ssd.set_ssd_selective_gating(Some(0.0)).is_err());
    assert!(ssd.set_ssd_selective_gating(Some(-1.0)).is_err());
    assert!(ssd.set_ssd_selective_gating(Some(f64::NAN)).is_err());

    let query = &vectors[..dim];
    let sc = SearchConfig {
        top_k: 10,
        ef_search: 64,
        rerank_limit: Some(64),
    };
    let exact_before = ssd.ssd_io_stats().unwrap().read_calls;
    let exact_res = ssd.search(query, &sc).unwrap();
    ssd.drain_ssd_reads();
    let exact_delta = ssd.ssd_io_stats().unwrap().read_calls - exact_before;

    ssd.clear_ssd_cache();
    ssd.set_ssd_selective_gating(Some(2.5)).unwrap();
    assert_eq!(ssd.ssd_selective_gating(), Some(2.5));
    let gated_before = ssd.ssd_io_stats().unwrap().read_calls;
    let gated_res = ssd.search(query, &sc).unwrap();
    ssd.drain_ssd_reads();
    let gated_delta = ssd.ssd_io_stats().unwrap().read_calls - gated_before;

    assert!(exact_delta > 0 && gated_delta > 0);
    assert!(
        gated_delta <= exact_delta,
        "gated_delta={gated_delta} 不应超过 exact_delta={exact_delta}"
    );
    let exact_ids: std::collections::HashSet<u64> = exact_res.iter().map(|r| r.0).collect();
    let overlap = gated_res
        .iter()
        .filter(|r| exact_ids.contains(&r.0))
        .count();
    assert!(
        overlap >= 9,
        "选择性门控至少保留 9/10 无损重排结果，实际重合={overlap}"
    );

    ssd.set_ssd_selective_gating(None).unwrap();
    assert_eq!(ssd.ssd_selective_gating(), None);
    assert_eq!(ssd.search(query, &sc).unwrap(), exact_res);
}
