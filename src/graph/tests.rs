//! 图索引核心功能回归。

use super::*;

/// 构造 N 个正交基向量（dim 维），每个向量只在第 i%dim 维为 1.0
#[allow(dead_code)] // 测试辅助函数，保留备用
fn make_orthogonal_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut vecs = vec![0.0f32; n * dim];
    for i in 0..n {
        vecs[i * dim + (i % dim)] = 1.0;
    }
    vecs
}

/// 构造随机向量（确定性 LCG）
fn make_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut vecs = vec![0.0f32; n * dim];
    let mut lcg = seed;
    for v in vecs.iter_mut() {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = ((lcg >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0;
    }
    vecs
}

#[derive(Default)]
struct RecordingBqScorer {
    visited_ids: Vec<u64>,
}

impl NavigationScorer for RecordingBqScorer {
    fn score(&mut self, node_id: u64, bq_distance: u32) -> u32 {
        self.visited_ids.push(node_id);
        bq_distance
    }
}

fn legacy_beam_search_l0(
    index: &QuIVer,
    q_sig: &Bq2Signature,
    entry: u32,
    ef: usize,
) -> (Vec<(NonNanF32, u32)>, Vec<u64>) {
    let mut visited = Bitset::new(index.n);
    visited.clear();
    let mut candidates = BinaryHeap::with_capacity(ef * 2);
    let mut results = BinaryHeap::with_capacity(ef + 1);
    let distance = |node: u32| {
        index
            .bq_store
            .distance_to_sig_cheap(node as usize, q_sig, index.dim)
    };
    let mut sequence = vec![index.ids[entry as usize]];
    let initial = NonNanF32(distance(entry) as f32);
    visited.set(entry as usize);
    candidates.push(Reverse((initial, entry)));
    results.push((initial, entry));
    while let Some(Reverse((candidate_distance, current))) = candidates.pop() {
        if results.len() >= ef && candidate_distance > results.peek().unwrap().0 {
            break;
        }
        for &neighbor in index.layer0.neighbors(current) {
            if visited.test(neighbor as usize) {
                continue;
            }
            visited.set(neighbor as usize);
            sequence.push(index.ids[neighbor as usize]);
            let neighbor_distance = NonNanF32(distance(neighbor) as f32);
            if results.len() < ef || neighbor_distance < results.peek().unwrap().0 {
                candidates.push(Reverse((neighbor_distance, neighbor)));
                results.push((neighbor_distance, neighbor));
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }
    let mut output = results.into_vec();
    for item in &mut output {
        item.0 = NonNanF32(
            index
                .bq_store
                .distance_to_sig(item.1 as usize, q_sig, index.dim) as f32,
        );
    }
    output.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    (output, sequence)
}

#[test]
fn bq_有界预筛与全量排序结果逐项一致() {
    for dim in [64, 400, 768] {
        let n = 257;
        let vectors = make_random_vectors(n, dim, 0xB0_0DED + dim as u64);
        let ids = (20_000..20_000 + n as u64).collect::<Vec<_>>();
        let slots = (0..n).collect::<Vec<_>>();
        let mut index = QuIVer::new(dim, &BuildConfig::default());
        index.batch_build(&vectors, &ids, &slots);
        index.soft_delete(ids[31]);

        let query = &vectors[17 * dim..18 * dim];
        let signature = Bq2Signature::from_vector(query);
        let candidates = ids
            .iter()
            .rev()
            .copied()
            .chain([u64::MAX])
            .collect::<Vec<_>>();
        let mut expected = candidates
            .iter()
            .filter_map(|id| {
                let internal = *index.id_to_internal.get(id)? as usize;
                (!index.tombstones[internal]).then(|| {
                    (
                        *id,
                        index
                            .bq_store
                            .distance_to_sig(internal, &signature, index.dim),
                    )
                })
            })
            .collect::<Vec<_>>();
        expected.sort_unstable_by(|left, right| {
            left.1.cmp(&right.1).then_with(|| left.0.cmp(&right.0))
        });

        for limit in [1, 7, 64, n, n + 20] {
            let mut wanted = expected.clone();
            wanted.truncate(limit);
            assert_eq!(
                index.prefilter_candidates(query, &candidates, limit),
                wanted
            );
        }
    }
}

#[test]
fn bq_internal_slot_cache_严格遵守字节预算并按最近使用淘汰() {
    let posting = |start: u64| {
        Arc::new(
            (0..4)
                .map(|offset| (start + offset, offset as u32))
                .collect::<Vec<_>>(),
        )
    };
    let entry_bytes = 4 * std::mem::size_of::<(u64, u32)>();
    let mut cache = PrefilterSlotCacheState {
        max_bytes: entry_bytes * 2,
        ..PrefilterSlotCacheState::default()
    };
    cache.insert((1, 1), posting(10));
    cache.insert((2, 1), posting(20));
    assert!(cache.get((1, 1)).is_some());
    cache.insert((3, 1), posting(30));
    assert!(cache.bytes <= cache.max_bytes);
    assert!(cache.get((1, 1)).is_some());
    assert!(cache.get((2, 1)).is_none());
    assert!(cache.get((3, 1)).is_some());

    let oversized = Arc::new(vec![(1, 1); 9]);
    cache.insert((4, 1), oversized);
    assert!(cache.get((4, 1)).is_none());
    assert!(cache.bytes <= cache.max_bytes);
}

#[test]
fn bq_internal_slot_posting_命中缓存且增量变更后安全失效() {
    let dim = 64;
    let vectors = make_random_vectors(24, dim, 0x5107_CA5E);
    let ids = (500..524).collect::<Vec<_>>();
    let slots = (0..24).collect::<Vec<_>>();
    let mut index = QuIVer::new(dim, &BuildConfig::default());
    index.batch_build(&vectors, &ids, &slots);
    let query = &vectors[..dim];

    let (cold, cold_metrics) = index.prefilter_candidates_profiled(query, &ids, 11, 1, 8);
    let (warm, warm_metrics) = index.prefilter_candidates_profiled(query, &ids, 11, 1, 8);
    assert_eq!(cold, warm);
    assert!(!cold_metrics.cache_hit);
    assert!(warm_metrics.cache_hit);
    assert_eq!(warm_metrics.mapped_candidates, ids.len());

    index.soft_delete(ids[3]);
    let (after_delete, delete_metrics) = index.prefilter_candidates_profiled(query, &ids, 11, 2, 8);
    assert!(!delete_metrics.cache_hit);
    assert_eq!(delete_metrics.mapped_candidates, ids.len() - 1);
    assert!(after_delete.iter().all(|(id, _)| *id != ids[3]));

    let new_id = 900;
    let mut lcg = 7;
    index.insert(&vectors[dim..dim * 2], new_id, 24, &mut lcg);
    let mut extended = ids.clone();
    extended.push(new_id);
    let (_, insert_metrics) = index.prefilter_candidates_profiled(query, &extended, 11, 3, 8);
    assert!(!insert_metrics.cache_hit);
    assert_eq!(insert_metrics.mapped_candidates, ids.len());
}

#[test]
fn bq_有界预筛边界输入安全且排序确定() {
    let dim = 64;
    let vectors = make_random_vectors(8, dim, 0xB0_0E00);
    let ids = (100..108).collect::<Vec<_>>();
    let slots = (0..8).collect::<Vec<_>>();
    let mut index = QuIVer::new(dim, &BuildConfig::default());
    index.batch_build(&vectors, &ids, &slots);
    let query = &vectors[..dim];

    assert!(index.prefilter_candidates(query, &ids, 0).is_empty());
    assert!(
        index
            .prefilter_candidates(&query[..dim - 1], &ids, 4)
            .is_empty()
    );
    let first = index.prefilter_candidates(query, &ids, 4);
    let second = index.prefilter_candidates(query, &ids, 4);
    assert_eq!(first, second);
    assert!(first.windows(2).all(|pair| {
        pair[0].1 < pair[1].1 || (pair[0].1 == pair[1].1 && pair[0].0 < pair[1].0)
    }));
}

#[test]
fn 默认量化评分器与参考路径访问序列及候选一致() {
    for dim in [64, 384, 768, 1536, 3072] {
        let n = 96;
        let vectors = make_random_vectors(n, dim, 0xC1A0_0000 + dim as u64);
        let ids = (10_000..10_000 + n as u64).collect::<Vec<_>>();
        let slots = (0..n).collect::<Vec<_>>();
        let mut index = QuIVer::new(
            dim,
            &BuildConfig {
                m: 12,
                ef_construction: 32,
                alpha: 1.2,
            },
        );
        index.batch_build(&vectors, &ids, &slots);
        let query = &vectors[17 * dim..18 * dim];
        let query_signature = Bq2Signature::from_vector(query);
        let (legacy, legacy_sequence) =
            legacy_beam_search_l0(&index, &query_signature, index.entry_point, 48);
        let mut visited = Bitset::new(n);
        let mut scorer = RecordingBqScorer::default();
        let scored = index.beam_search_l0_impl::<false, Bitset, RecordingBqScorer>(
            &query_signature,
            index.entry_point,
            48,
            &mut visited,
            &mut scorer,
        );
        assert_eq!(scorer.visited_ids, legacy_sequence, "维度 {dim}");
        assert_eq!(scored, legacy, "维度 {dim}");
    }
}

#[test]
fn 公开余弦查询与量化评分器结果一致() {
    for dim in [64, 384, 768, 1536, 3072] {
        let n = 128;
        let vectors = make_random_vectors(n, dim, 0xC1A1_0000 + dim as u64);
        let ids = (20_000..20_000 + n as u64).collect::<Vec<_>>();
        let slots = (0..n).collect::<Vec<_>>();
        let mut index = QuIVer::new(
            dim,
            &BuildConfig {
                m: 12,
                ef_construction: 32,
                alpha: 1.2,
            },
        );
        index.batch_build(&vectors, &ids, &slots);
        let query = &vectors[29 * dim..30 * dim];
        let config = SearchConfig {
            top_k: 20,
            ef_search: 64,
            rerank_limit: Some(48),
        };
        let get_vector = |slot: usize, buffer: &mut Vec<f32>| {
            buffer.clear();
            buffer.extend_from_slice(&vectors[slot * dim..(slot + 1) * dim]);
            true
        };
        let legacy = index.search(query, get_vector, &config);
        let mut scorer = BqNavigationScorer;
        let scored = index.search_with_scorer(query, get_vector, &config, &mut scorer);
        assert_eq!(scored, legacy, "维度 {dim}");
    }
}

#[test]
fn sparse_bitset跨查询不串扰并支持扩容() {
    let mut visited = SparseBitset::new(2);
    visited.begin();
    visited.set(0);
    assert!(visited.test(0));
    assert!(!visited.test(1));
    visited.begin();
    assert!(!visited.test(0));
    visited.set(1);
    visited.grow(1000);
    assert!(visited.test(1));
    assert!(!visited.test(999));
    visited.set(999);
    assert!(visited.test(999));
}

#[test]
fn dim384专用搜索与通用路径结果一致() {
    let dim = 384;
    let n = 256;
    let vectors = make_random_vectors(n, dim, 0xD1B5_4A32_D192_ED03);
    let ids = (0..n as u64).collect::<Vec<_>>();
    let slots = (0..n).collect::<Vec<_>>();
    let mut index = QuIVer::new(
        dim,
        &BuildConfig {
            m: 16,
            ef_construction: 64,
            alpha: 1.2,
        },
    );
    index.batch_build(&vectors, &ids, &slots);
    let config = SearchConfig {
        top_k: 10,
        ef_search: 64,
        rerank_limit: None,
    };
    let query = &vectors[17 * dim..18 * dim];
    let specialized = index.search_flat(query, &vectors, &config);
    let q_sig = Bq2Signature::from_vector(query);
    let mut visited = Bitset::new(n);
    let mut scorer = BqNavigationScorer;
    let candidates = index.beam_search_l0_impl::<false, Bitset, BqNavigationScorer>(
        &q_sig,
        index.entry_point,
        config.ef_search,
        &mut visited,
        &mut scorer,
    );
    let mut generic = candidates
        .iter()
        .take(config.rerank_limit())
        .map(|&(_, node)| {
            let slot = index.slot_indices[node as usize];
            let vector = &vectors[slot * dim..(slot + 1) * dim];
            (
                index.ids[node as usize],
                cosine_similarity_f32(query, vector),
            )
        })
        .collect::<Vec<_>>();
    generic.sort_unstable_by(|left, right| right.1.partial_cmp(&left.1).unwrap());
    generic.truncate(config.top_k);
    assert_eq!(
        specialized.iter().map(|hit| hit.0).collect::<Vec<_>>(),
        generic.iter().map(|hit| hit.0).collect::<Vec<_>>()
    );
}

#[test]
fn test_quiver_基础建图与搜索() {
    let config = BuildConfig {
        m: 16,
        ef_construction: 32,
        alpha: 1.2,
    };
    let mut quiver = QuIVer::new(4, &config);

    let vectors = vec![
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let ids = vec![10, 20, 30, 40];
    let slots = vec![0, 1, 2, 3];

    quiver.build_parallel(&vectors, &ids, &slots);

    assert_eq!(quiver.active_count(), 4);
    assert_eq!(quiver.total_count(), 4);

    let query = vec![1.0f32, 0.0, 0.0, 0.0];
    let search_config = SearchConfig {
        top_k: 2,
        ef_search: 10,
        rerank_limit: None,
    };

    let results = quiver.search_flat(&query, &vectors, &search_config);
    assert!(!results.is_empty());
    assert_eq!(results[0].0, 10);

    // 测试软删除
    quiver.soft_delete(10);
    assert_eq!(quiver.active_count(), 3);

    let results_after = quiver.search_flat(&query, &vectors, &search_config);
    if !results_after.is_empty() {
        assert_ne!(results_after[0].0, 10);
    }

    let stats = quiver.stats();
    assert_eq!(stats.n, 4);
    assert_eq!(stats.tombstone_count, 1);

    assert!(!quiver.needs_rebuild());
}

#[test]
fn test_quiver_多路径建图与序列化() {
    let config = BuildConfig {
        m: 16,
        ef_construction: 32,
        alpha: 1.2,
    };
    let mut quiver = QuIVer::new(4, &config);

    let vectors = vec![
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let ids = vec![10, 20, 30, 40];
    let slots = vec![0, 1, 2, 3];

    // 测试批量建图
    quiver.batch_build(&vectors, &ids, &slots);
    assert_eq!(quiver.total_count(), 4);

    // 测试追加式批量建图
    quiver.build_parallel_eager(&vectors, &[50, 60, 70, 80], &[4, 5, 6, 7]);
    assert_eq!(quiver.total_count(), 8);

    // 测试带校验的批量建图（会重置图）
    quiver.build_parallel_checked(&vectors, &ids, &slots);
    assert_eq!(quiver.total_count(), 4);

    // 测试单节点增量插入
    let mut lcg = 12345;
    let v5 = vec![0.5, 0.5, 0.0, 0.0];
    quiver.insert(&v5, 50, 4, &mut lcg);
    assert_eq!(quiver.total_count(), 5);

    quiver.dirty_count_inc();
    quiver.dirty_count_inc();
    assert!(quiver.needs_rebuild());

    quiver.debug_connectivity();

    // 测试序列化与反序列化
    let path = std::path::Path::new("test_quiver_ser.quiv");
    quiver.save_to_file(path).unwrap();

    let loaded = QuIVer::load_from_file(path).unwrap();
    assert_eq!(loaded.total_count(), 5);
    assert_eq!(loaded.active_count(), 5);
    assert_eq!(loaded.dim, 4);
    assert_eq!(loaded.entry_point, quiver.entry_point);
    assert_eq!(loaded.ids, vec![10, 20, 30, 40, 50]);

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_quiver_中等规模搜索() {
    // 50 个节点，8 维，测试更完整的 beam search 路径
    let dim = 8;
    let n = 50;
    let config = BuildConfig {
        m: 8,
        ef_construction: 32,
        alpha: 1.2,
    };
    let mut quiver = QuIVer::new(dim, &config);

    let vectors = make_random_vectors(n, dim, 42);
    let ids: Vec<u64> = (0..n as u64).collect();
    let slots: Vec<usize> = (0..n).collect();

    quiver.build_parallel(&vectors, &ids, &slots);
    assert_eq!(quiver.total_count(), n);

    let stats = quiver.stats();
    assert!(stats.avg_degree_l0 > 0.0);
    assert!(stats.hot_bytes > 0);

    // 用第 0 号向量自身做查询，应该能搜到自己
    let query = vectors[0..dim].to_vec();
    let search_config = SearchConfig {
        top_k: 5,
        ef_search: 32,
        rerank_limit: Some(20),
    };
    let results = quiver.search_flat(&query, &vectors, &search_config);
    assert!(!results.is_empty());
    assert_eq!(results[0].0, 0); // 自身应为 Top-1

    // 删除一半节点再搜索
    for i in 0..n / 2 {
        quiver.soft_delete(i as u64);
    }
    assert_eq!(quiver.active_count(), n - n / 2);
    let results2 = quiver.search_flat(&query, &vectors, &search_config);
    for (id, _) in &results2 {
        assert!(*id >= n as u64 / 2, "已删除节点不应出现在结果中");
    }
}

#[test]
fn test_quiver_空图搜索() {
    let config = BuildConfig::default();
    let quiver = QuIVer::new(4, &config);
    assert_eq!(quiver.total_count(), 0);
    assert_eq!(quiver.active_count(), 0);

    let query = vec![1.0f32, 0.0, 0.0, 0.0];
    let search_config = SearchConfig {
        top_k: 5,
        ef_search: 10,
        rerank_limit: None,
    };
    let results = quiver.search_flat(&query, &[], &search_config);
    assert!(results.is_empty());
}

#[test]
fn test_quiver_单节点图() {
    let config = BuildConfig::default();
    let mut quiver = QuIVer::new(4, &config);

    let vectors = vec![1.0f32, 0.0, 0.0, 0.0];
    quiver.build_parallel(&vectors, &[42], &[0]);
    assert_eq!(quiver.total_count(), 1);

    let search_config = SearchConfig {
        top_k: 1,
        ef_search: 10,
        rerank_limit: None,
    };
    let results = quiver.search_flat(&[1.0, 0.0, 0.0, 0.0], &vectors, &search_config);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 42);
}

#[test]
fn test_quiver_soft_delete_不存在的id() {
    let config = BuildConfig::default();
    let mut quiver = QuIVer::new(4, &config);
    let vectors = vec![1.0, 0.0, 0.0, 0.0];
    quiver.build_parallel(&vectors, &[1], &[0]);

    // 删除不存在的 ID
    assert!(!quiver.soft_delete(999));
    // 删除存在的 ID
    assert!(quiver.soft_delete(1));
    // 重复删除（已经是 tombstone）
    assert!(quiver.soft_delete(1));
}

#[test]
fn test_quiver_序列化错误路径() {
    // 加载不存在的文件
    let result = QuIVer::load_from_file(std::path::Path::new("nonexistent.quiv"));
    assert!(result.is_err());
}

#[test]
fn sidecar入口节点越界时安全拒绝() {
    let dir = std::env::temp_dir().join("quiver_invalid_entry");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("index.quiver");
    let vectors = make_orthogonal_vectors(4, 4);
    let ids = vec![1, 2, 3, 4];
    let slots = vec![0, 1, 2, 3];
    let mut index = QuIVer::new(4, &BuildConfig::default());
    index.batch_build(&vectors, &ids, &slots);
    index.save_to_file(&path).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[32..36].copy_from_slice(&u32::MAX.to_le_bytes());
    let result = std::panic::catch_unwind(|| QuIVer::load_from_bytes(&bytes));
    assert!(result.is_ok());
    assert!(result.unwrap().is_err());
    std::fs::remove_file(path).ok();
}

#[test]
fn sidecar畸形计数不得触发panic或巨量分配() {
    let mut bytes = vec![0u8; 64];
    bytes[0..4].copy_from_slice(QuIVer::QUIVER_MAGIC);
    bytes[4..8].copy_from_slice(&QuIVer::QUIVER_VERSION.to_le_bytes());
    bytes[8..12].copy_from_slice(&4u32.to_le_bytes());
    bytes[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    bytes[16..20].copy_from_slice(&16u32.to_le_bytes());
    bytes[20..24].copy_from_slice(&32u32.to_le_bytes());
    bytes[24..28].copy_from_slice(&128u32.to_le_bytes());
    bytes[28..32].copy_from_slice(&1.2f32.to_le_bytes());
    bytes[48..52].copy_from_slice(&1u32.to_le_bytes());
    let result = std::panic::catch_unwind(|| QuIVer::load_from_bytes(&bytes));
    assert!(result.is_ok());
    assert!(result.unwrap().is_err());
}

#[test]
fn sidecar固定头部逐字节截断不得触发panic() {
    let mut bytes = [0u8; 52];
    bytes[0..4].copy_from_slice(QuIVer::QUIVER_MAGIC);
    bytes[4..8].copy_from_slice(&QuIVer::QUIVER_VERSION.to_le_bytes());
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    bytes[12..16].copy_from_slice(&1u32.to_le_bytes());
    bytes[16..20].copy_from_slice(&1u32.to_le_bytes());
    bytes[20..24].copy_from_slice(&1u32.to_le_bytes());
    bytes[24..28].copy_from_slice(&1u32.to_le_bytes());
    bytes[28..32].copy_from_slice(&1.0f32.to_le_bytes());
    bytes[48..52].copy_from_slice(&1u32.to_le_bytes());

    for length in 0..52 {
        let result = std::panic::catch_unwind(|| QuIVer::load_from_bytes(&bytes[..length]));
        assert!(result.is_ok(), "固定头部截断至 {length} 字节时发生 panic");
        assert!(result.unwrap().is_err());
    }
}

#[test]
fn fuzz发现的四十八字节样本不得触发panic() {
    let bytes = [
        81, 85, 73, 86, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 130, 130, 130, 130, 130, 130, 81, 85, 73,
        86, 1, 0, 142, 0, 1, 99, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 153, 93,
    ];
    let result = std::panic::catch_unwind(|| QuIVer::load_from_bytes(&bytes));
    assert!(result.is_ok());
    assert!(result.unwrap().is_err());
}

#[test]
fn test_cosine_sim_函数() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [1.0f32, 0.0, 0.0];
    assert!((cosine_sim(&a, &b) - 1.0).abs() < 1e-5);

    let c = [0.0f32, 1.0, 0.0];
    assert!(cosine_sim(&a, &c).abs() < 1e-5);

    let d = [-1.0f32, 0.0, 0.0];
    assert!((cosine_sim(&a, &d) + 1.0).abs() < 1e-5);
}

#[test]
fn test_quiver_config_default() {
    let config = BuildConfig::default();
    assert_eq!(config.m, 16);
    assert_eq!(config.ef_construction, 128);
    assert!((config.alpha - 1.2).abs() < 1e-5);
}

#[test]
fn test_quiver_search_config_rerank_limit() {
    let cfg = SearchConfig {
        top_k: 10,
        ef_search: 50,
        rerank_limit: None,
    };
    assert_eq!(cfg.rerank_limit(), 50); // 默认取 ef_search

    let cfg2 = SearchConfig {
        top_k: 10,
        ef_search: 50,
        rerank_limit: Some(30),
    };
    assert_eq!(cfg2.rerank_limit(), 30);

    let cfg3 = SearchConfig {
        top_k: 100,
        ef_search: 50,
        rerank_limit: Some(30),
    };
    assert_eq!(cfg3.rerank_limit(), 100); // max(rerank_limit, top_k)
}
