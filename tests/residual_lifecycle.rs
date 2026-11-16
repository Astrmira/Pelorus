#![cfg(feature = "research")]

use quiver_ann::research::{
    ExecutionConfig, HyperbolicStrategy, RecallConfig, StorageConfig, set_execution_config,
    set_hyperbolic_strategy, set_recall_config, set_storage_config,
};
use quiver_ann::{BuildConfig, Index, PoincareNavigation, SearchConfig};

fn clustered_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut state = 20261114u64;
    let mut vectors = Vec::with_capacity(n * dim);
    for row in 0..n {
        let mut noise = Vec::with_capacity(dim);
        for _ in 0..dim {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            noise.push(((state >> 32) as u32) as f64 / f64::from(u32::MAX) - 0.5);
        }
        let noise_norm = noise.iter().map(|x| x * x).sum::<f64>().sqrt();
        for value in &mut noise {
            *value *= 0.16 / noise_norm;
        }
        noise[row % 12] += 1.0;
        let norm = noise.iter().map(|x| x * x).sum::<f64>().sqrt();
        let radius = (2.0 + (row % 7) as f64 * 0.003).tanh();
        vectors.extend(noise.into_iter().map(|x| (x * radius / norm) as f32));
    }
    vectors
}

fn assert_full_search(index: &Index, queries: &[Vec<f32>]) {
    let config = SearchConfig {
        top_k: 10,
        ef_search: 2048,
        rerank_limit: None,
    };
    for query in queries {
        assert_eq!(
            index.search(query, &config).unwrap(),
            index.search_exact(query, 10).unwrap()
        );
    }
}

#[test]
fn 残差编码在布局切换增删压缩和显式配置的保存加载中保持正确() {
    set_hyperbolic_strategy(HyperbolicStrategy::Full);
    set_execution_config(ExecutionConfig::default());
    set_storage_config(StorageConfig::default());
    set_recall_config(RecallConfig::adaptive());
    let (n, dim) = (1536, 64);
    let vectors = clustered_vectors(n, dim);
    let mut queries: Vec<Vec<f32>> = (0..12)
        .map(|i| vectors[i * 127 * dim..(i * 127 + 1) * dim].to_vec())
        .collect();
    let mut index = Index::build_poincare(
        dim,
        &(0..n as u64).collect::<Vec<_>>(),
        &vectors,
        &BuildConfig {
            m: 16,
            ef_construction: 64,
            alpha: 1.2,
        },
        1.0,
        PoincareNavigation::GeometryBq2,
    )
    .unwrap();
    assert!(
        index.residual_cone_count() > 0,
        "测试索引必须实际采用残差编码"
    );
    assert_full_search(&index, &queries);

    index.prepare_colocated_hot_store(true).unwrap();
    set_storage_config(StorageConfig {
        colocated_store: true,
        ..StorageConfig::default()
    });
    assert_full_search(&index, &queries);
    index.restore_split_hot_store();
    set_storage_config(StorageConfig::default());
    assert_full_search(&index, &queries);
    index.prepare_colocated_hot_store(true).unwrap();
    set_storage_config(StorageConfig {
        colocated_store: true,
        ..StorageConfig::default()
    });

    let mut inserted = queries[0].clone();
    inserted[dim - 1] += 0.003;
    index.insert(10_000, &inserted).unwrap();
    queries.push(inserted);
    assert_full_search(&index, &queries);
    assert!(index.remove(2));
    assert_full_search(&index, &queries);
    index.compact().unwrap();
    assert_full_search(&index, &queries);

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("残差生命周期.quiver");
    index.save(&path).unwrap();
    let loaded = Index::load(&path).unwrap();
    assert!(
        loaded.residual_cone_count() > 0,
        "显式启用救援配置后加载应重新建立残差编码"
    );
    assert_full_search(&loaded, &queries);
    set_recall_config(RecallConfig::default());
    assert_full_search(&loaded, &queries);
    let loaded_default = Index::load(&path).unwrap();
    assert_eq!(
        loaded_default.residual_cone_count(),
        loaded.residual_cone_count(),
        "自描述持久化在全局救援开关关闭时也必须自动恢复相同的切锥正交残差字典"
    );
    assert_full_search(&loaded_default, &queries);
    eprintln!(
        "显式救援配置加载的残差主轴数为 {}，默认配置加载的主轴数为 {}",
        loaded.residual_cone_count(),
        loaded_default.residual_cone_count()
    );

    set_storage_config(StorageConfig::default());
    set_recall_config(RecallConfig::default());
    set_execution_config(ExecutionConfig::default());
}

#[test]
fn 残差字典拓扑与单页合表可从映射快照直接恢复() {
    use quiver_ann::{RerankConfig, RerankPrecision, RerankRepresentation, StorageMode};
    set_hyperbolic_strategy(HyperbolicStrategy::Full);
    set_execution_config(ExecutionConfig::default());
    set_recall_config(RecallConfig::adaptive());
    set_storage_config(StorageConfig::default());
    let vectors = clustered_vectors(1536, 64);
    let mut index = Index::build_poincare(
        64,
        &(0..1536u64).collect::<Vec<_>>(),
        &vectors,
        &BuildConfig {
            m: 16,
            ef_construction: 64,
            alpha: 1.2,
        },
        1.0,
        PoincareNavigation::GeometryBq2,
    )
    .unwrap();
    assert!(index.residual_cone_count() > 0);
    index.prepare_colocated_hot_store(true).unwrap();
    set_storage_config(StorageConfig {
        colocated_store: true,
        ..StorageConfig::default()
    });
    index
        .set_rerank_config(RerankConfig {
            precision: RerankPrecision::Fp16,
            representation: RerankRepresentation::Polar,
            ..RerankConfig::default()
        })
        .unwrap();
    let query = &vectors[0..64];
    let search = SearchConfig {
        top_k: 10,
        ef_search: 512,
        rerank_limit: Some(256),
    };
    let expected = index.search(query, &search).unwrap();
    let topology = index.graph_diagnostics().topology_crc32;
    let hot = index.navigation_storage_bytes();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("residual.pelorus");
    index.save_snapshot(&path).unwrap();
    set_recall_config(RecallConfig::default());
    let loaded = Index::load_mmap(&path).unwrap();
    assert_eq!(loaded.storage_mode(), StorageMode::Mmap);
    assert_eq!(loaded.residual_cone_count(), index.residual_cone_count());
    assert_eq!(loaded.navigation_storage_bytes(), hot);
    assert_eq!(loaded.graph_diagnostics().topology_crc32, topology);
    assert_eq!(loaded.search(query, &search).unwrap(), expected);
    set_storage_config(StorageConfig::default());
    set_recall_config(RecallConfig::default());
    set_execution_config(ExecutionConfig::default());
}
