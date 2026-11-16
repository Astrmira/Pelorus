use quiver_ann::{
    BuildConfig, Index, PoincareNavigation, RerankConfig, RerankPrecision, ScalarQuantization,
    SearchConfig, StorageMode,
};

fn main() -> quiver_ann::Result<()> {
    let mut index = Index::build_poincare(
        3,
        &[100, 200, 300],
        &[0.1, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.3, 0.0],
        &BuildConfig::default(),
        1.0,
        PoincareNavigation::GeometryBq2,
    )?;
    index.set_rerank_config(RerankConfig {
        precision: RerankPrecision::Sq8,
        scalar_quantization: ScalarQuantization::Centered,
        ..RerankConfig::default()
    })?;
    index.set_storage_mode(StorageMode::Mmap)?;
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("vectors.pelorus");
    index.save_snapshot(&path)?;
    #[cfg(target_os = "linux")]
    let mut restored = Index::open_ssd(
        &path,
        quiver_ann::SsdCacheConfig {
            cache_bytes: 4 * 1024 * 1024,
            inflight_bytes: 1024 * 1024,
            ..quiver_ann::SsdCacheConfig::default()
        },
    )?;
    #[cfg(not(target_os = "linux"))]
    let mut restored = Index::load_mmap(&path)?;
    restored.insert(400, &[0.6, 0.0, 0.0])?;
    restored.save(&path)?;
    let result = restored.search(
        &[0.12, 0.0, 0.0],
        &SearchConfig {
            top_k: 2,
            ..SearchConfig::default()
        },
    )?;
    assert_eq!(result[0].0, 100);
    println!(
        "存储查询完成，返回 {} 条结果，向量存储统计为 {:?}，直接读取统计为 {:?}",
        result.len(),
        restored.vector_storage_stats(),
        restored.ssd_io_stats()
    );
    Ok(())
}
