#![cfg(feature = "research")]
use quiver_ann::research::{HyperbolicStrategy as Mode, set_hyperbolic_strategy};
use quiver_ann::{BuildConfig, Index, PoincareNavigation, SearchConfig};

#[test]
fn 预编码方向在查询插入重建与持久化中保持有效() {
    let cfg = SearchConfig {
        top_k: 2,
        ef_search: 64,
        rerank_limit: None,
    };
    for mode in [Mode::ScalarQuantizedDirection, Mode::FullPrecisionDirection] {
        set_hyperbolic_strategy(mode);
        let mut index = Index::build_poincare(
            3,
            &[9, 2, 5],
            &[0.3, 0.2, 0.1, 0.1, 0.5, 0.2, -0.3, 0.1, 0.5],
            &BuildConfig::default(),
            1.0,
            PoincareNavigation::GeometryBq2,
        )
        .unwrap();
        let query = [0.2, 0.3, 0.1];
        assert_eq!(
            index.search(&query, &cfg).unwrap(),
            index.search_exact(&query, 2).unwrap()
        );
        index.insert(3, &query).unwrap();
        assert_eq!(
            index.search(&query, &cfg).unwrap(),
            index.search_exact(&query, 2).unwrap()
        );
        index.remove(9);
        index.rebuild().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("方向对照.quiver");
        index.save(&path).unwrap();
        let mut loaded = Index::load(&path).unwrap();
        loaded.prepare_direction_representation(mode).unwrap();
        assert_eq!(
            loaded.search(&query, &cfg).unwrap(),
            loaded.search_exact(&query, 2).unwrap()
        );
    }
    set_hyperbolic_strategy(Mode::GeodesicPortals);
    let vectors: Vec<f32> = (0..120)
        .flat_map(|i| {
            let a = i as f32 * 0.2;
            [a.cos() * 0.7, a.sin() * 0.7, 0.1]
        })
        .collect();
    let index = Index::build_poincare(
        3,
        &(0..120).collect::<Vec<_>>(),
        &vectors,
        &BuildConfig::default(),
        1.0,
        PoincareNavigation::Exact,
    )
    .unwrap();
    let query = [0.4, 0.3, 0.1];
    let truth: Vec<_> = index
        .search_exact(&query, 2)
        .unwrap()
        .iter()
        .map(|r| r.0)
        .collect();
    let stats = index
        .search_loss_decomposition(&query, &cfg, PoincareNavigation::Exact, &truth)
        .unwrap();
    assert!(stats.approximate_evals > 0, "传统门户方向评分必须计入诊断");
    assert!(stats.hits_visited >= stats.hits_in_beam);
    set_hyperbolic_strategy(Mode::Full);
}
