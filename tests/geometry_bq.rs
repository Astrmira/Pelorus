use quiver_ann::{BuildConfig, Index, PoincareNavigation, SearchConfig};

#[test]
fn 原点附近径向下界保持最近邻() {
    for curvature in [1.0, 1e-300] {
        for scale in [1e-10f32, 1e-20] {
            let query = [0.5 * scale, 0.501 * scale, scale];
            let vectors = [
                0.501 * scale,
                0.5 * scale,
                scale,
                0.5001 * scale,
                0.501 * scale,
                scale,
            ];
            let index = Index::build_poincare(
                3,
                &[10, 20],
                &vectors,
                &BuildConfig::default(),
                curvature,
                PoincareNavigation::GeometryBq2,
            )
            .unwrap();
            assert_eq!(
                index.search(&query, &config(1)).unwrap(),
                index.search_exact(&query, 1).unwrap()
            );
        }
    }
}

#[test]
fn 精排距离相同按外部标识排序() {
    let vectors = [0.5f32, 0.0, 0.5, 0.0, -0.5, 0.0];
    for navigation in [PoincareNavigation::Exact, PoincareNavigation::GeometryBq2] {
        let index = Index::build_poincare(
            2,
            &[90, 10, 20],
            &vectors,
            &BuildConfig::default(),
            1.0,
            navigation,
        )
        .unwrap();
        assert_eq!(
            index.search(&[0.0, 0.0], &config(2)).unwrap(),
            index.search_exact(&[0.0, 0.0], 2).unwrap()
        );
    }
}

#[cfg(feature = "research")]
#[test]
fn 诊断复用实际搜索并区分候选池与预算() {
    let vectors: Vec<f32> = (0..120)
        .flat_map(|i| {
            let angle = i as f32 * 0.19;
            [angle.cos() * 0.6, angle.sin() * 0.6, i as f32 * 0.002]
        })
        .collect();
    let index = Index::build_poincare(
        3,
        &(0..120).collect::<Vec<u64>>(),
        &vectors,
        &BuildConfig::default(),
        1.0,
        PoincareNavigation::GeometryBq2,
    )
    .unwrap();
    for navigation in [PoincareNavigation::Exact, PoincareNavigation::GeometryBq2] {
        for budget in [2, 10, 64] {
            let cfg = SearchConfig {
                top_k: 2,
                ef_search: 64,
                rerank_limit: Some(budget),
            };
            let query = [0.4, 0.3, 0.1];
            let truth: Vec<_> = index
                .search_exact(&query, 2)
                .unwrap()
                .iter()
                .map(|h| h.0)
                .collect();
            let result = match navigation {
                PoincareNavigation::Exact => index.search_poincare_exact_navigation(&query, &cfg),
                _ => index.search(&query, &cfg),
            }
            .unwrap();
            let stats = index
                .search_loss_decomposition(&query, &cfg, navigation, &truth)
                .unwrap();
            assert_eq!(
                stats,
                index
                    .search_loss_decomposition(&query, &cfg, navigation, &truth.repeat(2))
                    .unwrap()
            );
            assert_eq!(
                stats.hits_returned,
                result.iter().filter(|h| truth.contains(&h.0)).count()
            );
            assert!(stats.hits_visited >= stats.hits_in_beam);
            assert!(stats.hits_in_beam >= stats.hits_selected_for_rerank);
            assert_eq!(stats.exact_rerank_loss(2), 0.0);
            assert!(stats.beam_candidates >= stats.selected_for_rerank);
            let loss = stats.reachability_loss(2)
                + stats.beam_retention_loss(2)
                + stats.budget_truncation_loss(2)
                + stats.exact_rerank_loss(2);
            assert!((loss - (1.0 - stats.hits_returned as f64 / 2.0)).abs() < 1e-12);
        }
    }
}

fn config(k: usize) -> SearchConfig {
    SearchConfig {
        top_k: k,
        ef_search: 256,
        rerank_limit: None,
    }
}

#[test]
fn 几何量化的径向差异尾块与最终精确分数有效() {
    for dimensions in [3, 65, 768] {
        let direction = (0..dimensions)
            .map(|i| ((i + 1) as f32).sin())
            .collect::<Vec<_>>();
        let norm = direction
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .sqrt();
        let vectors = [0., 0.2, 0.6, -0.9]
            .into_iter()
            .flat_map(|r| {
                direction
                    .iter()
                    .map(move |&v| (f64::from(v) * r / norm) as f32)
            })
            .collect::<Vec<_>>();
        let index = Index::build_poincare(
            dimensions,
            &[9, 2, 7, 1],
            &vectors,
            &BuildConfig::default(),
            1.0,
            PoincareNavigation::GeometryBq2,
        )
        .unwrap();
        let query = direction
            .iter()
            .map(|&v| (f64::from(v) * 0.5 / norm) as f32)
            .collect::<Vec<_>>();
        assert_eq!(
            index.search(&query, &config(4)).unwrap(),
            index.search_exact(&query, 4).unwrap()
        );
        assert_eq!(
            index
                .search(
                    &query,
                    &SearchConfig {
                        rerank_limit: Some(2),
                        ..config(2)
                    }
                )
                .unwrap()
                .len(),
            2
        );
    }
}

#[test]
fn 几何量化空图插入增删重建及持久化保留模式() {
    let mut index = Index::build_poincare(
        1,
        &[],
        &[],
        &BuildConfig::default(),
        0.5,
        PoincareNavigation::GeometryBq2,
    )
    .unwrap();
    assert!(index.search(&[0.2], &config(1)).unwrap().is_empty());
    for (id, value) in [(10, 0.0), (20, 0.2), (30, 0.8), (40, -0.8)] {
        index.insert(id, &[value]).unwrap();
    }
    assert!(index.insert(50, &[2.0]).is_err());
    assert_eq!(index.len(), 4);
    assert!(index.remove(20));
    assert_eq!(
        index.search(&[0.3], &config(3)).unwrap(),
        index.search_exact(&[0.3], 3).unwrap()
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("几何量化索引.quiver");
    index.save(&path).unwrap();
    let mut loaded = Index::load(&path).unwrap();
    assert_eq!(
        loaded.poincare_navigation(),
        PoincareNavigation::GeometryBq2
    );
    assert_eq!(
        loaded.search(&[0.3], &config(3)).unwrap(),
        index.search_exact(&[0.3], 3).unwrap()
    );
    assert_eq!(index.compact().unwrap(), 1);
    loaded.rebuild().unwrap();
    for current in [&index, &loaded] {
        assert_eq!(
            current.poincare_navigation(),
            PoincareNavigation::GeometryBq2
        );
        assert_eq!(
            current.search(&[0.3], &config(3)).unwrap(),
            current.search_exact(&[0.3], 3).unwrap()
        );
    }
}
