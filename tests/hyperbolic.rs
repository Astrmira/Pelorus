use quiver_ann::{BuildConfig, Error, Index, Metric, SearchConfig};
use rayon::prelude::*;

const GEOMETRY: Metric = Metric::Poincare { curvature: 1.0 };

#[test]
fn 并行构图与并发查询的访问状态不串扰() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        let vectors = (0..128)
            .flat_map(|i| {
                let radius = 0.01 + 0.98 * i as f32 / 128.;
                let angle = i as f32 * 2.399963;
                [radius * angle.cos(), radius * angle.sin()]
            })
            .collect::<Vec<_>>();
        let ids = (0..128).collect::<Vec<u64>>();
        let index =
            Index::build_with_metric(2, &ids, &vectors, &BuildConfig::default(), GEOMETRY).unwrap();
        (0..64).into_par_iter().for_each(|i| {
            let query = [vectors[i * 2] * 0.97, vectors[i * 2 + 1] * 0.97];
            assert_eq!(
                index.search(&query, &search_config(10)).unwrap(),
                index.search_exact(&query, 10).unwrap()
            );
        });
    });
}

fn search_config(k: usize) -> SearchConfig {
    SearchConfig {
        top_k: k,
        ef_search: 256,
        rerank_limit: None,
    }
}

#[test]
fn 同方向不同半径仍可区分且球心是合法点() {
    let index = Index::build_with_metric(
        2,
        &[10, 20, 30, 40],
        &[0.1, 0., 0.79, 0.05, 0.95, 0., 0., 0.],
        &BuildConfig::default(),
        GEOMETRY,
    )
    .unwrap();
    let query = [0.8, 0.];
    let actual = index.search(&query, &search_config(4)).unwrap();
    let exact = index.search_exact(&query, 4).unwrap();
    assert_eq!(actual, exact);
    assert_eq!(actual[0].0, 20);
    assert_eq!(
        index.search(&[0., 0.], &search_config(1)).unwrap()[0],
        (40, -0.0)
    );
}

#[test]
fn 曲率与独立闭式测地距离一致() {
    let curvature = 0.25;
    let index = Index::build_with_metric(
        1,
        &[1],
        &[0.5],
        &BuildConfig::default(),
        Metric::Poincare { curvature },
    )
    .unwrap();
    let expected = -4.0 * 0.25f64.atanh();
    let result = index.search(&[0.], &search_config(1)).unwrap();
    assert!((f64::from(result[0].1) - expected).abs() < 1e-6);
    assert_eq!(result, index.search_exact(&[0.], 1).unwrap());
}

#[test]
fn 拒绝非法曲率和边界且失败插入不改变索引() {
    for curvature in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(matches!(
            Index::build_with_metric(
                2,
                &[],
                &[],
                &BuildConfig::default(),
                Metric::Poincare { curvature }
            ),
            Err(Error::InvalidMetric(_))
        ));
    }
    for vector in [[1., 0.], [1.1, 0.]] {
        assert!(matches!(
            Index::build_with_metric(2, &[1], &vector, &BuildConfig::default(), GEOMETRY),
            Err(Error::OutsideManifold { index: 0 })
        ));
    }
    let mut index =
        Index::build_with_metric(2, &[], &[], &BuildConfig::default(), GEOMETRY).unwrap();
    assert!(index.insert(1, &[1., 0.]).is_err());
    assert!(index.is_empty());
    assert!(matches!(
        index.search(&[1., 0.], &search_config(1)),
        Err(Error::OutsideManifold { .. })
    ));
    index.insert(1, &[0., 0.]).unwrap();
    assert!(index.insert(2, &[f32::NAN, 0.]).is_err());
    assert_eq!(index.len(), 1);
}

#[test]
fn 增删重建和持久化保持双曲语义() {
    let mut index =
        Index::build_with_metric(2, &[], &[], &BuildConfig::default(), GEOMETRY).unwrap();
    for (id, vector) in [
        (0, [0., 0.]),
        (1, [0.2, 0.]),
        (2, [0.8, 0.]),
        (3, [-0.8, 0.]),
    ] {
        index.insert(id, &vector).unwrap();
    }
    assert!(index.remove(0));
    assert!(index.remove(1));
    assert!(index.remove(2));
    assert_eq!(
        index
            .search(&[0.8, 0.], &search_config(4))
            .unwrap()
            .iter()
            .map(|x| x.0)
            .collect::<Vec<_>>(),
        vec![3]
    );
    index.insert(0, &[0.7, 0.]).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("双曲索引.quiver");
    index.save(&path).unwrap();
    let mut loaded = Index::load(&path).unwrap();
    assert_eq!(loaded.metric(), GEOMETRY);
    assert_eq!(
        loaded.search(&[0.8, 0.], &search_config(2)).unwrap(),
        index.search_exact(&[0.8, 0.], 2).unwrap()
    );
    assert_eq!(index.compact().unwrap(), 3);
    loaded
        .rebuild_with_config(&BuildConfig {
            m: 4,
            ef_construction: 16,
            alpha: 1.1,
        })
        .unwrap();
    assert_eq!(loaded.metric(), GEOMETRY);
    assert_eq!(
        loaded.search(&[0.8, 0.], &search_config(2)).unwrap(),
        index.search_exact(&[0.8, 0.], 2).unwrap()
    );
}

#[test]
fn 宽维度与混合半径查询对照独立真值() {
    for dim in [3, 65, 384] {
        let mut vectors = Vec::new();
        for row in 0..80 {
            let mut vector = (0..dim)
                .map(|col| ((row * 71 + col * 19) as f64 * 0.13).sin() as f32)
                .collect::<Vec<_>>();
            let norm = vector
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt();
            let radius = 0.1 + 0.88 * (row as f64 / 80.);
            for value in &mut vector {
                *value = (f64::from(*value) * radius / norm) as f32;
            }
            vectors.extend(vector);
        }
        let ids = (0..80).collect::<Vec<u64>>();
        let index =
            Index::build_with_metric(dim, &ids, &vectors, &BuildConfig::default(), GEOMETRY)
                .unwrap();
        for qid in [7, 31, 79] {
            let query = vectors[qid * dim..(qid + 1) * dim]
                .iter()
                .map(|v| v * 0.93)
                .collect::<Vec<_>>();
            let qnorm = query.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
            let mut independent = vectors
                .chunks_exact(dim)
                .enumerate()
                .map(|(i, v)| {
                    let norm = v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
                    let delta = v
                        .iter()
                        .zip(&query)
                        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                        .sum::<f64>();
                    (
                        (1. + 2. * delta / ((1. - norm) * (1. - qnorm))).acosh(),
                        i as u64,
                    )
                })
                .collect::<Vec<_>>();
            independent.sort_by(|a, b| a.0.total_cmp(&b.0));
            let actual = index.search(&query, &search_config(10)).unwrap();
            assert_eq!(
                actual.iter().map(|x| x.0).collect::<Vec<_>>(),
                independent[..10].iter().map(|x| x.1).collect::<Vec<_>>()
            );
            for (result, expected) in actual.iter().zip(&independent) {
                assert!((f64::from(result.1) + expected.0).abs() < 1e-5);
            }
        }
    }
}

#[test]
fn 近球边界与极小曲率保持数值意义() {
    for (curvature, value) in [(1., f32::from_bits(1.0f32.to_bits() - 1)), (1e-300, 1e-20)] {
        let index = Index::build_with_metric(
            1,
            &[1],
            &[value],
            &BuildConfig::default(),
            Metric::Poincare { curvature },
        )
        .unwrap();
        let score = index.search(&[0.], &search_config(1)).unwrap()[0].1;
        assert!(score.is_finite() && score < 0.);
        let expected = -2. * ((curvature.sqrt() * f64::from(value)).atanh() / curvature.sqrt());
        assert!(((f64::from(score) - expected) / expected).abs() < 1e-6);
    }
}

#[test]
fn 老格式可读且几何字段受校验和保护() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("索引.quiver");
    Index::build(1, &[1], &[1.], &BuildConfig::default())
        .unwrap()
        .save(&path)
        .unwrap();
    assert_eq!(
        u32::from_le_bytes(std::fs::read(&path).unwrap()[8..12].try_into().unwrap()),
        1
    );
    assert_eq!(Index::load(&path).unwrap().metric(), Metric::Cosine);
    Index::build_with_metric(1, &[1], &[0.2], &BuildConfig::default(), GEOMETRY)
        .unwrap()
        .save(&path)
        .unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 2);
    bytes[28..36].copy_from_slice(&0f64.to_bits().to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    assert!(matches!(Index::load(&path), Err(Error::CorruptIndex(_))));
    let checksum = crc32fast::hash(&bytes[24..]);
    bytes[20..24].copy_from_slice(&checksum.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    assert!(matches!(Index::load(&path), Err(Error::InvalidMetric(_))));
}

#[test]
fn 黎曼扭曲乘积双纤维构图消除外层出度坍缩并提升精确与量化检索召回率() {
    use quiver_ann::PoincareNavigation;
    use std::collections::HashSet;

    let dim = 32;
    let count = 600usize;
    let mut vectors = Vec::with_capacity(count * dim);
    for row in 0..count {
        let mut dir = (0..dim)
            .map(|col| ((row * 97 + col * 31 + 17) as f64 * 0.173).sin() as f32)
            .collect::<Vec<_>>();
        let norm = dir
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .sqrt();
        // 覆盖内层、中层、外层三个径向壳层（测地深度最高超过 R > 3.0）
        let r = match row % 3 {
            0 => 0.15 + 0.25 * ((row / 3) as f64 / (count / 3) as f64),
            1 => 0.55 + 0.20 * ((row / 3) as f64 / (count / 3) as f64),
            _ => 0.82 + 0.14 * ((row / 3) as f64 / (count / 3) as f64),
        };
        for value in &mut dir {
            *value = (f64::from(*value) * r / norm) as f32;
        }
        vectors.extend(dir);
    }
    let ids = (0..count as u64).collect::<Vec<_>>();
    let index =
        Index::build_with_metric(dim, &ids, &vectors, &BuildConfig::default(), GEOMETRY).unwrap();

    #[cfg(feature = "research")]
    {
        let diag = index.graph_diagnostics();
        let mut shell_counts = [0usize; 3];
        let mut shell_out = [0usize; 3];
        for node in &diag.nodes {
            let s = (node.id as usize) % 3;
            shell_counts[s] += 1;
            shell_out[s] += node.out_degree;
        }
        for s in 0..3 {
            let avg_out = shell_out[s] as f64 / shell_counts[s] as f64;
            assert!(
                avg_out >= 30.0,
                "径向壳层 {s} 平均出度不应坍缩，实测: {avg_out:.2}"
            );
        }
    }

    let queries = (0..30usize)
        .map(|q| {
            let base = &vectors[(q * 19) * dim..(q * 19 + 1) * dim];
            base.iter().map(|&v| v * 0.96).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let eval_recall = |idx: &Index, cfg: &SearchConfig| -> f64 {
        let mut hit = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let truth: HashSet<u64> = idx
                .search_exact(q, 10)
                .unwrap()
                .into_iter()
                .map(|x| x.0)
                .collect();
            let got = idx.search(q, cfg).unwrap();
            hit += got.iter().filter(|x| truth.contains(&x.0)).count();
            total += truth.len();
        }
        hit as f64 / total as f64
    };

    let exact_cfg = SearchConfig {
        top_k: 10,
        ef_search: 64,
        rerank_limit: None,
    };
    assert!(eval_recall(&index, &exact_cfg) >= 0.95);

    let index_bq2 = Index::build_poincare(
        dim,
        &ids,
        &vectors,
        &BuildConfig::default(),
        1.0,
        PoincareNavigation::GeometryBq2,
    )
    .unwrap();
    let bq2_cfg = SearchConfig {
        top_k: 10,
        ef_search: 64,
        rerank_limit: Some(64),
    };
    assert!(eval_recall(&index_bq2, &bq2_cfg) >= 0.90);
}
