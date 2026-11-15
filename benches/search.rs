use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use quiver_ann::{BuildConfig, Index, SearchConfig};
use std::hint::black_box;

fn 搜索基准(criterion: &mut Criterion) {
    let dimensions = 384;
    let count = 1_000;
    let vectors = (0..dimensions * count)
        .map(|index| ((index as f32) * 0.013).sin())
        .collect::<Vec<_>>();
    let ids = (0..count as u64).collect::<Vec<_>>();
    let index = Index::build(dimensions, &ids, &vectors, &BuildConfig::default()).unwrap();
    let query = &vectors[..dimensions];
    let config = SearchConfig {
        top_k: 10,
        ef_search: 64,
        rerank_limit: None,
    };

    criterion.bench_with_input(BenchmarkId::new("搜索", count), &count, |bencher, _| {
        bencher.iter(|| index.search(black_box(query), black_box(&config)).unwrap());
    });
}

criterion_group!(基准组, 搜索基准);
criterion_main!(基准组);
