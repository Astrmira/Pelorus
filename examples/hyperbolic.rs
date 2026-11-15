use quiver_ann::{BuildConfig, Index, PoincareNavigation, SearchConfig};

fn main() -> quiver_ann::Result<()> {
    let index = Index::build_poincare(
        2,
        &[100, 200, 300],
        &[0.1, 0.0, 0.4, 0.0, 0.0, 0.3],
        &BuildConfig::default(),
        1.0,
        PoincareNavigation::GeometryBq2,
    )?;
    let results = index.search(
        &[0.12, 0.0],
        &SearchConfig {
            top_k: 2,
            ef_search: 16,
            rerank_limit: None,
        },
    )?;
    println!("双曲近邻及负测地距离分数：{results:?}");
    Ok(())
}
