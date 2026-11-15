use quiver_ann::{BuildConfig, Index, SearchConfig};

fn main() -> quiver_ann::Result<()> {
    let vectors = [1.0, 0.0, 0.0, 1.0];
    let index = Index::build(2, &[100, 200], &vectors, &BuildConfig::default())?;
    let results = index.search(
        &[1.0, 0.0],
        &SearchConfig {
            top_k: 1,
            ef_search: 8,
            rerank_limit: None,
        },
    )?;
    println!("最近邻：{:?}", results);
    Ok(())
}
