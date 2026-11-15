//! 索引构建与空索引维护回归。

use super::*;

#[test]
fn 独立索引支持超过三千零七十二维() {
    let dimensions = 4097;
    let mut vectors = vec![0.0; dimensions * 3];
    vectors[0] = 1.0;
    vectors[dimensions + 1] = 1.0;
    vectors[dimensions * 2 + 2] = 1.0;
    let index = Index::build(dimensions, &[1, 2, 3], &vectors, &BuildConfig::default()).unwrap();
    let result = index
        .search(
            &vectors[..dimensions],
            &SearchConfig {
                top_k: 1,
                ef_search: 8,
                rerank_limit: None,
            },
        )
        .unwrap();
    assert_eq!(result[0].0, 1);
}

#[test]
fn 空索引可以插入() {
    let mut index = Index::build(2, &[], &[], &BuildConfig::default()).unwrap();
    index.insert(0, &[1.0, 0.0]).unwrap();
    assert!(index.contains(0));
}
