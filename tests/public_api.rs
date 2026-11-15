use quiver_ann::{BuildConfig, Error, Index, SearchConfig};

fn search_config() -> SearchConfig {
    SearchConfig {
        top_k: 1,
        ef_search: 8,
        rerank_limit: None,
    }
}

#[test]
fn 构建搜索与零号标识可用() {
    let vectors = [1.0, 0.0, 0.0, 1.0];
    let index = Index::build(2, &[0, 2], &vectors, &BuildConfig::default()).unwrap();
    let result = index.search(&[1.0, 0.0], &search_config()).unwrap();
    assert_eq!(result[0].0, 0);
    assert_eq!(index.vector(2), Some(&[0.0, 1.0][..]));
}

#[test]
fn 拒绝重复标识和非有限数据() {
    assert!(matches!(
        Index::build(2, &[7, 7], &[1.0, 0.0, 0.0, 1.0], &BuildConfig::default()),
        Err(Error::DuplicateId(7))
    ));
    assert!(matches!(
        Index::build(2, &[1], &[f32::NAN, 0.0], &BuildConfig::default()),
        Err(Error::NonFiniteVector { index: 0 })
    ));
}

#[test]
fn 校验构建与搜索配置() {
    let invalid = BuildConfig {
        m: 1,
        ..BuildConfig::default()
    };
    assert!(matches!(
        Index::build(2, &[], &[], &invalid),
        Err(Error::InvalidBuildConfig(_))
    ));

    let index = Index::build(2, &[], &[], &BuildConfig::default()).unwrap();
    assert!(matches!(
        index.search(
            &[0.0, 1.0],
            &SearchConfig {
                top_k: 2,
                ef_search: 1,
                rerank_limit: None,
            }
        ),
        Err(Error::InvalidSearchConfig(_))
    ));
}

#[test]
fn 空索引搜索与后续插入可用() {
    let mut index = Index::build(2, &[], &[], &BuildConfig::default()).unwrap();
    assert!(
        index
            .search(&[1.0, 0.0], &search_config())
            .unwrap()
            .is_empty()
    );
    index.insert(0, &[1.0, 0.0]).unwrap();
    assert_eq!(index.search(&[1.0, 0.0], &search_config()).unwrap()[0].0, 0);
    assert_eq!(index.insert(0, &[0.0, 1.0]), Err(Error::DuplicateId(0)));
}

#[test]
fn 保存加载并拒绝损坏文件() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("index.quiver");
    let index = Index::build(
        3,
        &[10, 20],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &BuildConfig::default(),
    )
    .unwrap();
    index.save(&path).unwrap();
    index.save(&path).unwrap();

    let loaded = Index::load(&path).unwrap();
    assert_eq!(loaded.dimensions(), 3);
    assert_eq!(loaded.ids().collect::<Vec<_>>(), vec![10, 20]);
    assert_eq!(
        loaded.search(&[0.0, 1.0, 0.0], &search_config()).unwrap()[0].0,
        20
    );

    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&path, bytes).unwrap();
    assert!(matches!(Index::load(&path), Err(Error::CorruptIndex(_))));
}

#[test]
fn 压缩与重建移除墓碑() {
    let mut index = Index::build(
        2,
        &[1, 2, 3],
        &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0],
        &BuildConfig::default(),
    )
    .unwrap();
    assert!(index.remove(2));
    assert!(!index.remove(2));
    assert_eq!(index.compact().unwrap(), 1);
    assert_eq!(index.ids().collect::<Vec<_>>(), vec![1, 3]);
    assert!(!index.contains(2));
    assert!(!index.needs_rebuild());
}

#[test]
fn 四千零九十七维可持久化往返() {
    let dimensions = 4097;
    let mut vectors = vec![0.0; dimensions * 2];
    vectors[0] = 1.0;
    vectors[dimensions + 1] = 1.0;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("wide.quiver");
    let index = Index::build(dimensions, &[1, 2], &vectors, &BuildConfig::default()).unwrap();
    index.save(&path).unwrap();
    let loaded = Index::load(&path).unwrap();
    assert_eq!(
        loaded
            .search(&vectors[..dimensions], &search_config())
            .unwrap()[0]
            .0,
        1
    );
}
