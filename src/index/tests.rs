//! 索引边界校验与诊断前置条件回归。

use super::*;

#[test]
fn 精确检索与固定重排及双曲诊断校验非法输入() {
    let index = Index::build(2, &[1, 2], &[1.0, 0.0, 0.0, 1.0], &BuildConfig::default()).unwrap();
    assert!(matches!(
        index.search_exact(&[1.0, 0.0], 0),
        Err(Error::InvalidSearchConfig(_))
    ));
    assert!(matches!(
        index.rerank_candidates(&[1.0, 0.0], &[1, 2], 0),
        Err(Error::InvalidSearchConfig(_))
    ));
    #[cfg(feature = "research")]
    {
        let cfg = SearchConfig::default();
        assert!(matches!(
            index.search_loss_decomposition(&[1.0, 0.0], &cfg, PoincareNavigation::Exact, &[1]),
            Err(Error::InvalidSearchConfig(_))
        ));
        assert!(matches!(
            index.search_stage_profile(&[1.0, 0.0], &cfg, PoincareNavigation::Exact),
            Err(Error::InvalidSearchConfig(_))
        ));
    }
}
