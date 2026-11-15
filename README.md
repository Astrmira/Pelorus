# QuIVer

QuIVer 是使用两位符号幅值量化与图导航的 Rust ANN 库，支持余弦相似度和 Poincaré 球上的原生双曲检索。

索引提供并行构建、近似与精确查询、插入、软删除、压缩、重建和带校验和的原子保存。双曲查询返回按精确测地距离排序的候选。

## 使用源码

在仓库根目录执行以下命令。

```powershell
cargo test --release --lib --tests
cargo run --release --example basic
cargo run --release --example hyperbolic
```

通过本地路径依赖集成。

```toml
[dependencies]
quiver-ann = { path = "../QuIVer" }
```

### 余弦检索

```rust
use quiver_ann::{BuildConfig, Index, SearchConfig};

fn main() -> quiver_ann::Result<()> {
    let index = Index::build(
        2,
        &[100, 200],
        &[1.0, 0.0, 0.0, 1.0],
        &BuildConfig::default(),
    )?;
    let results = index.search(
        &[1.0, 0.0],
        &SearchConfig {
            top_k: 1,
            ef_search: 8,
            rerank_limit: None,
        },
    )?;
    assert_eq!(results[0].0, 100);
    Ok(())
}
```

### 双曲检索

```rust
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
    assert_eq!(results[0].0, 100);
    Ok(())
}
```

曲率参数 `c` 必须为正有限数，输入坐标满足 `c * ||x||² < 1`。向量按行连续存放，外部标识为唯一的 `u64`，零号标识可用。

查询结果为 `(id, score)`，按分数降序排列。余弦模式的分数为余弦相似度，双曲模式的分数为负测地距离。`search_exact` 使用相同的几何、删除状态和分数约定进行全库扫描。

`BuildConfig` 控制图度数、构建搜索宽度和剪枝系数。`SearchConfig` 控制返回数量、查询搜索宽度和精排预算。构建使用当前 Rayon 线程池，可通过调用方的线程池设置构图并行度。

### 维护与保存

```rust,no_run
use quiver_ann::{BuildConfig, Index};

fn main() -> quiver_ann::Result<()> {
    let mut index = Index::build(2, &[100], &[1.0, 0.0], &BuildConfig::default())?;
    index.insert(200, &[0.0, 1.0])?;
    index.remove(100);
    index.compact()?;
    index.save("vectors.quiver")?;
    let restored = Index::load("vectors.quiver")?;
    assert!(restored.contains(200));
    Ok(())
}
```

文件保存原始向量、外部标识和构建参数。`load` 校验文件后重建内存图，兼容既有的索引文件版本。`navigation_storage_bytes` 返回图与导航表示的有效载荷字节数，原始向量的内存单独计算。

## 源码结构

```text
src/
  config.rs              公开构建与查询配置
  index/                 公开 API、输入校验、维护与持久化
  graph/                 图存储、并行构建、余弦查询与属性预筛
    hyperbolic/          极坐标路由、残差编码、热存储与双曲搜索
  quantization/          两位签名、符号幅值重构与 SIMD 内核
  geometry.rs            几何量化表示与导航模式
  metric.rs              距离计算与流形约束
  vector.rs              向量算子
  research.rs            研究特性下的配置与诊断接口
examples/                常规使用示例
tests/                   核心功能与公共契约回归
benches/                 Criterion 查询基准
research/                测量工具、实验索引与第三方来源
paper/vldb/              论文源码与图表
```

`research` feature 提供方向表示对照、组件配置、损失分解和阶段计时，入口集中于 `quiver_ann::research`。研究配置中的进程级开关应在构图与并发查询之前设置，并在使用期间固定。[研究目录说明](research/README.md)

## 开发验证

```powershell
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --release --lib --tests
cargo test --release --all-features --lib --tests -- --test-threads=1
cargo doc --no-deps --all-features
```

研究特性的测试串行运行，以隔离进程级策略配置。现有 CI 覆盖 Linux、Windows 和 macOS 的核心构建与回归。

命名迁移和研究入口调整记录在迁移说明中。[源码迁移说明](docs/source-migration.md)

## 许可证

Apache-2.0。[许可证](LICENSE) · [上游版权声明](NOTICE)
