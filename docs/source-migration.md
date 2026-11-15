# 源码命名与目录迁移

2026-11-15 的整理将源码按索引 API、图构建、几何导航和量化算子拆分。公开的 `Index`、`BuildConfig`、`SearchConfig`、`Metric` 和 `PoincareNavigation` 使用方式继续适用于常规构建和查询。

## 研究接口

研究调用使用 `research` feature，并从 `quiver_ann::research` 导入配置和诊断类型。

| 原名称 | 当前名称 |
| --- | --- |
| `ablation` feature | `research` feature |
| `RiemannAblationMode` | `HyperbolicStrategy` |
| `RiemannComponents` | `GeometryComponents` |
| `SpeedOptConfig` | `ExecutionConfig` |
| `HotPathOptConfig` | `StorageConfig` |
| `RescueOptConfig` | `RecallConfig` |
| `HeaderEarlyGate` | `SketchFilter` |
| `set_riemann_ablation_mode` | `set_hyperbolic_strategy` |
| `set_riemann_components` | `set_geometry_components` |
| `set_speed_opt_config` | `set_execution_config` |
| `set_hot_path_opt_config` | `set_storage_config` |
| `set_rescue_opt_config` | `set_recall_config` |
| `prepare_direction_ablation` | `prepare_direction_representation` |
| `TraversalProfile::V3Default` | `TraversalProfile::Standard` |
| `RescueOptConfig::full_rescue()` | `RecallConfig::adaptive()` |

历史策略变体按机制命名，例如 `FixedShellCompass`、`PolarRouting` 和 `CalibratedCompass`。底层并行构图统一使用 `build_parallel`，内部构建上下文命名为 `ParallelBuildContext`。

## 基准程序

Rust 测量入口集中在 `research/benchmarks`，通过 Cargo 中的 `benchmark_*` example 运行。普通余弦和双曲示例位于 `examples`。

```powershell
cargo run --release --features research --example benchmark_geometry_sweep -- --data DATA --n 1000000 --dim 768 --queries 1000 --build-threads 32 --mode a8 --plan PLAN.json --output RESULT.jsonl
```

测量入口保留原有结果标签，用于对照历史记录。具体运行参数和数据文件约定见研究工具说明。[基准入口](../research/benchmarks/README.md)

## 本地研究材料

第三方系统检出归入 `research/third_party/systems`，表示模型与几何研究检出归入 `research/third_party/representations`。当前 Windows 工作区保留 `others` 和 `research/code_repositories` 的目录兼容链接，使既有脚本和记录中的路径可继续解析。

冻结实验的源码、结果、指纹与数据位于原有 `research/artifacts` 路径。实验索引记录其职责和入口，旧记录中的版本标识对应实际测量来源。[实验资料索引](../research/experiments/README.md)

环境变量使用 `QUIVER_DISABLE_AVX512`、`QUIVER_DISABLE_SPECIALIZED_KERNEL`、`QUIVER_EAGER_PRUNE`、`QUIVER_VERIFY_GRAPH` 和 `QUIVER_PROFILE_BUILD`。对应的早期 `TRIVIUM_*` 名称仍可读取，当前名称具有优先级。
