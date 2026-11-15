# ANN 测量入口

这些入口以 `research` feature 编译，配置类型与诊断函数从 `quiver_ann::research` 导入。运行时策略与数据布局在计时窗口之外设置。

| Cargo example | 文件 | 用途 |
| --- | --- | --- |
| `benchmark_geometry_sweep` | `geometry_sweep.rs` | 查询计划、逐请求记录、目标 Recall 与线程扩展 |
| `benchmark_controlled_geometry` | `controlled_geometry.rs` | 查询策略与几何表示的受控对照 |
| `benchmark_cohere_geometry` | `cohere_geometry.rs` | Cohere 数据的几何表示测量 |
| `benchmark_hyperbolic_navigation` | `hyperbolic_navigation.rs` | 精确与量化双曲导航 |
| `benchmark_geometry_components` | `geometry_components.rs` | 剪枝、路由与表示组件比较 |
| `benchmark_hot_storage` | `hot_storage.rs` | 合表布局与草图过滤 |
| `benchmark_execution_strategies` | `execution_strategies.rs` | 查询执行策略与阶段计时 |
| `benchmark_kernel_throughput` | `kernel_throughput.rs` | 符号幅值位计数内核 |
| `benchmark_distribution_adaptation` | `distribution_adaptation.rs` | 不同合成分布与残差编码 |

## 数据约定

`geometry_sweep` 从数据目录读取以下小端文件。

| 文件 | 形状与类型 |
| --- | --- |
| `documents.f32` | 底库向量，`N × D` 个 `f32` |
| `queries.f32` | 测试查询，`Q × D` 个 `f32` |
| `truth.u64` | 每条测试查询的前十近邻外部标识，`Q × 10` 个 `u64` |
| `tuning_queries.f32` | 独立调参查询，选择调参划分时读取 |
| `tuning_truth.u64` | 独立调参查询的前十近邻标识 |

构图与查询条件由 `--plan` 指定，结果写入 `--output`。逐请求延迟、命中与返回标识写入相邻的请求文件。构图参数、线程数、查询划分与结果指纹在测量记录中保留。

历史协议中的 `a8`、`a7`、`sq8`、`fp32` 和 `vamana` 是测量模式标签。它们分别映射到当前具名的极坐标、固定壳层罗盘、标量量化方向、全精度方向和测地剪枝策略。

已部署实验使用各自冻结目录中的入口。新目录中的程序面向后续源码开发与测量。[冻结材料位置](../experiments/README.md)
