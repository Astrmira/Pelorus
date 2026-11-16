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
| `benchmark_variable_curvature_50k` | `variable_curvature_50k.rs` | 纯合成 50k 变曲率（Tier 2 乘积 / Tier 3 径向翘曲）工程探索原型（仅 `research`，不转正） |

## 冷向量重排精度

`rerank_precision.py` 从 MERU 发布目录的 `mapping.json`、`documents.f32` 和 `queries.f32` 读取数据，固定种子抽取文档与查询。每种表示对相同文档子集全扫描，原始 FP32 查询保持固定，统一使用 FP64 距离累加，以该子集原始 FP32 坐标的精确前十名为真值。

```powershell
python research/benchmarks/rerank_precision.py --data DATA_DIR --output research/artifacts/rerank_precision --documents 100000 --queries 1000 --threads 8
```

运行环境需要 NumPy 和 threadpoolctl。结果包含逐查询命中、前十名标识、Recall@10、损失百分点、Top-1 与完整顺序一致率、无效向量数、编码载荷和输入指纹。全扫描耗时用于记录实验执行情况，SSD I/O 与 ANN 吞吐需要另外测量。

编码比较覆盖 FP32、FP16、BF16、逐向量对称 SQ8/SQ4、文档集合均值中心化 SQ8/SQ4，以及 128/64 维分组 minmax 仿射 SQ8/SQ4。分组仿射编码保存每组 FP32 尺度和中点偏置，与 TDB SSD 实验中的 scale/bias 表示对应。此处查询保持原始精度，评分通过解码后的 FP64 距离进行，尚未复现 TDB 的查询 INT8 量化和 VNNI 内核。

`polar` 模式额外保存每条向量的原始 FP64 欧氏半径，压缩单位方向并在解码后重新归一化。`cartesian_norm` 模式额外保存 FP32 范数恢复系数。中心化方案的共享逐维 FP32 均值空间单列，量化参数、半径与范数系数计入逐向量载荷，页面对齐和文件头需要在持久化实验中另计。

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

## 实际冷向量存储入口

`benchmark_rerank_storage` 使用公开 Rust API，在同一张图上遍历全部 26 种编码配置。输入目录提供 `documents.f32`、外部标识 `ids.u64`、`queries.f32` 和独立精确子集真值 `truth.u64`。每查询的固定重排池取原始 FP32 精确前 128 名，普通查询使用 `ef_search=256`、`rerank_limit=128`。

```powershell
cargo run --release --features research --example benchmark_rerank_storage -- --data DATA_DIR --output RESULT.json --dim 512 --queries 1000 --threads 8
```

每档均保存完整快照并映射重开，逐查询核对普通查询与候选重排的标识、顺序和分数。输出区分纯重排 Recall、普通 ANN Recall、原始与重排映射字节数和编码准备耗时。映射测量未清空文件页缓存，包含结果核对，正式冷 SSD 吞吐与尾延迟需另行测量。

`benchmark_storage_sweep` 提供 `prepare` 与 `run` 两个阶段。前者以32线程构建一张图并冻结四种重排精度快照，后者使用相同快照测量热内存、128 MiB 和512 MiB显式 SSD 缓存。Linux SSD 测量使用 `O_DIRECT`，10,000条测试查询按固定顺序打乱，另用2,000条独立调参查询预热。每条件清空显式页缓存后执行独立预热，计时包含查询及已提交预读排空，逐请求文件写入和正确性核对在计时外执行。

`run_storage_campaign.py` 顺序执行1线程曲线，再按目标Recall选择同精度共同工作点，测量2/4/8/16/24线程。查询与 I/O 工作线程共享每条件指定的CPU预算。控制器保留协议、数据和二进制哈希、计划、逐请求延迟及返回结果、进程I/O和缓存计数。完成记录验证后可续跑，所有存储与线程的结果指纹必须一致。每条件只做一次计时，曲线和多线程使用的条件保存在结果中。

当前 SSD 驱动使用 QuIVer 的单调度线程 io_uring 固定文件、固定缓冲、深度256及批量完成回收路径。结果包含实际驱动、注册字节数、在途峰值、提交与完成批次及 IOWQ 注册能力。显式页读取字节数与进程 `/proc/self/io` 独立记录，内核异步工作线程的 I/O 记账归属可能不同。AVX2 与原生 AVX-512 的对照分别冻结二进制，FP32 与压缩记录使用对应的 SIMD 内核。

`benchmark_storage_profile` 对同一快照执行1线程诊断，先记录普通查询耗时，再记录阶段计时，最后单独统计公开查询入口的堆分配。逐查询CSV包含高层下降、路由、图扩展、Stage-0.5、预算选择、重排整体及距离核耗时，以及实际重排数、径向剔除、扩展次数和分配次数。阶段计时含逐候选时钟采样开销，普通查询耗时单列。使用 `--data DATA_DIR --plan PLAN.json --ef 128 --output PROFILE.json` 调用，ef128与ef256及两种SIMD分别保存结果。
