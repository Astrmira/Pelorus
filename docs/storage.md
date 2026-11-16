# 冷向量存储与重排精度

Pelorus 将图、2-bit 导航编码、极坐标元数据及路由状态保存在内存中。量化内存模式仅将预编码重排记录加载到堆中，原始向量采用文件后备，供原始读取和维护按需访问。SSD模式将重排记录也放在文件中。新增向量进入内存增量层。

## 配置与普通查询

```rust
use quiver_ann::{
    BuildConfig, Index, PoincareNavigation, RerankConfig, RerankPrecision,
    ScalarQuantization, SearchConfig, StorageMode,
};

fn main() -> quiver_ann::Result<()> {
    let mut index = Index::build_poincare(
        3,
        &[100, 200],
        &[0.1, 0.0, 0.0, 0.4, 0.0, 0.0],
        &BuildConfig::default(),
        1.0,
        PoincareNavigation::GeometryBq2,
    )?;
    index.set_rerank_config(RerankConfig {
        precision: RerankPrecision::Sq8,
        scalar_quantization: ScalarQuantization::Centered,
        ..RerankConfig::default()
    })?;
    index.set_storage_mode(StorageMode::Mmap)?;
    index.save_snapshot("vectors.pelorus")?;

    let restored = Index::load_mmap("vectors.pelorus")?;
    let hits = restored.search(&[0.12, 0.0, 0.0], &SearchConfig::default())?;
    assert_eq!(hits[0].0, 100);
    Ok(())
}
```

`set_storage_mode` 可在 `Memory` 和 `Mmap` 之间切换。运行时切换建立索引独占的临时映射，其文件由索引生命周期管理。`save_snapshot` 发布持久快照，`load_mmap` 直接映射持久向量并恢复保存的实际图。

双曲索引可通过 `prepare_colocated_hot_store(true)` 准备单页合表并释放分离签名及元数据。完整快照会保存该布局及残差签名，重开后直接恢复。

完整可执行示例位于 [`examples/ssd.rs`](../examples/ssd.rs)，使用 `cargo run --release --example ssd` 运行。

## 直接 SSD 读取与有界缓存池

Linux构建需要C编译器、归档器及Linux io_uring UAPI头文件，默认调用 `cc` 与 `ar`，可通过 `CC` 和 `AR` 指定工具。

Linux 可用 `Index::open_ssd(path, SsdCacheConfig::default())` 打开直接读取后端。重排记录通过 `O_DIRECT` 按 4096 B 页读取，共享缓存默认 128 MiB，在途缓冲预算默认 32 MiB。512 MiB 缓存使用下列配置。

```rust,no_run
use quiver_ann::{Index, SsdCacheConfig};

# fn main() -> quiver_ann::Result<()> {
let mut index = Index::open_ssd("vectors.pelorus", SsdCacheConfig {
    cache_bytes: 512 * 1024 * 1024,
    ..SsdCacheConfig::default()
})?;
let before = index.ssd_io_stats().unwrap();
// 执行查询后，等待已提交的预读完成，再记录完整 I/O 增量。
index.drain_ssd_reads();
let after = index.ssd_io_stats().unwrap();
assert!(after.read_bytes >= before.read_bytes);
# Ok(())
# }
```

缓存采用 16 分片 FIFO，同页在途读取合并。默认 `SsdIoDriver::IoUring` 沿用 QuIVer 的单调度线程协议，创建深度 256 的 ring，注册只读文件和对齐匿名缓冲池，批量提交 `READ_FIXED` 并回收完成队列。查询按 32 个候选的滑动窗口预读，`prefetch_window=0` 关闭候选预读。需求读取在队列满时等待，额外预读跳过并计数。调度器有在途 DMA 时采用非阻塞租约获取，空闲缓存页先淘汰再复用。读取错误唤醒相关等待者，关闭时取消未提交预读并排空已提交 DMA。

固定缓冲注册覆盖整个 `cache_bytes + inflight_bytes` 池，128 MiB 缓存对应 160 MiB 注册内存，512 MiB 缓存对应 544 MiB。操作系统会锁定这些页，进程需有足够的 `RLIMIT_MEMLOCK`。池中的页在最后一个读者释放后复用。`registered_buffer_bytes` 单列注册容量，`pool_live_bytes` 单列被缓存、查询或 DMA 租用的页。

`ssd_io_stats()` 报告实际驱动、调度线程数、提交和完成数、批次数、在途峰值及错误。支持 IOWQ 控制的内核会注册当前调度线程的 CPU 亲和性及工作线程上限。旧内核缺少这些注册时继续使用固定缓冲批量读取，能力字段 `io_wq_affinity_registered` 和 `io_wq_workers_registered` 记录实际状态。ring 创建或固定缓冲注册失败直接返回错误。显式选择 `SsdIoDriver::Pread` 可使用阻塞读取驱动，其线程数由 `io_workers` 配置。

`ssd_io_stats()` 报告缓存命中、等待在途页、查询内复用、物理读取次数及字节、预读使用与浪费、淘汰、缓存载荷及池使用峰值。`cache_hits` 统计需求读取命中共享缓存，`pending_hits` 统计需求读取等待已提交的页，`query_page_hits` 统计查询内复用上一页。`prefetch_requests` 包含命中及合并的预读请求。进程 RSS 另包含图、映射、查询工作区、队列和缓存目录。

`set_ssd_cache_config` 在已映射索引上创建新的缓存运行时，支持调整缓存容量。`clear_ssd_cache` 等待在途请求排空后清除缓存，累计计数继续保留。切换回 `Memory` 会关闭直接读取运行时。配置重排精度与重建会沿用已启用的 SSD 配置。新增记录使用内存增量层，保存重开后进入文件基础层。

直接读取后端当前支持 Linux，其他平台可使用 `load_mmap`。缓存大小及 I/O 线程属于打开句柄的运行时配置，通过 `open_ssd` 指定。快照校验、原始向量访问和维护继续通过映射执行，普通双曲量化导航的最终重排从显式页缓存及直接读取后端取数。

## 重排表示

`RerankPrecision` 提供 `Fp32`、`Fp16`、`Bf16`、`Sq8` 和 `Sq4`。FP16 与 BF16 使用最近偶数舍入。SQ8 使用 −127 至 127，SQ4 使用 −7 至 7，并实际按半字节打包。SQ4 的未使用码字与奇数维填充会在读取时验证。

`ScalarQuantization` 为 SQ 编码提供三种参数布局。

| 配置 | 编码方式 | 附加参数 |
| --- | --- | --- |
| `Symmetric` | 每向量最大绝对值对称量化 | 每向量 4 B 尺度 |
| `Centered` | 训练文档集合的逐维均值，再对每向量残差对称量化 | 共享 `4D` B 中心，每向量 4 B 尺度 |
| `Affine { group_size }` | 每向量每组独立 minmax 仿射量化 | 每组 4 B 尺度及 4 B 中点偏置 |

分组支持任意正组宽及非整组尾部。128/64 维组对应 TDB SSD 实验使用的分组仿射思路。SQ 编码配置影响最终重排，构图与导航继续读取原始输入或现有热编码。

`RerankRepresentation::Cartesian` 压缩笛卡尔坐标。`norm_correction` 可额外保存一个 FP32 系数，将重构范数恢复到原始范数附近。`Polar` 额外保存原始 FP64 半径平方并压缩单位方向。编码或加载时预计算每条记录的FP64缩放和评分因子，共16 B/向量，查询直接在寄存器中重构码字并评分。保存半径平方可直接恢复边界分母，避免再次平方半径引入临界边界舍入。

笛卡尔压缩在准备阶段使用解码坐标计算边界因子。原始坐标的径向下界在这条路径关闭。极坐标重排保留原始径向信息，可继续使用原径向下界。

压缩精排提供AVX2/FMA/F16C、AVX-512及标量路径，运行时按CPU特征分派。FP16、BF16、SQ8、打包SQ4及可对齐分组均使用融合码字重构与FP64距离/内积累加。任意分组和维度尾部有标量回退。`QUIVER_DISABLE_AVX512=1` 同时关闭导航和FP32、压缩精排的显式AVX-512分派。

所有压缩重排的评分使用 FP64 计算，查询保持调用方输入的 FP32 精度。配置完成前会验证编码结果的有限性和双曲定义域。失败会返回错误并保留当前配置。`rerank_candidates(query, ids, top_k)` 对固定候选执行同一套重排，按分数和外部标识排序，并跳过不存在、已删除和重复标识。

## 原始向量与空间统计

原始 FP32 向量独立保存在文件中，供 `vector()`、`search_exact()`、精确导航、插入及重建按需访问。启用编码后，内存模式也采用原始文件映射，普通几何量化查询只读取热导航与当前编码记录。回到默认FP32内存配置时，原始向量恢复为堆存储。

压缩比描述重排向量载荷。持久快照同时包括原始FP32向量、图和导航状态。`vector_storage_stats()` 分别报告原始与重排向量的堆载荷、逻辑映射长度、共享中心、`rerank_metadata_bytes` 预计算元数据及单条记录长度。映射驻留、进程RSS、文件头和图内存另计。维护或显式原始向量访问会按需触及原始文件页。

## 插入、删除与重建

插入先验证原始输入和新重排记录，再预留空间并更新索引。映射基础层保持只读，原始与压缩记录同时追加到内存增量层。软删除同时作用于普通查询、精确扫描和候选重排，删除后的外部标识可重新使用。

中心化编码在插入时沿用已冻结的中心。显式调用 `set_rerank_config` 或重建会根据当前活跃文档重新训练中心。`compact` 和 `rebuild_with_config` 保留重排配置及运行时存储模式。重建会读取活跃原始向量并使用构图工作区。

## 持久化与发布

`save_snapshot` 将实际邻接表、层级、入口、墓碑、BQ 签名、极坐标路由、残差字典、单页合表状态、原始向量和重排编码一起保存。加载会验证入口与快照的 CRC、长度、目录、记录和坐标定义域，并直接恢复图状态。

一次快照包含一个用户命名的入口文件及其同目录数据文件。

```text
vectors.pelorus
.pelorus-<generation>.snapshot
```

数据文件写入并同步后，入口通过临时文件原子替换。后续保存生成新的不可变数据文件，现有读者继续访问旧代。已发布的数据文件保持只读，应用需保留当前入口引用的文件及现有读者使用的代际。旧数据文件保留在目录中，由宿主在读者关闭并暂停打开、发布操作后清理。备份可对已打开索引调用 `save_snapshot`，在目标目录生成完整的一致快照。

`Index::open(path, StorageMode::Memory)` 恢复热导航和编码重排记录到堆，编码索引的原始FP32保留只读映射。默认FP32索引恢复原始向量到堆。`Index::open(path, StorageMode::Mmap)` 和 `load_mmap` 使用映射。加载完成完整校验与评分元数据准备，查询直接使用准备结果。

`save` 在压缩重排或 mmap 模式下采用完整快照。默认内存 FP32 索引继续使用历史单文件保存协议。`load` 自动识别两种入口，已有 v1–v3 文件继续可读，并按历史协议重建派生图。

## 验证与测量

核心回归覆盖多精度与分组尾部、独立测地公式、编码失败原子性、映射增量、删除与标识复用、存储切换、保存重开、损坏拒绝、并发旧读者，以及残差字典与单页合表的实际状态恢复。

[`research/benchmarks/rerank_storage.rs`](../research/benchmarks/rerank_storage.rs) 在 MERU 子集上调用实际公开 API，对内存和映射的普通查询及固定候选重排逐请求核对。该测量复用文件页缓存，耗时包含结果对账，冷 SSD 吞吐与尾延迟需要独立协议测量。
