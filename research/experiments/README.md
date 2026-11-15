# 实验资料索引

`catalog.json` 按 ANN 主实验、组件消融、补充数据集和早期表示读出研究组织实验目录。路径相对于本目录。

## 当前 ANN 材料

| 目录标识 | 职责 |
| --- | --- |
| `zen4_full_matrix_v1` | 冻结主矩阵、原有消融与实例一原生基线 |
| `westb_systems_v1` | 实例二原阶段的本地材料与系统基线记录 |
| `zen4_a9_followup_v1` | 七数据集新增组件测量与组合消融 |
| `zen4_wordnet_v1` | WordNet 公开嵌入的 QuIVer 测量 |
| `westc_extreme_v1` | Uniform-Wide、Clustered-Narrow 的三系统基线分片 |
| `westb_extreme_v1` | Uniform-Narrow、Clustered-Wide 的三系统基线分片 |

每个实验以实际保存的 `source_manifest.json`、`measurement_manifest.json` 或部署清单定位源码版本，以 `matrix.json`、`plans/` 和 `assignment.json` 定位实际计划与机器分配。

`completion_verification.json` 与请求审计记录给出完成核验，`archive_manifest.json` 与文件哈希给出归档指纹。双机分片的完成作业集合共同覆盖十二组外部基线。

## 测量口径

当前外部基线分片采用 `M=32`、`efConstruction=256`，构图 32 线程，查询 1/2/4/8/16/24 线程。后续扫描使用十八个单线程 ef 测点，上限 4096。两组已启动的旧计划以其原计划指纹记录，超出上限的结果由补充表保存。

独立构图重复与同图计时轮次分别记录。当前正式矩阵使用单次独立构图，已有额外构图重复按原种子保存。新增组件的收益对比使用共同参数网格中的同 Recall 测点。

冻结材料中的版本标签保留实际含义。后续源码重命名通过迁移说明追溯。[源码迁移说明](../../docs/source-migration.md)
