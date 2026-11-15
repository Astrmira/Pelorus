# 研究资料

| 目录 | 内容 |
| --- | --- |
| `benchmarks/` | Rust ANN 测量入口，使用根目录 Cargo 工程构建 |
| `experiments/` | 冻结实验的索引、数据集与材料定位 |
| `third_party/` | 相关系统与表示研究的仓库来源和提交记录 |
| `scripts/`、`src/`、`tests/`、`configs/` | 早期 embedding 读出与受控几何实验的 Python 工程 |
| `papers/` | 文献索引与本地阅读材料 |
| `data/` | 本地原始数据和预处理数据 |
| `artifacts/` | 向量、测量输出、冻结源码、指纹和归档 |

根目录 Rust 库使用 `quiver_ann`。本目录 Python 工程使用 `pyproject.toml` 与 `uv.lock` 管理依赖。

## 编码服务配置

Embedding 编码脚本从 `QUIVER_EMBEDDING_API_BASE_URL` 和 `QUIVER_EMBEDDING_API_KEY` 读取服务地址与密钥。环境变量具有优先级，其次读取本目录 `.env` 中的同名字段。首次配置可参考 `.env.example`。本地 `.env` 由 Git 忽略规则管理。

## ANN 测量

```powershell
cargo run --release --features research --example benchmark_geometry_sweep -- --data DATA --n 1000000 --dim 768 --queries 1000 --build-threads 32 --mode a8 --plan PLAN.json --output RESULT.jsonl
```

测量时固定源码与二进制指纹、查询划分、构图参数和实际线程数。不同冻结版本的原始记录按各自的来源读取，同 Recall 的比较使用实际达标测点。[基准工具说明](benchmarks/README.md)

## 实验材料

`experiments/catalog.json` 提供当前 ANN 实验和早期读出实验的相对路径索引。完成情况、查询核验与归档指纹由相应实验目录中的清单记录。[实验资料索引](experiments/README.md)

大型数据、生成结果与第三方完整检出由本地工作区保存。源码仓库记录工具、协议入口、文献及第三方提交信息。[第三方仓库记录](third_party/README.md)
