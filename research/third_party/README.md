# 第三方资料

`repositories.json` 记录本地相关工作检出的上游地址与提交。系统检出位于 `systems`，表示模型和几何研究检出位于 `representations`。

第三方完整检出由 Git 忽略规则保存在本地，检出内的许可证和独立 Git 元数据随原仓库保存。源码工作区通过记录的上游地址和提交定位相应版本。

当前 Windows 工作区的兼容链接如下。

| 原路径 | 当前存放位置 |
| --- | --- |
| `others` | `research/third_party/systems` |
| `research/code_repositories` | `research/third_party/representations` |

`systems/inspection_20261112` 保存早期系统数学与接口核验记录。WordNet 权重来源和数据转换信息记录在对应实验的 `source_provenance.json` 中。[WordNet 与其他实验](../experiments/README.md)
