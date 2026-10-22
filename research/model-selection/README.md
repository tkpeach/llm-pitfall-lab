# 前期选型与可行性调研归档

本目录集中保存正式项目启动前的候选筛选、配置快照、环境与核心算子探测，以及相应依赖环境。它们是决策过程记录，不属于正式推理、训练、资源保护或暂停恢复实现，也不应作为正式项目运行时依赖。

## 内容索引

| 目录/文件 | 内容 |
| --- | --- |
| [docs/feasibility.md](docs/feasibility.md) | 硬件检查、Granite 等候选筛选及早期验证记录 |
| [docs/qwen35-assessment.md](docs/qwen35-assessment.md) | Qwen3.5 资源、训练、优化限制与 4B→9B 路线讨论 |
| [scripts/probe_environment.py](scripts/probe_environment.py) | GPU 算术、梯度和 Granite 配置解析探测 |
| [scripts/probe_qwen35_recurrence.py](scripts/probe_qwen35_recurrence.py) | Qwen3.5 单递推算子的前向/反向探测 |
| [evidence/](evidence/) | 原始 JSON 测量结果、配置 revision 与包大小快照，归档时未改写 |
| [pyproject.toml](pyproject.toml)、[uv.lock](uv.lock)、[.python-version](.python-version) | 选型探测环境及完整依赖锁，不代表正式工程依赖已定稿 |
| `.venv/`、`.cache/` | 本机探测环境与缓存，已忽略，不加入 Git，仍计入项目磁盘预算 |

文档中的“当前”“下一步”、候选优先级与待定事项是讨论时的记录，可能已经被后续决策取代。正式需求以 [学习规格](../../docs/learning-spec.md)、[模型路线](../../docs/model-plan.md)、[资源规则](../../docs/resource-policy.md) 和根目录 [AGENTS.md](../../AGENTS.md) 为准。

## 复现探测

在本目录执行，使用允许访问 Metal GPU 的环境：

```sh
UV_CACHE_DIR="$PWD/.cache/uv" uv sync --frozen
.venv/bin/python scripts/probe_environment.py --output runs/environment-probe.json
.venv/bin/python scripts/probe_qwen35_recurrence.py --output runs/qwen35-recurrence-probe.json
```

输出到忽略的 `runs/`，避免覆盖原始 `evidence/`。脚本不下载预训练权重，不验证完整模型训练能力，也没有正式项目资源预检与暂停恢复机制；复现前应确认资源预算。默认沙箱可能无法访问 Metal，不得以 CPU 回退冒充通过。

归档时仅调整文件位置、内部路径和说明，原始证据的时间、数值及身份信息保留。之后如果正式项目需要类似探测，应在正式实验设计中重新定义接口、资源约束和验收，而不是直接引用本目录作为主路径。
