# Qwen3.5 模型制品工具

这里维护固定 Qwen3.5-4B 的来源获取、格式转换、完整性审计和复现工具。这些工具属于显式制品构建流程，不依赖测试入口，也不被正式 C++ 推理或训练调用。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| [prepare_qwen35.py](prepare_qwen35.py) | 固定 revision 下载、源摘要校验、资源预检，以及调用上游转换主路径生成语言制品 |
| [qwen35_conversion_rules.json](qwen35_conversion_rules.json) | 上游来源、依赖版本及摘要、参数筛选/命名/布局/归一化与量化规则 |
| [qwen35_conversion.py](qwen35_conversion.py) | 核对转换器身份与上游实际输出，生成逐参数审计；不替代上游转换 |
| [verify_qwen35_reproduction.py](verify_qwen35_reproduction.py) | 核验两份制品完整性，比较权重、配置、索引及附属文件的大小与摘要 |

## 使用与记录

从项目根使用辅助环境运行 `prepare_qwen35.py download` 获取固定源，再使用 `convert` 写入新的执行目录。再次转换时显式指定不存在的 `--output` 目录，最后运行比较器核对基线和新制品。完整参数、当前路径、资源边界及命令统一维护在 [制品准备与复现说明](../../../docs/model-validation.md)，不在此复制第二套操作规范。

规则中的 sanitizer 来源为 MLX-LM 0.32.2 的 Qwen3.5 实现，量化、加载与保存也沿用固定上游。输入权重来源、命名和数值变换的含义见 [Qwen 专题](../../../learning/topics/qwen35-components-and-weight-conversion.html)。

脚本、规则和依赖锁随 Git 管理。每次转换的 `conversion-recipe.json`、`converter-identity.json`、`conversion-audit.json` 与 artifact 清单随产物保存在 `runs/`；必要的小型验收摘要由 [当前证据](../../../evidence/current.json) 维护。不能为了整理目录而改写既有制品的历史来源记录。

## 与测试的关系

准备工具测试留在 [test_model_preparation.py](../../../tests/checks/test_model_preparation.py)，小张量规则测试留在 [test_qwen35_conversion.py](../../../tests/checks/test_qwen35_conversion.py)，通过 CTest 的 `checks.model_preparation`、`conversion.rules` 运行。独立模型数值对照属于 [开发诊断工具](../../numerical_validation/qwen35/README.md)。测试可以调用这里的工具；工具不导入测试代码。
