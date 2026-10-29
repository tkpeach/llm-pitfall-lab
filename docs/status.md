# 仓库现状

本页描述当前代码，而非未来设计。规范与目录导航见 [总入口](README.md)，验证结果见 [当前快照](../evidence/current.json)，已学内容见 [学习索引](../learning/README.html)。维护要求见 [现状与证据审查](maintenance.md#现状与证据的对应审查)。

## 当前能力与证据审查

当前 D0–D3 基础设施与合成安全暂停/恢复机制，以及 D4 固定 4B 语言模型短前向已验证。D4 为单条 22-token 输入、三种分块方式与独立参考对照；不代表目标训练或 D5 恢复已验收。新增评估种子集仍待独立复核。正式实现使用 C++23 + MLX；模型目标和硬件见 [模型路线](model-plan.md)。

| 能力与状态 | 对应证据 | 覆盖与充分性边界 |
| --- | --- | --- |
| 原生构建、依赖与模块边界已验证 | 构建命令成功；`project.build_boundary`、`dependencies.sdk`、`dependencies.lock` | 正式源文件按 C++23 编译，SDK 固定，直接动态链接不含 Python；静态检查不证明任意动态路径安全 |
| CLI 帮助、版本、错误用法已验证 | `cli.contract` | 从无关目录调用；尚无生成、训练、暂停或恢复命令 |
| 严格任务评估与离线命令已验证 | `task.contract`、`task.evaluate` | 62 项原生断言及六条合成契约样例；覆盖重复键、类型/日期、部分评分、空值分组、非法及缺失答案计错和输入拒绝；不证明模型效果或标注语义正确 |
| 候选评估数据自动检查已通过，独立复核待完成 | `task.dataset`、`checks.dataset_failure_cases` | 52 条、26 个改写组，dev/test 候选各 26 条、13 组；检查摘要、分组、字段证据与归一化，并以 6 项破坏用例验证拒绝行为；不证明全部标注语义正确或数据代表性 |
| 轻量只读资源预检已验证 | `resources.preflight`、`resources.cli` | 34 项原生断言及 CLI 检查通过；读取配置、路径元数据和卷可用空间，项目占用返回未知，不遍历文件 |
| 简化写入会话已验证 | `resources.writer`、`checkpoint.transactions` | 18 项原生断言及 37 项 CLI 检查通过；无跨进程租约和分块峰值记账，保留暂存、文件校验和不覆盖发布 |
| 检查点容器与 CLI 已验证 | `checkpoint.store`、`checkpoint.transactions` | 19 项原生断言及新进程核验通过，覆盖摘要/身份/版本、损坏和非普通文件、真实部分写入失败、旧版本保留；合成二进制材料不证明张量语义或训练状态完整 |
| GPU 微型数值链路已验证 | `gpu.smoke` | 沙箱内因 Metal 设备不可见而失败；沙箱外重跑通过，Apple M4 固定矩阵前向和自动微分最大绝对误差均为 0；不代表完整模型正确或性能达标 |
| D3 安全控制机制已验证 | `runner.control`、`training.driver_cleanup`、`training.lifecycle` | 正式 runner 在入口与每个完整更新后的安全边界处理 SIGINT/SIGTERM；已有暂停请求时不再执行更新，仅在请求暂停时调用检查点回调；保存失败返回错误，成功后运行循环返回暂停。驱动的 4 项真实子进程失败回归验证超时及异常时终止并回收 worker。8 步合成 LoRA/Adam 对照在第 3 步暂停，新进程恢复后参数、Adam、RNG、loss 与样本进度 9 类张量逐项匹配，worker 正常退出，请求到退出约 4.3 ms。该结果不代表目标 LLM 训练或 D5 验收 |
| D4 执行制品与目标短前向已验证 | `model.config`、`checks.model_preparation`、`conversion.rules`、`compute.gated_delta`、`model.reference` | 固定官方源摘要通过，语言 affine 4-bit/group-64 产物约 2.23 GiB；单条 22-token 输入下，完整、逐 token、三 token 分块与各自独立参考的 229 项张量误差均为 0，覆盖全部层与混合状态，短生成均为 `2` 后结束。跨方式 BF16 logits 原始漂移约 1.66%–1.68%，与参考自身漂移完全一致，不宣称跨方式逐位相等。原生 MLX 峰值约 2.32–2.33 GiB、最大 RSS 约 2.30 GiB；单次带观测时间不是性能优化基线。BF16 递推仅前向，长上下文、批量、任务效果与训练尚未验收 |
| 正式文档导航与检查器拒绝行为已验证 | `project.docs`、`checks.failure_cases` | 后者包含 8 个单元测试；不证明自然语言规范、状态或学习笔记的语义正确 |

人工审查结论：当前快照源码清单与哈希已同步工作区；完整 CTest 24/24 通过。数值对照已作为独立开发工具迁入 `tools/numerical_validation/qwen35/`，驱动、参考执行、进程管理和原生探针不依赖测试；CTest 的 `model.reference` 仅通过固定输入验收入口调用该能力并核对设计要求。`BUILD_TESTING=OFF` 的独立构建已成功，测试注册数为 0；另用该构建对一条 23-token 输入独立运行对照，229 项张量比较通过。制品工具仍位于 `tools/model_artifacts/qwen35/`；此前完整转换和 9 个文件逐字节一致的证据保留。D3 证据支持合成训练暂停与新进程续接；D4 支持目标模型短前向与参考对照，不覆盖目标训练和长训练。转换沿用固定上游主路径；规则和逐参数审计已进入仓库，另目录完整复现的 9 个有效载荷文件与原制品逐字节一致，来源审计与限制见 [模型短验证](model-validation.md#转换来源与可复现审计)。项目占用本次按阶段实测，具体值及卷空间见当前快照；正式轻量预检仍报告项目用量未知。新增评估种子集仍待独立语义复核；未建立状态—证据的自动语义检查。

## 代码入口与真实调用关系

- [根构建](../CMakeLists.txt) 配置标准与 SDK，进入 [源码入口](../src/CMakeLists.txt) 和 [测试入口](../tests/CMakeLists.txt)；计算库由 [模块构建](../src/compute/CMakeLists.txt) 定义。
- [CLI](../src/main.cpp) 处理帮助、版本和离线评估；评估调用 [任务模块](../src/task/evaluator.cpp)，读取 gold 与原始预测文本，输出分阶段校验、准确率与逐样例诊断。该路径不调用 GPU 或 Python。
- [原生 GPU 集成测试](../tests/integration/gpu_smoke.cpp) 使用 [计算模块](../src/compute/mlx_runtime.cpp)，执行固定矩阵运算、梯度计算和断言；固定样例不属于产品接口。
- [模型制品工具](../tools/model_artifacts/qwen35/README.md) 独立维护官方源下载、转换规则、完整性审计和复现比较；工具不依赖测试，测试从新位置调用工具验证行为。实际转换仍沿用固定上游主路径。
- [固定模型模块](../src/model/qwen35.cpp) 校验执行制品并运行量化语言前向，维护独立请求的 KV、卷积和递推状态；[GDN 接入](../src/compute/gated_delta.cpp) 保留原生 float32 路径并复用有 MIT 来源的混合精度前向内核。[目标探针](../tools/numerical_validation/qwen35/probe.cpp) 由独立数值对照工具调用，不是正式自然语言生成 CLI；Python 显式准备输入和运行独立参考，正式计算不调用 Python。
- [训练运行控制模块](../src/runner/control.cpp) 安装暂停信号处理，在完整更新边界调用保存回调；[生命周期集成测试](../tests/integration/training_lifecycle.cpp) 用合成 LoRA/Adam 步骤验证连续运行与新进程续接。
- [MLX SDK 定位器](../cmake/mlx/locate_sdk.py) 为 [SDK 集成](../cmake/mlx/MLXSDK.cmake) 提供定位及校验；[仓库检查器](../tests/checks/check_project.py) 由 CTest 调用。正式原生程序直接链接 MLX，不调用 MLX-LM。
- [评估种子入口](../fixtures/order-evaluation/v1/README.md) 说明数据身份、标注规则及复核/冻结流程；其检查器仅用于辅助数据审查，正式评分继续使用原有 C++ 实现。
- [资源模块](../src/resources/preflight.cpp) 从 TOML 加载策略，检查项目根/显式外部归属目录及卷空间；当前不遍历文件，项目用量报告为未知。
- [写入会话](../src/resources/writer.cpp) 提供暂存、文件写入和不覆盖发布；[检查点模块](../src/checkpoint/store.cpp) 在保存开始时做一次卷空间预检，再暂存、回读验证并原子发布。[文件原语](../src/storage/files.cpp) 提供描述符、同步、相对路径及 SHA-256，详见 [检查点专题](checkpoints.md)。

准备环境、配置、构建、执行 CLI 和 CTest 的当前命令统一见 [构建与检查](build.md)。当前复验构建目录为 `runs/d2/build`（已配置 `LAB_QWEN35_ARTIFACT`）；D4 执行产物在 `runs/models/qwen35-4b-text-4bit/`，短验证报告在构建目录，准备及过程材料在 `runs/d4/`，数值观测的大张量临时文件位于 `runs/numerical-validation/`。已核验的量化复现副本为恢复卷空间安全线而清理，必要审计小文件保存在 `runs/d4/reproduction-receipts/`，完整源和活动制品保留。产物准备命令见 [固定目标模型短验证](model-validation.md)。资源报告及检查点示例请求/报告在 `runs/d2/`，示例发布版本在 `runs/checkpoints/`，评估示例仍在 `runs/d1/`。这些目录可清理，不作为代码依赖；候选数据复核与正式功能验收不能依赖它们永久存在。

## 尚未实现与已知限制

- 已有固定 4B 语言前向与短生成验收探针，尚无正式自然语言生成 CLI 或原生 tokenizer、LoRA、在线强化学习与可解释性正式运行成果。当前只验收一条短、无 padding 文本输入。
- [磁盘配置](../configs/resources.toml) 已接入轻量预检和检查点写入；预检只计算卷可用空间，项目总占用未知。`checkpoint-save` 开始时做一次峰值检查，无并发租约或写入中的空间复核。D4 显式准备程序和原生探针已按阶段调用同一预检；仍没有系统磁盘配额或并发写入预留。
- 目标模型训练尚未接入通用 runner；BF16 GDN 前向内核尚无反向定义，需 D5 单独实现与验证。当前 `resources` 命令只提供卷空间和指定峰值检查，不提供项目用量估算。
- 已实现的是通用训练运行控制和检查点容器；已有 4B 执行权重，尚无模型训练步骤或正式训练命令。没有自动选择“最新”版本、后台恢复或崩溃残留自动清理。未实测断电与所有存储硬件故障。
- 目标模型训练状态恢复属于 D5；目标模型状态、内存释放量和长训练均未验收。
- 原生计算绑定当前 SDK 路径，移动环境需要重新配置和构建；Metal 测试需要可访问本机 GPU 的执行环境，不支持 CPU 兜底。

## 下一步与接手检查点

数据方面先按种子集说明完成独立人工复核，修正歧义后再冻结 test；不得将候选测试用于提示、few-shot 或训练选择。现阶段保持严格匹配；新增诊断指标、语义裁判等仅为后续参考，不抢先实现。当前未建立训练集，尚不能验证与未来训练数据的隔离。

下一步推进 [D5 训练与恢复](design.md#首轮关口)：先验证混合精度递推反向、目标 loss 与可训练参数梯度，再对照一次 LoRA 更新、adapter 重载及完整检查点的新进程恢复。保留 [D4 短前向基线](model-validation.md)；长训练仍须等待 D5 验收，不能用 D3 合成链路代替。

接手时先核对工作区差异、快照身份与验证环境；受影响而未复验的条目标为待复验。
