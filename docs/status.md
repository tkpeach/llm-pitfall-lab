# 仓库现状

本页描述当前代码，而非未来设计。规范与目录导航见 [总入口](README.md)，验证结果见 [当前快照](../evidence/current.json)，已学内容见 [学习索引](../learning/README.md)。维护要求见 [现状与证据审查](maintenance.md#现状与证据的对应审查)。

## 当前能力与证据审查

当前处于 D0 完成、D1 尚未开始的位置。正式实现使用 C++23 + MLX；模型目标和硬件见 [模型路线](model-plan.md)。下表检查名称对应快照中的 `ctest.tests`。

| 能力与状态 | 对应证据 | 覆盖与充分性边界 |
| --- | --- | --- |
| 原生构建、依赖与模块边界已验证 | 构建命令成功；`project.build_boundary`、`dependencies.sdk`、`dependencies.lock` | 正式源文件按 C++23 编译，SDK 固定，直接动态链接不含 Python；静态检查不证明任意动态路径安全 |
| CLI 帮助、版本、错误用法已验证 | `cli.contract` | 从无关目录调用；尚无生成、训练、暂停或恢复命令 |
| GPU 初始化与微型数值链路已验证 | `gpu.smoke` | Apple M4 GPU 上固定矩阵前向和自动微分最大绝对误差均为 0，阈值为 1e-5；不代表完整模型正确或性能达标 |
| 正式文档导航与检查器拒绝行为已验证 | `project.docs`、`checks.failure_cases` | 后者包含 8 个单元测试；不证明自然语言规范、状态或学习笔记的语义正确 |

人工审查结论：当前快照的源码清单、哈希与本次工作区一致，上述 7 项检查足以支持 D0 范围；不足以支持目标模型、资源保护、训练或恢复验收。文档调整后的链接检查纳入本次验证。未建立状态—证据的自动语义检查。

## 代码入口与真实调用关系

- [根构建](../CMakeLists.txt) 配置标准与 SDK，进入 [源码入口](../src/CMakeLists.txt) 和 [测试入口](../tests/CMakeLists.txt)；计算库由 [模块构建](../src/compute/CMakeLists.txt) 定义。
- [CLI](../src/main.cpp) 当前只读取生成的构建信息，处理帮助、版本与用法错误，不调用 GPU 初始化。
- [原生 GPU 集成测试](../tests/integration/gpu_smoke.cpp) 使用 [计算模块](../src/compute/mlx_runtime.cpp)，执行固定矩阵运算、梯度计算和断言；固定样例不属于产品接口。
- [MLX SDK 定位器](../cmake/mlx/locate_sdk.py) 为 [SDK 集成](../cmake/mlx/MLXSDK.cmake) 提供定位及校验；[仓库检查器](../tests/checks/check_project.py) 由 CTest 调用。正式原生程序直接链接 MLX，不调用 MLX-LM。

准备环境、配置、构建、执行 CLI 和 CTest 的当前命令统一见 [构建与检查](build.md)。当前复验构建目录为 `runs/d0/module-layout`；目录可清理，重新构建即可，不作为代码依赖。

## 尚未实现与已知限制

- 尚无完整目标权重下载或模型加载、生成、LoRA、在线强化学习与可解释性正式运行成果。参考机制实验不计入这些能力的验收。
- [磁盘配置](../configs/resources.toml) 和 [任务样例](../fixtures/task_contract.json) 已有；正式 C++ 任务评估、资源额度保护和完整检查点尚未实现。安装及构建前仍须人工预检资源。
- 暂停、退出释放资源和新进程恢复尚未实现；不能启动长训练来代替短验证。
- 原生计算绑定当前 SDK 路径，移动环境需要重新配置和构建；Metal 测试需要可访问本机 GPU 的执行环境，不支持 CPU 兜底。

## 下一步与接手检查点

下一步实现 [D1 任务契约](design.md#首轮关口)：读取 [任务规范](task-spec.md) 与手写样例，从一个确定性失败用例开始，建立 C++ 严格 JSON 评估器。完成条件是 schema、归一化与评分符合规范，正确和错误样例均通过行为测试，接入 CTest，并更新本页及当前快照；不以 loss 或模型输出替代契约检查。

接手时先核对工作区差异、快照身份与验证环境；受影响而未复验的条目标为待复验。D1 后依次推进 D2 资源与检查点、D3 安全控制，再进入 D4/D5 固定模型短验证。
