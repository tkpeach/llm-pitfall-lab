# 构建与检查

本页定义正式 C++ 工程的构建、检查与离线评估入口。适用于当前 Apple Silicon macOS 和固定工具环境，不是完整模型推理或训练指南。

## 准备环境

需要本机 Xcode 命令行工具、CMake 3.25 或更新版本、uv，以及能编译项目所用 C++23 特性的编译器和标准库。CMake 要求 C++23 并关闭 GNU 扩展；正式目标的编译和链接验证实际使用的标准库能力，不另写逐项特性探测程序。

CMake 在当前 Apple Clang 上使用 `-std=c++2b` 别名；正式源文件断言 `__cplusplus >= 202302L`，因此按实际语言级别验收，不只比较标志拼写。不启用 GNU 扩展模式。

从仓库根目录操作。按 [资源规则](resource-policy.md) 规划项目占用，并在大额写入前核对卷空闲和预计峰值。`resources` 命令只检查卷空间，不估算项目总占用；首次安装、依赖同步和构建仍须人工预检。已有 `.venv` 时可直接核对环境，不需要反复安装：

```sh
UV_CACHE_DIR="$PWD/.cache/uv" uv sync --locked
.venv/bin/python cmake/mlx/locate_sdk.py
```

`locate_sdk.py` 不导入 MLX 执行 GPU 代码。它检查辅助环境的 Python 次版本、直接依赖精确版本和 SDK 必要文件，输出 SDK 根目录。环境缺失或版本不符时明确失败。依赖锁的有效性由下述统一检查验证。

## 配置与构建

```sh
cmake -S . -B runs/d0/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -DLAB_MLX_ROOT="$(.venv/bin/python cmake/mlx/locate_sdk.py)"
cmake --build runs/d0/build -j 2
```

构建目录必须独立于源码目录。SDK 路径须显式提供，并与选定 Python 环境中的固定版本一致；不会自动改用系统 MLX 或研究环境。`cmake/mlx/MLXSDK.cmake` 集中处理 SDK 集成及本机 framework 路径，不修改 SDK 包文件。

构建定义按职责和模块组织，源文件在所属目标中显式列出，不递归扫描整个源码目录：

| 文件 | 职责 |
| --- | --- |
| 根 `CMakeLists.txt` | 项目、平台、C++ 标准、SDK、构建信息生成及源码/测试入口 |
| `cmake/mlx/MLXSDK.cmake`、`cmake/mlx/locate_sdk.py` | MLX 构建集成及固定工具环境的 SDK 定位、校验 |
| `src/CMakeLists.txt` | 正式模块入口和 CLI 目标 |
| `src/compute/CMakeLists.txt` | 计算模块目标、头文件路径及 MLX 链接依赖 |
| `src/task/CMakeLists.txt` | 严格任务评估模块与私有 JSON 依赖 |
| `src/resources/CMakeLists.txt` | 磁盘策略、轻量卷空间预检和检查点写入会话，私有使用 TOML/JSON 依赖 |
| `src/storage/CMakeLists.txt`、`src/checkpoint/CMakeLists.txt` | 原生文件原语及检查点容器，资源协调和发布路径见检查点专题 |
| `tests/CMakeLists.txt` | 测试可执行文件和 CTest 注册 |
| `tests/checks/` | 文档、构建边界检查器及检查器自身的错误用例测试 |

后续模块在实现时建立自己的 `CMakeLists.txt`，由源码入口接入；依赖和编译选项在目标上声明，不预建空模块或通用构建框架。根目录保留 `include(CTest)`，通过 `BUILD_TESTING` 控制是否进入测试子目录；默认开启。CLI 与 GPU 测试可执行文件继续输出到构建根目录，现有命令保持不变。

GPU 运行时及其集成测试绑定配置时的 SDK 动态库和 `mlx.metallib` 路径。移动/重建工具环境后，应重新配置和构建；D0 不提供独立安装包或可移动分发能力。原生程序与测试均不需要启动 Python 解释器。

## 原生程序

```sh
./runs/d0/build/pitfall-lab --help
./runs/d0/build/pitfall-lab --version
```

`--version` 中的 MLX SDK 版本是构建时身份，不表示执行了 GPU 初始化。GPU 初始化由计算模块负责，固定数值样例属于独立集成测试。

CLI 还提供下述离线评估和资源预检。退出码：`0` 操作完成（评估得分可以为 0），`1` 输入/读写错误或资源预检拒绝，`2` 用法错误。无参数、未知命令或多余参数均拒绝。帮助、版本、评估和资源预检命令不初始化 GPU。

## 离线任务评估

先准备符合 [任务契约](task-spec.md#离线评估输入与报告) 的预测文件，再执行：

```sh
./runs/d0/build/pitfall-lab evaluate \
  --gold fixtures/task_contract.json \
  --predictions predictions.json
```

`predictions.json` 是用户准备的输入路径，不是仓库已有文件。命令只读取输入并将报告输出到标准输出，不提取或修复答案。契约、输入边界、错误区分和指标解释统一见任务规范。正式任务模块使用仓库内固定版本的 [JSON 依赖](../third_party/nlohmann/README.md)，无需额外安装或构建时下载。

## 只读资源预检

```sh
./runs/d0/build/pitfall-lab resources \
  --root "$PWD" --config configs/resources.toml --peak-bytes 0
```

`--peak-bytes` 是本次操作预计新增的临时峰值，须包括转换中间产物、最终文件和恢复预留；`0` 仅检查当前状态，不代表任何后续操作已经获准。项目用量未测量，因此命令不产生项目预算警告；卷剩余空间安全线或本次峰值不满足时输出拒绝原因并退出 `1`，配置或目录错误在 stderr 报告。

资源模块严格读取四个配置项，拒绝缺失、额外键、错误类型和无效阈值；配置文件上限为 64 KiB。轻量实现检查项目根和显式外部归属目录本身，不遍历其中的文件；项目占用字段为 `null`，表示尚未测量。外部路径须为同卷真实绝对目录，缺失、目录符号链接或跨卷均拒绝。

预检只用卷可用空间底线判断峰值，并报告项目占用为未测量。此只读命令不写入产物、不删除文件、不修改系统设置，也不创建额度预留。卷空间查询不能阻止其他进程随后写盘；检查点保存会在开始时进行一次卷空间检查，但没有跨进程写入租约。

## 检查点保存与核验

`checkpoint-save` 在一次卷空间预检后暂存、校验并原子发布；`checkpoint-verify` 在新进程核对显式版本的身份与文件摘要，不启动训练。请求格式、命令及失败语义见 [受控写入与检查点](checkpoints.md)。

## 统一检查

```sh
ctest --test-dir runs/d0/build --output-on-failure --no-tests=error
```

| 测试 | 检查内容 |
| --- | --- |
| `project.docs` | 正式文档的本地内联链接、标题锚点和总入口索引覆盖；不校验远端网页 |
| `project.build_boundary` | 实际编译数据库中的 C++23、研究目录依赖边界，以及原生 CLI 和 GPU 集成测试的直接动态链接 |
| `dependencies.sdk` | 所选辅助环境与 SDK 文件及版本 |
| `dependencies.lock` | `uv lock --offline --check`，锁文件过期或离线所需元数据缺失时报错；不自动更新或下载 |
| `checks.failure_cases` | 用受控错误验证导航、编译标准、SDK 与研究目录依赖检查能拒绝问题 |
| `cli.contract` | 从无关工作目录验证原生帮助、版本及错误用法 |
| `gpu.smoke` | CTest 直接从独立工作目录运行 C++ 集成测试，调用正式初始化后验证 GPU 前向/反向 |
| `task.contract` | 原生严格解析、字段类型、日期、非法输出与部分计分检查 |
| `task.evaluate` | 调用原生 CLI 验证六条契约样例、ID 匹配、空值分组、阶段指标、缺失/非法输出分母及输入错误 |
| `task.dataset` | 检查候选评估种子的身份、分组隔离、标注证据/归一化及原生 gold 合法性；不替代独立人工语义复核 |
| `checks.dataset_failure_cases` | 用破坏哈希、证据、标注、分组和缺失理由的临时副本验证检查器拒绝错误 |
| `resources.preflight` | 配置拒绝、项目用量未测量状态、卷空间峰值边界及额外归属路径校验 |
| `resources.cli` | 从无关工作目录执行轻量只读预检，检查允许/拒绝、参数/配置错误及不改写产物 |
| `resources.writer` | 暂存写入/清理、路径边界与不覆盖发布 |
| `checkpoint.store` | 状态往返、已知 SHA-256、身份/格式错误拒绝及旧检查点保留 |
| `checkpoint.transactions` | 原生 CLI、新进程核验、真实部分写入失败以及损坏/路径错误拒绝 |

`tests/integration/gpu_smoke.cpp` 保存固定输入、解析结果、容差断言和测试报告；这些不属于正式计算接口。测试检查小矩阵乘法与 loss 对权重的梯度，最大绝对误差不得超过 `1e-5`，失败退出码为 `1`。GPU 不可用时明确失败，不执行 CPU 替代路径。需要单独诊断时可直接运行：

```sh
./runs/d0/build/lab_gpu_smoke
```

该测试报告设备、实际 MLX/C++ 版本、两类数值误差及 MLX 分配峰值；它不是模型性能基线，也不证明完整模型可行。原生 CLI 不调用测试程序。

仅检查不需要 GPU 的部分时，可执行：

```sh
ctest --test-dir runs/d0/build -L host --output-on-failure --no-tests=error
```

验收保持 `BUILD_TESTING=ON`（默认值）；`--no-tests=error` 防止零测试被当作通过。只运行 `host` 不算 D0 全部通过。`gpu` 标签的测试必须在具有本机 Metal 访问能力的执行环境中运行；失败不能改为跳过后宣称验收通过。各测试有超时，GPU 测试串行执行，不创建常驻进程。

构建文件、编译数据库和 CTest 日志存于 `runs/`，不提交 Git。静态边界检查只能发现显式依赖，不能证明任意动态构造的路径都安全；任务契约、资源保护和恢复行为的检查在对应模块建立后接入。
