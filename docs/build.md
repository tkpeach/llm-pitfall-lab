# 构建与检查

本页定义正式 C++ 工程的操作入口。适用于当前 Apple Silicon macOS 和固定工具环境，不是完整模型推理或训练指南。

## 准备环境

需要本机 Xcode 命令行工具、CMake 3.25 或更新版本、uv，以及能编译项目所用 C++23 特性的编译器和标准库。CMake 要求 C++23 并关闭 GNU 扩展；正式目标的编译和链接验证实际使用的标准库能力，不另写逐项特性探测程序。

CMake 在当前 Apple Clang 上使用 `-std=c++2b` 别名；正式源文件断言 `__cplusplus >= 202302L`，因此按实际语言级别验收，不只比较标志拼写。不启用 GNU 扩展模式。

从仓库根目录操作。先按 [资源规则](resource-policy.md) 检查项目占用、卷空闲和预计写入峰值；D0 尚未实现正式资源保护。已有 `.venv` 时可直接核对环境，不需要反复安装：

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
| `tests/CMakeLists.txt` | 测试可执行文件和 CTest 注册 |
| `tests/checks/` | 文档、构建边界检查器及检查器自身的错误用例测试 |

后续模块在实现时建立自己的 `CMakeLists.txt`，由源码入口接入；依赖和编译选项在目标上声明，不预建空模块或通用构建框架。根目录保留 `include(CTest)`，通过 `BUILD_TESTING` 控制是否进入测试子目录；默认开启。CLI 与 GPU 测试可执行文件继续输出到构建根目录，现有命令保持不变。

GPU 运行时及其集成测试绑定配置时的 SDK 动态库和 `mlx.metallib` 路径。移动/重建工具环境后，应重新配置和构建；D0 不提供独立安装包或可移动分发能力。原生程序与测试均不需要启动 Python 解释器。

## 原生程序

```sh
./runs/d0/build/pitfall-lab --help
./runs/d0/build/pitfall-lab --version
```

正式 CLI 当前仅提供帮助和构建版本信息，`--version` 中的 MLX SDK 版本是构建时身份，不表示执行了 GPU 初始化。GPU 初始化接口位于 `src/compute/`，在需要计算的工作进程启动时调用，不执行测试样例。

CLI 退出码：`0` 成功，`2` 用法错误；无参数、未知命令或多余参数均拒绝。帮助和版本命令不初始化 GPU。

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
