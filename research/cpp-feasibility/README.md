# C++ / MLX 可行性验证

本目录保留 C++ / MLX 的有限验证，不属于正式模型执行代码，也不是完整 Qwen 实现。实测结果见 [验证报告](report.md)，当前实现约定见 [正式文档](../../docs/implementation.md)。未下载任何模型权重。

## 环境与边界

- Apple M4 / 24 GiB，macOS 15.7.3，Apple Clang 17，C++17。
- MLX 0.32.4；Python 对照采用 MLX LM 0.32.2。
- 明确使用项目根 `.venv` 中 MLX wheel 提供的 C++ SDK、`libmlx.dylib` 和 `mlx.metallib`；不依赖 `research/model-selection/` 环境。
- C++ 工作进程直接链接 MLX，不嵌入 Python、不启动 Python 训练程序。Python 仅承担输入生成、参考计算、子进程控制和结果比较。
- 复现环境按根目录辅助工具锁文件准备。正式主语言为 C++，根目录环境只提供 SDK 与 Python 对照工具。
- 编译、张量和检查点存于被 Git 忽略的 `runs/cpp-feasibility/`，同样计入项目磁盘预算。每次验证写入新目录，保留失败记录。

## 复现

从仓库根目录执行。需要本机 Metal GPU；GPU 不可用时明确失败，无 CPU 自动替代。

```sh
cmake -S research/cpp-feasibility -B runs/cpp-feasibility/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/.venv/lib/python3.12/site-packages/mlx"
cmake --build runs/cpp-feasibility/build -j 2
.venv/bin/python research/cpp-feasibility/validate.py
```

验证器读取根目录 `configs/resources.toml`，在生成输入前检查项目物理占用及卷空闲，预留 128 MiB 本次新增文件空间，执行资源额度约束。此预检覆盖本目录的小实验，不是正式下载器的磁盘保护；编译前也应核对资源规则。当前构建目录很小，不下载或编译 MLX 源码。

`CMakeLists.txt` 针对已安装 wheel 中不存在的构建机 SDK framework 路径，查找本机相同 framework；不修改 wheel，不改变计算后端。缺失 framework 时构建失败。

## 内容与证据

- `probe.cpp`：原生 C++ 计算、量化 LoRA、小型 Adam 更新、原子检查点、信号暂停和恢复。
- `validate.py`：参考实现与梯度对照、有限差分、跨进程一致性和错误路径。
- `evidence/result.json`：2026-10-22 最终成功报告，含源码、动态库和二进制摘要，完整检查结果与进程输出。
- `evidence/after-organization.json`：目录整理、移除 Python 应用入口并改为读取统一资源配置后的复现报告；65 项检查再次通过。
- `evidence/initial-failure.json`：首次检查点路径失败的摘要；原始完整记录仍在本机 `runs/`。

训练小实验固定常量和身份，仅用于恢复机制。尚无通用参数注册、数据集加载、模型身份清单、张量内容哈希、并发运行锁或完整 4B 训练恢复能力。信号测试在第 3 步设置测试会合点，验证安全边界退出；它的毫秒级延迟不能用于估计真实模型暂停延迟。

注意力测试使用完整序列 K/V 的前缀视图，验证因果注意力与逐 token 计算一致；没有实现动态 cache 分配或缓存持久化。递推分块测试显式携带状态，不能代替完整模型的混合 cache 验收。

参考来源：[MLX](https://github.com/ml-explore/mlx)、[MLX LM](https://github.com/ml-explore/mlx-lm)，包括本机 MLX LM 0.32.2 的 `models/gated_delta.py` 中 `gated_delta_ops`。原生融合递推与组合算子递推作为两个独立实验路径比较，没有失败后静默切换。框架共享意味着跨语言对照不能排除共享底层缺陷，因此另加量化/反量化等价式和递推门的有限差分检查。
