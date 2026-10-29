# Qwen3.5 参考执行的分块漂移诊断

2026-10-29 在 Apple M4 / 24 GiB、macOS 15.7.3、MLX 0.32.4、MLX-LM 0.32.2，对固定 4B 语言制品和 `fixtures/qwen35-4b/numerical-input.json` 的 22 个 token 进行观察。制品 `artifact.json` SHA-256 为 `a4fbcefc2262ae474a3ead3ecbe9ca14bccb6b2b82f247d134e128e4e1029978`；源码基础为 `be24b7e`，这三个辅助脚本原样保存在本目录。正式 C++ 与 MLX-LM 的验收状态不由此实验改变。

- `layers.py`：运行未修改参考进程和临时注入逐层输出观测的参考进程，分别使用完整和逐 token 模式，先核对同模式共同输出与缓存及生成结果，再比较 32 层。结果见 `layer-result.json`。
- `stages.py`：对第 0 层选定模块的原始调用添加观测；同样先核对共同输出、状态和生成结果。结果见 `stage-result.json`。
- `projection.py`：同一已物化归一化输入、同一 QKV 权重，在单进程比较一次 22 token、逐 token 和三 token 投影。结果见 `projection-result.json`。

脚本输出写入 `runs/numerical-validation/diagnostic/`；前两个在该目录的临时子目录运行 worker 并在结束时清除大张量。它们会覆盖同名报告，保留旧报告前请另选位置。运行前按正式资源规则检查项目占用、卷空间和新增峰值；本次使用 1 GiB 新增峰值预检，模型进程顺序运行。前两个脚本每个 worker 前再次预检；`projection.py` 自身不做资源预检，必须由调用者先执行。需要固定制品、依赖和本机 Metal，不下载模型、不提供 CPU 兜底：

```sh
./runs/d2/build/pitfall-lab resources --root "$PWD" --config "$PWD/configs/resources.toml" --peak-bytes 1073741824
.venv/bin/python research/cpp-feasibility/cross-mode-drift/layers.py
.venv/bin/python research/cpp-feasibility/cross-mode-drift/stages.py
.venv/bin/python research/cpp-feasibility/cross-mode-drift/projection.py
```

脚本中断或失败时应确认子进程已退出、临时目录已清理后再继续。观测版与基线在共同保存项完全相同，支持本次观测不改变这些输出，但未检查所有内部张量。隔离投影重现了不同形状的差异，不证明具体 GPU kernel、归约或指令根因，也不证明投影差异是全部 logits 漂移的唯一原因。结果及限制的面向学习者解释见 [数值对照笔记](../../../learning/topics/fixed-model-numerical-validation.html#cross-mode-observation-20261029)。

## Metal 命令行追踪与高精度参考

`trace_one_shape.py` 每进程仅运行一个形状的 QKV 投影。实际使用 Xcode 16.4 的 `xctrace record --template 'Metal System Trace'`，分别追踪 1、3、22 token，结束后用 `xctrace export --xpath` 导出 `metal-shader-profiler-shader-list`。`xctrace-result.json` 保存独立进程的输出与对应 shader 列表；追踪在本机 `runs/numerical-validation/diagnostic/metal-{1,3,22}.trace`，总约 105 MiB，不纳入 Git。示例，以一个 token 为例：

```sh
./runs/d2/build/pitfall-lab resources --root "$PWD" --config "$PWD/configs/resources.toml" --peak-bytes 1073741824
xcrun xctrace record --template 'Metal System Trace' --no-prompt --output runs/numerical-validation/diagnostic/metal-1-new.trace --time-limit 30s --target-stdout - --launch -- "$PWD/.venv/bin/python" "$PWD/research/cpp-feasibility/cross-mode-drift/trace_one_shape.py" 1
xcrun xctrace export --input runs/numerical-validation/diagnostic/metal-1-new.trace --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-shader-profiler-shader-list"]' --output runs/numerical-validation/diagnostic/metal-1-new-shaders.xml
```

每份追踪报告包含 backdated signpost 警告；3 token 的 shader profiler interval 和 sample 表为空。shader 列表与隔离脚本共同提供路径选择的强证据，但列表本身不能当成逐次 dispatch 或指定路径因果干预的记录。名称末尾括号数字不作含义推断。三个脚本输出同一目标分量分别为 17.5、17.5、17.625。

`capture_and_reference.py` 使用 `mx.metal.start_capture` 捕获 GPU 工作，并在结束捕获前 `mx.synchronize()`。捕获文件引用全模型缓冲，单份实测约 2.52 GB；脚本逐次检查卷剩余和项目占用，按 4 GiB 峰值预留，读取后立即删除，不能并行运行。捕获文件内的字符串不能可靠地归属到实际调用，未据此断定 dispatch。脚本另外通过 float32 反量化选定的 4-bit 权重行、CPU float64 求点积，结果见 `capture-reference-result.json`。它仅用于比较基于量化制品的数学参考，不等于原始权重的参考或 GPU 内部中间值。运行方式：

```sh
./runs/d2/build/pitfall-lab resources --root "$PWD" --config "$PWD/configs/resources.toml" --peak-bytes 4294967296
MTL_CAPTURE_ENABLED=1 .venv/bin/python research/cpp-feasibility/cross-mode-drift/capture_and_reference.py
```

本机 `mlx-metal` 0.32.4 的 `libmlx.dylib` 和 `mlx.metallib` 的 SHA-256 均与该已安装包的 RECORD 一致；这只校验已安装文件相对于包记录的完整性，不证明 wheel 与 GitHub v0.32.4 标签的某个源码 commit 同源。高精度对照、shader 列表、实际分派和对 kernel 的因果控制是不同证据层级，不能相互替代。
