# 资源与可行性验证

> 前期调研归档：本文件包含历史候选和讨论中间状态，不是正式执行规格。当前路线见 [模型路线](../../../docs/model-plan.md)，资源要求见 [资源规则](../../../docs/resource-policy.md)。

状态：已确定一期 Qwen3.5-4B 完整学习、二期 Qwen3.5-9B 迁移，目标为当前 M4 / 24 GiB Mac。F0 环境与核心算子验证已完成；尚未下载或运行完整预训练模型，具体量化产物和资源阈值待开工准备确认。

当前决策依据见 [Qwen3.5 专项评估](qwen35-assessment.md)，资源约束见 [资源使用规则](../../../docs/resource-policy.md)。下文 Granite 筛选、预算与 F1–F4 候选表作为早期调研记录保留，不再据此推进 Granite 下载，也不继续开展无边界的候选比较。

## 1. 当前机器只读检查

检查日期：2026-10-22。用户已确认以当前工作机器作为运行目标。可用空间与软件环境仍是检查时的快照。

| 项目 | 实测信息 |
| --- | --- |
| 设备 | MacBook Pro，Apple M4 |
| CPU | 10 核（4 性能核、6 能效核） |
| GPU | 10 核，Metal 支持 |
| 内存 | 24 GB，不能直接视为模型可用内存预算 |
| 系统 | macOS 15.7.3，Darwin 24.6.0，arm64 |
| 项目所在卷剩余空间 | `df -h .` 报告约 66 GiB，动态变化，不能视为全部可供本项目使用 |
| 当前 `python3` | Xcode 提供的 Python 3.9.6 |
| 当前解释器包发现检查 | 未发现 torch、mlx、transformers、safetensors；不代表其他环境未安装 |
| 工具路径检查 | 可发现 uv、Homebrew、xcrun；未在当前 PATH 发现 nvidia-smi |

本节为初次只读检查；后续已创建项目独立环境并安装依赖，见第 6 节。未下载模型权重或启动模型训练。检查命令包括 `uname`、`system_profiler`（硬件与显示信息）、`sw_vers`、`df`、命令路径检查及 Python 包发现检查。

## 2. 已确认的学习与资源约束

- 运行设备：当前 M4 / 24 GB Mac，不依赖额外设备作为当前方案成立的条件。
- 知识基础：按从零开始安排，不预设 Python、张量、反向传播或 Transformer 熟练度。
- 等待时间：常规实验接受小时级；少量特定实验可在空闲时执行，约 8 小时。
- 每周投入时间和项目周期尚未确定，因此不承诺日历工期；不将约 8 小时理解为每周投入。

等待预算约束实际运行时间，不等同于保证训练收敛或任务效果提升。先用短实验测量，再估计长实验，不直接启动整夜任务。

## 3. 本轮验证原则

- 优先选择较新发布的开放文本模型；核对原模型发布日期、许可证和架构，不能把转换包更新时间当作模型发布日期。
- 候选验证阶段允许有限比较，正式优化前固定一个主模型及其权重 revision、tokenizer/模板和目标硬件。
- 本轮比较仅保存两个 Granite 候选的小型配置，不同时下载两套权重。主模型改变时重建基线，不将跨模型差异算作优化收益。
- 训练检查点、精度与量化是可追踪的受控变体，不要求建设多模型兼容层。
- MoE 后训练和多请求调度、离线偏好优化仍在二期，不因新模型选择自动改变范围。

## 4. 新候选筛选

调研日期：2026-10-22。下表是满足本项目约束的有限筛选，不宣称覆盖所有发布或选出市场最优模型。

| 候选 | 核实的时间与结构 | 本轮处理 |
| --- | --- | --- |
| Granite 4.1 3B | 2026-04-29 发布；稠密 decoder-only 文本模型，Apache 2.0 | 优先进入完整模型验证的候选，尚未冻结 |
| Granite 4.1 8B | 同日发布，同家族稠密文本模型 | 资源比较候选，不预设后续必须升级到它 |
| Nemotron 3 Nano 4B | 模型卡发布日 2026-03-16；文本、Mamba2/Attention 混合结构 | 暂不优先：需额外验证混合结构在 Mac 上的训练/观察路径 |
| Qwen3.5 4B/9B | 2026-03-02 发布；带视觉编码器，语言部分为混合结构 | 当前文本模型限制下不直接采用，不能静默按“只输入文本”放宽范围 |
| LFM2.5 8B-A1B | 2026 年发布的 MoE 方向 | 不作为一期主训练对象，避免提前引入二期 MoE 后训练 |
| Qwen3 4B/8B | 前轮已调研的候选 | 本轮不再优先；保留此前调研背景，不启动其权重下载 |

主要来源：[Granite 官方发布说明](https://research.ibm.com/blog/granite-4-1-ai-foundation-models)、[Granite 3B 模型卡](https://huggingface.co/ibm-granite/granite-4.1-3b)、[Nemotron 模型卡](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)、[Qwen 官方发布记录](https://github.com/QwenLM/Qwen3.8#news)、[Qwen3.5 结构](https://huggingface.co/Qwen/Qwen3.5-4B)、[LFM2.5 官方说明](https://www.liquid.ai/blog/lfm2-5-8b-a1b)。

Granite 同一家族还发布视觉/语音产品，不代表本次语言模型包含这些组件；本次核实的配置为 `GraniteForCausalLM`、`model_type=granite`。选择它是学习覆盖与资源方面的判断，不采信官方跑分作为本项目实测。

### 组件复用边界

优先验证 MLX / MLX LM：复用张量后端、自动求导、tokenizer、模型加载与参考执行；项目负责明确配置、实验操作、观察、对照及必要局部修改。MLX LM 已安装版本包含 Granite 的 Attention、cache 更新、逐层执行和 LoRA 转换入口，便于追踪。

这只是可检查的实现路径，不能由此推断完整权重加载、量化 LoRA、可解释性工具或在线强化学习均已验证。PyTorch MPS 保留为针对具体缺口的比较方案，不提前维护双框架。

参考：[MLX LM](https://github.com/ml-explore/mlx-lm)、[Granite 实现](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/granite.py)、[LoRA 入口](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/utils.py)、[PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html)。

## 5. 基于实际配置的资源预算

元数据、权重 revision、配置与估算见 [model-candidates.json](../evidence/model-candidates.json)。所有数字都是权重或缓存的理论数据量，不是运行峰值。GB 为十进制，MiB/GiB 为二进制。

| 项目 | Granite 4.1 3B | Granite 4.1 8B |
| --- | --- | --- |
| 元数据参数量 | 3,402,836,480 | 8,791,592,960 |
| 纯 BF16 权重 | 6.81 GB，约 6.34 GiB | 17.58 GB，约 16.38 GiB |
| 理想 4-bit 权重 | 1.70 GB | 4.40 GB |
| 层数 / KV heads / head dimension | 40 / 8 / 64 | 40 / 8 / 128 |
| 16-bit KV，batch=1，1024 个已缓存 token | 80 MiB | 160 MiB |
| 同条件，4096 个已缓存 token | 320 MiB | 640 MiB |

权重公式为实际参数量 × 每参数字节。4-bit 估算未包含未量化层、scale/bias 和分组开销，激活与 KV 不随权重自动变为 4-bit。KV 公式为 `2 × 层数 × KV heads × head dimension × 已缓存 token 数 × 每元素字节`，不含分配余量、临时缓冲；缓存长度包括已生成的 token。

配置获取使用明确 revision；Hub 仓库创建时间单独保存在证据中，不将其当作对外发布日期。原始来源链接见证据 JSON。

实测 MLX 返回当前 GPU 的推荐工作集约 **16 GiB**，系统总内存为 **24 GiB**。推荐工作集不是硬上限、预留额度或当前空闲内存，也不是完整进程 RSS。[device_info 文档](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.device_info.html)

因此，8B 的 BF16 权重本身已接近或超过该推荐工作集；这不证明 8B 无法运行，但意味着不能把它作为宽裕的默认训练路径。量化后是否适合训练仍需实测。3B 在保留非量化对照和观察余量方面更值得先验证；这是预算判断，不是性能或效果结论。

训练另计 adapter、梯度、优化器、保存的激活、logits 和临时副本。在线强化学习另计同题多个回答、总生成量、概率计算和所选算法的其他状态；不能用 SFT 峰值直接推断。首轮训练采用 LoRA/量化 LoRA 候选，不默认进行主模型全参训练。

### 磁盘与长实验预算

当前项目所在卷约有 66 GiB 余量。独立环境与包缓存合计约 610 MiB（安装后快照，硬链接共享可能影响实际物理占用）。首个候选的原始 BF16 权重约 6.8 GB；如之后保存量化副本、adapter 和观察结果，应分别计数，转换临时峰值也须计入。

首轮仅下载一个模型，暂按不超过约 15 GiB 新增占用进行规划；这是本轮暂定操作预算，实际下载前核对文件清单和空间，不代表用户授权用满磁盘或需要删除其他文件。

先用短输入、batch=1 和少量训练步测量，之后再确定正式上下文、batch、样本数量与小时级运行规模。约 8 小时长实验仅用于已通过短验证的明确问题，不由理论参数量推算训练完成时间。

## 6. F0 环境验证结果

已完成：

- 使用已存在的 Python 3.12.13 创建项目 `.venv`；未修改系统 Python。
- 安装并锁定 MLX 0.32.4、MLX Metal 0.32.4、MLX LM 0.32.2；完整依赖见本归档目录的 `uv.lock`。
- 在明确指定的 M4 GPU 上执行矩阵乘法，与手算结果一致。
- 用可解析求导的平方均值验证自动求导，最大绝对误差 0；一次更新后 loss 从 7.5 降到约 6.76875。
- 已安装的 Granite 参数类能够解析两个候选的真实配置；没有实例化完整模型。

证据：[environment-probe.json](../evidence/environment-probe.json)。上述 loss 仅为小型数学探测，不是 LLM 训练效果；MLX 内存计数仅为该探测的分配指标，不是模型运行峰值。

遇到的问题：默认工具沙箱内导入 MLX 报告无法访问 Metal；在获准访问 GPU 的运行环境执行相同脚本后通过。未使用 CPU 兜底，也未据沙箱结果误判硬件不支持。

复现步骤（本归档目录 research/model-selection，需允许访问 Metal；首次同步需要网络）：

```sh
UV_CACHE_DIR="$PWD/.cache/uv" uv sync --frozen
.venv/bin/python scripts/probe_environment.py --output runs/environment-probe.json
```

脚本会明确拒绝 Metal 不可用，不自动切换 CPU。若在无 GPU 权限的执行环境失败，应在可访问 GPU 的环境重试；不修改操作系统 GPU 内存限制。

## 7. 验证进度与下一关口

| 关口 | 状态 | 完成依据或待办 |
| --- | --- | --- |
| 元数据与预算 | 已完成本轮初筛 | 两个 Granite 候选的明确 revision、参数量和架构配置，尚非全市场比较 |
| F0 环境 | 已通过 | GPU 算术、自动求导及配置解析记录 |
| F1 推理 | 待执行 | 优先验证 Granite 4.1 3B 的完整权重、模板、短生成、KV cache 和阶段计时 |
| F2 训练 | 待执行 | 少量 LoRA 更新，检查梯度/参数；保存并在新进程加载 |
| F3 扩展风险 | 待执行 | 在线强化学习的生成/概率/梯度路径、可解释性采集和 MoE 观察可行性 |
| F4 固定主目标 | 待决定 | 根据实际负载、资源、任务表现及时间预算冻结主模型 |

F1 不要求先运行 8B，只有首个候选不满足目标或有明确比较问题时才推进另一候选。F1/F2 分别记录冷启动、预热、峰值内存、内存压力与步耗时；推断长实验成本时列明假设。

从零学习按“说明问题 → 补充够用基础 → 预测结果 → 短实验 → 检查过程与结果 → 记录结论”推进。原始模型与任务未实测前，不承诺任务改善、训练收敛或在线强化学习一定在约定时间内有效。

早期结论：当时已具备进入模型短验证的环境依据，但没有模型实测。现已决定采用 Qwen3.5 的 4B→9B 分期路线；完整一期可行性仍由后续关口验证。
