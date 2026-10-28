# 固定目标模型短验证

本页定义 D4 的制品准备、正式执行边界和短验证入口。目标身份见 [模型路线](model-plan.md)，验收要求见 [设计](design.md#首轮关口)，资源约束见 [资源规则](resource-policy.md)。实际完成程度以 [仓库现状](status.md) 与 [当前证据](../evidence/current.json) 为准。

## 模型组成与任务路径

当前固定模型的完整权重包含语言主干、视觉组件和 MTP 辅助分支。语言主干为 32 层，每四层中三个 GDN 层、一个全注意力层，共 24 个递推层和 8 个全注意力层；各层还包含归一化、MLP 与残差连接。视觉侧包含 patch embedding、24 个视觉编码块和 merger，后者将视觉特征接到语言输入空间。输入词嵌入和输出词表投影共享权重，不应按两个参数矩阵重复计数。

| 组件 | 对应内容 | 官方源中已下载 | 当前语言执行产物保留 |
| --- | --- | --- | --- |
| 文本输入/输出规则 | tokenizer、词表、merges、聊天模板 | 是，作为附属文件 | 是；当前 Python 辅助准备 token IDs 和解码报告 |
| 语言网络 | embedding、32 层混合主干、最终归一化、共享词表输出投影 | 是，426 个原始张量 | 是；量化后随源布局适配组成 924 个张量 |
| 视觉编码与合并 | `model.visual.*`，包括 patch embedding、视觉块和 merger | 是，297 个原始张量 | 否 |
| MTP 辅助预测 | `mtp.*`，与多 token 预测方案相关 | 是，15 个原始张量 | 否 |
| 像素图像生成网络/解码器 | 扩散生成器、图像 codec 解码器等 | 当前模型权重未提供这条生成路径 | 否 |

语言张量数量增加来自量化将矩阵展开为 packed weight、scales、biases，不是增加模型层。视觉和 MTP 权重只保留在官方源文件中；正式 C++ 模型从语言执行产物加载，不会因源文件在磁盘上就把这两个分支常驻 GPU。

| 使用目的 | 参与的主要路径 | 本项目边界 |
| --- | --- | --- |
| 文本问答、抽取、文本生成 | 文本编码 → embedding → 语言主干 → 归一化/共享输出投影 → 文本解码 | 已完成 D4 短前向对照；尚无正式自然语言生成 CLI |
| 图片问答、OCR 等图文理解 | 图片预处理 → 视觉编码/merger，与文本表示组合 → 同一语言主干 → 文字输出 | 不开展视觉流程；不加载视觉执行权重 |
| 视频理解 | 视频帧处理 → 同一视觉网络 → 时空位置/序列组织 → 语言主干 → 文字输出 | 不开展视频流程；不是另下载一个视频语言主干 |
| 纯文本 SFT/LoRA | 文本前向，加 loss、反向、可训练参数更新和优化器状态；普通文本任务不要求视觉/MTP 分支 | 目标训练属 D5，尚未验收；冻结基础权重仍需其参与前向及必要的梯度传播 |
| MTP 相关训练或解码方案 | 普通语言主路径加 MTP 分支与配套执行逻辑 | 未接入，不能由权重存在推断加速已可用 |
| 原生图片生成 | 需要相应生成网络/输出表示、图像解码及训练 | 当前模型路径与项目范围均不提供 |

源目录保存了完整的两片原始权重，但没有下载仓库全部附属文件，例如未保存 `preprocessor_config.json` 和 `video_preprocessor_config.json`；因此“视觉参数已在磁盘”不表示图像/视频输入流程已准备好。两片权重是存储分片，两片都含语言张量，不能按文件名把其中一片视为视觉模型。

## 官方源与语言执行目录的具体差别

源目录为 `runs/models/qwen35-4b-official/`，语言执行目录为 `runs/models/qwen35-4b-text-4bit/`。它们是同一固定来源的原始材料与派生执行产物，不是训练前、训练后的两个模型。

| 项目 | 官方源目录 | 当前语言执行目录 |
| --- | --- | --- |
| 参数组成 | 语言、视觉、MTP 的完整原始权重 | 仅语言网络 |
| 参数表示 | 主要为 BF16，含少量 float32 参数 | 可量化矩阵为 affine 4-bit/group-64，另有 scales/biases；未量化小张量保留相应 dtype |
| 张量数量 | 语言 426、视觉 297、MTP 15，共 738 | 924；249 个矩阵各增加 scales、biases 两个张量 |
| 权重文件 | 两片原始 `.safetensors` | 一片 `model.safetensors`，仍是 safetensors 格式 |
| 语言名称映射 | `model.language_model.*` | `language_model.model.*` |
| 卷积布局 | 例如 `[8192, 1, 4]` | 对应 MLX `[8192, 4, 1]` |
| 特定 RMSNorm 参数 | 源实现使用 `1 + weight` 的约定 | 转换器对指定归一化权重加 1，适配 MLX 直接使用 weight 的约定 |
| 配置 | 官方原始配置，含 vision_config | 移除 vision_config，加入量化配置并适配加载约定；某些原始特殊 token/辅助字段仍可保留，不代表对应分支已加载 |
| tokenizer 与模板 | 官方文件 | 原样复制；已核对 tokenizer、tokenizer_config、chat_template、merges、vocab 的 SHA-256 相同 |
| 校验材料 | 本地生成的 source_manifest.json 记录固定 revision、源文件摘要和下载预检 | 本地生成的 artifact.json 记录转换版本、源/执行文件摘要与组件排除声明 |
| 占用 | 约 8.7 GiB | 含附属文件约 2.23 GiB；其中原始语言张量在量化前仍约 7.83 GiB |

本次转换进行参数筛选、布局/约定适配和有损量化，没有运行 SFT、训练 adapter 或执行基于训练样本的梯度更新。其输出是待后训练的量化基座，不含本项目目标模型训练产生的 LoRA 增量或优化器恢复状态。

当前数值对照确认 C++ 与同一量化执行产物的参考路径一致，不等于已经评估量化产物相对原始 BF16 模型的独立任务效果。判断实际能力应联合检查配置、实际参数集合、构造的模块与调用路径，不能只根据目录名或某个保留字段判断。

## 制品与准备

制品构建工具、规则与完整性审计位于 [tools/model_artifacts/qwen35](../tools/model_artifacts/qwen35/README.md)，与数值对照开发工具分开。测试调用工具验证行为，工具本身不依赖 `tests/`。

唯一目标为固定官方 revision 的 `Qwen/Qwen3.5-4B`。辅助准备程序 [prepare_qwen35.py](../tools/model_artifacts/qwen35/prepare_qwen35.py) 从官方文件清单读取大小和摘要：LFS 文件核对 SHA-256，小文件核对 Git blob ID；不执行远端代码。原始文件位于 `runs/models/qwen35-4b-official/`，供转换使用，不能作为量化运行目录。

辅助环境中的 MLX-LM 0.32.2 执行权重命名及归一化布局处理、移除视觉与 MTP 参数，再转换为 MLX affine 4-bit、group size 64，未量化参数保留原始 dtype。只有一个活动语言执行产物，位于 `runs/models/qwen35-4b-text-4bit/`。`artifact.json` 记录原始身份、来源文件摘要、转换版本和执行文件摘要；转换中断后没有有效清单的目录不能进入正式执行。现有转换目录不自动覆盖或删除。

下载按 21 GiB 总准备峰值预检（续传扣除本目录已保留文件），覆盖约 8.7 GiB 官方源、分段重组、转换产物、暂存及恢复余量；每片下载前另查分段与重组的并存空间。转换前另查 7 GiB 新增峰值。原始源、分段、执行产物和辅助环境都计入项目占用；预检不能预留其他进程可能消耗的空间。两片文件并发，每片最多八个连接，读取截断可在当前分段内有限重试，完整文件通过官方摘要后才发布。中断时取消未开始的分段，正在读取的连接关闭后结束，不留下后台下载服务。

从项目根执行，`--binary` 指向已构建的资源预检程序：

```sh
.venv/bin/python tools/model_artifacts/qwen35/prepare_qwen35.py download \
  --root "$PWD" --binary "$PWD/runs/d2/build/pitfall-lab"
.venv/bin/python tools/model_artifacts/qwen35/prepare_qwen35.py convert \
  --root "$PWD" --binary "$PWD/runs/d2/build/pitfall-lab"
```

转换和数值对照需要本机 Metal；下载只使用网络和磁盘。准备程序属于显式数据准备，不被正式 C++ 模块导入或调用。

## 转换来源与可复现审计

官方源文件保持不变，转换写入独立执行目录。命名和布局适配沿用固定 [MLX-LM 0.32.2 的 Qwen3.5 sanitizer](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/models/qwen3_5.py)，量化与保存沿用其 [utils](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/utils.py)，不以项目自写转换替换这条主路径。

| 变换 | 固定源上的实际行为 | 依据 |
| --- | --- | --- |
| 参数筛选与重命名 | 保留 426 个语言张量，前缀 `model.language_model.` → `language_model.model.`；排除 297 个视觉、15 个 MTP 张量 | 上游 Model / TextModel 的 sanitize |
| 卷积布局 | 24 个 conv1d 权重从 `[8192, 1, 4]` 转为 `[8192, 4, 1]`，移动轴 2 到轴 1 | 上游适配 MLX 卷积布局 |
| 归一化约定 | 81 个指定 RMSNorm 权重加 1；不包括所有名字带 norm 的参数 | 官方源的 `1 + weight` 约定转换为 MLX 直接使用 weight；上游在检测到原始卷积布局时执行 |
| 量化存储 | 249 个矩阵转为 affine 4-bit/group-64；每个增加 scales、biases，得到 924 个输出张量 | 上游 quantize_model 和 MLX 量化模块；属于有损变换 |
| 配置和附属文件 | 删除 vision_config、保留上游配置适配和量化信息；tokenizer、模板及许可证原样复制 | 配置适配来自上游，语言制品范围及文件复制由项目准备脚本规定 |

仓库保存可审查的 [转换规则](../tools/model_artifacts/qwen35/qwen35_conversion_rules.json)、[准备脚本](../tools/model_artifacts/qwen35/prepare_qwen35.py)、[审计代码](../tools/model_artifacts/qwen35/qwen35_conversion.py)、[规则测试](../tests/checks/test_qwen35_conversion.py) 和 [复现比较器](../tools/model_artifacts/qwen35/verify_qwen35_reproduction.py)。规则固定模型 revision、包版本、关键上游文件与 MLX 二进制摘要；环境不匹配时拒绝转换。审计核对上游输出的逐参数名称、shape、dtype 和声明的转轴/加一数值；仅重命名的数组必须保持原对象。量化审计核对输出集合与布局，数值结果另由完整制品复现比较验证。

新转换目录内生成 `conversion-recipe.json`、`converter-identity.json`、`conversion-audit.json`，记录规则、转换代码身份和逐参数变化，纳入 artifact 清单。转换前后均核对源文件摘要。代码、规则、依赖锁和必要证据摘要纳入 Git；大权重及本地审计产物保留在 `runs/`。原有活动制品没有追补这些文件，保持已验收身份。

复现时选择一个不存在的 `runs/models/` 直属子目录，先按资源规则预检，再转换和比较：

```sh
.venv/bin/python tools/model_artifacts/qwen35/prepare_qwen35.py convert \
  --root "$PWD" --binary "$PWD/runs/d2/build/pitfall-lab" \
  --output "$PWD/runs/models/qwen35-4b-text-4bit-repro"
.venv/bin/python tools/model_artifacts/qwen35/verify_qwen35_reproduction.py \
  --baseline "$PWD/runs/models/qwen35-4b-text-4bit" \
  --candidate "$PWD/runs/models/qwen35-4b-text-4bit-repro" \
  --report "$PWD/runs/d4/conversion-reproduction.json"
```

本机固定环境已从官方源完整重跑：上述 9 个权重、索引、配置与附属文件逐字节一致；新增的 3 个来源审计文件及 manifest 按设计不同。该结果证明当前脚本复现了既有制品，不承诺更换后端、硬件或版本后仍逐字节一致。复现目录是核验副本，不自动替代活动制品；保留期间其占用计入预算。完成核验后可保存必要来源/审计小文件并按资源规则清理副本；当前保留情况见仓库现状。

## 正式执行边界

`src/model/` 仅支持已固定的 Qwen3.5-4B 语言结构与上述量化布局，拒绝不支持的几何、层序列、RoPE、量化和嵌入配置。加载核对制品身份、文件大小和摘要、分片索引、全部张量名、shape 与 dtype；不隐式选择其他模型或回退 Python。摘要检查保证完整性，不认证被恶意整体改写的清单。

前向 API 接收准备好的非空 token ID 序列，当前只处理单条无 padding 文本；每个请求使用独立状态，包含 KV、卷积历史、float32 递推矩阵和共同位置。一次调用可处理完整序列或分块追加。标准文本位置使用部分 RoPE；不执行视觉位置编码、MTP 或任意多模型适配。

正式计算复用 MLX 原生 RMSNorm、量化矩阵乘、卷积与注意力，不运行 Python。`compute` 中的 float32 GDN 保留原生接口；BF16 GDN 通过 MLX C++ custom-kernel 接口复用固定参考的混合精度递推，直接保留 BF16 q/k/v 与 float32 衰减和状态。来源与 MIT 许可见 [内核说明](../third_party/mlx-lm-gdn/README.md)。BF16 接入当前仅定义前向，目标训练与反向仍须 D5 单独实现和验收。小型解析及分块测试为 `compute.gated_delta`，不能据此认定目标模型已经通过。

## 独立进程数值对照

[辅助对照驱动](../tools/numerical_validation/qwen35/compare.py) 通过 `--input` 接受文本消息 JSON，在开发过程中独立检查数值正确性，不读取候选 test 集。固定验收样例单独保存在 fixtures，由验收测试选择。模板渲染及 token IDs 由固定 tokenizer 生成，并与 tokenizer 文件的直接编码路径核对。它先运行独立 MLX-LM 参考进程，再运行调用正式 C++ 模块的 [探针](../tools/numerical_validation/qwen35/probe.cpp)；参考进程退出后才启动原生 worker。

```sh
.venv/bin/python tools/numerical_validation/qwen35/compare.py \
  --root "$PWD" \
  --artifact "$PWD/runs/models/qwen35-4b-text-4bit" \
  --input "$PWD/fixtures/qwen35-4b/numerical-input.json" \
  --binary "$PWD/runs/d2/build/tools/numerical_validation/qwen35/lab_qwen35_probe" \
  --resource-binary "$PWD/runs/d2/build/pitfall-lab" \
  --report "$PWD/runs/d4/model-reference.json"
```

完整序列模式检查 embedding、全部 32 个层输出、最终 hidden 与 logits；三种模式均核对全部层的 KV、卷积及递推状态。完整序列、逐 token 与三 token 分块分别使用相同分块方式的独立参考。短程贪心生成最多八个 token，遇结束标志停止，要求跨语言及跨方式 token IDs 完全一致。

同种方式的跨语言数值 scaled RMSE 不超过 0.01，logits 最大绝对误差不超过 0.5；shape、dtype 和离散结果严格匹配。BF16 跨方式受矩阵形状与计算顺序影响，另行报告原始漂移，比较 `(C++ 分块 - C++ 完整) - (参考分块 - 参考完整)`，使用同样的数值上限，以参考完整结果为归一化尺度。不能把跨语言误差和参考自身的跨方式漂移混成一个检查，也不宣称 BF16 跨方式逐位相等；本次测得原始 logits 漂移约 1.66%–1.68%，对应的跨语言漂移差为 0。误差失败需定位，不能直接放宽容差以制造通过。

对照工具不依赖 CTest；`BUILD_TESTING=OFF` 时仍可构建 `lab_qwen35_probe`。若要验证仓库是否满足固定 D4 场景的设计目标，可在开启测试并准备好制品后，用显式 CMake 选项注册验收测试。该 [验收入口](../tests/model/test_qwen35_comparison.py) 调用工具，核对固定输入、三种模式、全部层/状态的覆盖、既定判据和贪心结果，不在测试中实现参考模型或对照算法：

```sh
cmake -S . -B runs/d2/build \
  -DLAB_QWEN35_ARTIFACT="$PWD/runs/models/qwen35-4b-text-4bit"
cmake --build runs/d2/build -j 2
ctest --test-dir runs/d2/build -R '^model.reference$' --output-on-failure
```

不配置 `LAB_QWEN35_ARTIFACT` 时只注册小型检查，不能据其全通过宣布 D4 完成。配置后如果制品不完整则明确失败，不自动下载、不跳过，也不使用合成模型替代。

报告同时记录每个进程的耗时、MLX 分配峰值与最大 RSS；带观测和保存输出的短验证不是性能优化基线。父进程使用有限超时并在异常时终止和回收 worker，不常驻模型；临时比较张量在完成后移除，必要的小型结果摘要同步当前证据。

尚无正式自然语言生成 CLI 或原生 tokenizer；Python 在此只准备输入与解码报告，不承担正式模型计算。D4 不包含目标模型训练、长上下文、变长批量或 D5 检查点恢复。
