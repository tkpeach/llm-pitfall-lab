# Qwen3.5-4B 配置契约样例

[config.json](config.json) 来自官方 `Qwen/Qwen3.5-4B` 的固定 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，只额外添加统一 `quantization` 字段：MLX affine、4-bit、group size 64。官方配置来源见 [模型路线](../../docs/model-plan.md#模型身份与执行产物)。

该文件供 `model.config` 的小型确定性契约测试使用，不包含权重，不代表已加载模型或已通过目标模型数值对照。实际执行配置由已核验官方来源经固定 MLX-LM 转换产生，制品规则见 [固定目标模型短验证](../../docs/model-validation.md)。

官方模型的 Apache 2.0 许可随 [LICENSE](LICENSE) 保存。

`numerical-input.json` 是 D4 数值对照能力验收使用的手写文本消息样例，不是训练或任务效果评估集。独立对照工具接受调用者提供的消息输入；验收测试选择该文件并核对固定覆盖与结果。
