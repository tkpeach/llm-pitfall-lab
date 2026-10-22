# Python 原型参考

本目录保存正式 C++ 实现之前的 Python 原型及对应测试，供阅读、数值契约和局部实现参考。它不是第二套正式实现，不继续扩展为产品入口；正式代码不得导入或依赖它。

## 内容

- `src/llm_pitfall_lab/`：任务评分、资源预检、JSON 检查点、运行控制和小型 MLX 恢复实验。
- `tests/`：对应契约与生命周期测试。
- `configs/model-4b.toml`：未执行的 Python 加载草稿配置，包含社区量化包身份；不属于已验收执行产物，也不授权自动下载。
- `pyproject.toml`、`uv.lock`、`.python-version`：原型依赖快照，不作为当前正式工具环境的配置来源。

其中任务规则、资源规则和检查点思路可以用于 C++ 实现；Python 实现的通过结果不能计入正式 C++ 模块验收。`model.py` 尚未完成执行验证，原型 CLI 也未接入该加载流程，不将它视为可运行的模型后端。

## 参考测试

从仓库根目录显式指定原型模块路径，不安装为正式应用：

```sh
PYTHONPATH="$PWD/research/python-prototype/src" \
  .venv/bin/python -m unittest discover -s research/python-prototype/tests -v
```

默认只运行不需要 GPU 的契约测试；生命周期测试须显式设置 `RUN_MLX_TESTS=1` 并允许 Metal 访问。测试沿用根目录的已确认资源配置和任务样例，运行文件写入根目录 `runs/`，仍计入项目预算。原型不是正式资源保护工具，不用于启动完整模型或长训练。
