# 开发标准

## 完成定义

代码变更必须包含测试、配置影响、文档影响和一条迭代记录。训练/数据变更还必须创建 run，记录命令、环境、输入 hash、数据混合、指标与结论。未运行的测试写明 `not_run` 及原因，不能写“通过”。

## 代码

- Python 3.11+，公开接口带类型标注；Ruff + mypy strict。
- 随机过程显式 seed；设备、dtype、shape 不依赖隐式全局状态。
- loss 按完整问题归一化，不按候选行数放大。
- 模型代码不做网络下载；下载和许可确认属于显式 prepare 步骤。
- 错误必须可操作：sample_id、字段路径、期望与实际值。

## 配置

配置是运行输入，运行启动后复制到 run 目录并计算 SHA-256。禁止覆盖已完成 run。改变数据、权重 revision、视觉预算、随机种子或代码都创建新 run_id。

## 分支和评审

提交遵循 Conventional Commits。架构/数据语义/指标口径改变时新增 `docs/decisions/NNNN-*.md`。PR 描述必须区分：实现完成、结构验证、真实训练、待验证。

## 结果语言

允许的证据状态：`planned`、`structural_check`、`measured`、`reproduced`、`released`。所有表格每行都带状态；吞吐、显存和准确率没有 run_id 时不得标为实测。
