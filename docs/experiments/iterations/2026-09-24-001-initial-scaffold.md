# 2026-09-24-001：M0 初始仓库

- 状态：structural_check
- 代码 revision：当前工作区（尚无可用 Git 元数据）
- run_id：无训练 run

## 假设与变化

先建立严格的数据、运行与证据契约，能让后续模型开发从第一步起可复现。新增项目结构、配置、schema、标准库 CLI、决策头代码、测试和完整教程。

## 运行结果

先在基础环境执行 `doctor`、样例数据校验、`unittest`、仓库 JSON/必需文件检查和 `compileall`。随后创建 `vision-jev` Conda 环境并安装完整依赖，环境内 6/6 测试通过（含 PyTorch 决策头排列等价性），Ruff、格式、mypy strict、`pip check` 与仓库检查全部通过。样例的 3 道完整问题覆盖 Choice/Noul/Score。

PyTorch 安装为 2.14.0+cu130。最初在受限沙箱中检查时 NVML/GPU 不可见，曾被错误记录为本机没有驱动；改用宿主权限复核后，确认驱动 580.173.02、CUDA 13.0、2 张 RTX 4090 均可被 PyTorch 使用。两张卡分别完成 1024 × 1024 BF16 矩阵乘法与同步 smoke test。驱动报告每张卡 49,140 MiB。

本次只验证 GPU 环境可用；没有模型权重、准确率、训练吞吐、峰值显存或多进程 NCCL 实测。

## 结论

M0 脚手架保留。下一步是来源/许可冻结和 Qwen 原生 processor 的最小对接；该工作需进入 `vision-jev` Conda 环境并安装训练依赖。
