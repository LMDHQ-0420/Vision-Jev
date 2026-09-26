# 2026-09-26-014：Score 排查与全量 LoRA 框架

- 状态：structural_check / measured smoke
- 正式训练状态：未启动
- 基座：Qwen3.5-0.8B
- 方法：单卡 LoRA SFT

## Score 排查

清洗后 12k pilot 的 Score holdout 只有 29 条，严格 exact match 为 41.38%。进一步按有序等级检查：平均绝对误差为 0.655，相邻一级内命中率为 96.55%，跨两级及以上只有 1 条。混淆主要发生在 2/3、3/4、4/5 相邻档；验证集中第 1 档只有 1 条，因此 29 条 exact match 不足以稳定描述 Score 能力。

KonIQ 标签由 train MOS 五等分位产生，阈值为 2.7048、3.1402、3.4158、3.6422。阈值附近视觉质量相近的图片会被切到相邻档，因此正式评测增加 MAE、相邻一级命中率和 5×5 混淆矩阵，exact match 仍保留但不再作为唯一指标。

完整数据中 Choice/Noul/Score 为 96k/18k/6k，即 80%/15%/5%。该配额有意不均衡：Score 只有一个 KonIQ 来源，扩充到三等分会重复同一来源。完整 6k 已是 pilot 600 条的十倍，主配置先不重采样；训练器保留 `task_repeat`，只有正式评测证明暴露不足时才启用。

## 全量训练数据

新增 `data-build-training`，对完整 120k 按 group 确定性分配角色。实测得到训练 114,229、验证 5,771，验证含 Choice 4,601、Noul 893、Score 277；4,021 个 eval group，无 group 跨角色。Score 训练五档为 1,169/1,115/1,134/1,153/1,152，验证五档为 55/51/64/43/64。完整 schema、资源存在性和 group 检查通过。清单 SHA-256 为 `93c331b4f3416f3e36b47beace9ad8b5d46e6c515bff53b06b53416367b0e6c5`。

## 单卡正式框架

- LoRA rank 16、alpha 32、dropout 0.05；视觉塔冻结。
- 单张 RTX 4090，micro batch 1，gradient accumulation 32，global batch 32。
- 2 epochs，预计 7,140 optimizer steps；LR 3e-5，warmup 3%，BF16。
- 普通视觉预算约 256 token，Score/区域约 576 token。
- 每 500 step 保存 LoRA 和完整恢复状态；输出目录禁止静默覆盖。
- `--resume-from` 恢复模型、优化器、scheduler、随机状态、epoch、step 和 batch 游标，并要求 world size 与原 run 一致。
- `sft_main_profile.json` 使用相同单卡 batch、视觉预算和优化器执行 200 step，作为正式运行前的性能门禁。

## Smoke 结果

8 条训练、4 条验证、2 optimizer step 的真实 GPU smoke 通过。连续运行从 loss 0.662566 降到 0.436142；从 step 1 checkpoint 恢复后，step 2 loss 与学习率轨迹一致，能够生成新的最终 LoRA 和完整状态。跨独立 CUDA 进程比较的最大 LoRA 张量差约 `5e-5`，属于非位级确定性；恢复功能通过，但不声明逐 bit 复现。

代码检查：Ruff 通过，mypy 21 个源文件通过，pytest 40 passed。

## 停止条件

本轮只搭建框架和排查 Score，不启动 120k 正式训练。启动前先确认 Score 指标口径、正式配置、训练清单 hash、GPU 0 空闲和输出目录不存在。
