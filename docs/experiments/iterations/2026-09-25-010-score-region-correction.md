# Score 与区域定位纠偏

- 日期：2026-09-25
- 代码状态：实现完成，结构与 smoke 验证通过
- 训练证据：纠偏 smoke 已测量；正式双卡纠偏运行中
- 父 checkpoint：`qwen35-08b-sft-pilot-12k/checkpoint-last`

## 根因

- Score prompt 只有无解释的 `relative quality level 1–5`，输出也是不具备显式顺序的字符串 ID。
- RefCOCO 系列 canonical 样本带候选 box，但旧训练 prompt 没有序列化 box。
- 所有样本统一压到约 256 个视觉 token，不利于画质缺陷和细粒度区域判断。

## 修正

- Score 输出改为整数 1–5，并加入从 very poor 到 excellent 的等级含义和画质判断维度。
- 区域 box 按原图归一化到 0–1000 并进入候选文本。
- Score/区域视觉预算提高到约 576 token，其他任务保持约 256。
- 正式纠偏只包含 KonIQ-10k、RefCOCO、RefCOCOg、RefCOCO+，Score 重复 3 倍。
- 唯一训练样本 2,264；有效训练样本 3,406，其中 Score 1,713、区域 Choice 1,693。

## Smoke

- 配置：`configs/train/sft_corrective_smoke.json`
- 路径：`/mnt/sda1/sol_data/vision-jev/runs/qwen35-08b-sft-corrective-smoke`
- 唯一弱项样本：32；重采样后有效样本：40
- 更新步：20
- eval loss：0.0390762
- 结果：父 adapter 恢复、梯度更新、动态视觉预算和 checkpoint 保存均通过。

## 正式纠偏

- 配置：`configs/train/sft_corrective.json`
- 路径：`/mnt/sda1/sol_data/vision-jev/runs/qwen35-08b-sft-corrective`
- 硬件：RTX 4090 × 2，BF16
- 计划：2 epochs，global batch 32，LR 1e-5
- 状态：已启动；完成后在原 653 条 group-safe eval 上复测，与总体 75.80%、Choice 77.53%、Noul 80.00%、Score 31.03% 的基线直接比较。

## 验证

- Ruff lint：通过
- mypy strict：通过
- pytest：33 passed
- 真实 processor 检查：区域框进入 prompt；Score 标签为整数；两类视觉预算约 576 token
