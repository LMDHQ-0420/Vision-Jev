# 0006：Score 有序表示与区域候选坐标

## 状态

已采用，2026-09-25。

## 背景

首个 12k SFT pilot 的总体 exact match 为 75.80%，但 Score 只有 31.03%，RefCOCO、RefCOCOg 和 RefCOCO+ 约为 35%–45%。检查训练输入后确认：Score 仅使用缺少质量语义的 `score_1` 至 `score_5` 字符串；区域候选的框存在于 canonical 数据中，却没有进入模型 prompt，模型只能从随机 region ID 猜测目标。

## 决策

- 生成式 SFT 的 Score 答案改为有序整数 JSON，例如 `{"score":3}`；数据层继续保留 `score_3` 候选 ID，以兼容统一 schema。
- Prompt 明确写出 1–5 从 very poor 到 excellent 的有序质量语义，并列出 blur、noise、exposure、color、compression 等判断维度。
- 区域候选框按原图宽高归一化到 0–1000，以 `[x1,y1,x2,y2]` 写入每个候选。
- 普通样本保持约 256 个视觉 token；Score 和含 box 的区域样本提高到约 576 个视觉 token。
- 纠偏训练从已完成的 12k adapter 继续，以较小学习率训练 KonIQ-10k 与 RefCOCO 三来源；Score 做 3 倍重复，使有效 Score/区域样本数接近均衡。

## 影响

修正不改变 canonical 数据标签，也不伪造新标注。生成式 Score 输出需要在服务层映射回 `score_<level>`；后续概率头仍必须输出完整有序等级分布并用 RPS 等有序指标评测。区域坐标增加 prompt 长度和视觉计算量，因此只对确有 box 的样本启用更高预算。
