# 2026-09-29：SFT Preview 发布与完整测评

- 状态：SFT Preview 已准备发布，完整 holdout 测评运行中
- 基座：`Qwen/Qwen3.5-0.8B`
- 基座 revision：`2fc06364715b967f1860aea9cf38778875588b17`
- 训练：双卡 LoRA，120,000 道完整问题，2 轮，7,140 optimizer step
- 最终验证 loss：0.0387403
- 最终 checkpoint：`/mnt/sda1/sol_data/vision-jev/runs/qwen35-08b-sft-main/checkpoint-last`

## 发布边界

当前发布物是生成式 SFT adapter，只生成 Choice、Noul 或 Score 的紧凑 JSON。它不输出经过校准的原生候选概率，也没有完成 RLCD 或闭环策略强化学习。README、STATUS 和模型卡均明确标注该边界，RLCD-inspired 校准决策后训练保留为 TODO。

仓库内发布目录为 `release/sft-preview/`，只包含约 16 MB 的 LoRA adapter、去除本地绝对路径的 adapter 配置及 SHA-256。完整训练状态和 optimizer 不进入公开权重。

## 测评

30 条 smoke test 已完成：JSON 合法率 100%，exact match 83.33%，平均生成延迟 0.401 秒，峰值分配显存约 2.10 GB。该样本只证明加载和生成链路可用，不作为最终模型质量结论。

完整评测使用全部 5,771 条 group-safe holdout，在 GPU 0 的 tmux `vision-jev-eval-sft-main` 中运行。逐条结果、汇总和日志写入主训练 run 目录；完成后需要把总体、分任务、分来源和 Score 序数指标同步回模型卡与状态页。
