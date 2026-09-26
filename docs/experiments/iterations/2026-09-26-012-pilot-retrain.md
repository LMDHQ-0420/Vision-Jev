# 2026-09-26-012：清洗数据 Pilot 重训

- 状态：measured
- 数据：`/mnt/sda1/sol_data/vision-jev/manifests/pilot-12k.jsonl`
- 配置：`configs/train/sft_pilot_12k.json`
- 输出：`/mnt/sda1/sol_data/vision-jev/runs/qwen35-08b-sft-pilot-clean-dual`
- tmux：`vision-jev-sft-pilot`

## 目的

使用完成质量重建后的 12k pilot，从 Qwen3.5-0.8B 基座重新执行双卡 LoRA SFT，不继承旧清单训练得到的 adapter。

## 启动检查

- 训练集 11,339 条，评测集 661 条。
- 两张 RTX 4090 均正常加载模型。
- 首个优化 step 成功，训练损失为 `0.1535733640`。

## 训练结果

- 710 steps，双卡训练耗时 5,688.68 秒。
- 最终训练损失 `0.0769909471`，评测损失 `0.0407800957`。
- checkpoint：`qwen35-08b-sft-pilot-clean-dual/checkpoint-last`。

## 完整结构化评测

- 661 条全部完成，JSON 合法率 100%。
- 总体 exact match：82.90%。
- Choice：85.42%；Noul：81.11%；Score：41.38%。
- 平均生成延迟 0.389 秒，P95 0.645 秒，峰值显存约 2.16 GB。
- 逐样本结果：`eval-predictions.jsonl`。
- 汇总：`eval-predictions.summary.json`。

旧 pilot 与新 pilot 的 holdout 已因数据重建发生变化，因此历史指标只作方向性参考，不能视为严格配对实验。视觉依赖对照已完成，结果见 `2026-09-26-013-image-ablation.md`。
