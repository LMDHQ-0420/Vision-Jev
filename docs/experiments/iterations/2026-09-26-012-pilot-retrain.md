# 2026-09-26-012：清洗数据 Pilot 重训

- 状态：running
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
- 完成后需运行完整结构化评测和 image-ablation，并在本文件补充最终结果。
