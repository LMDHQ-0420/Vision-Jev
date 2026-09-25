# 监督训练教程

## 门禁顺序

1. `doctor` 与数据校验；2. 合成张量测试；3. 50–200 样本过拟合；4. 12k pilot；5. 200 步 GPU profile；6. 120k × 2 主训练；7. 独立校准。任一门禁失败就停止扩大。

默认配置见 `configs/train/sft_main.json`：语言 LoRA rank 16/alpha 32/dropout 0.05，头 LR 1e-4，主干 LR 3e-5，global batch 64，BF16，FP32 概率/loss，gradient clipping 1.0。视觉编码器、连接器与 embedding 初始冻结。LoRA target 必须从真实语言层白名单审计，不允许粗暴 `all-linear`。

训练目标：Choice CE（多正确答案用负 log 概率和）、Noul BCE、Score CE + 0.1 RPS。先按问题等权；任何任务重权必须有独立消融。保存 best 与 last checkpoint、optimizer/scheduler、RNG 状态、训练样本游标以及逐来源指标。

启动训练前 `init-run`；运行命令写入 `commands.log`；过程中追加 JSONL metrics；完成后用 `finalize-run` 写状态与结果。OOM、NaN、数据异常和人工中止也必须 finalize 为 `failed`/`aborted` 并说明。

## Qwen3.5 原生多模态 SFT

首个主干固定为 `Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17`。模型必须先通过显式 prepare 命令下载到数据盘，训练代码只使用本地快照，不在模型构造期间联网：

```bash
vision-jev model-prepare --config configs/model/qwen35_08b.json
```

图片由模型原生 `AutoProcessor` 处理。普通样本最多约 256 个合并视觉 token，Score 和区域候选样本约 576 个。问题、可见状态和完整动态候选集合进入 user message；训练 loss 只覆盖 assistant 的紧凑 JSON 答案，不覆盖提示词。Choice 返回候选 ID，Noul 返回布尔值，生成式 Score 返回具有明确顺序的整数 1–5。含 box 的候选会把原图坐标归一化到 0–1000 后写入 prompt。视觉塔冻结，LoRA 只注入经过白名单审计的语言层 `q/k/v/o_proj`、`in_proj_qkv` 与 `out_proj`。

12k pilot 从完整 120k manifest 按来源和任务比例确定性抽取，验证集按 `group_id` 隔离：

```bash
vision-jev data-build-pilot \
  --input /mnt/sda1/sol_data/vision-jev/manifests/sft-120k.jsonl \
  --output /mnt/sda1/sol_data/vision-jev/manifests/pilot-12k.jsonl \
  --questions 12000 --seed vision-jev-pilot-12k
```

训练门禁命令：

```bash
vision-jev train-sft \
  --config configs/train/sft_overfit.json \
  --data /mnt/sda1/sol_data/vision-jev/manifests/pilot-12k.jsonl \
  --output /mnt/sda1/sol_data/vision-jev/runs/qwen35-08b-overfit-32

accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
  -m vision_jev.cli train-sft \
  --config configs/train/sft_pilot_12k.json \
  --data /mnt/sda1/sol_data/vision-jev/manifests/pilot-12k.jsonl \
  --output /mnt/sda1/sol_data/vision-jev/runs/qwen35-08b-sft-pilot-12k
```

训练完成后必须运行 `vision-jev eval-sft`，同时报告验证 loss、JSON 合法率、总体 exact match 和三类任务分项 exact match。

弱项纠偏从已完成的 pilot adapter 恢复，只训练 KonIQ-10k 和 RefCOCO 三来源；配置见 `configs/train/sft_corrective.json`。该实验用于验证表示修正，不替代后续完整数据重训。

仓库也提供可直接运行的独立脚本。`--maximum 0` 表示评测完整 holdout；逐样本预测和汇总文件写入训练 run 目录：

```bash
python scripts/evaluate_sft.py --maximum 0 --progress-every 25
```
