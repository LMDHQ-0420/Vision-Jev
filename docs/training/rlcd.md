# RLCD-inspired 静态决策头训练

本阶段从已经完成并通过 holdout 门禁的 SFT adapter 提取冻结隐藏状态，只训练
Choice、Noul 和 Score 三个原生决策头。它不更新视觉塔、语言主干或 SFT LoRA，
也不训练交互环境中的 policy/value 分支。

## 数据边界

- `train-views.jsonl` 的 156k 确定性视图用于梯度更新。
- `base-72k.jsonl` 中的 6k dev 用于训练过程检查和最终开发集报告。
- 3k calibration 只在参数冻结后拟合每任务温度。
- threshold、audit 和 test 不参与训练、选点或温度拟合。

Choice 使用正确候选集合的总概率质量损失，兼容交互数据中的多个同样最优动作；
Noul 使用二元交叉熵；Score 使用 NLL 与 ranked probability score。报告按任务分别包含
Accuracy、NLL、Brier 和 15-bin ECE，Score 额外包含 RPS 与 MAE。

## SFT 门禁

完整 5% group-safe holdout 评测完成后，用冻结配置检查门禁：

```bash
python scripts/check_sft_gate.py \
  --summary /data/vision-jev/runs/qwen35-08b-sft-main-117k/eval-full/predictions.summary.json \
  --gate configs/eval/sft_gate.json \
  --output /data/vision-jev/runs/qwen35-08b-sft-main-117k/eval-full/gate.json
```

门禁要求 JSON 有效率至少 99.5%、总体 exact match 至少 80%、Choice 和 Noul 分别
至少 80%，Score 相邻一级命中率至少 80%。任一条件失败就停止，不启动 RLCD。

## 训练

先用 `configs/train/rlcd_smoke.json` 完成双卡两步 smoke，再启动正式配置：

```bash
CUDA_VISIBLE_DEVICES=0,1 VISION_JEV_PROCESS_NAME=comfyui \
accelerate launch --num_processes 2 --multi_gpu --mixed_precision bf16 \
  -m vision_jev.cli train-rlcd \
  --config configs/train/rlcd_main.json \
  --output /data/vision-jev/runs/qwen35-08b-rlcd-main-72k
```

正式阶段每卡 microbatch 1、累积 16 步、global batch 32，训练 4,875 个优化步。
每 500 步保存一次 `decision_heads.safetensors`，并在 `metrics.jsonl` 中记录训练 loss
和 dev preview 的分任务可靠性指标。最终目录包含完整 dev 报告、温度参数、来源 SFT
checkpoint 和头部结构配置。

交互 16k 数据不混入本阶段。MiniGrid、BabyAI、Procgen Maze 和 Boxoban 的
policy/value 闭环训练属于独立 R0 阶段。

训练完成后，`eval-rlcd` 先在 threshold 角色上按任务选择满足 95% 经验准确率时的最大
覆盖阈值，再冻结阈值用于 audit 和 test。逐样本概率、置信度、高置信错误与覆盖率均写入
run 的 `evaluation/` 目录。`scripts/run_model_pipeline.sh` 固定执行顺序，并保证已经产生
test summary 后不会重复运行一次性 test。
