# 监督训练教程

## 门禁顺序

1. `doctor` 与数据校验；2. 合成张量测试；3. 50–200 样本过拟合；4. 12k pilot；5. 200 步 GPU profile；6. 120k × 2 主训练；7. 独立校准。任一门禁失败就停止扩大。

默认配置见 `configs/train/sft_main.json`：语言 LoRA rank 16/alpha 32/dropout 0.05，头 LR 1e-4，主干 LR 3e-5，global batch 64，BF16，FP32 概率/loss，gradient clipping 1.0。视觉编码器、连接器与 embedding 初始冻结。LoRA target 必须从真实语言层白名单审计，不允许粗暴 `all-linear`。

训练目标：Choice CE（多正确答案用负 log 概率和）、Noul BCE、Score CE + 0.1 RPS。先按问题等权；任何任务重权必须有独立消融。保存 best 与 last checkpoint、optimizer/scheduler、RNG 状态、训练样本游标以及逐来源指标。

启动训练前 `init-run`；运行命令写入 `commands.log`；过程中追加 JSONL metrics；完成后用 `finalize-run` 写状态与结果。OOM、NaN、数据异常和人工中止也必须 finalize 为 `failed`/`aborted` 并说明。
