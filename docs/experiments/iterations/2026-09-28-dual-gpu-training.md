# 2026-09-28：双卡全量训练准备

- 状态：等待双卡空闲
- 训练：Qwen3.5-0.8B LoRA，双卡 DDP
- 数据：114,229 train / 5,771 eval
- global batch：32（每卡 1，累积 16 步）

## 问题与修正

单卡恢复训练再次在 step 1050 后停止。堆栈确认故障发生在语言输出层：模型只监督末尾的紧凑 JSON 答案，但原实现仍为整段图文 prompt 计算全词表 logits，一次异常分配达到 67.16 GiB。全量扫描进一步定位到 Mind2Web 候选构造异常：193 条超过 32 个候选，最大一条有 3,114 个候选、约 40 万字符。

Mind2Web 适配器现先确定唯一目标，再从其余 DOM 节点确定性采样，保证候选总数不超过 32 且打散目标位置；随后覆盖重建 canonical、public、final 和 train/eval 清单。已有 API 改写只保留改写后的问题文本，其他不可变字段会从重建后的父样本自动刷新，不需要重新调用 API。

训练前向现只为可能参与答案 loss 的末尾区间计算 logits，监督目标保持不变。collator 同时增加 4096 token 的处理后序列硬门禁，超过门禁时会报告具体 sample ID 并停止，不会静默截断导致标签含义变化。

## 数据重建结果

- Mind2Web canonical：7,341 条。
- public 117k SHA-256：`bd52b63569977d0054b67103b3b181f1f63555d4edab04d249f3f8c696356e29`。
- final 120k SHA-256：`2e2d2e77dd4b4bc76765083671e7a7e11eb28ae862c7ed5d1e64a3dee760e0b9`。
- train/eval 清单 SHA-256：`9e0354fc6d7f5f72d0a8550dc6f8b1cf9f2c402d224700903868697c238946a4`。
- 重建后 120,000 条中候选超过 32 的数量为 0；最大候选数为 32。
- 对文本最长的四条样本执行原生 processor 抽查，处理后序列为 2,846–3,936 token，均低于 4,096 门禁。
- 3,274 条已有 API rewrite 已从新父样本刷新不可变字段，审计违规为 0；未产生新 API 请求。

## 启动策略

单卡 checkpoint 记录 `world_size=1`，与双卡状态不兼容，因此不恢复旧优化器、scheduler 或数据游标。正式训练从头开始，仍使用原完整训练清单。`scripts/wait_for_dual_gpu_sft.sh` 每 10 秒检查一次两张 GPU；两张卡显存占用均低于 10 GiB 后，先执行双卡 2-step smoke test，成功后才通过 Accelerate 启动两个正式训练进程。等待和训练均在 tmux 中后台进行。

运行目录：`/mnt/sda1/sol_data/vision-jev/runs/qwen35-08b-sft-main`。

等待日志：`/mnt/sda1/sol_data/vision-jev/runs/qwen35-08b-sft-main.wait.log`。

Smoke test 日志：`/mnt/sda1/sol_data/vision-jev/runs/qwen35-08b-sft-smoke-dual.log`。

训练日志：`/mnt/sda1/sol_data/vision-jev/runs/qwen35-08b-sft-main.log`。
