# 2026-09-27-015：全量 LoRA 中断与恢复

- 状态：superseded_by_dual_gpu_restart
- 原 run：`qwen35-08b-sft-main`
- 恢复 run：`qwen35-08b-sft-main-resumed-1000`
- tmux：`vision-jev-sft-main`

## 中断状态

原单卡训练最后写入 step 1050 / 7140，约完成 14.7%，对应约 33,600 条训练问题。第二次恢复的持久化堆栈确认 Mind2Web 超大候选样本使语言输出层尝试申请 67.16 GiB，训练因 OOM 退出。该单卡恢复方案已由修复数据后的双卡从头训练替代。

step 1000 checkpoint 已完整保存，包含 LoRA、Accelerate 模型状态、优化器、scheduler、随机状态和 batch 游标。记录的游标为 epoch 0、batch 32,000、world size 1。

## 恢复策略

旧 step 1000 checkpoint 的 `world_size=1` 与双卡训练不兼容，不再恢复。旧单卡 run 在双卡任务启动前清理。

等待日志：`/mnt/sda1/sol_data/vision-jev/runs/qwen35-08b-sft-main-resumed-1000.wait.log`。

训练日志：`/mnt/sda1/sol_data/vision-jev/runs/qwen35-08b-sft-main-resumed-1000.log`。

## 恢复执行

等待任务于 2026-09-27 15:01 首次尝试启动，但 GPU 0 随即被另一个约占 38.61 GiB 的进程使用，本项目首批前向只剩 1.78 GiB 可用并明确 OOM；该尝试没有产生新 optimizer step。2026-09-28 确认两张 GPU 空闲后再次从 step 1000 启动，加入 `expandable_segments` 显存分配配置并继续追加同一日志。第二次恢复已成功加载完整 checkpoint，GPU 0 正常计算；GPU 1 保持空闲。
