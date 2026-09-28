# 2026-09-27-015：全量 LoRA 中断与恢复

- 状态：resumed_running
- 原 run：`qwen35-08b-sft-main`
- 恢复 run：`qwen35-08b-sft-main-resumed-1000`
- tmux：`vision-jev-sft-main`

## 中断状态

原单卡训练最后写入 step 1050 / 7140，约完成 14.7%，对应约 33,600 条训练问题。最后指标写入时间为 2026-09-27 02:27 CST，tmux 会话随后不存在且没有生成最终 summary。终端日志未持久化，结合运行中曾出现的显存分配警告，怀疑极端样本触发 OOM，但缺少退出堆栈，不能确认为唯一原因。

step 1000 checkpoint 已完整保存，包含 LoRA、Accelerate 模型状态、优化器、scheduler、随机状态和 batch 游标。记录的游标为 epoch 0、batch 32,000、world size 1。

## 恢复策略

两张 GPU 当前均由其他用户任务占用，不抢占或终止其进程。已建立 tmux 等待任务，每 60 秒检查 GPU 0；释放后自动从 step 1000 恢复到新的输出目录，不覆盖原 run。恢复要求继续使用单卡、相同配置和同一训练清单。

等待日志：`/mnt/sda1/sol_data/vision-jev/runs/qwen35-08b-sft-main-resumed-1000.wait.log`。

训练日志：`/mnt/sda1/sol_data/vision-jev/runs/qwen35-08b-sft-main-resumed-1000.log`。

## 恢复执行

等待任务于 2026-09-27 15:01 首次尝试启动，但 GPU 0 随即被另一个约占 38.61 GiB 的进程使用，本项目首批前向只剩 1.78 GiB 可用并明确 OOM；该尝试没有产生新 optimizer step。2026-09-28 确认两张 GPU 空闲后再次从 step 1000 启动，加入 `expandable_segments` 显存分配配置并继续追加同一日志。第二次恢复已成功加载完整 checkpoint，GPU 0 正常计算；GPU 1 保持空闲。
