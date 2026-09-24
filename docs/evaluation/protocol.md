# 评测协议

## 判断

Choice：accuracy、macro-F1、NLL、Brier、高置信错误；Noul 增加不平衡指标；Score：等级误差与 RPS。按 source、task、language、K、visual token、文字/空间/计数/区域/图表/信息不足分层。

结构测试包括候选同步换序、集合增删、近义候选、K=2/8/16/32/64、全 mask、末级 Score、缺失区域、反事实成对正确率。区域任务同时报告 proposal recall 与给定候选的条件准确率。

## 性能

同卡、同精度、同 processor、同视觉 token、同 K/长度分布比较 reference/shared/基线。预热后记录 p50/p95、问题/秒、候选/秒、视觉 token/秒、峰值显存、GPU 利用率和数据等待；覆盖预处理到输出，不只选 CUDA 最快片段。

## 闭环

固定未见 seed，快速 300 episode，最终尽量 1,000 配对 episode并报告置信区间。保存失败轨迹/截图序列。静态 VQA、oracle proposal 与闭环成功率不能汇总成一个含糊总分。
