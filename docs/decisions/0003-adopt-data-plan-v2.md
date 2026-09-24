# ADR-0003：采用 VisJev 数据方案 v2.0

- 状态：accepted
- 日期：2026-09-24

## 决定

主 SFT 仍为 120k 整题，但改为公开 90k、程序/环境 24k、可核验增强 6k。新增 AndroidControl 和 WebLINX，Mind2Web 上限降到 6k，VQAv2 降到 8k，SST 从主训练移除。主 manifest 禁止 teacher-only 标签。

## 理由与影响

该结构把预算转向真实 GUI 动作、集合比较、视觉等级和已知不确定性，并修正 Mind2Web 官方训练动作不足以支持旧配额的问题。代价是 raw 数据更大、WebLINX 有非商业许可限制、AndroidControl 需要官方 GCS 哈希复核，且 GUI 数据必须实施严格历史截断、PII 和候选可见性检查。

旧 `sft-120k-v1` 不再用于新 run；已有 SST 快照不删除，但被标记 inactive，不能进入 v2 manifest。
