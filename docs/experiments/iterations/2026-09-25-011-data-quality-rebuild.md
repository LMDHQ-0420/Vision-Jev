# 2026-09-25-011：训练数据质量重建

- 状态：measured
- 关联 issue/ADR：数据质量抽样检查
- 代码 revision：本迭代提交
- run_id：data-quality-rebuild-20260925

## 假设与单一变量

保持来源配额和人工标签不变，只修复候选构造、区域目标消歧和训练输入遗漏。

## 代码/配置/数据变化

- 开放问答干扰项按问题族与答案类型构建，无法形成有效选择题的长尾行自动剔除。
- AndroidControl 与 Mind2Web 的重叠合法框统一为面积最小的目标框。
- GUI 目标框必须至少 50% 位于截图内，且不得覆盖超过 90% 的截图；不满足时自动剔除。
- Visual7W 候选文本改为中性区域名称，避免 box 名称提供错误语义暗示。
- SFT prompt 加入 `allowed_history`，顺序为从旧到新。
- 从原始数据覆盖重建受影响的 canonical 文件，并重建 117k public manifest。
- API 改写候选从新 public manifest 重新选择父样本并重新生成。
- API 续跑会自动清除不再属于当前父样本集合的缓存行。

## 命令与环境

- 环境：`vision-jev`
- 数据根目录：`/mnt/sda1/sol_data/vision-jev`
- 执行：`data-normalize`、`data-build-public`、`data-rewrite-api`、`data-audit-rewrites`、`data-build-final`

## 结果（计划值与实测值分表）

| 检查 | 计划 | 实测 |
|---|---:|---:|
| public 问题数 | 117,000 | 117,000 |
| public Choice | 94,000 | 94,000 |
| public Noul | 17,000 | 17,000 |
| public Score | 6,000 | 6,000 |
| 多目标监督 | 0 | 0 |
| 开放问答候选答案类型混杂 | 0 | 0 |
| 非空 GUI 历史 | 保留并进入 prompt | 26,520 |
| GUI 目标大半位于截图外 | 0 | 0 |
| API 候选 | 至少 3,000 | 3,274 |
| API 父子不变量违规 | 0 | 0 |
| 最终问题数 | 120,000 | 120,000 |

Public manifest SHA-256：`3fe3375cca4cb2756d4345ce163a64e4e60578203f9212ec08e577163ea103c7`。

最终 manifest SHA-256：`b0aa7cfcb3ca965df849303a5146ebdcbb1c7ffafb800d381dfcf2d0b1ebaa42`。

## 失败样本和限制

- ChartQA 与 VQAv2 各发现极少数无法形成兼容负例的长尾格式；预处理选择剔除，不注入弱负例。
- 问题族是确定性规则，不等价于完整语义理解；仍需用训练后 image-ablation 评测检查视觉依赖。
- 规则式问题族不能替代完整语义理解；国家/地区等边界类别仍可能互为负例，但不会再混入公司名、比例或年份。

## 结论：保留 / 回退 / 待复现

保留。public 清单已通过全量 schema、资源、目标映射、配额和新增质量统计。

## 下一步

基于新 120k 清单重新生成训练 pilot，并执行 image-ablation 检查视觉依赖。
