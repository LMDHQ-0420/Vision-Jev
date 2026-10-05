# RLCD-inspired 数据计划

更新日期：2026-10-03。

## 范围

本项目采用 **RLCD-inspired calibrated-decision post-training** 这一名称。这里的 RLCD 指面向 Choice、Noul 与 Score 原生概率头的校准决策后训练；TypeSafe 没有公开完整训练方法，因此本计划不声称复现其专有 RLCD。用户口语中的 RFCD 在本项目内统一记为 RLCD-inspired。

SFT 的 117,000 道完整问题用于学习任务和输出格式。RLCD 阶段的目标不同：在保持准确率的同时，让预测概率与长期正确频率一致，并为自动执行、人工复核和拒答阈值提供独立数据。样本单位始终是唯一 `root_id` 对应的一道完整问题；候选换序、压力视图、重复采样和 rollout 都不计作新增问题。

机器可读配方见 `configs/data/rlcd_72k.json`。

完成 canonical 预处理后，可用纯 CPU 冻结根问题和生成确定性候选顺序视图：

```bash
python -m vision_jev.cli data-build-rlcd \
  --data-root /data/vision-jev \
  --sft-manifest /data/vision-jev/manifests/public-117k-training.jsonl \
  --output /data/vision-jev/manifests/rlcd/base-72k.jsonl \
  --views-output /data/vision-jev/manifests/rlcd/train-views.jsonl
```

构建器先冻结所有非训练角色，再从 SFT train 根样本中选择 48k train；任何评测
`root_id`/`group_id` 与 SFT 的交叉、train 根缺失或跨角色 group 都会令构建失败。

在主配方之前，先用 CLEVR、NLVR、KADID-10k 等低歧义来源跑通 20k 简单场景启动集；候选来源、配额、难度视图和验收规则见 [RLCD-inspired 简单场景启动集](rlcd-simple-scenarios.md)。

迷宫等程序环境采用单独的 16k 状态决策集和未见 seed 闭环评测，不计入 72k 静态根问题；见 [RLCD-inspired 交互场景计划](rlcd-interactive-scenarios.md)。

## 配额

总配额为 72,000 个唯一根问题。训练部分允许复用 SFT 的公开训练根样本，避免重新下载或伪造标签；其余 24,000 个根问题必须从 canonical pool 中尚未进入任何 SFT manifest 的组里冻结。

| 角色 | Choice | Noul | Score | 合计 | 用途 |
|---|---:|---:|---:|---:|---|
| train | 34,000 | 9,000 | 5,000 | 48,000 | RLCD 更新与 reward 消融 |
| dev | 4,200 | 1,200 | 600 | 6,000 | 训练中选 checkpoint；不拟合校准器 |
| calibration | 2,100 | 600 | 300 | 3,000 | 只拟合温度或其他冻结后校准参数 |
| threshold | 1,400 | 400 | 200 | 2,000 | 选择自动执行、复核和拒答阈值 |
| audit | 700 | 200 | 100 | 1,000 | 标签、泄漏、高置信错误和切片审计 |
| test | 8,400 | 2,400 | 1,200 | 12,000 | 一次性最终内部测评 |
| **合计** | **50,800** | **13,800** | **7,400** | **72,000** | |

70/20/10 的评测任务比例有意提高 Noul 和 Score 权重，使可靠性图和高置信错误不被 Choice 数量淹没。发布总体指标时同时给出未加权任务宏平均和按部署流量加权结果，不能只报混合微平均。

## 三层数据

### 1. 冻结根问题

`base-72k.jsonl` 保存原始图像引用、状态、问题、候选、标签、`root_id`、`group_id`、来源版本、许可与 `decision_role`。选择规则如下：

- train 只使用公开原始标签，并可从 SFT 的 `role=train` 部分复用根样本。
- dev/calibration/threshold/audit/test 的根样本不得出现在 public-117k 或 SFT 训练清单中。
- 任一 `root_id` 或 `group_id` 只能属于一个角色。共享图片、episode、网页演示、COCO image ID 和 KonIQ image ID 按来源适配器的最大关联组一起分配。
- 选择时按任务、来源、语言、候选数 K 与视觉/文本输入轨道分层。单一来源原则上不超过 train 的 20%；Score 在引入第二个 IQA 来源前必须记录例外。

### 2. 确定性训练视图

每个 train 根问题最多产生四个视图：一个 canonical、两个由固定 seed 生成的候选换序、一个标签保持的压力视图。压力视图只允许使用能程序验证标签不变的变换，例如非目标候选增删、候选同义措辞、关键区域外遮挡或无关历史裁剪。

以下变换不能静默进入训练：移除唯一正确候选、改变 Noul 命题真假、跨越 Score 等级阈值、错配图像，或使用隐藏答案决定候选。它们只作为带显式期望结果的反事实评测对。

### 3. 带版本的 rollout

RLCD rollout 由冻结根问题和视图生成，不覆盖 base manifest。每行至少记录：

- `sample_id`, `root_id`, `view_id`, `decision_role`；
- model、processor、head、policy 与 calibration revision；
- 按候选 ID 对齐的 logits、概率、mask 和候选顺序；
- 真实 target、argmax 是否正确，以及 Choice/Noul 的 NLL/Brier 或 Score 的 RPS；
- reward 各分量、KL/reference 项、采样 seed、时间和生成代码 revision。

概率必须在合法候选集合内归一化。Noul 的 0.5 表示概率中点，不自动表示“不知道”；Score 保存完整等级分布，期望等级不能当作成功概率。

## 构建顺序

1. 完成 15 个已规划来源的下载、子集物化和 canonical 转换，冻结每个来源的 revision 与资产清单。
2. 根据全部 SFT manifest 的 `root_id` 和 `group_id` 建立排除集合；先冻结 24k 独立评测根样本，再从 SFT train 选择 48k RLCD train 根样本。
3. 输出 source/task/K/language/label 分布、角色交叉、图片交叉和上游 split 使用报告。任一评测交叉非零则停止。
4. 生成固定训练视图，验证换序后的目标 ID、mask 与候选框仍严格对齐。
5. 用 SFT checkpoint 产生首轮原生概率基线 rollout；保存未校准 NLL、Brier、RPS、ECE、准确率和高置信错误。
6. RLCD 更新只查看 train；checkpoint 选择只看 dev；训练结束后才拟合 calibration；阈值只在 threshold 上选择。
7. audit 用于人工和程序审计，test 只在方案冻结后运行一次。所有报告记录 manifest SHA-256 和代码 commit。

## 发布门禁

- Choice/Noul/Score 的准确率不能相对固定 SFT 回归基线显著退化；容忍度在训练前写入运行配置。
- 必须同时报告 NLL、Brier、ECE、最大校准误差和高置信错误；Score 额外报告 RPS、MAE 与逐等级可靠性。
- 候选换序前后概率按候选 ID 对齐后应数值等价；K=2/8/16/32/64、信息不足、全 mask 和缺失区域分别报告。
- calibration、threshold、audit 和 test 不得用于梯度更新、早停或 reward 设计。
- 当前 Score 全部来自 KonIQ-10k，不能把单来源校准解释为跨域视觉质量校准。正式对外声明前应增加至少一个许可清晰、与 KonIQ 图像不重叠的 IQA 来源，并将其保留为 source-shift 测试。

该 72k 计划解决训练与内部校准数据预留。ScreenSpot/ScreenSuite 等外部视觉基准继续作为独立 zero-shot manifest，不计入 72k，也不参与校准参数或阈值拟合。
