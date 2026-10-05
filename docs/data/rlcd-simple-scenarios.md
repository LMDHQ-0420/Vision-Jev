# RLCD-inspired 简单场景启动集

更新日期：2026-10-03。

## 目标

这份方案用于先把校准决策训练闭环跑通，而不是替代 `rlcd_72k` 主计划。启动集优先选择标签可程序验证、视觉元素少、候选空间小、难度可控的数据。项目沿用仓库术语 **RLCD-inspired**；用户口语中的 RFCD 指同一阶段。

简单数据不能全部保持“干净且一眼可答”。那样只能训练高置信正确，无法覆盖低置信、错误和阈值路由。每个根问题应保留 canonical 图像，并生成固定 seed 的轻度、中度、重度困难视图；视图不计作新根问题。

## 现有来源：无需新增下载

| 优先级 | 来源 | 任务 | 简单场景 | 使用建议 |
|---|---|---|---|---|
| P0 | CLEVR | Choice / Noul | 颜色、形状、材质、大小、存在性、计数比较、左右关系 | 首选。按 functional program 长度、对象数和关系跳数分层，标签由程序和场景图验证 |
| P0 | NLVR | Noul | 合成图形上的真假命题、集合比较和空间关系 | 使用原始 NLVR；六个 image permutation 必须保持同组 |
| P1 | GQA Balanced | Choice / Noul | 单属性、对象存在、简单空间关系 | 只取 scene graph 可验证、短程序、短答案样本；不完整场景图样本进入审计而非校准集 |
| P1 | Visual7W / RefCOCO | Choice | 四个候选框中的目标定位 | 使用 oracle box 轨；按目标面积、候选 IoU 和同类干扰物数量控制难度 |
| P1 | KonIQ-10k | Score | 五级相对图像质量 | 保留人类评分分布；只作为自然失真来源，不能单独代表跨域 Score 校准 |

GUI 操作、长网页历史、OCR 密集图片和开放域科学题暂不进入第一阶段。它们适合在简单场景的概率头、proper scoring loss 和阈值流程稳定后再加入。

## 建议新增的三个小来源

### easy-VQA

- 4,000 张训练图、1,000 张测试图，合计 48,248 个问题；图像为 64x64 的简单彩色形状。
- 可直接构造颜色/形状 Choice，以及存在性 Noul；代码和仓库数据采用 MIT 许可。
- 只用于训练、开发和校准冒烟测试，不把它当成真实世界泛化证据。

### FigureQA

- 五类合成图表、十五类关系问题，训练和验证问题均为 yes/no，适合 Noul。
- 可覆盖最大/最小、大小比较、中位数、曲线粗糙度和面积关系等简单图表判断。
- 生成代码为 MIT；正式纳入前仍需单独确认下载数据包的发布条款。若条款不清晰，可用官方生成代码生成资产，并记录配置和 seed。

### KADID-10k

- 81 张 pristine 图像，每张经过 25 种失真、5 个强度，共 10,125 张失真图；每图有人类 DMOS 和方差。
- 适合 Score：既有明确失真阶梯，又有人类感知分数，可补足 KonIQ 的自然失真单来源问题。
- 必须按 81 个 pristine reference 分组切分，绝不能把同一原图的不同失真放到不同角色。
- 数据面向研究社区开放；权重或数据再分发前继续执行许可复核。

TID2013 只有 25 张 reference，内容组过少，不建议训练；可作为不参与调参的外部 source-shift 测试。

## 20k 启动配方

以下均为唯一根问题。它是 `rlcd_72k` 的先导切片，后续并入 72k 时必须按 `root_id` 和 `group_id` 去重。

| 任务 | 来源与数量 | 合计 |
|---|---|---:|
| Choice | CLEVR 4,000；easy-VQA 3,000；GQA-simple 2,000；Visual7W/RefCOCO oracle 1,000 | 10,000 |
| Noul | CLEVR 2,000；NLVR 2,000；easy-VQA 1,500；FigureQA 1,500 | 7,000 |
| Score | KADID-10k 2,000；KonIQ-10k 1,000 | 3,000 |
| **总计** |  | **20,000** |

建议角色为 train 12,000、dev 2,000、calibration 2,000、threshold 1,000、audit 500、test 2,500。非 train 的根问题必须从所有 SFT manifest 中排除；新增来源虽然天然不与 SFT 重叠，仍需做图片哈希和近重复检查。

## 难度视图

每个 train 根问题最多四个视图：

1. `canonical`：原图和原始候选。
2. `easy`：仅候选换序，不改变视觉输入。
3. `medium`：轻度缩放、JPEG、模糊或增加同类型干扰候选。
4. `hard`：更强但仍可审计的遮挡、降采样、低对比度或相近候选。

变换参数必须写入 manifest。Choice 的 target 始终按 candidate ID 重映射；Noul 真值不改变；Score 不对同一张 KADID 图再施加失真，以免原始 DMOS 标签失效。若变换让证据完全不可见，该视图应标记为 `insufficient_evidence_stress`，只用于阈值和拒答评测，不能静默当作普通训练样本。

## 采样和验收

- 先用 SFT checkpoint 对候选池跑 baseline，再按正确/错误与置信度分桶采样；不能只按类别均衡。
- 启动集至少覆盖置信度区间 `[0.5,0.6)` 到 `[0.9,1.0]`，每个区间单独报告 accuracy、NLL、Brier/ECE；Score 报 RPS。
- 评测同时报告 canonical 与困难视图，避免通过大量简单样本获得虚假的低 ECE。
- easy-VQA、CLEVR 和 NLVR 的合成域结果只证明训练闭环有效；最终结论必须回到 GQA、区域定位、KonIQ/KADID 和后续真实场景切片。
- 第一轮建议只实现 CLEVR + NLVR + KADID-10k。三者标签强、场景简单、任务覆盖完整，最适合快速验证 Choice/Noul/Score 三个原生概率头。

迷宫、钥匙门、动态避障和语言指令等实际交互任务采用独立 seed、oracle 与在线 rollout 管理，不计入本页 20k 静态根问题；见 [RLCD-inspired 交互场景计划](rlcd-interactive-scenarios.md)。
