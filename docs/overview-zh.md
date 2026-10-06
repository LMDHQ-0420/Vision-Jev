<div align="center">
  <img src="../asset/Vision-Jev.svg" alt="Vision-Jev" width="380">
  <h3>一个支持视觉的开源 JEV-like 模型</h3>
  <p>训练代码、数据配额、评测流程与模型权重，全流程开放。</p>
  <p><a href="../README.md">English</a> · <b>简体中文</b> · <a href="index.md">项目文档</a> · <a href="../LICENSE">Apache-2.0</a></p>
</div>

---

## 项目简介

Vision-Jev 是一个面向**动态候选判断与行动**的开源视觉语言基础项目。模型接收图片、可见状态、问题和数量不断变化的候选集合，输出三类结构化结果：

- **Choice**：选择或排序最合适的候选；
- **Noul**：判断信息是否充分，或某个命题是否成立；
- **Score**：输出有序的质量或置信等级。

同一套视觉表示还会连接独立的 policy/value 分支，用于闭环环境中的行动学习。Vision-Jev 重点解决可审计、可校准、可评测并能连接真实动作的视觉判断，而不是开放式聊天。

## 两个核心亮点

### 1. 训练一个开源 Vision-Jev

Vision-Jev 将视觉理解引入 JEV-like 模型范式，并完整开放从数据到模型权重的实现过程。

### 2. 开源完整训练链路

项目将开放数据处理、精确配额、固定来源、确定性 manifest、SFT 与策略训练代码、评测协议、实验结果、模型权重、模型卡和数据卡。第三方原始数据继续遵守各自许可；本项目公开可复现的处理路径，不把上游资产声明为自有数据。

## 开放数据配方

SFT 数据包含 **117,000 道公开来源完整问题**，本项目新增人工标注为 0。

| 数据块 | Choice | Noul | Score | 合计 |
|---|---:|---:|---:|---:|
| 公开数据集 | 94,000 | 17,000 | 6,000 | 117,000 |
| 本地合成 | 0 | 0 | 0 | 0 |
| **合计** | **94,000** | **17,000** | **6,000** | **117,000** |

数据覆盖 GUI 操作、区域定位、组合推理、VQA、OCR、图表理解、文本逻辑和视觉质量。全部样本来自固定版本的公开来源，并保留原始语言。

- [数据来源账本](data/sources.md)
- [数据混合方案](data/mixtures.md)
- [机器可读来源](../configs/data/sources.json)
- [训练配方](../configs/data/sft_117k.json)

<!-- showcase:start -->
## 静态 RLCD 结果

原始 Qwen、完成 SFT 但未接入静态 RLCD 决策头的 checkpoint, 以及完整 Vision-Jev 均在同一份冻结的 12,000 题测试集上评测。以下汇总保留全部成功与失败样本, 因此不同阶段的原始准确率回退也会直接展示。原始 Qwen 与 SFT-only 生成不提供校准决策头概率, 因此阈值结果记为 N/A。

| 模型 | Choice 准确率 | Noul 准确率 | Score 准确率 | Choice 阈值结果 | Noul 阈值结果 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B (原始) | 24.5% | 39.5% | 2.9% | N/A | N/A |
| Qwen3.5-0.8B + SFT | 90.4% | 81.4% | 54.7% | N/A | N/A |
| Vision-Jev-0.8B | 80.8% | 81.2% | 55.5% | 93.9%, 覆盖率 65.9% | 94.7%, 覆盖率 53.2% |
| Qwen3.5-9B (原始) | 87.1% | 80.4% | 32.9% | N/A | N/A |
| Qwen3.5-9B + SFT | 95.4% | 89.3% | 60.5% | N/A | N/A |
| Vision-Jev-9B | 74.4% | 88.9% | 62.1% | 94.8%, 覆盖率 60.7% | 94.7%, 覆盖率 84.9% |

### 结构化输出有效率

| 模型 | Choice | Noul | Score |
| --- | ---: | ---: | ---: |
| Qwen3.5-0.8B | 37.4% | 61.0% | 22.2% |
| Qwen3.5-0.8B + SFT | 100.0% | 100.0% | 100.0% |
| Vision-Jev-0.8B | 100.0% | 100.0% | 100.0% |
| Qwen3.5-9B | 99.5% | 100.0% | 100.0% |
| Qwen3.5-9B + SFT | 100.0% | 100.0% | 100.0% |
| Vision-Jev-9B | 100.0% | 100.0% | 100.0% |

Vision-Jev 通过固定决策头直接返回候选, 因而结构化输出始终有效。

### Vision-Jev 概率与校准指标

| 模型 | 任务 | 题数 | 准确率 | NLL | Brier | ECE | RPS | MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vision-Jev-0.8B | Choice | 8,400 | 80.8% | 0.530 | 0.269 | 0.026 | N/A | N/A |
| Vision-Jev-0.8B | Noul | 2,400 | 81.2% | 0.396 | 0.253 | 0.031 | N/A | N/A |
| Vision-Jev-0.8B | Score | 1,200 | 55.5% | 1.003 | 0.552 | 0.031 | 0.090 | 0.514 |
| Vision-Jev-9B | Choice | 8,400 | 74.4% | 0.610 | 0.309 | 0.016 | N/A | N/A |
| Vision-Jev-9B | Noul | 2,400 | 88.9% | 0.231 | 0.145 | 0.023 | N/A | N/A |
| Vision-Jev-9B | Score | 1,200 | 62.1% | 0.834 | 0.485 | 0.041 | 0.071 | 0.405 |

#### 阈值策略

| 模型 | 任务 | 阈值 | 覆盖率 | 接受准确率 |
| --- | --- | ---: | ---: | ---: |
| Vision-Jev-0.8B | Choice | 0.762 | 65.9% | 93.9% |
| Vision-Jev-0.8B | Noul | 0.797 | 53.2% | 94.7% |
| Vision-Jev-0.8B | Score | 0.886 | 9.2% | 97.3% |
| Vision-Jev-9B | Choice | 0.665 | 60.7% | 94.8% |
| Vision-Jev-9B | Noul | 0.697 | 84.9% | 94.7% |
| Vision-Jev-9B | Score | 0.915 | 14.3% | 96.5% |

#### 高置信错误

统计置信度不低于 0.9 但预测错误的完整测试样本。

| 模型 | 错误数 | 测试题数 | 占比 |
| --- | ---: | ---: | ---: |
| Vision-Jev-0.8B | 163 | 12,000 | 1.4% |
| Vision-Jev-9B | 86 | 12,000 | 0.7% |

### 推理耗时

原始 Qwen 与 SFT 的 JSONL 保留每一道题的生成耗时; Vision-Jev 的独立计时复测保留同步后的完整决策路径耗时。下表由全部逐题记录汇总, 两种口径分开标注。

| 模型 | 口径 | 总计 (s) | 均值 (ms) | P50 | P95 | P99 | 最小 | 最大 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 逐题生成 | 3718.1 | 309.8 | 241.9 | 542.3 | 550.7 | 171.3 | 671.0 |
| Qwen3.5-0.8B + SFT | 逐题生成 | 3802.8 | 316.9 | 286.6 | 444.9 | 501.7 | 248.0 | 569.6 |
| Vision-Jev-0.8B | 完整决策循环 | 1051.6 | 87.6 | 83.2 | 147.3 | 193.8 | 41.7 | 292.2 |
| Qwen3.5-9B | 逐题生成 | 4372.3 | 364.4 | 326.9 | 507.7 | 530.2 | 263.7 | 804.1 |
| Qwen3.5-9B + SFT | 逐题生成 | 5259.5 | 438.3 | 392.0 | 609.1 | 691.4 | 344.2 | 754.0 |
| Vision-Jev-9B | 完整决策循环 | 1384.5 | 115.4 | 105.6 | 197.3 | 273.0 | 55.1 | 318.5 |

#### 分任务耗时

| 模型 | 任务 | 样本 | 均值 (ms) | P50 | P95 | P99 | 最小 | 最大 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | Choice | 8,400 | 298.2 | 241.4 | 535.7 | 542.2 | 190.0 | 652.2 |
| Qwen3.5-0.8B | Noul | 2,400 | 270.5 | 218.9 | 505.7 | 528.8 | 210.0 | 636.3 |
| Qwen3.5-0.8B | Score | 1,200 | 469.8 | 541.3 | 548.5 | 656.2 | 171.3 | 671.0 |
| Qwen3.5-0.8B + SFT | Choice | 8,400 | 341.4 | 290.7 | 447.1 | 508.1 | 267.4 | 569.6 |
| Qwen3.5-0.8B + SFT | Noul | 2,400 | 262.2 | 258.9 | 290.8 | 317.8 | 252.4 | 565.8 |
| Qwen3.5-0.8B + SFT | Score | 1,200 | 255.2 | 251.7 | 301.8 | 305.6 | 248.0 | 308.8 |
| Vision-Jev-0.8B | Choice | 8,400 | 88.5 | 84.1 | 167.6 | 199.0 | 41.7 | 250.2 |
| Vision-Jev-0.8B | Noul | 2,400 | 72.0 | 73.7 | 78.9 | 83.0 | 52.1 | 292.2 |
| Vision-Jev-0.8B | Score | 1,200 | 112.9 | 113.5 | 116.8 | 119.2 | 90.7 | 138.9 |
| Qwen3.5-9B | Choice | 8,400 | 390.9 | 330.3 | 510.5 | 563.5 | 263.7 | 804.1 |
| Qwen3.5-9B | Noul | 2,400 | 295.9 | 293.3 | 302.7 | 357.7 | 285.9 | 365.7 |
| Qwen3.5-9B | Score | 1,200 | 315.7 | 314.0 | 316.9 | 359.1 | 311.5 | 363.0 |
| Qwen3.5-9B + SFT | Choice | 8,400 | 471.3 | 396.5 | 612.6 | 696.3 | 362.9 | 754.0 |
| Qwen3.5-9B + SFT | Noul | 2,400 | 357.9 | 351.9 | 424.5 | 431.1 | 344.2 | 624.3 |
| Qwen3.5-9B + SFT | Score | 1,200 | 368.3 | 363.5 | 420.8 | 424.9 | 360.3 | 432.0 |
| Vision-Jev-9B | Choice | 8,400 | 116.0 | 110.7 | 227.3 | 276.7 | 55.1 | 318.5 |
| Vision-Jev-9B | Noul | 2,400 | 94.1 | 96.0 | 111.8 | 122.6 | 72.7 | 133.2 |
| Vision-Jev-9B | Score | 1,200 | 153.9 | 158.0 | 180.6 | 185.9 | 133.2 | 192.4 |

### 0.8B 分数据集与任务结果

每个结果单元格依次为准确率 / 单样本平均耗时。

| 数据集 | 任务 | 题数 | Qwen3.5-0.8B | Qwen3.5-0.8B + SFT | Vision-Jev-0.8B |
| --- | --- | ---: | ---: | ---: | ---: |
| Android Control | Choice | 700 | 1.0% / 281.9 ms | 84.1% / 357.2 ms | 52.3% / 173.0 ms |
| ChartQA | Choice | 700 | 53.6% / 234.0 ms | 97.0% / 291.0 ms | 79.4% / 83.7 ms |
| ChartQA | Noul | 82 | 45.1% / 300.0 ms | 61.0% / 265.3 ms | 56.1% / 75.5 ms |
| CLEVR | Choice | 700 | 17.9% / 277.9 ms | 93.6% / 288.4 ms | 87.9% / 75.8 ms |
| CLEVR | Noul | 580 | 59.3% / 229.4 ms | 92.6% / 261.0 ms | 91.6% / 73.2 ms |
| GQA | Choice | 700 | 29.1% / 278.4 ms | 96.7% / 288.4 ms | 95.9% / 76.4 ms |
| GQA | Noul | 580 | 29.8% / 286.2 ms | 83.1% / 262.9 ms | 83.8% / 74.2 ms |
| KonIQ-10k | Score | 1,200 | 2.9% / 469.8 ms | 54.7% / 255.2 ms | 55.5% / 112.9 ms |
| MultiNLI | Choice | 700 | 44.3% / 226.9 ms | 76.3% / 276.7 ms | 72.6% / 52.5 ms |
| NLVR | Noul | 579 | 23.0% / 288.2 ms | 64.1% / 259.3 ms | 64.4% / 64.4 ms |
| RefCOCO | Choice | 700 | 0.0% / 413.4 ms | 95.0% / 434.3 ms | 83.1% / 91.3 ms |
| RefCOCOg | Choice | 700 | 0.0% / 401.8 ms | 90.4% / 431.1 ms | 80.1% / 90.0 ms |
| RefCOCO+ | Choice | 700 | 0.0% / 405.0 ms | 88.4% / 432.4 ms | 78.6% / 91.6 ms |
| ScienceQA | Choice | 700 | 47.1% / 233.9 ms | 85.3% / 289.9 ms | 83.1% / 76.3 ms |
| TextVQA | Choice | 700 | 54.0% / 228.5 ms | 98.0% / 291.7 ms | 92.0% / 85.1 ms |
| Visual7W | Choice | 700 | 20.6% / 340.1 ms | 80.7% / 423.4 ms | 67.7% / 85.3 ms |
| VQAv2 | Choice | 700 | 25.9% / 257.0 ms | 98.9% / 291.7 ms | 96.4% / 81.0 ms |
| VQAv2 | Noul | 579 | 44.9% / 274.1 ms | 88.6% / 265.1 ms | 88.8% / 75.8 ms |

### 9B 分数据集与任务结果

每个结果单元格依次为准确率 / 单样本平均耗时。

| 数据集 | 任务 | 题数 | Qwen3.5-9B | Qwen3.5-9B + SFT | Vision-Jev-9B |
| --- | --- | ---: | ---: | ---: | ---: |
| Android Control | Choice | 700 | 41.0% / 468.7 ms | 89.6% / 544.1 ms | 34.7% / 237.5 ms |
| ChartQA | Choice | 700 | 94.7% / 330.4 ms | 98.0% / 399.3 ms | 81.6% / 108.4 ms |
| ChartQA | Noul | 82 | 80.5% / 299.6 ms | 80.5% / 364.5 ms | 80.5% / 96.1 ms |
| CLEVR | Choice | 700 | 90.0% / 321.8 ms | 98.9% / 392.8 ms | 96.3% / 96.9 ms |
| CLEVR | Noul | 580 | 88.3% / 294.0 ms | 99.1% / 356.5 ms | 99.0% / 94.8 ms |
| GQA | Choice | 700 | 96.4% / 326.1 ms | 98.0% / 393.1 ms | 97.1% / 98.7 ms |
| GQA | Noul | 580 | 80.2% / 297.4 ms | 89.7% / 358.8 ms | 89.5% / 96.9 ms |
| KonIQ-10k | Score | 1,200 | 32.9% / 315.7 ms | 60.5% / 368.3 ms | 62.1% / 153.9 ms |
| MultiNLI | Choice | 700 | 73.0% / 306.3 ms | 86.6% / 374.5 ms | 87.1% / 67.0 ms |
| NLVR | Noul | 579 | 64.1% / 291.7 ms | 74.6% / 354.1 ms | 73.9% / 85.6 ms |
| RefCOCO | Choice | 700 | 96.7% / 492.4 ms | 97.6% / 595.9 ms | 60.7% / 119.5 ms |
| RefCOCOg | Choice | 700 | 93.1% / 487.7 ms | 94.9% / 588.8 ms | 59.7% / 117.6 ms |
| RefCOCO+ | Choice | 700 | 90.7% / 492.0 ms | 93.4% / 595.6 ms | 56.0% / 119.8 ms |
| ScienceQA | Choice | 700 | 94.6% / 326.7 ms | 97.0% / 394.6 ms | 95.6% / 99.5 ms |
| TextVQA | Choice | 700 | 98.4% / 331.9 ms | 99.6% / 400.1 ms | 94.7% / 109.9 ms |
| Visual7W | Choice | 700 | 80.4% / 476.5 ms | 91.4% / 577.0 ms | 30.3% / 111.0 ms |
| VQAv2 | Choice | 700 | 96.6% / 329.8 ms | 99.7% / 399.7 ms | 98.6% / 105.8 ms |
| VQAv2 | Noul | 579 | 89.1% / 300.1 ms | 95.0% / 361.3 ms | 94.5% / 98.6 ms |

## 冻结测试集展示

每段动画都让原始 Qwen3.5 checkpoint 与 Vision-Jev 使用同一个冻结测试样本和候选集。类别预先确定; 每个类别选择两套 Vision-Jev checkpoint 都回答正确、通过既定置信阈值且样本 ID 的 SHA-256 最小的样本, 选择过程不使用基线预测。每张 GIF 会依次播放全部固定样例; 两条进度条按照预热后 3 次推理耗时的中位数推进, 并在对应模型完成时显示答案。

覆盖能力: 通用视觉问答、自然图像文本读取、图表推理、组合视觉推理、图示科学推理、证据充分性判断、有序视觉质量评估。

### 0.8B

<p align="center"><img src="../asset/demos/0.8b.gif" alt="0.8B: 原始 Qwen3.5 与 Vision-Jev 冻结 RLCD 样例对比" width="900"></p>

### 9B

<p align="center"><img src="../asset/demos/9b.gif" alt="9B: 原始 Qwen3.5 与 Vision-Jev 冻结 RLCD 样例对比" width="900"></p>
<!-- showcase:end -->

## 快速开始

```bash
git clone git@github.com:LMDHQ-0420/Vision-Jev.git
cd Vision-Jev
conda env create -f environment.yml
conda activate vision-jev

vision-jev doctor
python scripts/check_repo.py
python -m unittest discover -s tests -p 'test_*.py'
```

默认数据目录为 `/mnt/sda1/sol_data/vision-jev`：

```bash
vision-jev data-download
vision-jev data-inventory
vision-jev data-download-weblinx-subset --target-rows 7000
vision-jev data-download-gui-odyssey-subset --target-rows 8500
vision-jev data-normalize visual7w
vision-jev data-build-public \
  --mixture configs/data/sft_117k.json \
  --output /mnt/sda1/sol_data/vision-jev/manifests/public-117k.jsonl
```

每行 JSONL 代表一道完整问题，不会把 K 个候选拆成 K 条虚假样本。

## 仓库结构

```text
asset/              项目视觉资产
showcase/           可复现模型对比与动图生成代码
configs/            数据、模型、训练和评测模板
data/               Schema、公开样例和来源元数据
docs/               架构、数据、训练、评测和发布文档
runs/               不可变运行记录和外部大文件引用
scripts/            仓库与资源检查工具
vision_jev/         数据、模型、运行时、训练和评测代码
tests/              单元测试与集成测试
```

原始数据、checkpoint 和大型运行产物不会提交 Git。

## 文档与许可

建议从[文档首页](index.md)、[数据流程](data/pipeline.md)、[SFT 教程](training/sft.md)、[评测协议](evaluation/protocol.md)和[发布清单](release/checklist.md)开始。

Vision-Jev 源代码采用 [Apache License 2.0](../LICENSE)。数据资产和衍生发布可能受到上游许可的额外限制，重新分发或商用前请检查[来源账本](data/sources.md)。首个模型发布时会补充正式引用信息；在此之前，请引用仓库地址和准确 Git commit。
