<div align="center">
  <img src="asset/Vision-Jev.svg" alt="Vision-Jev" width="380">
  <h3>一个支持视觉的开源 JEV-like 模型</h3>
  <p>训练代码、数据配额、评测流程与模型权重，全流程开放。</p>
  <p><a href="README.md">English</a> · <b>简体中文</b> · <a href="https://huggingface.co/LMDHQ-0420/vision-jev">模型权重</a> · <a href="docs/index.md">项目文档</a> · <a href="LICENSE">Apache-2.0</a></p>
  <p><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white"> <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.7%2B-EE4C2C?logo=pytorch&logoColor=white"> <img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-2ea44f"> <img alt="Status" src="https://img.shields.io/badge/status-Models%20Released-2ea44f"></p>
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

## 模型权重

最终 0.8B 和 9B 模型已发布到 Hugging Face：[LMDHQ-0420/vision-jev](https://huggingface.co/LMDHQ-0420/vision-jev)。每个版本都包含最终 SFT LoRA 适配器、静态 Choice/Noul/Score 决策头、冻结校准温度，以及对应的 tokenizer 和 processor 配置。

| 发布版本 | 基础模型 | Hugging Face 文件 |
| --- | --- | --- |
| Vision-Jev-0.8B | `Qwen/Qwen3.5-0.8B` | [`vision-jev-0.8b/`](https://huggingface.co/LMDHQ-0420/vision-jev/tree/main/vision-jev-0.8b) |
| Vision-Jev-9B | `Qwen/Qwen3.5-9B` | [`vision-jev-9b/`](https://huggingface.co/LMDHQ-0420/vision-jev/tree/main/vision-jev-9b) |

发布内容采用 PEFT 适配器加项目决策头的形式，不重复上传合并后的 Qwen 权重，因此还需要对应的 Qwen3.5 基础模型。

安装 Hugging Face CLI，然后下载指定版本及完整测试报告：

```bash
python -m pip install -U huggingface_hub

# 0.8B
hf download LMDHQ-0420/vision-jev \
  vision-jev-0.8b/ results/static-rlcd-test.json \
  --revision b42c2f29eb6572f4a2ee31bd99d85d45ae8510c7 \
  --local-dir ./models/vision-jev

# 9B
hf download LMDHQ-0420/vision-jev \
  vision-jev-9b/ results/static-rlcd-test.json \
  --revision b42c2f29eb6572f4a2ee31bd99d85d45ae8510c7 \
  --local-dir ./models/vision-jev
```

一次下载两个模型、模型卡、许可证和测试结果：

```bash
hf download LMDHQ-0420/vision-jev \
  --revision b42c2f29eb6572f4a2ee31bd99d85d45ae8510c7 \
  --local-dir ./models/vision-jev
```

[Hugging Face 模型卡](https://huggingface.co/LMDHQ-0420/vision-jev)提供发布目录和 PEFT 加载方式说明。

## 冻结测试集展示

每段动画都让原始 Qwen3.5 checkpoint 与 Vision-Jev 使用同一个冻结测试样本和候选集。类别预先确定; 每个类别选择两套 Vision-Jev checkpoint 都回答正确、通过既定置信阈值且样本 ID 的 SHA-256 最小的样本, 选择过程不使用基线预测。每张 GIF 会依次播放全部固定样例; 两条进度条按照预热后 3 次推理耗时的中位数推进, 并在对应模型完成时显示答案。

覆盖能力: 通用视觉问答、自然图像文本读取、图表推理、组合视觉推理、图示科学推理、证据充分性判断、有序视觉质量评估。

### 0.8B

<p align="center"><img src="asset/demos/0.8b.gif" alt="0.8B: 原始 Qwen3.5 与 Vision-Jev 冻结 RLCD 样例对比" width="900"></p>

### 9B

<p align="center"><img src="asset/demos/9b.gif" alt="9B: 原始 Qwen3.5 与 Vision-Jev 冻结 RLCD 样例对比" width="900"></p>

## 静态 RLCD 结果

原始 Qwen、完成 SFT 但未接入静态 RLCD 决策头的 checkpoint, 以及完整 Vision-Jev 均在同一份冻结的 12,000 题测试集上评测。以下汇总保留全部成功与失败样本, 因此不同阶段的原始准确率回退也会直接展示。原始 Qwen 与 SFT-only 生成不提供校准决策头概率, 因此接受集校准只对 Vision-Jev 报告。

[下载完整机器可读测试报告](results/static-rlcd-test.json)，其中包含分任务、分数据集、校准、阈值、高置信错误和推理耗时结果。

### 0.8B 汇总

| 训练阶段 | Choice | Noul | Score |
| --- | ---: | ---: | ---: |
| Qwen3.5-0.8B (原始) | 24.5% | 39.5% | 2.9% |
| Qwen3.5-0.8B + SFT | **90.4%** | **81.4%** | 54.7% |
| **Vision-Jev-0.8B** | 80.8% | 81.2% | **55.5%** |

### 9B 汇总

| 训练阶段 | Choice | Noul | Score |
| --- | ---: | ---: | ---: |
| Qwen3.5-9B (原始) | 87.1% | 80.4% | 32.9% |
| Qwen3.5-9B + SFT | **95.4%** | **89.3%** | 60.5% |
| **Vision-Jev-9B** | 74.4% | 88.9% | **62.1%** |

### 接受集校准

#### Vision-Jev-0.8B

| 任务 | 阈值 | 覆盖率 | **接受准确率** |
| --- | ---: | ---: | ---: |
| Choice | 0.762 | 65.9% | **93.9%** |
| Noul | 0.797 | 53.2% | **94.7%** |
| Score | 0.886 | 9.2% | **97.3%** |

#### Vision-Jev-9B

| 任务 | 阈值 | 覆盖率 | **接受准确率** |
| --- | ---: | ---: | ---: |
| Choice | 0.665 | 60.7% | **94.8%** |
| Noul | 0.697 | 84.9% | **94.7%** |
| Score | 0.915 | 14.3% | **96.5%** |

#### 高置信错误

统计置信度不低于 0.9 但预测错误的完整测试样本。

| 模型 | 错误数 | 测试题数 | 占比 |
| --- | ---: | ---: | ---: |
| Vision-Jev-0.8B | 163 | 12,000 | 1.4% |
| Vision-Jev-9B | 86 | 12,000 | 0.7% |

### 推理耗时

原始 Qwen 与 SFT 的 JSONL 保留每一道题的生成耗时; Vision-Jev 的独立计时复测保留同步后的完整决策路径耗时。下表由全部逐题记录汇总, 两种口径分开标注。

#### 0.8B 总体耗时

| 指标 | 原始 Qwen | SFT | **Vision-Jev** |
| --- | ---: | ---: | ---: |
| 计时口径 | 逐题生成 | 逐题生成 | 完整决策循环 |
| 总计 (s) | 3718.1 | 3802.8 | **1051.6** |
| 均值 (ms) | 309.8 | 316.9 | **87.6** |
| P50 (ms) | 241.9 | 286.6 | **83.2** |
| P95 (ms) | 542.3 | 444.9 | **147.3** |
| P99 (ms) | 550.7 | 501.7 | **193.8** |
| 最小值 (ms) | 171.3 | 248.0 | **41.7** |
| 最大值 (ms) | 671.0 | 569.6 | **292.2** |

#### 9B 总体耗时

| 指标 | 原始 Qwen | SFT | **Vision-Jev** |
| --- | ---: | ---: | ---: |
| 计时口径 | 逐题生成 | 逐题生成 | 完整决策循环 |
| 总计 (s) | 4372.3 | 5259.5 | **1384.5** |
| 均值 (ms) | 364.4 | 438.3 | **115.4** |
| P50 (ms) | 326.9 | 392.0 | **105.6** |
| P95 (ms) | 507.7 | 609.1 | **197.3** |
| P99 (ms) | 530.2 | 691.4 | **273.0** |
| 最小值 (ms) | 263.7 | 344.2 | **55.1** |
| 最大值 (ms) | 804.1 | 754.0 | **318.5** |

### 分任务耗时

以下耗时均以毫秒为单位。均值使用粗体突出，同时保留完整分布。

#### 0.8B

| 任务 | 样本 | 指标 | 原始 Qwen | SFT | **Vision-Jev** |
| --- | ---: | --- | ---: | ---: | ---: |
| Choice | 8,400 | **均值** | 298.2 | 341.4 | **88.5** |
| Choice | 8,400 | P50 | 241.4 | 290.7 | 84.1 |
| Choice | 8,400 | P95 | 535.7 | 447.1 | 167.6 |
| Choice | 8,400 | P99 | 542.2 | 508.1 | 199.0 |
| Choice | 8,400 | 最小值 | 190.0 | 267.4 | 41.7 |
| Choice | 8,400 | 最大值 | 652.2 | 569.6 | 250.2 |
| Noul | 2,400 | **均值** | 270.5 | 262.2 | **72.0** |
| Noul | 2,400 | P50 | 218.9 | 258.9 | 73.7 |
| Noul | 2,400 | P95 | 505.7 | 290.8 | 78.9 |
| Noul | 2,400 | P99 | 528.8 | 317.8 | 83.0 |
| Noul | 2,400 | 最小值 | 210.0 | 252.4 | 52.1 |
| Noul | 2,400 | 最大值 | 636.3 | 565.8 | 292.2 |
| Score | 1,200 | **均值** | 469.8 | 255.2 | **112.9** |
| Score | 1,200 | P50 | 541.3 | 251.7 | 113.5 |
| Score | 1,200 | P95 | 548.5 | 301.8 | 116.8 |
| Score | 1,200 | P99 | 656.2 | 305.6 | 119.2 |
| Score | 1,200 | 最小值 | 171.3 | 248.0 | 90.7 |
| Score | 1,200 | 最大值 | 671.0 | 308.8 | 138.9 |

#### 9B

| 任务 | 样本 | 指标 | 原始 Qwen | SFT | **Vision-Jev** |
| --- | ---: | --- | ---: | ---: | ---: |
| Choice | 8,400 | **均值** | 390.9 | 471.3 | **116.0** |
| Choice | 8,400 | P50 | 330.3 | 396.5 | 110.7 |
| Choice | 8,400 | P95 | 510.5 | 612.6 | 227.3 |
| Choice | 8,400 | P99 | 563.5 | 696.3 | 276.7 |
| Choice | 8,400 | 最小值 | 263.7 | 362.9 | 55.1 |
| Choice | 8,400 | 最大值 | 804.1 | 754.0 | 318.5 |
| Noul | 2,400 | **均值** | 295.9 | 357.9 | **94.1** |
| Noul | 2,400 | P50 | 293.3 | 351.9 | 96.0 |
| Noul | 2,400 | P95 | 302.7 | 424.5 | 111.8 |
| Noul | 2,400 | P99 | 357.7 | 431.1 | 122.6 |
| Noul | 2,400 | 最小值 | 285.9 | 344.2 | 72.7 |
| Noul | 2,400 | 最大值 | 365.7 | 624.3 | 133.2 |
| Score | 1,200 | **均值** | 315.7 | 368.3 | **153.9** |
| Score | 1,200 | P50 | 314.0 | 363.5 | 158.0 |
| Score | 1,200 | P95 | 316.9 | 420.8 | 180.6 |
| Score | 1,200 | P99 | 359.1 | 424.9 | 185.9 |
| Score | 1,200 | 最小值 | 311.5 | 360.3 | 133.2 |
| Score | 1,200 | 最大值 | 363.0 | 432.0 | 192.4 |

### 校准详细指标

#### 结构化输出有效率

| 模型 | Choice | Noul | Score |
| --- | ---: | ---: | ---: |
| Qwen3.5-0.8B | 37.4% | 61.0% | 22.2% |
| Qwen3.5-0.8B + SFT | 100.0% | 100.0% | 100.0% |
| Vision-Jev-0.8B | 100.0% | 100.0% | 100.0% |
| Qwen3.5-9B | 99.5% | 100.0% | 100.0% |
| Qwen3.5-9B + SFT | 100.0% | 100.0% | 100.0% |
| Vision-Jev-9B | 100.0% | 100.0% | 100.0% |

Vision-Jev 通过固定决策头直接返回候选, 因而结构化输出始终有效。

#### Vision-Jev-0.8B 概率指标

| 指标 | Choice | Noul | Score |
| --- | ---: | ---: | ---: |
| 题数 | 8,400 | 2,400 | 1,200 |
| **准确率** | **80.8%** | **81.2%** | **55.5%** |
| NLL | 0.530 | 0.396 | 1.003 |
| Brier | 0.269 | 0.253 | 0.552 |
| ECE | 0.026 | 0.031 | 0.031 |
| RPS | N/A | N/A | 0.090 |
| MAE | N/A | N/A | 0.514 |

#### Vision-Jev-9B 概率指标

| 指标 | Choice | Noul | Score |
| --- | ---: | ---: | ---: |
| 题数 | 8,400 | 2,400 | 1,200 |
| **准确率** | **74.4%** | **88.9%** | **62.1%** |
| NLL | 0.610 | 0.231 | 0.834 |
| Brier | 0.309 | 0.145 | 0.485 |
| ECE | 0.016 | 0.023 | 0.041 |
| RPS | N/A | N/A | 0.071 |
| MAE | N/A | N/A | 0.405 |

### 0.8B 分数据集结果

每行最佳准确率和最低平均耗时使用粗体标出。

#### 准确率 (%)

| 数据集 | 任务 | 题数 | 原始 Qwen | SFT | **Vision-Jev** |
| --- | --- | ---: | ---: | ---: | ---: |
| Android Control | Choice | 700 | 1.0 | **84.1** | 52.3 |
| ChartQA | Choice | 700 | 53.6 | **97.0** | 79.4 |
| CLEVR | Choice | 700 | 17.9 | **93.6** | 87.9 |
| GQA | Choice | 700 | 29.1 | **96.7** | 95.9 |
| MultiNLI | Choice | 700 | 44.3 | **76.3** | 72.6 |
| RefCOCO | Choice | 700 | 0.0 | **95.0** | 83.1 |
| RefCOCOg | Choice | 700 | 0.0 | **90.4** | 80.1 |
| RefCOCO+ | Choice | 700 | 0.0 | **88.4** | 78.6 |
| ScienceQA | Choice | 700 | 47.1 | **85.3** | 83.1 |
| TextVQA | Choice | 700 | 54.0 | **98.0** | 92.0 |
| Visual7W | Choice | 700 | 20.6 | **80.7** | 67.7 |
| VQAv2 | Choice | 700 | 25.9 | **98.9** | 96.4 |
| ChartQA | Noul | 82 | 45.1 | **61.0** | 56.1 |
| CLEVR | Noul | 580 | 59.3 | **92.6** | 91.6 |
| GQA | Noul | 580 | 29.8 | 83.1 | **83.8** |
| NLVR | Noul | 579 | 23.0 | 64.1 | **64.4** |
| VQAv2 | Noul | 579 | 44.9 | 88.6 | **88.8** |
| KonIQ-10k | Score | 1,200 | 2.9 | 54.7 | **55.5** |

#### 平均耗时 (ms)

| 数据集 | 任务 | 题数 | 原始 Qwen | SFT | **Vision-Jev** |
| --- | --- | ---: | ---: | ---: | ---: |
| Android Control | Choice | 700 | 281.9 | 357.2 | **173.0** |
| ChartQA | Choice | 700 | 234.0 | 291.0 | **83.7** |
| CLEVR | Choice | 700 | 277.9 | 288.4 | **75.8** |
| GQA | Choice | 700 | 278.4 | 288.4 | **76.4** |
| MultiNLI | Choice | 700 | 226.9 | 276.7 | **52.5** |
| RefCOCO | Choice | 700 | 413.4 | 434.3 | **91.3** |
| RefCOCOg | Choice | 700 | 401.8 | 431.1 | **90.0** |
| RefCOCO+ | Choice | 700 | 405.0 | 432.4 | **91.6** |
| ScienceQA | Choice | 700 | 233.9 | 289.9 | **76.3** |
| TextVQA | Choice | 700 | 228.5 | 291.7 | **85.1** |
| Visual7W | Choice | 700 | 340.1 | 423.4 | **85.3** |
| VQAv2 | Choice | 700 | 257.0 | 291.7 | **81.0** |
| ChartQA | Noul | 82 | 300.0 | 265.3 | **75.5** |
| CLEVR | Noul | 580 | 229.4 | 261.0 | **73.2** |
| GQA | Noul | 580 | 286.2 | 262.9 | **74.2** |
| NLVR | Noul | 579 | 288.2 | 259.3 | **64.4** |
| VQAv2 | Noul | 579 | 274.1 | 265.1 | **75.8** |
| KonIQ-10k | Score | 1,200 | 469.8 | 255.2 | **112.9** |

### 9B 分数据集结果

每行最佳准确率和最低平均耗时使用粗体标出。

#### 准确率 (%)

| 数据集 | 任务 | 题数 | 原始 Qwen | SFT | **Vision-Jev** |
| --- | --- | ---: | ---: | ---: | ---: |
| Android Control | Choice | 700 | 41.0 | **89.6** | 34.7 |
| ChartQA | Choice | 700 | 94.7 | **98.0** | 81.6 |
| CLEVR | Choice | 700 | 90.0 | **98.9** | 96.3 |
| GQA | Choice | 700 | 96.4 | **98.0** | 97.1 |
| MultiNLI | Choice | 700 | 73.0 | 86.6 | **87.1** |
| RefCOCO | Choice | 700 | 96.7 | **97.6** | 60.7 |
| RefCOCOg | Choice | 700 | 93.1 | **94.9** | 59.7 |
| RefCOCO+ | Choice | 700 | 90.7 | **93.4** | 56.0 |
| ScienceQA | Choice | 700 | 94.6 | **97.0** | 95.6 |
| TextVQA | Choice | 700 | 98.4 | **99.6** | 94.7 |
| Visual7W | Choice | 700 | 80.4 | **91.4** | 30.3 |
| VQAv2 | Choice | 700 | 96.6 | **99.7** | 98.6 |
| ChartQA | Noul | 82 | **80.5** | **80.5** | **80.5** |
| CLEVR | Noul | 580 | 88.3 | **99.1** | 99.0 |
| GQA | Noul | 580 | 80.2 | **89.7** | 89.5 |
| NLVR | Noul | 579 | 64.1 | **74.6** | 73.9 |
| VQAv2 | Noul | 579 | 89.1 | **95.0** | 94.5 |
| KonIQ-10k | Score | 1,200 | 32.9 | 60.5 | **62.1** |

#### 平均耗时 (ms)

| 数据集 | 任务 | 题数 | 原始 Qwen | SFT | **Vision-Jev** |
| --- | --- | ---: | ---: | ---: | ---: |
| Android Control | Choice | 700 | 468.7 | 544.1 | **237.5** |
| ChartQA | Choice | 700 | 330.4 | 399.3 | **108.4** |
| CLEVR | Choice | 700 | 321.8 | 392.8 | **96.9** |
| GQA | Choice | 700 | 326.1 | 393.1 | **98.7** |
| MultiNLI | Choice | 700 | 306.3 | 374.5 | **67.0** |
| RefCOCO | Choice | 700 | 492.4 | 595.9 | **119.5** |
| RefCOCOg | Choice | 700 | 487.7 | 588.8 | **117.6** |
| RefCOCO+ | Choice | 700 | 492.0 | 595.6 | **119.8** |
| ScienceQA | Choice | 700 | 326.7 | 394.6 | **99.5** |
| TextVQA | Choice | 700 | 331.9 | 400.1 | **109.9** |
| Visual7W | Choice | 700 | 476.5 | 577.0 | **111.0** |
| VQAv2 | Choice | 700 | 329.8 | 399.7 | **105.8** |
| ChartQA | Noul | 82 | 299.6 | 364.5 | **96.1** |
| CLEVR | Noul | 580 | 294.0 | 356.5 | **94.8** |
| GQA | Noul | 580 | 297.4 | 358.8 | **96.9** |
| NLVR | Noul | 579 | 291.7 | 354.1 | **85.6** |
| VQAv2 | Noul | 579 | 300.1 | 361.3 | **98.6** |
| KonIQ-10k | Score | 1,200 | 315.7 | 368.3 | **153.9** |

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

默认数据目录为 `/data/vision-jev`：

```bash
vision-jev data-download
vision-jev data-inventory
vision-jev data-download-weblinx-subset --target-rows 7000
vision-jev data-download-gui-odyssey-subset --target-rows 8500
vision-jev data-normalize visual7w
vision-jev data-build-public \
  --mixture configs/data/sft_117k.json \
  --output /data/vision-jev/manifests/public-117k.jsonl
```

每行 JSONL 代表一道完整问题，不会把 K 个候选拆成 K 条虚假样本。

## 仓库结构

```text
asset/              项目视觉资产
showcase/           可复现模型对比与动图生成代码
configs/            数据、模型、训练和评测模板
data/               Schema 与公开样例
docs/               架构、数据、训练、评测和发布文档
results/            已发布的机器可读评测结果
scripts/            仓库与资源检查工具
vision_jev/         数据、模型、运行时、训练和评测代码
tests/              单元测试与集成测试
```

原始数据、checkpoint 和大型运行产物不会提交 Git。

## 训练数据

SFT 数据包含 **117,000 道公开来源完整问题**，本项目新增人工标注为 0。

| 数据块 | Choice | Noul | Score | 合计 |
|---|---:|---:|---:|---:|
| 公开数据集 | 94,000 | 17,000 | 6,000 | 117,000 |
| 本地合成 | 0 | 0 | 0 | 0 |
| **合计** | **94,000** | **17,000** | **6,000** | **117,000** |

数据覆盖 GUI 操作、区域定位、组合推理、VQA、OCR、图表理解、文本逻辑和视觉质量。全部样本来自固定版本的公开来源，并保留原始语言。

- [数据来源账本](docs/data/sources.md)
- [数据混合方案](docs/data/mixtures.md)
- [机器可读来源](configs/data/sources.json)
- [训练配方](configs/data/sft_117k.json)

## 文档与许可

建议从[文档首页](docs/index.md)、[数据流程](docs/data/pipeline.md)、[SFT 教程](docs/training/sft.md)、[评测协议](docs/evaluation/protocol.md)和[发布清单](docs/release/checklist.md)开始。

Vision-Jev 源代码采用 [Apache License 2.0](LICENSE)。数据资产和衍生发布可能受到上游许可的额外限制，重新分发或商用前请检查[来源账本](docs/data/sources.md)。引用时请同时注明 [Hugging Face 模型发布页](https://huggingface.co/LMDHQ-0420/vision-jev)和所用的准确 Git commit。
