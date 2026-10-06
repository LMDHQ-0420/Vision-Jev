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

## 当前进度

公开数据 117k 的 Qwen3.5-0.8B 与 Qwen3.5-9B SFT、静态 RLCD-inspired 训练均已完成。
Qwen 官方没有发布 Qwen3.5-7B checkpoint。

<!-- showcase:start -->
## 静态 RLCD 结果

已发布 checkpoint 在完整冻结的 12,000 题静态 RLCD 测试集上评测。以下汇总包含全部成功与失败样本。

| 模型 | Choice 准确率 | Noul 准确率 | Score 准确率 | Choice 阈值结果 | Noul 阈值结果 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vision-Jev-0.8B | 80.8% | 81.2% | 55.5% | 93.9%, 覆盖率 65.9% | 94.7%, 覆盖率 53.2% |
| Vision-Jev-9B | 74.4% | 88.9% | 62.1% | 94.8%, 覆盖率 60.7% | 94.7%, 覆盖率 84.9% |

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
