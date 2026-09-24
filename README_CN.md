<div align="center">
  <img src="asset/Vision-Jev.svg" alt="Vision-Jev" width="760">
  <h3>面向动态候选判断与行动的开源视觉语言模型</h3>
  <p>模型架构、训练代码、数据配额、评测流程与模型权重，全流程开放。</p>
  <p><a href="README.md">English</a> · <a href="README_CN.md">简体中文</a> · <a href="docs/index.md">项目文档</a> · <a href="LICENSE">Apache-2.0</a></p>
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

模型由原生多模态基座、轻量判断头、候选集合交互和区域视觉证据组成。首个目标基座为 `Qwen/Qwen3.5-0.8B`，判断能力与策略优化相互分离。

### 2. 开源完整训练链路

项目将开放数据处理、精确配额、固定来源、确定性 manifest、SFT 与策略训练代码、评测协议、实验结果、模型权重、模型卡和数据卡。第三方原始数据继续遵守各自许可；本项目公开可复现的处理路径，不把上游资产声明为自有数据。

## 模型结构

```text
图片 ─ 原生 processor ─ 视觉编码器 ─┐
状态 + 问题 ─ tokenizer ─────────────┼─ 多模态主干
动态候选 ─ serializer ──────────────┘       │
                                      ┌──────┼──────┐
                                   Choice  Noul  Score
                                      │
                               policy + value 分支
```

详见[系统架构](docs/architecture/system.md)和[接口契约](docs/architecture/interfaces.md)。

## 开放数据配方

训练目标为 **120,000 道完整问题**，本项目新增人工标注为 0。

| 数据块 | Choice | Noul | Score | 合计 |
|---|---:|---:|---:|---:|
| 公开数据集 | 94,000 | 17,000 | 6,000 | 117,000 |
| API 同语言改写 | 2,000 | 1,000 | 0 | 3,000 |
| 本地合成 | 0 | 0 | 0 | 0 |
| **合计** | **96,000** | **18,000** | **6,000** | **120,000** |

数据覆盖 GUI 操作、区域定位、组合推理、VQA、OCR、图表理解、文本逻辑和视觉质量。公开数据保留原始语言；API 样本继承父样本语言和标签语义。

- [数据来源账本](docs/data/sources.md)
- [数据混合方案](docs/data/mixtures.md)
- [机器可读来源](configs/data/sources.json)
- [训练配方](configs/data/sft_120k.json)

## 当前进度

仓库已经具备数据管线、统一 schema、确定性 manifest 构建器、判断头原型、评测指标、实验追踪和经过验证的双 GPU 环境。原十来源已经转换 3,175,521 道 canonical 问题并生成 90k 核心清单；新五来源联合 pilot 已通过。

正式训练结果和模型权重**尚未发布**。计划指标不会冒充实测结果，证据状态见 [STATUS.md](STATUS.md)。

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
  --mixture configs/data/sft_120k.json \
  --output /mnt/sda1/sol_data/vision-jev/manifests/public-117k.jsonl
```

每行 JSONL 代表一道完整问题，不会把 K 个候选拆成 K 条虚假样本。

## 仓库结构

```text
asset/              项目视觉资产
configs/            数据、模型、训练、评测和 API 模板
data/               Schema、公开样例和来源元数据
docs/               架构、数据、训练、评测和发布文档
runs/               不可变运行记录和外部大文件引用
scripts/            仓库与资源检查工具
vision_jev/         数据、模型、运行时、训练和评测代码
tests/              单元测试与集成测试
```

原始数据、checkpoint、密钥和大型运行产物不会提交 Git。API 配置放在 `configs/local/api.toml`，模板见 [configs/api.example.toml](configs/api.example.toml)。

## TODO / Roadmap

- [x] 建立仓库、schema、文档和运行记录契约
- [x] 完成固定来源下载与 canonical 转换管线
- [x] 生成并验证 90k 公开核心清单
- [x] 验证新五来源联合 pilot
- [ ] 完成 GUI-Odyssey 正式训练子集截图物化
- [ ] 冻结并验证 117k 公开数据 manifest
- [ ] 生成 3k 程序验证的同语言 API 改写数据
- [ ] 接入 Qwen3.5-0.8B 原生 processor 和主干
- [ ] 完成小批量过拟合与 12k SFT pilot
- [ ] 训练并评测 120k SFT 模型
- [ ] 实现共享前缀推理与概率校准
- [ ] 在闭环环境中训练 policy/value 分支
- [ ] 发布模型权重、模型卡、数据卡和复现报告

只有存在可核验的运行证据后，TODO 才会标记完成。

## 文档与许可

建议从[文档首页](docs/index.md)、[数据流程](docs/data/pipeline.md)、[SFT 教程](docs/training/sft.md)、[评测协议](docs/evaluation/protocol.md)和[发布清单](docs/release/checklist.md)开始。

Vision-Jev 源代码采用 [Apache License 2.0](LICENSE)。数据资产和衍生发布可能受到上游许可的额外限制，重新分发或商用前请检查[来源账本](docs/data/sources.md)。首个模型发布时会补充正式引用信息；在此之前，请引用仓库地址和准确 Git commit。
