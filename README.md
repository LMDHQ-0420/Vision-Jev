# Vision-JEV

Vision-JEV 是一个面向动态候选决策的轻量视觉语言模型研究与教程仓库。项目的两条同等重要的主线是：

1. 开源一个真正读取图像、状态、问题与动态候选的 Jev 风格模型；
2. 给出从数据登记、监督训练、闭环 PPO、评测、校准到模型发布的可复现教程。

当前状态：**M0 数据获取与管线实现中**。仓库已建立数据契约、实验账本、配置、决策头原型和文档门禁；v2 公开数据正在固定版本下载，尚未构造 120k 训练 manifest，也未声称模型指标。所有规划指标与实测结果严格分开。

## 核心设计

- 基座：固定 revision 的 `Qwen/Qwen3.5-0.8B` 原生 VLM。
- 判断：隐藏表示决策头，Choice 支持独立证据与可选集合交互；Noul 与 Score 使用独立语义。
- 视觉：单图起步，区域分支复用原生视觉网格，不重复运行视觉编码器。
- 策略：判断概率和动作策略分离；R0 只训练独立 actor/value，真实环境回报用于 PPO。
- 证据：计划、估算、CPU 结构检查、GPU 实测与最终结论使用不同状态标签。

## 快速开始（无需下载模型）

```bash
PYTHONPATH=src python -m vision_jev.cli doctor
PYTHONPATH=src python -m vision_jev.cli validate-data data/samples/example.jsonl
PYTHONPATH=src python -m vision_jev.cli init-run --config configs/train/sft_smoke.json --data data/samples/example.jsonl
python scripts/check_repo.py
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

源码位于 `src/`，未安装时可用 `PYTHONPATH=src` 运行；正式开发建议：

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

训练依赖单独安装：`pip install -e '.[train]'`。具体流程从 [教程入口](docs/index.md) 开始。

数据依赖与下载命令：`pip install -e '.[data]'`，然后运行 `vision-jev data-download`。WebLINX 使用 `vision-jev data-download-weblinx-subset` 做训练集 PII 预筛和按需截图下载，不应拉取全量小文件快照。默认数据根目录是 `/mnt/sda1/sol_data/vision-jev`；可用 `vision-jev data-inventory` 查看只有完成记录才计入的实际状态。

固定环境名为 `vision-jev`：`conda env create -f environment.yml`，然后 `conda activate vision-jev`。若环境已存在，用 `conda env update -n vision-jev -f environment.yml --prune`，并在 run 中记录变更后的完整包版本。当前实际安装版本与硬件边界见 `docs/development/environment-baseline.md`。

## 里程碑

| 阶段 | 交付 | 当前状态 |
|---|---|---|
| M0 | 来源、版本、许可、环境与仓库规范 | 进行中 |
| M1 | 12k manifest、三类输出、SFT smoke | 未开始 |
| M2 | 隐藏/LM/集合/区域消融 | 未开始 |
| M3 | 共享前缀后端与 GPU profile | 未开始 |
| M4 | 120k × 2 SFT、校准 | 未开始 |
| M5 | R0 PPO 与闭环评测 | 未开始 |
| M6 | 模型卡、数据卡、权重与教程发布 | 未开始 |

## 研究边界

本仓库不会把规划吞吐写成实测，不会把 GT box 条件准确率写成端到端定位能力，不会把 policy probability 称为 confidence，也不会将候选展开行数当作独立问题数。训练数据本体默认不入库；只提交许可、来源、hash、转换版本与 manifest。

## 开源

代码按 Apache-2.0 发布。第三方模型、数据集和图片遵循各自许可；本仓库许可不覆盖它们。参见 [数据来源账本](docs/data/sources.md) 与 [安全策略](SECURITY.md)。
