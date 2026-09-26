# 2026-09-26-013：清洗 Pilot 视觉消融

- 状态：measured
- 代码 revision：`3cb6a47`
- checkpoint：`qwen35-08b-sft-pilot-clean-dual/checkpoint-last`
- 数据：`pilot-12k.jsonl` 的 661 条 group-safe holdout

## 假设与单一变量

保持问题、状态文字、历史、候选、区域坐标、标签、checkpoint、视觉 token 配置和 greedy 解码参数不变，仅将每张图替换为保持原始宽高比的中性灰图。灰图最大边限制为 1,024 像素，防止超长网页截图产生无意义的内存开销；区域坐标继续按原图尺寸归一化。

若原图准确率显著高于灰图，说明模型在当前 holdout 上实际使用了视觉证据。该实验只能测量视觉输入的边际贡献，不能单独证明完整视觉理解能力。

## 结果

| 范围 | 题数 | 原图 EM | 灰图 EM | 下降 |
|---|---:|---:|---:|---:|
| 总体 | 661 | 82.90% | 49.62% | 33.28 pp |
| Choice | 542 | 85.42% | 50.74% | 34.69 pp |
| Noul | 90 | 81.11% | 57.78% | 23.33 pp |
| Score | 29 | 41.38% | 3.45% | 37.93 pp |

两组 JSON 合法率均为 100%。灰图评测耗时 233.59 秒，平均生成延迟 0.346 秒，P95 0.506 秒，峰值 GPU 显存 2,135,130,624 bytes。

## 分来源视觉贡献

| 来源 | 题数 | 原图 EM | 灰图 EM | 下降 |
|---|---:|---:|---:|---:|
| AndroidControl | 86 | 80.23% | 25.58% | 54.65 pp |
| API rewrite | 20 | 90.00% | 40.00% | 50.00 pp |
| ChartQA | 37 | 91.89% | 45.95% | 45.95 pp |
| CLEVR | 25 | 88.00% | 44.00% | 44.00 pp |
| GQA | 106 | 90.57% | 66.98% | 23.58 pp |
| GUI-Odyssey | 51 | 84.31% | 41.18% | 43.14 pp |
| KonIQ-10k | 29 | 41.38% | 3.45% | 37.93 pp |
| MultiNLI | 12 | 58.33% | 58.33% | 0.00 pp |
| Multimodal-Mind2Web | 36 | 86.11% | 86.11% | 0.00 pp |
| NLVR | 15 | 60.00% | 53.33% | 6.67 pp |
| RefCOCO | 38 | 92.11% | 50.00% | 42.11 pp |
| RefCOCOg | 29 | 75.86% | 58.62% | 17.24 pp |
| RefCOCO+ | 40 | 87.50% | 47.50% | 40.00 pp |
| ScienceQA | 26 | 73.08% | 57.69% | 15.38 pp |
| TextVQA | 26 | 96.15% | 42.31% | 53.85 pp |
| Visual7W | 32 | 65.62% | 25.00% | 40.62 pp |
| VQAv2 | 37 | 91.89% | 70.27% | 21.62 pp |
| WebLINX | 16 | 100.00% | 100.00% | 0.00 pp |

MultiNLI 本身没有视觉输入，0 pp 符合预期。Mind2Web 和 WebLINX 在本次小 holdout 上完全不降，说明现有状态文字、历史或候选已经足以预测标签，不能把其高分当成视觉能力证据。GQA、VQAv2、ScienceQA、RefCOCOg 和 NLVR 的下降偏小，也提示语言先验、候选结构或样本量可能影响指标。AndroidControl、TextVQA、API rewrite、ChartQA、CLEVR、GUI-Odyssey、RefCOCO/RefCOCO+、Visual7W 和 KonIQ-10k 显示了明确视觉依赖。

## 产物

- 逐样本：`eval-image-ablation.jsonl`，SHA-256 `4200906222210bc05b49878895c5c43c6f428d2e66539a08b789c59fb7ec3260`
- 汇总：`eval-image-ablation.summary.json`，SHA-256 `50ae1df1da3ccbf43a5d564406129b7fcc4698a0413c580d065f8ee7d5b27e69`
- 路径：`/mnt/sda1/sol_data/vision-jev/runs/qwen35-08b-sft-pilot-clean-dual/`

## 结论

视觉消融门禁整体通过：原图带来 33.28 个百分点的准确率增益，模型总体确实使用视觉输入。但当前总体 82.90% 仍混合了真正视觉贡献、文本可解任务和语言/候选先验；后续发布评测必须同时报告原图结果、灰图结果和两者差值，不能只报告原图总体分数。
