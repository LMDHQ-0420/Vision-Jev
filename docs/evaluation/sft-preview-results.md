# SFT Preview完整测评结果

## 范围

- Checkpoint：`qwen35-08b-sft-main/checkpoint-last`
- 数据：5,771 道 group-safe holdout 完整问题
- 硬件：单张 RTX 4090
- 解码：greedy，最多 24 个新 token
- 逐条结果 SHA-256：`413a53e365140992de53db095408ad23d8c53a8ab4749f618cee6c691815285e`
- 汇总 SHA-256：`33445c24d948a8c7a4df4d02f54304d1e22db7f37299412e843e2e8a57b01434`

## 总体与任务

| 指标 | 结果 |
|---|---:|
| 总体 exact match | 85.55% |
| JSON 合法率 | 100.00% |
| Choice exact match | 88.44%（4,601题） |
| Noul exact match | 78.72%（893题） |
| Score exact match | 59.57%（277题） |
| Score MAE | 0.477级 |
| Score 相邻一级内准确率 | 94.58% |

## 分来源

| 来源 | 题数 | Exact match |
|---|---:|---:|
| WebLINX | 271 | 100.00% |
| ChartQA | 275 | 96.36% |
| TextVQA | 328 | 96.04% |
| RefCOCO | 323 | 92.26% |
| CLEVR | 169 | 91.12% |
| VQAv2 | 399 | 89.97% |
| GQA | 899 | 89.43% |
| GUI-Odyssey | 337 | 88.43% |
| RefCOCOg | 207 | 87.92% |
| AndroidControl | 803 | 84.43% |
| API rewrite | 148 | 83.11% |
| RefCOCO+ | 350 | 82.57% |
| ScienceQA | 198 | 82.32% |
| Visual7W | 297 | 79.80% |
| MultiNLI | 73 | 72.60% |
| Mind2Web | 277 | 72.56% |
| KonIQ-10k | 277 | 59.57% |
| NLVR | 140 | 58.57% |

## 性能

| 指标 | 结果 |
|---|---:|
| 平均延迟 | 0.391秒/题 |
| p50 | 0.372秒 |
| p95 | 0.667秒 |
| 峰值分配显存 | 2.14 GB |
| 完整测评耗时 | 2,349.9秒 |

这些数字来自当前逐题生成式实现，不等同于后续原生决策头、批处理或共享视觉前缀的优化性能。WebLINX 100%需要通过候选位置、文字捷径和重复样本审计后再解释；NLVR、KonIQ-10k、Mind2Web和MultiNLI是后续错误分析重点。
