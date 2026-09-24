# Implementation status

| Component | Status | Evidence |
|---|---|---|
| Repository/data/run contracts | structural_check | iterations 2026-09-24-001/002 |
| Hidden Choice/Noul/Score/value heads | implemented and CPU-tested in `vision-jev` env | repository suite 12/12 passed |
| Region evidence fusion | implemented, native grid mapping pending | code only |
| Qwen3.5 native processor/backbone adapter | planned M1 | none |
| Raw public datasets | all 10 active sources have pinned archives/indexes; WebLINX v2.2 selection has 7,000 rows and 6,719 screenshots | data-root `_state/downloads` and `_state/selections`; iteration 004 |
| Canonical v2.2 public data | 3,175,521 validated questions across all 10 sources | processed JSONL validators; iteration 004 |
| v2.2 public manifest | complete: 90k = 76k Choice + 14k Noul; SHA-256 `4f1b0d54b9f9e9c020b43355c0c18be53d0e762d88d9ee73e8c5448c06d713f5` | `public-90k-v2.2.build-report.json`; iteration 004 |
| v2.2 programmatic/augmentation blocks | 27k local generator implemented; 72-question r2 pilot measured, full 27k not run; 3k API block planned; 0 project human annotation | iteration 005; `configs/data/sft_120k.json`; ADR-0004 |
| SFT loop | planned M1 | config/tutorial only |
| Shared-prefix runtime | planned M3 | cache identity only |
| PPO rollout/update | planned M5 | transition contract/config/tutorial only |
| GPU environment smoke test | passed on 2 × RTX 4090 with BF16 matmul | environment baseline |
| Model accuracy/latency/peak-memory results | not measured | no model run yet |
| Released weights/model card | not released | none |

This file prevents scaffolding from being mistaken for a trained model. Update it only when a linked run provides evidence.
