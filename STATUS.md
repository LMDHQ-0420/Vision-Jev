# Implementation status

| Component | Status | Evidence |
|---|---|---|
| Repository/data/run contracts | structural_check | iterations 2026-09-24-001/002 |
| Hidden Choice/Noul/Score/value heads | implemented and CPU-tested in `vision-jev` env | 6/6 test suite passed |
| Region evidence fusion | implemented, native grid mapping pending | code only |
| Qwen3.5 native processor/backbone adapter | planned M1 | none |
| Raw public datasets | 10 active sources have pinned archives/indexes; WebLINX selected screenshots remain in progress | data-root `_state/downloads`; iteration 002 |
| Canonical v2 public data | 3,168,658 validated questions across 9 sources; WebLINX pending | processed JSONL validators; iteration 002 |
| v2 12k/120k manifests | planned; not built | `configs/data/sft_120k.json` |
| SFT loop | planned M1 | config/tutorial only |
| Shared-prefix runtime | planned M3 | cache identity only |
| PPO rollout/update | planned M5 | transition contract/config/tutorial only |
| GPU environment smoke test | passed on 2 × RTX 4090 with BF16 matmul | environment baseline |
| Model accuracy/latency/peak-memory results | not measured | no model run yet |
| Released weights/model card | not released | none |

This file prevents scaffolding from being mistaken for a trained model. Update it only when a linked run provides evidence.
