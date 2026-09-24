# Implementation status

| Component | Status | Evidence |
|---|---|---|
| Repository/data/run contracts | structural_check | iterations 2026-09-24-001/002 |
| Hidden Choice/Noul/Score/value heads | implemented and CPU-tested in `vision-jev` env | repository suite 25/25 passed |
| Region evidence fusion | implemented, native grid mapping pending | code only |
| Qwen3.5 native processor/backbone adapter | planned M1 | none |
| Raw public datasets | 15 active sources registered; GUI-Odyssey 8,500-step subset and all other required source assets complete | data-root `_state/downloads`; source ledger |
| Canonical public data | 3,175,521 validated questions across the original 10 sources | processed JSONL validators |
| Public core manifest | complete: 90k = 76k Choice + 14k Noul; SHA-256 `4f1b0d54b9f9e9c020b43355c0c18be53d0e762d88d9ee73e8c5448c06d713f5` | build report and manifest |
| Public 117k manifest | complete: 94k Choice + 17k Noul + 6k Score; SHA-256 `5a3275395e9df1c78ec7a43fcaaabf7fbb540a8a80723eb1528916bcaeac851c` | build report and manifest |
| API block | complete: 3,251 checked candidates; final quota 2k Choice + 1k Noul; 0 immutable parent-field violations | iteration 2026-09-25-008 |
| Final SFT manifest | complete: 120k = 96k Choice + 18k Noul + 6k Score; SHA-256 `d0ddde7a861f0e9588e6dff37156c3593c7c390747aca84b2fbe254b931f13c8` | final build report and manifest |
| SFT loop | planned M1 | config/tutorial only |
| Shared-prefix runtime | planned M3 | cache identity only |
| PPO rollout/update | planned M5 | transition contract/config/tutorial only |
| GPU environment smoke test | passed on 2 × RTX 4090 with BF16 matmul | environment baseline |
| Model accuracy/latency/peak-memory results | not measured | no model run yet |
| Released weights/model card | not released | none |

This file prevents scaffolding from being mistaken for a trained model. Update it only when a linked run provides evidence.
