# Implementation status

| Component | Status | Evidence |
|---|---|---|
| Repository/data/run contracts | structural_check | iterations 2026-09-24-001/002 |
| Hidden Choice/Noul/Score/value heads | implemented and CPU-tested in `vision-jev` env | repository suite 27/27 passed |
| Region evidence fusion | implemented, native grid mapping pending | code only |
| Qwen3.5 native processor/backbone adapter | implemented and trained | SFT Preview run |
| Raw public datasets | 15 active sources registered; GUI-Odyssey 8,500-step subset and all other required source assets complete | data-root `_state/downloads`; source ledger |
| Canonical public data | 3,175,521 validated questions across the original 10 sources | processed JSONL validators |
| Public core manifest | complete: 90k = 76k Choice + 14k Noul; SHA-256 `4f1b0d54b9f9e9c020b43355c0c18be53d0e762d88d9ee73e8c5448c06d713f5` | build report and manifest |
| Public 117k manifest | complete: 94k Choice + 17k Noul + 6k Score; SHA-256 `bd52b63569977d0054b67103b3b181f1f63555d4edab04d249f3f8c696356e29` | build report and manifest |
| API block | complete: 3,274 checked candidates; final quota 2k Choice + 1k Noul; 0 immutable parent-field violations | rebuilt parent audit |
| Final SFT manifest | complete: 120k = 96k Choice + 18k Noul + 6k Score; SHA-256 `2e2d2e77dd4b4bc76765083671e7a7e11eb28ae862c7ed5d1e64a3dee760e0b9` | final build report and manifest |
| SFT loop | complete: 7,140 optimizer steps, 2 epochs, world size 2, global batch 32 | `qwen35-08b-sft-main/summary.json` |
| Shared-prefix runtime | planned M3 | cache identity only |
| RLCD-inspired calibration | TODO | no training code or frozen manifest yet |
| PPO rollout/update | TODO M5 | transition contract/config/tutorial only |
| GPU environment smoke test | passed on 2 × RTX 4090 with BF16 matmul | environment baseline |
| Model accuracy/latency/peak-memory results | smoke measured; full 5,771-question evaluation running | 30-question smoke: 83.33% exact match, 100% valid JSON, 0.401 s mean latency, 2.10 GB peak GPU memory |
| Released weights/model card | SFT Preview published in repository | `release/sft-preview/`, SFT Preview model card |

This file prevents scaffolding from being mistaken for a trained model. Update it only when a linked run provides evidence.
