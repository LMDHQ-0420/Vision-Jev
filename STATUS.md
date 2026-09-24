# Implementation status

| Component | Status | Evidence |
|---|---|---|
| Repository/data/run contracts | structural_check | iterations 2026-09-24-001/002 |
| Hidden Choice/Noul/Score/value heads | implemented and CPU-tested in `vision-jev` env | repository suite 17/17 passed |
| Region evidence fusion | implemented, native grid mapping pending | code only |
| Qwen3.5 native processor/backbone adapter | planned M1 | none |
| Raw public datasets | 15 active sources registered; Visual7W, ScienceQA, NLVR and KonIQ-10k complete; GUI-Odyssey annotations/split and 64-screenshot pilot complete | data-root `_state/downloads`; source ledger |
| Canonical public data | 3,175,521 validated questions across the original 10 sources | processed JSONL validators |
| Public core manifest | complete: 90k = 76k Choice + 14k Noul; SHA-256 `4f1b0d54b9f9e9c020b43355c0c18be53d0e762d88d9ee73e8c5448c06d713f5` | build report and manifest |
| Public extension | local generation disabled; five-source 1,964-question pilot passed with Choice 1,064/Noul 300/Score 600; full GUI-Odyssey 8k screenshot materialization pending | `configs/data/sft_120k.json`; ADR-0005 |
| API block | 3k same-language rewrites planned; no translation and 0 project human annotation | `configs/data/sft_120k.json`; ADR-0005 |
| SFT loop | planned M1 | config/tutorial only |
| Shared-prefix runtime | planned M3 | cache identity only |
| PPO rollout/update | planned M5 | transition contract/config/tutorial only |
| GPU environment smoke test | passed on 2 × RTX 4090 with BF16 matmul | environment baseline |
| Model accuracy/latency/peak-memory results | not measured | no model run yet |
| Released weights/model card | not released | none |

This file prevents scaffolding from being mistaken for a trained model. Update it only when a linked run provides evidence.
