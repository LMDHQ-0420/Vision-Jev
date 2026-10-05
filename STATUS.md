# Implementation status

| Component | Status | Evidence |
|---|---|---|
| Repository/data/run contracts | structural_check | iterations 2026-09-24-001/002 |
| Hidden Choice/Noul/Score/value heads | implemented; static heads passed single- and dual-GPU end-to-end smoke | repository suite 41/41 passed; RLCD smoke summaries |
| Region evidence fusion | implemented, native grid mapping pending | code only |
| Qwen3.5 native processor/backbone adapter | implemented and trained on public-only 117k | `qwen35-08b-sft-main-117k/summary.json` |
| Raw public datasets | 15 active sources registered; GUI-Odyssey 8,500-step subset and all other required source assets complete | data-root `_state/downloads`; source ledger |
| Canonical public data | 3,175,521 validated questions across the original 10 sources | processed JSONL validators |
| Public core manifest | complete: 90k = 76k Choice + 14k Noul; SHA-256 `4f1b0d54b9f9e9c020b43355c0c18be53d0e762d88d9ee73e8c5448c06d713f5` | build report and manifest |
| Public 117k manifest | complete: 94k Choice + 17k Noul + 6k Score; SHA-256 `bd52b63569977d0054b67103b3b181f1f63555d4edab04d249f3f8c696356e29` | build report and manifest |
| Final SFT manifest | complete: 117k = 94k Choice + 17k Noul + 6k Score; SHA-256 `be04f9178f3c7672b4dac9a42e42da25bc33e04fbcb4690efd0a8c3f31ac9ba4` | public build report and manifest |
| SFT loop | complete: 6,946 optimizer steps, 2 epochs, world size 2, global batch 32 | `qwen35-08b-sft-main-117k/summary.json` |
| Shared-prefix runtime | planned M3 | cache identity only |
| RLCD-inspired calibration | implemented; 72k roots/156k train views frozen; 0.8B dual-GPU run complete | `configs/train/rlcd_main.json`; `qwen35-08b-rlcd-main-72k/summary.json` |
| Qwen3.5-9B reproduction | model revision frozen; download and sequential 117k SFT/RLCD pipeline in progress | `configs/model/qwen35_9b.json`; `scripts/run_model_pipeline.sh` |
| PPO rollout/update | TODO M5 | transition contract/config/tutorial only |
| GPU environment smoke test | passed on 2 × RTX 5090 with BF16 matmul | SFT and RLCD smoke logs |
| Model accuracy/latency/peak-memory results | measured on all 5,886 holdout questions | 85.54% exact match, 100% valid JSON, 0.348 s mean / 0.643 s p95, 2.06 GiB peak GPU memory |
| Released weights/model card | final SFT adapter trained; release packaging pending | `qwen35-08b-sft-main-117k/checkpoint-last` |

This file prevents scaffolding from being mistaken for a trained model. Update it only when a linked run provides evidence.
