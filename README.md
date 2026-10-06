<div align="center">
  <img src="asset/Vision-Jev.svg" alt="Vision-Jev" width="380">
  <h3>An open vision-enabled JEV-like model</h3>
  <p>Training code, data recipe, evaluation protocol, and weights — developed in the open.</p>
  <p><b>English</b> · <a href="docs/overview-zh.md">简体中文</a> · <a href="docs/index.md">Documentation</a> · <a href="LICENSE">Apache-2.0</a></p>
  <p><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white"> <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.7%2B-EE4C2C?logo=pytorch&logoColor=white"> <img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-2ea44f"> <img alt="Status" src="https://img.shields.io/badge/status-SFT%20Preview-2ea44f"></p>
</div>

---

## Overview

Vision-Jev is an open foundation project for **judging and acting over a dynamic candidate set**. Given an image, visible state, question, and changing candidates, the model produces three structured outputs:

- **Choice** — select or rank the best candidate;
- **Noul** — decide whether the evidence is sufficient or a proposition is supported;
- **Score** — estimate an ordered quality or confidence level.

The same visual representation is designed to support a separate policy/value branch for closed-loop interaction. Vision-Jev focuses on auditable, calibratable decisions rather than unconstrained chat.

## Two commitments

### Train an open Vision-Jev

Vision-Jev brings visual understanding to the JEV-like modeling paradigm and is developed as an open model from data to weights.

### Open the complete training path

The project will publish the data pipeline, exact quotas, source revisions, deterministic manifests, SFT and policy-training code, evaluation protocol, experiment evidence, model weights, and release cards. Third-party datasets retain their original licenses; this repository publishes the source ledger and reproducible processing path rather than claiming ownership of upstream assets.

## Open data recipe

The SFT mixture contains **117,000 complete public-data questions** and introduces no project-specific human annotation.

| Block | Choice | Noul | Score | Total |
|---|---:|---:|---:|---:|
| Public datasets | 94,000 | 17,000 | 6,000 | 117,000 |
| Local synthetic generation | 0 | 0 | 0 | 0 |
| **Total** | **94,000** | **17,000** | **6,000** | **117,000** |

The mixture covers GUI actions, region grounding, compositional reasoning, VQA, OCR, charts, language inference, and visual quality. Every sample comes from a pinned public source and preserves its source language.

- [Source ledger](docs/data/sources.md)
- [Mixture specification](docs/data/mixtures.md)
- [Machine-readable sources](configs/data/sources.json)
- [Training recipe](configs/data/sft_117k.json)

<!-- showcase:start -->
## Static RLCD results

Original Qwen, the completed SFT-only checkpoint, and Vision-Jev with its static RLCD decision heads are evaluated on the same complete frozen 12,000-question test split. These aggregate results include every success and failure, so raw-accuracy regressions between stages remain visible. Original Qwen and SFT-only generation do not provide calibrated decision-head probabilities, so their threshold columns are N/A.

| Model | Choice accuracy | Noul accuracy | Score accuracy | Choice accepted accuracy | Noul accepted accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B (original) | 24.5% | 39.5% | 2.9% | N/A | N/A |
| Qwen3.5-0.8B + SFT | 90.4% | 81.4% | 54.7% | N/A | N/A |
| Vision-Jev-0.8B | 80.8% | 81.2% | 55.5% | 93.9% at 65.9% coverage | 94.7% at 53.2% coverage |
| Qwen3.5-9B (original) | 87.1% | 80.4% | 32.9% | N/A | N/A |
| Qwen3.5-9B + SFT | 95.4% | 89.3% | 60.5% | N/A | N/A |
| Vision-Jev-9B | 74.4% | 88.9% | 62.1% | 94.8% at 60.7% coverage | 94.7% at 84.9% coverage |

### Structured-output validity

| Model | Choice | Noul | Score |
| --- | ---: | ---: | ---: |
| Qwen3.5-0.8B | 37.4% | 61.0% | 22.2% |
| Qwen3.5-0.8B + SFT | 100.0% | 100.0% | 100.0% |
| Vision-Jev-0.8B | 100.0% | 100.0% | 100.0% |
| Qwen3.5-9B | 99.5% | 100.0% | 100.0% |
| Qwen3.5-9B + SFT | 100.0% | 100.0% | 100.0% |
| Vision-Jev-9B | 100.0% | 100.0% | 100.0% |

Vision-Jev returns candidates through fixed decision heads, so its structured output is always valid.

### Vision-Jev probability and calibration metrics

| Model | Task | Questions | Accuracy | NLL | Brier | ECE | RPS | MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vision-Jev-0.8B | Choice | 8,400 | 80.8% | 0.530 | 0.269 | 0.026 | N/A | N/A |
| Vision-Jev-0.8B | Noul | 2,400 | 81.2% | 0.396 | 0.253 | 0.031 | N/A | N/A |
| Vision-Jev-0.8B | Score | 1,200 | 55.5% | 1.003 | 0.552 | 0.031 | 0.090 | 0.514 |
| Vision-Jev-9B | Choice | 8,400 | 74.4% | 0.610 | 0.309 | 0.016 | N/A | N/A |
| Vision-Jev-9B | Noul | 2,400 | 88.9% | 0.231 | 0.145 | 0.023 | N/A | N/A |
| Vision-Jev-9B | Score | 1,200 | 62.1% | 0.834 | 0.485 | 0.041 | 0.071 | 0.405 |

#### Threshold policy

| Model | Task | Threshold | Coverage | Accepted accuracy |
| --- | --- | ---: | ---: | ---: |
| Vision-Jev-0.8B | Choice | 0.762 | 65.9% | 93.9% |
| Vision-Jev-0.8B | Noul | 0.797 | 53.2% | 94.7% |
| Vision-Jev-0.8B | Score | 0.886 | 9.2% | 97.3% |
| Vision-Jev-9B | Choice | 0.665 | 60.7% | 94.8% |
| Vision-Jev-9B | Noul | 0.697 | 84.9% | 94.7% |
| Vision-Jev-9B | Score | 0.915 | 14.3% | 96.5% |

#### High-confidence errors

Counts cover every incorrect test prediction with confidence at least 0.9.

| Model | Errors | Test questions | Rate |
| --- | ---: | ---: | ---: |
| Vision-Jev-0.8B | 163 | 12,000 | 1.4% |
| Vision-Jev-9B | 86 | 12,000 | 0.7% |

### Inference timing

The original-Qwen and SFT JSONL files retain generation latency for every question. The independent Vision-Jev timing runs retain synchronized full decision-path latency. The table summarizes every per-question record and labels the two scopes separately.

| Model | Scope | Total (s) | Mean (ms) | P50 | P95 | P99 | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | Per-question generation | 3718.1 | 309.8 | 241.9 | 542.3 | 550.7 | 171.3 | 671.0 |
| Qwen3.5-0.8B + SFT | Per-question generation | 3802.8 | 316.9 | 286.6 | 444.9 | 501.7 | 248.0 | 569.6 |
| Vision-Jev-0.8B | Full decision loop | 1051.6 | 87.6 | 83.2 | 147.3 | 193.8 | 41.7 | 292.2 |
| Qwen3.5-9B | Per-question generation | 4372.3 | 364.4 | 326.9 | 507.7 | 530.2 | 263.7 | 804.1 |
| Qwen3.5-9B + SFT | Per-question generation | 5259.5 | 438.3 | 392.0 | 609.1 | 691.4 | 344.2 | 754.0 |
| Vision-Jev-9B | Full decision loop | 1384.5 | 115.4 | 105.6 | 197.3 | 273.0 | 55.1 | 318.5 |

#### Timing by task

| Model | Task | Samples | Mean (ms) | P50 | P95 | P99 | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | Choice | 8,400 | 298.2 | 241.4 | 535.7 | 542.2 | 190.0 | 652.2 |
| Qwen3.5-0.8B | Noul | 2,400 | 270.5 | 218.9 | 505.7 | 528.8 | 210.0 | 636.3 |
| Qwen3.5-0.8B | Score | 1,200 | 469.8 | 541.3 | 548.5 | 656.2 | 171.3 | 671.0 |
| Qwen3.5-0.8B + SFT | Choice | 8,400 | 341.4 | 290.7 | 447.1 | 508.1 | 267.4 | 569.6 |
| Qwen3.5-0.8B + SFT | Noul | 2,400 | 262.2 | 258.9 | 290.8 | 317.8 | 252.4 | 565.8 |
| Qwen3.5-0.8B + SFT | Score | 1,200 | 255.2 | 251.7 | 301.8 | 305.6 | 248.0 | 308.8 |
| Vision-Jev-0.8B | Choice | 8,400 | 88.5 | 84.1 | 167.6 | 199.0 | 41.7 | 250.2 |
| Vision-Jev-0.8B | Noul | 2,400 | 72.0 | 73.7 | 78.9 | 83.0 | 52.1 | 292.2 |
| Vision-Jev-0.8B | Score | 1,200 | 112.9 | 113.5 | 116.8 | 119.2 | 90.7 | 138.9 |
| Qwen3.5-9B | Choice | 8,400 | 390.9 | 330.3 | 510.5 | 563.5 | 263.7 | 804.1 |
| Qwen3.5-9B | Noul | 2,400 | 295.9 | 293.3 | 302.7 | 357.7 | 285.9 | 365.7 |
| Qwen3.5-9B | Score | 1,200 | 315.7 | 314.0 | 316.9 | 359.1 | 311.5 | 363.0 |
| Qwen3.5-9B + SFT | Choice | 8,400 | 471.3 | 396.5 | 612.6 | 696.3 | 362.9 | 754.0 |
| Qwen3.5-9B + SFT | Noul | 2,400 | 357.9 | 351.9 | 424.5 | 431.1 | 344.2 | 624.3 |
| Qwen3.5-9B + SFT | Score | 1,200 | 368.3 | 363.5 | 420.8 | 424.9 | 360.3 | 432.0 |
| Vision-Jev-9B | Choice | 8,400 | 116.0 | 110.7 | 227.3 | 276.7 | 55.1 | 318.5 |
| Vision-Jev-9B | Noul | 2,400 | 94.1 | 96.0 | 111.8 | 122.6 | 72.7 | 133.2 |
| Vision-Jev-9B | Score | 1,200 | 153.9 | 158.0 | 180.6 | 185.9 | 133.2 | 192.4 |

### 0.8B results by dataset and task

Each result cell reports accuracy / mean per-sample latency.

| Dataset | Task | Questions | Qwen3.5-0.8B | Qwen3.5-0.8B + SFT | Vision-Jev-0.8B |
| --- | --- | ---: | ---: | ---: | ---: |
| Android Control | Choice | 700 | 1.0% / 281.9 ms | 84.1% / 357.2 ms | 52.3% / 173.0 ms |
| ChartQA | Choice | 700 | 53.6% / 234.0 ms | 97.0% / 291.0 ms | 79.4% / 83.7 ms |
| ChartQA | Noul | 82 | 45.1% / 300.0 ms | 61.0% / 265.3 ms | 56.1% / 75.5 ms |
| CLEVR | Choice | 700 | 17.9% / 277.9 ms | 93.6% / 288.4 ms | 87.9% / 75.8 ms |
| CLEVR | Noul | 580 | 59.3% / 229.4 ms | 92.6% / 261.0 ms | 91.6% / 73.2 ms |
| GQA | Choice | 700 | 29.1% / 278.4 ms | 96.7% / 288.4 ms | 95.9% / 76.4 ms |
| GQA | Noul | 580 | 29.8% / 286.2 ms | 83.1% / 262.9 ms | 83.8% / 74.2 ms |
| KonIQ-10k | Score | 1,200 | 2.9% / 469.8 ms | 54.7% / 255.2 ms | 55.5% / 112.9 ms |
| MultiNLI | Choice | 700 | 44.3% / 226.9 ms | 76.3% / 276.7 ms | 72.6% / 52.5 ms |
| NLVR | Noul | 579 | 23.0% / 288.2 ms | 64.1% / 259.3 ms | 64.4% / 64.4 ms |
| RefCOCO | Choice | 700 | 0.0% / 413.4 ms | 95.0% / 434.3 ms | 83.1% / 91.3 ms |
| RefCOCOg | Choice | 700 | 0.0% / 401.8 ms | 90.4% / 431.1 ms | 80.1% / 90.0 ms |
| RefCOCO+ | Choice | 700 | 0.0% / 405.0 ms | 88.4% / 432.4 ms | 78.6% / 91.6 ms |
| ScienceQA | Choice | 700 | 47.1% / 233.9 ms | 85.3% / 289.9 ms | 83.1% / 76.3 ms |
| TextVQA | Choice | 700 | 54.0% / 228.5 ms | 98.0% / 291.7 ms | 92.0% / 85.1 ms |
| Visual7W | Choice | 700 | 20.6% / 340.1 ms | 80.7% / 423.4 ms | 67.7% / 85.3 ms |
| VQAv2 | Choice | 700 | 25.9% / 257.0 ms | 98.9% / 291.7 ms | 96.4% / 81.0 ms |
| VQAv2 | Noul | 579 | 44.9% / 274.1 ms | 88.6% / 265.1 ms | 88.8% / 75.8 ms |

### 9B results by dataset and task

Each result cell reports accuracy / mean per-sample latency.

| Dataset | Task | Questions | Qwen3.5-9B | Qwen3.5-9B + SFT | Vision-Jev-9B |
| --- | --- | ---: | ---: | ---: | ---: |
| Android Control | Choice | 700 | 41.0% / 468.7 ms | 89.6% / 544.1 ms | 34.7% / 237.5 ms |
| ChartQA | Choice | 700 | 94.7% / 330.4 ms | 98.0% / 399.3 ms | 81.6% / 108.4 ms |
| ChartQA | Noul | 82 | 80.5% / 299.6 ms | 80.5% / 364.5 ms | 80.5% / 96.1 ms |
| CLEVR | Choice | 700 | 90.0% / 321.8 ms | 98.9% / 392.8 ms | 96.3% / 96.9 ms |
| CLEVR | Noul | 580 | 88.3% / 294.0 ms | 99.1% / 356.5 ms | 99.0% / 94.8 ms |
| GQA | Choice | 700 | 96.4% / 326.1 ms | 98.0% / 393.1 ms | 97.1% / 98.7 ms |
| GQA | Noul | 580 | 80.2% / 297.4 ms | 89.7% / 358.8 ms | 89.5% / 96.9 ms |
| KonIQ-10k | Score | 1,200 | 32.9% / 315.7 ms | 60.5% / 368.3 ms | 62.1% / 153.9 ms |
| MultiNLI | Choice | 700 | 73.0% / 306.3 ms | 86.6% / 374.5 ms | 87.1% / 67.0 ms |
| NLVR | Noul | 579 | 64.1% / 291.7 ms | 74.6% / 354.1 ms | 73.9% / 85.6 ms |
| RefCOCO | Choice | 700 | 96.7% / 492.4 ms | 97.6% / 595.9 ms | 60.7% / 119.5 ms |
| RefCOCOg | Choice | 700 | 93.1% / 487.7 ms | 94.9% / 588.8 ms | 59.7% / 117.6 ms |
| RefCOCO+ | Choice | 700 | 90.7% / 492.0 ms | 93.4% / 595.6 ms | 56.0% / 119.8 ms |
| ScienceQA | Choice | 700 | 94.6% / 326.7 ms | 97.0% / 394.6 ms | 95.6% / 99.5 ms |
| TextVQA | Choice | 700 | 98.4% / 331.9 ms | 99.6% / 400.1 ms | 94.7% / 109.9 ms |
| Visual7W | Choice | 700 | 80.4% / 476.5 ms | 91.4% / 577.0 ms | 30.3% / 111.0 ms |
| VQAv2 | Choice | 700 | 96.6% / 329.8 ms | 99.7% / 399.7 ms | 98.6% / 105.8 ms |
| VQAv2 | Noul | 579 | 89.1% / 300.1 ms | 95.0% / 361.3 ms | 94.5% / 98.6 ms |

## Frozen test showcase

Each animation uses one identical frozen test sample and candidate set for the original Qwen3.5 checkpoint and Vision-Jev. Categories were declared first. Within each category, the sample is the minimum SHA-256 sample ID for which both Vision-Jev checkpoints are correct and pass their already-frozen confidence threshold; baseline predictions were not used for selection. Each GIF cycles through every configured example. The two progress bars advance on the measured median of three post-warmup inference runs, then reveal each model's answer.

Included capabilities: General visual question answering, Text reading in natural images, Chart reasoning, Compositional visual reasoning, Diagram-grounded science reasoning, Evidence sufficiency judgment, Ordered visual quality assessment.

### 0.8B

<p align="center"><img src="asset/demos/0.8b.gif" alt="0.8B: original Qwen3.5 versus Vision-Jev across frozen RLCD examples" width="900"></p>

### 9B

<p align="center"><img src="asset/demos/9b.gif" alt="9B: original Qwen3.5 versus Vision-Jev across frozen RLCD examples" width="900"></p>
<!-- showcase:end -->

## Quick start

```bash
git clone git@github.com:LMDHQ-0420/Vision-Jev.git
cd Vision-Jev
conda env create -f environment.yml
conda activate vision-jev

vision-jev doctor
python scripts/check_repo.py
python -m unittest discover -s tests -p 'test_*.py'
```

The default external data root is `/mnt/sda1/sol_data/vision-jev`:

```bash
vision-jev data-download
vision-jev data-inventory
vision-jev data-download-weblinx-subset --target-rows 7000
vision-jev data-download-gui-odyssey-subset --target-rows 8500
vision-jev data-normalize visual7w
vision-jev data-build-public \
  --mixture configs/data/sft_117k.json \
  --output /mnt/sda1/sol_data/vision-jev/manifests/public-117k.jsonl
```

Every JSONL row is one complete question; candidates are never expanded into fake independent samples.

## Repository layout

```text
asset/              Brand assets
showcase/           Reproducible side-by-side demo implementation
configs/            Data, model, training, and evaluation templates
data/               Schemas, public examples, and source metadata
docs/               Architecture, data, training, evaluation, and release guides
runs/               Immutable run metadata and external artifact references
scripts/            Repository and resource checks
vision_jev/         Data, model, runtime, training, and evaluation code
tests/              Unit and integration tests
```

Raw data, checkpoints, and large run artifacts are excluded from Git.

## Documentation and license

Start with the [documentation index](docs/index.md), [data pipeline](docs/data/pipeline.md), [SFT guide](docs/training/sft.md), [evaluation protocol](docs/evaluation/protocol.md), and [release checklist](docs/release/checklist.md).

Vision-Jev source code is released under the [Apache License 2.0](LICENSE). Dataset assets and derived releases may carry additional upstream restrictions; consult the [source ledger](docs/data/sources.md) before redistribution or commercial use.

Formal citation metadata will be added with the first model release. Until then, cite the repository URL and exact Git commit used.
