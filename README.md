<div align="center">
  <img src="asset/Vision-Jev.svg" alt="Vision-Jev" width="760">
  <h3>An open vision-language decision model for dynamic candidates</h3>
  <p>Model architecture, training code, data recipe, evaluation protocol, and weights — developed in the open.</p>
  <p><a href="README.md">English</a> · <a href="README_CN.md">简体中文</a> · <a href="docs/index.md">Documentation</a> · <a href="LICENSE">Apache-2.0</a></p>
  <p><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white"> <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.7%2B-EE4C2C?logo=pytorch&logoColor=white"> <img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-2ea44f"> <img alt="Status" src="https://img.shields.io/badge/status-data%20pipeline-f59e0b"></p>
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

The model combines a native multimodal backbone with lightweight decision heads, optional candidate-set interaction, and region-aware visual evidence. The first target backbone is `Qwen/Qwen3.5-0.8B`; judgment and policy optimization remain separate.

### Open the complete training path

The project will publish the data pipeline, exact quotas, source revisions, deterministic manifests, SFT and policy-training code, evaluation protocol, experiment evidence, model weights, and release cards. Third-party datasets retain their original licenses; this repository publishes the source ledger and reproducible processing path rather than claiming ownership of upstream assets.

## Architecture

```text
image ─ native processor ─ vision encoder ─┐
state + question ─ tokenizer ──────────────┼─ multimodal backbone
dynamic candidates ─ serializer ──────────┘          │
                                         ┌────────────┼────────────┐
                                      Choice        Noul        Score
                                         │
                                  policy + value branch
```

See the [system architecture](docs/architecture/system.md) and [interface contract](docs/architecture/interfaces.md).

## Open data recipe

The target contains **120,000 complete questions** and introduces no project-specific human annotation.

| Block | Choice | Noul | Score | Total |
|---|---:|---:|---:|---:|
| Public datasets | 94,000 | 17,000 | 6,000 | 117,000 |
| Same-language API rewrites | 2,000 | 1,000 | 0 | 3,000 |
| Local synthetic generation | 0 | 0 | 0 | 0 |
| **Total** | **96,000** | **18,000** | **6,000** | **120,000** |

The mixture covers GUI actions, region grounding, compositional reasoning, VQA, OCR, charts, language inference, and visual quality. Public samples preserve their source language; API-assisted samples inherit their parent language and label semantics.

- [Source ledger](docs/data/sources.md)
- [Mixture specification](docs/data/mixtures.md)
- [Machine-readable sources](configs/data/sources.json)
- [Training recipe](configs/data/sft_120k.json)

## Status

The repository currently provides the data pipeline, canonical schemas, deterministic manifest builder, decision-head prototypes, metrics, experiment tracking, and a validated two-GPU environment. The original ten-source pool contains 3,175,521 canonical questions and a validated 90k core manifest. The five-source public extension has passed a combined pilot.

Training results and weights are **not published yet**. Planned results are never presented as measured results; see [STATUS.md](STATUS.md).

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
  --mixture configs/data/sft_120k.json \
  --output /mnt/sda1/sol_data/vision-jev/manifests/public-117k.jsonl
```

Every JSONL row is one complete question; candidates are never expanded into fake independent samples.

## Repository layout

```text
asset/              Brand assets
configs/            Data, model, training, evaluation, and API templates
data/               Schemas, public examples, and source metadata
docs/               Architecture, data, training, evaluation, and release guides
runs/               Immutable run metadata and external artifact references
scripts/            Repository and resource checks
vision_jev/         Data, model, runtime, training, and evaluation code
tests/              Unit and integration tests
```

Raw data, checkpoints, credentials, and large run artifacts are excluded from Git. API configuration belongs in `configs/local/api.toml`; start from [configs/api.example.toml](configs/api.example.toml).

## Roadmap

- [x] Establish repository, schemas, documentation, and run contracts
- [x] Build fixed-source download and canonical normalization pipelines
- [x] Produce and validate the 90k public core manifest
- [x] Validate the five-source public extension pilot
- [ ] Materialize the full GUI-Odyssey training subset
- [ ] Freeze and validate the 117k public manifest
- [ ] Produce 3k program-verified, same-language API rewrites
- [ ] Integrate the native Qwen3.5-0.8B processor and backbone
- [ ] Complete small-batch overfitting and the 12k SFT pilot
- [ ] Train and evaluate the 120k SFT model
- [ ] Implement shared-prefix inference and calibration
- [ ] Train the policy/value branch in closed-loop environments
- [ ] Publish weights, model card, data card, and reproducibility report

The checklist changes only when linked evidence exists.

## Documentation and license

Start with the [documentation index](docs/index.md), [data pipeline](docs/data/pipeline.md), [SFT guide](docs/training/sft.md), [evaluation protocol](docs/evaluation/protocol.md), and [release checklist](docs/release/checklist.md).

Vision-Jev source code is released under the [Apache License 2.0](LICENSE). Dataset assets and derived releases may carry additional upstream restrictions; consult the [source ledger](docs/data/sources.md) before redistribution or commercial use.

Formal citation metadata will be added with the first model release. Until then, cite the repository URL and exact Git commit used.
