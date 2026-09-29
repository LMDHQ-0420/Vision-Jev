# Vision-Jev SFT Preview Model Card

## Model

This preview is a language-side LoRA adapter for `Qwen/Qwen3.5-0.8B`, pinned to revision `2fc06364715b967f1860aea9cf38778875588b17`. The visual encoder is frozen. LoRA rank is 16, alpha is 32, and dropout is 0.05. The adapter targets audited language projection layers only.

This is an SFT checkpoint, not a completed JEV-like calibrated decision model. It generates one compact JSON answer for Choice, Noul, or five-level image-quality Score tasks. It does not expose calibrated candidate probabilities, an RLCD-trained decision head, or a closed-loop policy.

## Training

- Data: 120,000 complete questions; 114,229 train and 5,771 group-safe holdout.
- Composition: 96,000 Choice, 18,000 Noul, and 6,000 Score questions.
- Procedure: two epochs, two RTX 4090 GPUs, BF16, global batch 32, 7,140 optimizer steps.
- Duration: 58,183 seconds.
- Training-manifest SHA-256: `9e0354fc6d7f5f72d0a8550dc6f8b1cf9f2c402d224700903868697c238946a4`.
- Training-config SHA-256: `0fbe1a94772d6a9f6fbf34ed5099663c1548e05469798d4738ad1720ca7f7808`.

No project-specific human annotation was introduced. Public labels and 3,000 checked same-language API question rewrites were used. Upstream datasets retain their own licenses and restrictions.

## Evaluation

The complete 5,771-question group-safe holdout evaluation measured 85.55% exact match and 100% JSON validity. Choice reached 88.44%, Noul 78.72%, and five-level Score 59.57% exact match. Score mean absolute error was 0.477 levels and 94.58% of predictions were within one level.

On one RTX 4090, sequential end-to-end generation averaged 0.391 seconds per question, with 0.667-second p95 latency and 2.14 GB peak allocated GPU memory. These timings include native image processing and generation but do not represent an optimized batched or native decision-head runtime. Full per-source results and artifact hashes are in `docs/evaluation/sft-preview-results.md`.

## Files and integrity

The repository release directory contains the LoRA adapter only. Loading requires the pinned Qwen base model and the repository's native processor path. SHA-256 values are recorded in `release/sft-preview/SHA256SUMS`.

## Intended use and limitations

The preview is intended for reproducible research on multimodal structured SFT and for establishing the baseline before calibrated-decision post-training. It must not be described as calibrated, as an official reproduction of TypeSafe's proprietary RLCD method, or as a validated autonomous policy. Known risks include OCR and small-target failures, source imbalance, prompt sensitivity, unmeasured domain shift, and uncalibrated generated answers.

## Next work

- Complete failure-slice and external zero-shot analysis.
- Connect native Choice/Noul/Score probability heads.
- Build a group-isolated calibration manifest from unused public data.
- Implement and document an open RLCD-inspired calibrated-decision objective.
- Implement separate online policy/value training for closed-loop environments.
