# Reproducible static showcase

The README animations compare original Qwen3.5 with Vision-Jev on samples from the
frozen static RLCD test split. The showcase is inference-only: it reuses the released
SFT adapters, `ChoiceHead`, `NoulHead`, `ScoreHead`, and temperatures without creating
or modifying a checkpoint.

## Comparison contract

Each example is evaluated twice per parameter group:

| Group | Left panel | Right panel |
| --- | --- | --- |
| 0.8B | Original Qwen3.5-0.8B | Vision-Jev-0.8B |
| 9B | Original Qwen3.5-9B | Vision-Jev-9B |

Both panels receive the identical image, state text, question, and dynamic candidate
set. Original Qwen uses greedy structured generation. Vision-Jev scores the candidates
with its existing calibrated task head. Invalid baseline JSON remains an invalid result.

The seven categories are declared in `showcase/configs/examples.json`. Within each
category, the published sample is selected from the frozen test role by this auditable
rule:

1. Both released Vision-Jev checkpoints must be correct.
2. Both predictions must pass their previously frozen per-task confidence threshold.
3. The minimum `SHA-256(sample_id)` among eligible samples is selected.
4. Original Qwen predictions are never used during selection.

This deliberate accepted-set showcase is paired with the complete 12,000-question test
table in the README. One GIF per parameter group cycles through all configured samples.
Independent progress bars use the median of three post-warmup per-sample inference runs,
so each answer appears when that model finishes. All three predictions must agree. The
animations demonstrate calibrated strengths; the full test table remains the measure of
overall quality.

## Run

Set the external data root and validate every configured sample against the pinned
manifest hash:

```bash
export DATA_ROOT=/data/vision-jev
python -m showcase.cli validate
```

Build a source-aware report from the immutable completed test artifacts:

```bash
python -m showcase.cli build-test-report
```

Run every static comparison sequentially. Only one model is resident on the GPU,
complete prediction records are skipped when resuming, and the final output is
`asset/demos/0.8b.gif` plus `asset/demos/9b.gif`:

```bash
python -m showcase.cli run-all
```

Record one model prediction or render one completed pair for local inspection:

```bash
python -m showcase.cli record --example general-vqa --model qwen35-08b-base
python -m showcase.cli render \
  --baseline showcase/output/general-vqa/qwen35-08b-base/prediction.json \
  --trained showcase/output/general-vqa/vision-jev-08b/prediction.json \
  --output showcase/output/general-vqa-preview.gif
```

Raw prediction records and generated reports remain under ignored `showcase/output/`.
Reviewed GIFs belong in `asset/demos/`. Publish the generated section only after all
configured assets and the test report exist:

```bash
python -m showcase.cli readme
python -m showcase.cli publish-readme
```

## Integrity rules

- Never train, fine-tune, or create a showcase-specific decision head.
- Never change the sample or candidates between baseline and Vision-Jev.
- Keep the frozen manifest hash and exact sample IDs in version control.
- Preserve valid and invalid original-Qwen generations in the prediction records.
- Do not describe selected accepted-set examples as aggregate evaluation.
- Regenerate the full test report from immutable artifacts before publishing metrics.
