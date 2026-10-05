# Qwen3.5-0.8B SFT 117k holdout results

Measured on 2026-10-05 using the complete 5,886-question group-safe holdout from
`public-117k-training.jsonl`. The evaluated checkpoint is
`qwen35-08b-sft-main-117k/checkpoint-last`; predictions, the JSON summary and the frozen
RLCD admission decision are stored under the run's `eval-full/` directory.

| Metric | Result |
|---|---:|
| Valid compact JSON | 100.00% |
| Overall exact match | 85.54% |
| Choice exact match | 88.63% (4,652) |
| Noul exact match | 80.07% (918) |
| Score exact match | 56.01% (316) |
| Score within one level | 90.51% |
| Score mean absolute error | 0.551 |
| Mean generation latency | 0.348 s/question |
| P95 generation latency | 0.643 s/question |
| Peak allocated GPU memory | 2.06 GiB |

The SFT-to-RLCD gate passed all five checks. Noul exceeded its 80% admission threshold by
only 0.07 percentage points, so subsequent RLCD reports must show its Accuracy, NLL, Brier
and ECE separately; a macro or mixed overall score cannot hide a Noul regression.

Score is an ordinal quality task. Its strict five-class exact match is reported rather than
discarded, but the adjacent-level rate and MAE are the appropriate companion measures. All
Score holdout rows produced a valid integer level.
