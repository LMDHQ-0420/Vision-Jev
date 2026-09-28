#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="/mnt/sda1/sol_data/vision-jev"
output_dir="${data_root}/runs/qwen35-08b-sft-main"
smoke_output="${data_root}/runs/qwen35-08b-sft-smoke-dual"
wait_log="${data_root}/runs/qwen35-08b-sft-main.wait.log"
smoke_log="${data_root}/runs/qwen35-08b-sft-smoke-dual.log"
train_log="${data_root}/runs/qwen35-08b-sft-main.log"
lock_file="${data_root}/runs/qwen35-08b-sft-main.lock"
memory_limit_mib=10240
poll_seconds=10

exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "A main-training waiter is already active." >&2
  exit 1
fi

if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "Training output is not empty: ${output_dir}" >&2
  exit 1
fi

while true; do
  timestamp="$(date --iso-8601=seconds)"
  if gpu_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"; then
    gpu_count="$(printf '%s\n' "${gpu_memory}" | awk 'NF {count++} END {print count+0}')"
    busy_count="$(printf '%s\n' "${gpu_memory}" | awk -v limit="${memory_limit_mib}" '$1 >= limit {count++} END {print count+0}')"
    if [[ "${gpu_count}" -ge 2 && "${busy_count}" -eq 0 ]]; then
      printf '%s both GPUs are below 10 GiB; starting smoke test.\n' "${timestamp}" >>"${wait_log}"
      break
    else
      printf '%s waiting for two GPUs; memory MiB: %s\n' "${timestamp}" "${gpu_memory//$'\n'/, }" >>"${wait_log}"
    fi
  else
    printf '%s nvidia-smi is unavailable; waiting.\n' "${timestamp}" >>"${wait_log}"
  fi
  sleep "${poll_seconds}"
done

cd "${project_root}"
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -e "${smoke_output}" ]]; then
  echo "Smoke-test output already exists: ${smoke_output}" >&2
  exit 1
fi
if ! conda run --no-capture-output -n vision-jev \
  accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
  --main_process_port 29517 -m vision_jev.cli train-sft \
  --config configs/train/sft_framework_smoke.json \
  --data "${data_root}/manifests/sft-120k-training.jsonl" \
  --output "${smoke_output}" >>"${smoke_log}" 2>&1; then
  printf '%s two-GPU smoke test failed; full training was not started.\n' \
    "$(date --iso-8601=seconds)" >>"${wait_log}"
  exit 1
fi
printf '%s two-GPU smoke test passed; starting full SFT.\n' \
  "$(date --iso-8601=seconds)" >>"${wait_log}"
rm -rf "${smoke_output}"

conda run --no-capture-output -n vision-jev \
  accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
  --main_process_port 29517 -m vision_jev.cli train-sft \
  --config configs/train/sft_main.json \
  --data "${data_root}/manifests/sft-120k-training.jsonl" \
  --output "${output_dir}" >>"${train_log}" 2>&1
