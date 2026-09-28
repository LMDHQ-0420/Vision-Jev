#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="/mnt/sda1/sol_data/vision-jev"
output_dir="${data_root}/runs/qwen35-08b-sft-main"
wait_log="${data_root}/runs/qwen35-08b-sft-main.wait.log"
train_log="${data_root}/runs/qwen35-08b-sft-main.log"
lock_file="${data_root}/runs/qwen35-08b-sft-main.lock"

exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "A main-training waiter is already active." >&2
  exit 1
fi

if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "Training output is not empty: ${output_dir}" >&2
  exit 1
fi

free_checks=0
while (( free_checks < 2 )); do
  timestamp="$(date --iso-8601=seconds)"
  if gpu_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"; then
    gpu_count="$(printf '%s\n' "${gpu_memory}" | awk 'NF {count++} END {print count+0}')"
    busy_count="$(printf '%s\n' "${gpu_memory}" | awk '$1 > 1024 {count++} END {print count+0}')"
    if [[ "${gpu_count}" -ge 2 && "${busy_count}" -eq 0 ]]; then
      free_checks=$((free_checks + 1))
      printf '%s GPUs are available (%s/2 checks).\n' "${timestamp}" "${free_checks}" >>"${wait_log}"
    else
      free_checks=0
      printf '%s waiting for two GPUs; memory MiB: %s\n' "${timestamp}" "${gpu_memory//$'\n'/, }" >>"${wait_log}"
    fi
  else
    free_checks=0
    printf '%s nvidia-smi is unavailable; waiting.\n' "${timestamp}" >>"${wait_log}"
  fi
  if (( free_checks < 2 )); then
    sleep 60
  fi
done

cd "${project_root}"
printf '%s starting two-GPU SFT.\n' "$(date --iso-8601=seconds)" >>"${wait_log}"
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda run --no-capture-output -n vision-jev \
  accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
  --main_process_port 29517 -m vision_jev.cli train-sft \
  --config configs/train/sft_main.json \
  --data "${data_root}/manifests/sft-120k-training.jsonl" \
  --output "${output_dir}" >>"${train_log}" 2>&1
