#!/usr/bin/env bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="/data/vision-jev"
train_python="${data_root}/envs/train/bin/python"
model_root="${data_root}/models"
manifest="${data_root}/manifests/public-117k-training.jsonl"
output_dir="${data_root}/runs/qwen35-08b-sft-main-117k"
smoke_output="${data_root}/runs/qwen35-08b-sft-smoke-dual-117k"
wait_log="${data_root}/runs/qwen35-08b-sft-main-117k.wait.log"
smoke_log="${data_root}/runs/qwen35-08b-sft-smoke-dual-117k.log"
train_log="${data_root}/runs/qwen35-08b-sft-main-117k.log"
lock_file="${data_root}/runs/qwen35-08b-sft-main-117k.lock"
memory_limit_mib=10240
poll_seconds=10

mkdir -p "${data_root}/runs"
exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "A main-training launcher is already active." >&2
  exit 1
fi

for required in "${train_python}" "${manifest}" "${model_root}/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17/config.json"; do
  [[ -e "${required}" ]] || { echo "Missing training prerequisite: ${required}" >&2; exit 1; }
done
for destination in "${output_dir}" "${smoke_output}"; do
  if [[ -d "${destination}" ]] && [[ -n "$(find "${destination}" -mindepth 1 -print -quit)" ]]; then
    echo "Training output is not empty: ${destination}" >&2
    exit 1
  fi
done

while true; do
  timestamp="$(date --iso-8601=seconds)"
  if gpu_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"; then
    gpu_count="$(printf '%s\n' "${gpu_memory}" | awk 'NF {count++} END {print count+0}')"
    busy_count="$(printf '%s\n' "${gpu_memory}" | awk -v limit="${memory_limit_mib}" '$1 >= limit {count++} END {print count+0}')"
    if [[ "${gpu_count}" -ge 2 && "${busy_count}" -eq 0 ]]; then
      printf '%s both GPUs are below 10 GiB; starting smoke test.\n' "${timestamp}" >>"${wait_log}"
      break
    fi
    printf '%s waiting for two GPUs; memory MiB: %s\n' \
      "${timestamp}" "${gpu_memory//$'\n'/, }" >>"${wait_log}"
  else
    printf '%s nvidia-smi is unavailable; waiting.\n' "${timestamp}" >>"${wait_log}"
  fi
  sleep "${poll_seconds}"
done

cd "${project_root}"
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export VISION_JEV_PROCESS_NAME=comfyui

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "process_name=${VISION_JEV_PROCESS_NAME}"
  echo "manifest=${manifest}"
  sha256sum "${manifest}" configs/train/sft_main.json configs/model/qwen35_08b.json
  "${train_python}" --version
  "${train_python}" -c 'import accelerate, peft, torch, transformers; print({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "accelerate": accelerate.__version__, "peft": peft.__version__})'
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader
} >>"${train_log}" 2>&1

launch=(
  "${train_python}" -m accelerate.commands.launch
  --multi_gpu --num_processes 2 --mixed_precision bf16 --main_process_port 29517
  -m vision_jev.cli train-sft
)

if ! "${launch[@]}" \
  --config configs/train/sft_framework_smoke.json \
  --data "${manifest}" \
  --model-root "${model_root}" \
  --output "${smoke_output}" >>"${smoke_log}" 2>&1; then
  printf '%s two-GPU smoke test failed; full training was not started.\n' \
    "$(date --iso-8601=seconds)" >>"${wait_log}"
  exit 1
fi

printf '%s two-GPU smoke test passed; starting full SFT.\n' \
  "$(date --iso-8601=seconds)" >>"${wait_log}"

"${launch[@]}" \
  --config configs/train/sft_main.json \
  --data "${manifest}" \
  --model-root "${model_root}" \
  --output "${output_dir}" >>"${train_log}" 2>&1
