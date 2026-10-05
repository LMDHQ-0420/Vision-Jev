#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1:-}"
data_root="${2:-${DATA_ROOT:-/data/vision-jev}}"
train_python="${TRAIN_PYTHON:-${data_root}/envs/train/bin/python}"
accelerate="${ACCELERATE:-${data_root}/envs/train/bin/accelerate}"
manifest="${data_root}/manifests/public-117k-training.jsonl"
rlcd_views="${data_root}/manifests/rlcd/train-views.jsonl"
rlcd_roles="${data_root}/manifests/rlcd/base-72k.jsonl"
model_root="${data_root}/models"
runs="${data_root}/runs"

case "${profile}" in
  08b)
    run_slug="qwen35-08b"
    model_config="configs/model/qwen35_08b.json"
    sft_config="configs/train/sft_main.json"
    sft_smoke_config="configs/train/sft_smoke.json"
    rlcd_config="configs/train/rlcd_main.json"
    rlcd_smoke_config="configs/train/rlcd_smoke.json"
    ;;
  9b)
    run_slug="qwen35-9b"
    model_config="configs/model/qwen35_9b.json"
    sft_config="configs/train/sft_main_9b.json"
    sft_smoke_config="configs/train/sft_smoke_9b.json"
    rlcd_config="configs/train/rlcd_main_9b.json"
    rlcd_smoke_config="configs/train/rlcd_smoke_9b.json"
    ;;
  *)
    echo "usage: $0 {08b|9b} [data-root]" >&2
    exit 2
    ;;
esac

sft_smoke="${runs}/${run_slug}-sft-smoke-dual-117k"
sft_run="${runs}/${run_slug}-sft-main-117k"
sft_eval="${sft_run}/eval-full"
rlcd_smoke="${runs}/${run_slug}-rlcd-smoke-dual-72k"
rlcd_run="${runs}/${run_slug}-rlcd-main-72k"
rlcd_eval="${rlcd_run}/evaluation"
pipeline_log="${runs}/${run_slug}-pipeline.log"
lock_file="${runs}/${run_slug}-pipeline.lock"

mkdir -p "${runs}"
cd "${repo_root}"
exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "another ${profile} pipeline owns ${lock_file}" >&2
  exit 3
fi

for required in "${train_python}" "${accelerate}" "${manifest}" "${rlcd_views}" "${rlcd_roles}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required path: ${required}" >&2
    exit 4
  fi
done

timestamp() { date --iso-8601=seconds; }
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "${pipeline_log}"; }

require_clean_destination() {
  local directory="$1"
  local completion_file="$2"
  if [[ -f "${completion_file}" ]]; then
    return 1
  fi
  if [[ -e "${directory}" ]]; then
    log "refusing to overwrite incomplete stage: ${directory}"
    exit 5
  fi
  return 0
}

wait_for_gpus() {
  while true; do
    mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [[ "${#used[@]}" -ge 2 && "${used[0]}" -lt 2048 && "${used[1]}" -lt 2048 ]]; then
      return
    fi
    log "waiting for two GPUs below 2 GiB used"
    sleep 30
  done
}

run_dual() {
  env CUDA_VISIBLE_DEVICES=0,1 VISION_JEV_PROCESS_NAME=comfyui PYTHONUNBUFFERED=1 \
    "${accelerate}" launch --num_processes 2 --multi_gpu --mixed_precision bf16 \
    --num_machines 1 --dynamo_backend no "$@"
}

log "pipeline start profile=${profile} data_root=${data_root}"
HF_HOME="${data_root}/hf-cache" "${train_python}" -m vision_jev.cli model-prepare \
  --config "${model_config}" --model-root "${model_root}" 2>&1 | tee -a "${pipeline_log}"

wait_for_gpus
if require_clean_destination "${sft_smoke}" "${sft_smoke}/summary.json"; then
  log "starting dual-GPU SFT smoke"
  run_dual -m vision_jev.cli train-sft --config "${sft_smoke_config}" \
    --data "${manifest}" --output "${sft_smoke}" --model-root "${model_root}" \
    2>&1 | tee -a "${sft_smoke}.log" "${pipeline_log}"
else
  log "SFT smoke already complete; skipping"
fi

if require_clean_destination "${sft_run}" "${sft_run}/summary.json"; then
  log "starting full dual-GPU SFT"
  run_dual -m vision_jev.cli train-sft --config "${sft_config}" \
    --data "${manifest}" --output "${sft_run}" --model-root "${model_root}" \
    2>&1 | tee -a "${sft_run}.log" "${pipeline_log}"
else
  log "full SFT already complete; skipping"
fi

mkdir -p "${sft_eval}"
if [[ ! -f "${sft_eval}/predictions.summary.json" ]]; then
  if [[ -e "${sft_eval}/predictions.jsonl" ]]; then
    log "refusing to overwrite incomplete SFT evaluation"
    exit 6
  fi
  log "starting complete SFT holdout evaluation"
  env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "${train_python}" -m vision_jev.cli eval-sft \
    --config "${sft_config}" --data "${manifest}" \
    --checkpoint "${sft_run}/checkpoint-last" \
    --output "${sft_eval}/predictions.jsonl" --maximum 0 --progress-every 25 \
    --model-root "${model_root}" 2>&1 | tee -a "${sft_eval}/evaluation.log" "${pipeline_log}"
else
  log "SFT holdout evaluation already complete; skipping"
fi

if [[ ! -f "${sft_eval}/gate.json" ]]; then
  "${train_python}" scripts/check_sft_gate.py \
    --summary "${sft_eval}/predictions.summary.json" \
    --gate configs/eval/sft_gate.json --output "${sft_eval}/gate.json" \
    2>&1 | tee -a "${pipeline_log}"
fi

wait_for_gpus
if require_clean_destination "${rlcd_smoke}" "${rlcd_smoke}/summary.json"; then
  log "starting dual-GPU RLCD smoke"
  run_dual -m vision_jev.cli train-rlcd --config "${rlcd_smoke_config}" \
    --sft-checkpoint "${sft_run}/checkpoint-last" --train-manifest "${rlcd_views}" \
    --role-manifest "${rlcd_roles}" --output "${rlcd_smoke}" --model-root "${model_root}" \
    2>&1 | tee -a "${rlcd_smoke}.log" "${pipeline_log}"
else
  log "RLCD smoke already complete; skipping"
fi

if require_clean_destination "${rlcd_run}" "${rlcd_run}/summary.json"; then
  log "starting full dual-GPU RLCD"
  run_dual -m vision_jev.cli train-rlcd --config "${rlcd_config}" \
    --sft-checkpoint "${sft_run}/checkpoint-last" --train-manifest "${rlcd_views}" \
    --role-manifest "${rlcd_roles}" --output "${rlcd_run}" --model-root "${model_root}" \
    2>&1 | tee -a "${rlcd_run}.log" "${pipeline_log}"
else
  log "full RLCD already complete; skipping"
fi

mkdir -p "${rlcd_eval}"
thresholds="${rlcd_eval}/confidence-thresholds.json"
common_eval=(--config "${rlcd_config}" --checkpoint "${rlcd_run}/checkpoint-last"
  --sft-checkpoint "${sft_run}/checkpoint-last" --role-manifest "${rlcd_roles}"
  --model-root "${model_root}" --maximum 0 --progress-every 100)

if [[ ! -f "${rlcd_eval}/threshold.summary.json" ]]; then
  log "selecting confidence thresholds"
  env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "${train_python}" -m vision_jev.cli eval-rlcd \
    "${common_eval[@]}" --role threshold --output "${rlcd_eval}/threshold.jsonl" \
    --select-thresholds "${thresholds}" 2>&1 | tee -a "${rlcd_eval}/threshold.log" "${pipeline_log}"
fi
if [[ ! -f "${rlcd_eval}/audit.summary.json" ]]; then
  log "running frozen audit evaluation"
  env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "${train_python}" -m vision_jev.cli eval-rlcd \
    "${common_eval[@]}" --role audit --output "${rlcd_eval}/audit.jsonl" \
    --thresholds "${thresholds}" 2>&1 | tee -a "${rlcd_eval}/audit.log" "${pipeline_log}"
fi
if [[ ! -f "${rlcd_eval}/test.summary.json" ]]; then
  log "running one-time frozen test evaluation"
  env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "${train_python}" -m vision_jev.cli eval-rlcd \
    "${common_eval[@]}" --role test --output "${rlcd_eval}/test.jsonl" \
    --thresholds "${thresholds}" 2>&1 | tee -a "${rlcd_eval}/test.log" "${pipeline_log}"
fi

log "pipeline complete profile=${profile}"
