#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1:-all}"
data_root="${2:-${DATA_ROOT:-/data/vision-jev}}"
train_python="${TRAIN_PYTHON:-${data_root}/envs/train/bin/python}"
model_root="${data_root}/models"
role_manifest="${data_root}/manifests/rlcd-interactive/base-16k.jsonl"
runs="${data_root}/runs"
lock_file="${runs}/interactive-zero-shot-evaluation.lock"

case "${profile}" in
  08b) profiles=(08b) ;;
  9b) profiles=(9b) ;;
  all) profiles=(08b 9b) ;;
  *)
    echo "usage: $0 [08b|9b|all] [data-root]" >&2
    exit 2
    ;;
esac

for required in "${train_python}" "${role_manifest}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required path: ${required}" >&2
    exit 3
  fi
done

mkdir -p "${runs}"
cd "${repo_root}"
exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "another interactive zero-shot evaluation owns ${lock_file}" >&2
  exit 4
fi

timestamp() { date --iso-8601=seconds; }
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "${pipeline_log}"; }

require_clean_output() {
  local output="$1"
  local summary="$2"
  if [[ -f "${summary}" ]]; then
    return 1
  fi
  if [[ -e "${output}" ]]; then
    log "refusing to overwrite incomplete evaluation output: ${output}"
    exit 5
  fi
  return 0
}

run_profile() {
  local selected="$1"
  local run_slug config sft_checkpoint rlcd_checkpoint
  case "${selected}" in
    08b)
      run_slug="qwen35-08b"
      config="configs/train/rlcd_main.json"
      ;;
    9b)
      run_slug="qwen35-9b"
      config="configs/train/rlcd_main_9b.json"
      ;;
  esac
  sft_checkpoint="${runs}/${run_slug}-sft-main-117k/checkpoint-last"
  rlcd_checkpoint="${runs}/${run_slug}-rlcd-main-72k/checkpoint-last"
  output_dir="${runs}/${run_slug}-rlcd-main-72k/evaluation-interactive-zero-shot"
  pipeline_log="${output_dir}/pipeline.log"
  thresholds="${output_dir}/confidence-thresholds.json"

  for required in "${sft_checkpoint}" "${rlcd_checkpoint}"; do
    if [[ ! -d "${required}" ]]; then
      echo "missing required checkpoint: ${required}" >&2
      exit 6
    fi
  done
  mkdir -p "${output_dir}"
  common=(
    --config "${config}"
    --checkpoint "${rlcd_checkpoint}"
    --sft-checkpoint "${sft_checkpoint}"
    --role-manifest "${role_manifest}"
    --model-root "${model_root}"
    --maximum 0
    --progress-every 100
  )

  log "interactive zero-shot evaluation start profile=${selected}"
  if require_clean_output "${output_dir}/threshold.jsonl" "${output_dir}/threshold.summary.json"; then
    log "running frozen threshold role"
    env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "${train_python}" -m vision_jev.cli \
      eval-rlcd "${common[@]}" --role threshold \
      --output "${output_dir}/threshold.jsonl" --select-thresholds "${thresholds}" \
      2>&1 | tee -a "${output_dir}/threshold.log" "${pipeline_log}"
  fi
  if require_clean_output "${output_dir}/audit.jsonl" "${output_dir}/audit.summary.json"; then
    log "running frozen audit role"
    env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "${train_python}" -m vision_jev.cli \
      eval-rlcd "${common[@]}" --role audit --output "${output_dir}/audit.jsonl" \
      --thresholds "${thresholds}" \
      2>&1 | tee -a "${output_dir}/audit.log" "${pipeline_log}"
  fi
  if require_clean_output "${output_dir}/test.jsonl" "${output_dir}/test.summary.json"; then
    log "running frozen test role"
    env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "${train_python}" -m vision_jev.cli \
      eval-rlcd "${common[@]}" --role test --output "${output_dir}/test.jsonl" \
      --thresholds "${thresholds}" \
      2>&1 | tee -a "${output_dir}/test.log" "${pipeline_log}"
  fi
  log "interactive zero-shot evaluation complete profile=${selected}"
}

for selected in "${profiles[@]}"; do
  run_profile "${selected}"
done
