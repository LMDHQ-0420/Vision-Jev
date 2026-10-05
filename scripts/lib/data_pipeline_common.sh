#!/usr/bin/env bash

init_data_pipeline() {
  local repo_root="$1"
  local requested_root="$2"
  [[ -n "$requested_root" ]] || { echo "--data-root is required" >&2; return 2; }
  mkdir -p "$requested_root"

  DATA_REPO_ROOT="$repo_root"
  DATA_ROOT="$(cd "$requested_root" && pwd)"
  DATA_STATE_ROOT="$DATA_ROOT/_state"
  DATA_VENV_ROOT="$DATA_REPO_ROOT/.venv-data"
  DATA_VENV_PYTHON="$DATA_VENV_ROOT/bin/python"
  mkdir -p "$DATA_STATE_ROOT"

  export DATA_REPO_ROOT DATA_ROOT DATA_STATE_ROOT DATA_VENV_ROOT DATA_VENV_PYTHON
  export HF_HOME="$DATA_ROOT/_cache/huggingface"
  export PYTHONPATH="$DATA_REPO_ROOT"
  export PYTHONUNBUFFERED=1
  export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
  mkdir -p "$HF_HOME"
}

ensure_base_runtime() {
  if [[ ! -x "$DATA_VENV_PYTHON" ]]; then
    local bootstrap_python="${DATA_PIPELINE_PYTHON:-python3.12}"
    command -v "$bootstrap_python" >/dev/null 2>&1 || bootstrap_python=python3
    "$bootstrap_python" -m venv "$DATA_VENV_ROOT"
  fi
  if ! "$DATA_VENV_PYTHON" -c 'import huggingface_hub, pandas' >/dev/null 2>&1; then
    "$DATA_VENV_PYTHON" -m pip install 'huggingface_hub>=0.34,<2' 'pandas>=2.2'
  fi
}

ensure_prepare_runtime() {
  ensure_base_runtime
  if ! "$DATA_VENV_PYTHON" -c \
    'import android_env, minigrid, pandas, PIL, pyarrow, tfrecord' >/dev/null 2>&1; then
    "$DATA_VENV_PYTHON" -m pip install -e "$DATA_REPO_ROOT[data,rl]"
  fi
}

run_with_retries() {
  local max_attempts="${DOWNLOAD_MAX_ATTEMPTS:-0}"
  local retry_delay="${DOWNLOAD_RETRY_DELAY:-60}"
  local attempt=1
  local status

  until "$@"; do
    status=$?
    if ((max_attempts > 0 && attempt >= max_attempts)); then
      echo "command failed after $attempt attempts (status $status): $*" >&2
      return "$status"
    fi
    echo "command failed (status $status); retrying in ${retry_delay}s: $*" >&2
    sleep "$retry_delay"
    ((attempt += 1))
  done
}

run_data_cli() {
  "$DATA_VENV_PYTHON" -m vision_jev.cli "$@"
}
