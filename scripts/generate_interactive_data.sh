#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/data_pipeline_common.sh
source "$repo_root/scripts/lib/data_pipeline_common.sh"

data_root=""
max_workers="${INTERACTIVE_MAX_WORKERS:-32}"

usage() {
  cat <<'EOF'
Usage: scripts/generate_interactive_data.sh --data-root PATH [--max-workers N]

Generates all CPU-only interactive data with resumable stage boundaries. Procgen
runs in an isolated Python 3.10 environment. No external model or GPU is used.
EOF
}

while (($#)); do
  case "$1" in
    --data-root) data_root="$2"; shift 2 ;;
    --max-workers) max_workers="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

init_data_pipeline "$repo_root" "$data_root"
exec 9>"$DATA_STATE_ROOT/generate-interactive.lock"
flock -n 9 || { echo "another interactive generator holds the data lock" >&2; exit 1; }
ensure_prepare_runtime
cd "$repo_root"

stage_root="$DATA_ROOT/manifests/rlcd-interactive/stages"
image_root="$DATA_ROOT/processed/interactive/images"
mkdir -p "$stage_root" "$image_root"

stage_ready() {
  local path="$1"
  local expected="$2"
  [[ -f "$path" ]] || return 1
  [[ "$(wc -l <"$path")" -eq "$expected" ]] || return 1
  "$DATA_VENV_PYTHON" - "$path" <<'PY' >/dev/null
import sys
from vision_jev.data import validate_jsonl
validate_jsonl(sys.argv[1], check_assets=True)
PY
}

run_stage() {
  local path="$1"
  local expected="$2"
  shift 2
  if stage_ready "$path" "$expected"; then
    echo "[$(date -Is)] stage ready, skipping: $path"
  else
    echo "[$(date -Is)] generating stage: $path"
    "$@"
  fi
}

run_stage "$stage_root/minigrid-navigation-4k.jsonl" 4000 \
  run_data_cli data-generate-minigrid-navigation \
    --output "$stage_root/minigrid-navigation-4k.jsonl" \
    --image-root "$image_root/minigrid-navigation"

run_stage "$stage_root/minigrid-tools-3k.jsonl" 3000 \
  run_data_cli data-generate-minigrid-stage minigrid_tools \
    --output "$stage_root/minigrid-tools-3k.jsonl" \
    --image-root "$image_root/minigrid-tools" --max-workers "$max_workers"

run_stage "$stage_root/minigrid-hazards-static-2250.jsonl" 2250 \
  run_data_cli data-generate-minigrid-stage minigrid_hazards \
    --output "$stage_root/minigrid-hazards-static-2250.jsonl" \
    --image-root "$image_root/minigrid-hazards-static" --max-workers "$max_workers"

run_stage "$stage_root/minigrid-dynamic-obstacles-250.jsonl" 250 \
  run_data_cli data-generate-dynamic-obstacles \
    --output "$stage_root/minigrid-dynamic-obstacles-250.jsonl" \
    --image-root "$image_root/minigrid-dynamic-obstacles" --max-workers "$max_workers"

run_stage "$stage_root/babyai-grounded-2500.jsonl" 2500 \
  run_data_cli data-generate-minigrid-stage babyai_grounded \
    --output "$stage_root/babyai-grounded-2500.jsonl" \
    --image-root "$image_root/babyai-grounded" --max-workers "$max_workers"

procgen_python="$repo_root/.venv-procgen/bin/python"
if [[ ! -x "$procgen_python" ]]; then
  conda create -p "$repo_root/.venv-procgen" --override-channels -c conda-forge python=3.10 pip -y
fi
if ! "$procgen_python" -c 'import procgen; assert procgen.__version__ == "0.10.7"' 2>/dev/null; then
  "$repo_root/.venv-procgen/bin/pip" install 'procgen==0.10.7'
fi
run_stage "$stage_root/procgen-maze-2k.jsonl" 2000 \
  "$procgen_python" scripts/generate_procgen_maze.py \
    --output "$stage_root/procgen-maze-2k.jsonl" \
    --image-root "$image_root/procgen-maze" --max-workers "$max_workers"

run_stage "$stage_root/boxoban-2k.jsonl" 2000 \
  run_data_cli data-generate-boxoban \
    --level-index "$DATA_ROOT/processed/interactive/boxoban-levels.jsonl" \
    --output "$stage_root/boxoban-2k.jsonl" \
    --image-root "$image_root/boxoban" --max-workers "$max_workers"

run_data_cli data-finalize-interactive --data-root "$DATA_ROOT"
echo "[$(date -Is)] interactive 16k generation and validation complete"
