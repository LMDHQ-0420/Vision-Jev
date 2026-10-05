#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/data_pipeline_common.sh
source "$repo_root/scripts/lib/data_pipeline_common.sh"

data_root=""
detach=false
prepare=true
use_gqa_mirror=true

usage() {
  cat <<'EOF'
Usage: scripts/download_all_data.sh --data-root PATH [OPTIONS]

Downloads every active static and interactive source, materializes selective
subsets, verifies the inventory, and preprocesses all data. Re-running
the command resumes partial HTTP and Hugging Face downloads.

Options:
  --data-root PATH    Required destination root.
  --detach            Run in the background and print the PID and log path.
  --skip-prepare      Download and verify assets without preprocessing.
  --no-gqa-mirror     Use the slower Stanford endpoint for GQA images.
  -h, --help          Show this help.
EOF
}

parse_args() {
  while (($#)); do
    case "$1" in
      --data-root)
        [[ $# -ge 2 ]] || { echo "--data-root needs a path" >&2; exit 2; }
        data_root="$2"
        shift 2
        ;;
      --detach)
        detach=true
        shift
        ;;
      --skip-prepare)
        prepare=false
        shift
        ;;
      --no-gqa-mirror)
        use_gqa_mirror=false
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done
}

start_detached() {
  local pid_path="$DATA_STATE_ROOT/download-all.pid"
  local log_path="$DATA_STATE_ROOT/download-all.log"
  if [[ -s "$pid_path" ]]; then
    local existing_pid
    existing_pid="$(<"$pid_path")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "download already running: PID $existing_pid" >&2
      echo "log: $log_path" >&2
      return 1
    fi
  fi

  local child_args=(--data-root "$DATA_ROOT")
  [[ "$prepare" == true ]] || child_args+=(--skip-prepare)
  [[ "$use_gqa_mirror" == true ]] || child_args+=(--no-gqa-mirror)
  nohup "$repo_root/scripts/download_all_data.sh" "${child_args[@]}" \
    >>"$log_path" 2>&1 </dev/null &
  local child_pid=$!
  printf '%s\n' "$child_pid" >"$pid_path"
  echo "download started: PID $child_pid"
  echo "log: $log_path"
  echo "inventory: $DATA_STATE_ROOT/final-inventory.json"
}

acquire_lock() {
  exec 9>"$DATA_STATE_ROOT/download-all.lock"
  flock -n 9 || { echo "another download process holds the data lock" >&2; return 1; }
  printf '%s\n' "$$" >"$DATA_STATE_ROOT/download-all.pid"
}

cleanup() {
  local status=$?
  local pid_path="$DATA_STATE_ROOT/download-all.pid"
  if [[ -f "$pid_path" ]] && [[ "$(<"$pid_path")" == "$$" ]]; then
    rm -f "$pid_path"
  fi
  if ((status == 0)); then
    printf '[%s] download and preprocessing complete\n' "$(date -Is)"
  else
    printf '[%s] pipeline stopped with status %s; rerun the same command to resume\n' \
      "$(date -Is)" "$status" >&2
  fi
  return "$status"
}

prepare_gqa_questions() {
  local downloads="$DATA_ROOT/raw/gqa/downloads"
  local destination="$downloads/questions1.2.zip"
  local expected_bytes=1498616372
  mkdir -p "$downloads"
  if [[ -f "$destination" ]] && [[ "$(stat -c %s "$destination")" == "$expected_bytes" ]]; then
    return
  fi
  if ! command -v aria2c >/dev/null 2>&1; then
    [[ ! -f "$destination.part.aria2" ]] || {
      echo "aria2c is required to resume $destination.part safely" >&2
      return 1
    }
    return
  fi
  aria2c \
    --continue=true --max-connection-per-server=8 --split=8 --min-split-size=16M \
    --file-allocation=none --auto-file-renaming=false --allow-overwrite=true \
    --max-tries=0 --retry-wait=5 --dir="$downloads" --out=questions1.2.zip.part \
    https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
  [[ "$(stat -c %s "$destination.part")" == "$expected_bytes" ]]
  mv "$destination.part" "$destination"
}

prepare_gqa_images() {
  [[ "$use_gqa_mirror" == true ]] || return 0
  local downloads="$DATA_ROOT/raw/gqa/downloads"
  local destination="$downloads/images.zip"
  local expected_bytes=21817965542
  local expected_sha256="02ce5c49c793accd5305356de9c39a50f80a7aaac193b0203de30dbbc65bde62"
  mkdir -p "$downloads"
  if [[ ! -f "$destination" ]] || [[ "$(stat -c %s "$destination")" != "$expected_bytes" ]]; then
    "$DATA_VENV_PYTHON" -c \
      "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Feeky929/GQA-images', repo_type='dataset', filename='images.zip', revision='a30da776e9311362fe2691163dd74608578d58a1', local_dir=r'$downloads')"
  fi
  local actual_sha256
  actual_sha256="$(sha256sum "$destination" | cut -d' ' -f1)"
  [[ "$actual_sha256" == "$expected_sha256" ]] || {
    echo "GQA mirror checksum mismatch: $actual_sha256" >&2
    return 1
  }
  mkdir -p "$DATA_STATE_ROOT/mirrors"
  cat >"$DATA_STATE_ROOT/mirrors/gqa-images.json" <<EOF
{
  "source": "gqa",
  "artifact": "images",
  "repo_id": "Feeky929/GQA-images",
  "revision": "a30da776e9311362fe2691163dd74608578d58a1",
  "filename": "images.zip",
  "bytes": $expected_bytes,
  "sha256": "$expected_sha256"
}
EOF
}

active_sources() {
  "$DATA_VENV_PYTHON" - "$repo_root/configs/data/sources.json" <<'PY'
import json
import sys
from pathlib import Path

sources = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["sources"]
active = (item for item in sources if item.get("active", True))
for item in sorted(active, key=lambda row: (int(row.get("priority", 99)), str(row["id"]))):
    print(item["id"])
PY
}

download_catalog_sources() {
  local source
  while IFS= read -r source; do
    [[ -n "$source" ]] || continue
    echo "[$(date -Is)] downloading source: $source"
    run_with_retries run_data_cli data-download --data-root "$DATA_ROOT" --source "$source"
  done < <(active_sources)
}

materialize_selective_subsets() {
  run_with_retries run_data_cli data-download-weblinx-subset \
    --data-root "$DATA_ROOT" --target-rows 7000
  run_with_retries run_data_cli data-download-gui-odyssey-subset \
    --data-root "$DATA_ROOT" --target-rows 8500
}

write_and_verify_inventory() {
  local inventory_tmp="$DATA_STATE_ROOT/final-inventory.json.tmp"
  run_data_cli data-inventory --data-root "$DATA_ROOT" >"$inventory_tmp"
  mv "$inventory_tmp" "$DATA_STATE_ROOT/final-inventory.json"
  "$DATA_VENV_PYTHON" - "$repo_root/configs/data/sources.json" \
    "$DATA_STATE_ROOT/downloads" <<'PY'
import json
import sys
from pathlib import Path

catalog = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {row["id"] for row in catalog["sources"] if row.get("active", True)}
actual = {path.stem for path in Path(sys.argv[2]).glob("*.json")}
missing = sorted(expected - actual)
if missing:
    raise SystemExit(f"missing completed source states: {missing}")
print(f"verified {len(expected)} completed source states")
PY
}

main() {
  parse_args "$@"
  init_data_pipeline "$repo_root" "$data_root"
  if [[ "$detach" == true ]]; then
    start_detached
    return
  fi
  acquire_lock
  trap cleanup EXIT
  trap 'exit 130' INT TERM
  cd "$repo_root"
  ensure_base_runtime
  prepare_gqa_questions
  prepare_gqa_images
  download_catalog_sources
  materialize_selective_subsets
  write_and_verify_inventory
  if [[ "$prepare" == true ]]; then
    "$repo_root/scripts/prepare_all_data.sh" --data-root "$DATA_ROOT"
  fi
}

main "$@"
