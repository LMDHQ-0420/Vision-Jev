#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/data_pipeline_common.sh
source "$repo_root/scripts/lib/data_pipeline_common.sh"

data_root=""

usage() {
  cat <<'EOF'
Usage: scripts/prepare_all_data.sh --data-root PATH

Normalizes every static public source, validates/indexes interactive assets,
builds the public 117k manifest and group-safe training split, and records the
complete public-only preprocessing state.
EOF
}

while (($#)); do
  case "$1" in
    --data-root)
      [[ $# -ge 2 ]] || { echo "--data-root needs a path" >&2; exit 2; }
      data_root="$2"
      shift 2
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

init_data_pipeline "$repo_root" "$data_root"
exec 8>"$DATA_STATE_ROOT/prepare-all.lock"
flock -n 8 || { echo "another preprocessing process holds the data lock" >&2; exit 1; }
cd "$repo_root"
ensure_prepare_runtime

mapfile -t static_sources < <(
  "$DATA_VENV_PYTHON" -c \
    'from vision_jev.data.pipeline import ADAPTERS; print("\n".join(sorted(ADAPTERS)))'
)
for source in "${static_sources[@]}"; do
  echo "[$(date -Is)] normalizing source: $source"
  run_data_cli data-normalize "$source" --data-root "$DATA_ROOT"
done

echo "[$(date -Is)] indexing interactive assets"
run_data_cli data-prepare-interactive --data-root "$DATA_ROOT"

manifest_root="$DATA_ROOT/manifests"
mkdir -p "$manifest_root"
public_manifest="$manifest_root/public-117k.jsonl"
training_manifest="$manifest_root/public-117k-training.jsonl"
run_data_cli data-build-public --data-root "$DATA_ROOT" --output "$public_manifest"
run_data_cli data-build-training \
  --input "$public_manifest" --output "$training_manifest" --eval-percent 5

rlcd_root="$manifest_root/rlcd"
run_data_cli data-build-rlcd \
  --data-root "$DATA_ROOT" \
  --config "$repo_root/configs/data/rlcd_72k.json" \
  --sft-manifest "$training_manifest" \
  --output "$rlcd_root/base-72k.jsonl" \
  --views-output "$rlcd_root/train-views.jsonl"

"$repo_root/scripts/generate_interactive_data.sh" --data-root "$DATA_ROOT"

cat >"$DATA_STATE_ROOT/preprocessing.json" <<EOF
{
  "schema_version": 1,
  "status": "complete",
  "public_manifest": "$public_manifest",
  "training_manifest": "$training_manifest",
  "rlcd_base_manifest": "$rlcd_root/base-72k.jsonl",
  "rlcd_training_views": "$rlcd_root/train-views.jsonl",
  "interactive_index": "$DATA_ROOT/processed/interactive/environment-index.json",
  "boxoban_levels": "$DATA_ROOT/processed/interactive/boxoban-levels.jsonl",
  "interactive_manifest": "$DATA_ROOT/manifests/rlcd-interactive/base-16k.jsonl",
  "interactive_oracle_report": "$DATA_ROOT/manifests/rlcd-interactive/oracle-report.json"
}
EOF
echo "[$(date -Is)] all preprocessing complete"
