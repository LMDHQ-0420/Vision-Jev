"""Command line entry points for repository governance and data checks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from vision_jev.api_keys import DEFAULT_API_KEYS_PATH, load_api_keys
from vision_jev.data import DataValidationError, validate_jsonl
from vision_jev.data.api_rewrite import (
    audit_rewrites,
    generate_rewrites,
    select_parents,
)
from vision_jev.data.build import build_final_manifest, build_public_manifest
from vision_jev.data.download import (
    download_source,
    inventory,
    load_catalog,
    materialize_gui_odyssey_subset,
    materialize_weblinx_subset,
)
from vision_jev.data.pipeline import ADAPTERS, normalize_source
from vision_jev.tracking import create_run, finalize_run


def _root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "vision_jev").exists():
            return candidate
    raise RuntimeError("run inside the Vision-JEV repository")


def doctor(_: argparse.Namespace) -> int:
    checks = {
        "python>=3.11": sys.version_info >= (3, 11),
        "repository": (_root() / "pyproject.toml").exists(),
    }
    environment_note = {
        "conda_available": bool(shutil.which("conda")),
        "expected_training_environment": "vision-jev",
        "active_conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    print(json.dumps({"checks": checks, "environment": environment_note}, indent=2))
    return 0 if all(checks.values()) else 1


def validate_data(args: argparse.Namespace) -> int:
    try:
        report = validate_jsonl(args.path)
    except (OSError, DataValidationError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
    return 0


def init_run(args: argparse.Namespace) -> int:
    root = _root()
    validate_jsonl(args.data)
    run_dir = create_run(root, Path(args.config), Path(args.data), args.run_id)
    print(run_dir)
    return 0


def finish_run(args: argparse.Namespace) -> int:
    finalize_run(Path(args.run_dir), args.status, args.summary, args.evidence_status)
    print(Path(args.run_dir) / "run.json")
    return 0


def data_download(args: argparse.Namespace) -> int:
    root = _root()
    catalog = load_catalog(root / args.catalog)
    selected = args.source or [key for key, value in catalog.items() if value.get("active", True)]
    unknown = sorted(set(selected) - set(catalog))
    if unknown:
        raise ValueError(f"unknown data sources: {unknown}")
    for source_id in selected:
        download_source(catalog[source_id], args.data_root, extract=not args.no_extract)
    print(json.dumps(inventory(args.data_root), ensure_ascii=False, indent=2))
    return 0


def data_inventory(args: argparse.Namespace) -> int:
    print(json.dumps(inventory(args.data_root), ensure_ascii=False, indent=2))
    return 0


def data_download_weblinx_subset(args: argparse.Namespace) -> int:
    report = materialize_weblinx_subset(
        args.data_root, target_rows=args.target_rows, seed=args.seed
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def data_download_gui_odyssey_subset(args: argparse.Namespace) -> int:
    report = materialize_gui_odyssey_subset(
        args.data_root,
        target_rows=args.target_rows,
        seed=args.seed,
        max_workers=args.max_workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def data_normalize(args: argparse.Namespace) -> int:
    destination = args.output or args.data_root / "processed" / args.source / "canonical.jsonl"
    count = normalize_source(args.source, args.data_root, destination)
    report = validate_jsonl(destination, check_assets=True)
    if report.questions != count:
        raise RuntimeError(f"written {count} but validator counted {report.questions}")
    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
    return 0


def data_build_public(args: argparse.Namespace) -> int:
    try:
        report = build_public_manifest(
            data_root=args.data_root,
            mixture_config=args.mixture,
            destination=args.output,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"manifest build blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def data_rewrite_api(args: argparse.Namespace) -> int:
    if args.minimum_choice is not None and args.minimum_choice > args.choice:
        raise ValueError("minimum-choice cannot exceed choice candidates")
    if args.minimum_noul is not None and args.minimum_noul > args.noul:
        raise ValueError("minimum-noul cannot exceed noul candidates")
    if args.max_runtime_hours <= 0 or args.max_runtime_hours >= 5:
        raise ValueError("max-runtime-hours must be greater than 0 and less than 5")
    if args.min_interval_seconds < 20:
        raise ValueError("min-interval-seconds must be at least 20 for the Kimi 3 RPM limit")
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be positive")
    keys = load_api_keys(args.api_keys, required_providers=("kimi",))
    parents = select_parents(args.input, choice=args.choice, noul=args.noul, seed=args.seed)
    report = generate_rewrites(
        parents,
        keys=keys,
        destination=args.output,
        provider=args.provider,
        max_attempts=args.max_attempts,
        max_workers=args.max_workers,
        group_size=args.group_size,
        min_interval_seconds=args.min_interval_seconds,
        backoff_base_seconds=args.backoff_base_seconds,
        backoff_cap_seconds=args.backoff_cap_seconds,
        max_runtime_seconds=args.max_runtime_hours * 60 * 60,
    )
    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
    task_counts: Counter[str] = Counter()
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            for raw in handle:
                if raw.strip():
                    task_counts[str(json.loads(raw)["task_type"])] += 1
    minimum_choice = args.minimum_choice if args.minimum_choice is not None else args.choice
    minimum_noul = args.minimum_noul if args.minimum_noul is not None else args.noul
    return (
        0 if task_counts["choice"] >= minimum_choice and task_counts["noul"] >= minimum_noul else 2
    )


def data_build_final(args: argparse.Namespace) -> int:
    try:
        report = build_final_manifest(
            public_manifest=args.public,
            api_candidates=args.api_candidates,
            mixture_config=args.mixture,
            destination=args.output,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"final manifest build blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def data_audit_rewrites(args: argparse.Namespace) -> int:
    report = audit_rewrites(args.candidates, args.parents)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["invariant_violations"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vision-jev")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor_parser = sub.add_parser("doctor", help="check the lightweight local setup")
    doctor_parser.set_defaults(func=doctor)
    validate_parser = sub.add_parser("validate-data", help="validate complete-question JSONL")
    validate_parser.add_argument("path", type=Path)
    validate_parser.set_defaults(func=validate_data)
    init_parser = sub.add_parser("init-run", help="create an immutable run record")
    init_parser.add_argument("--config", type=Path, required=True)
    init_parser.add_argument("--data", type=Path, required=True)
    init_parser.add_argument("--run-id")
    init_parser.set_defaults(func=init_run)
    finish_parser = sub.add_parser("finalize-run", help="finalize an existing run")
    finish_parser.add_argument("run_dir", type=Path)
    finish_parser.add_argument(
        "--status", choices=["completed", "failed", "aborted"], required=True
    )
    finish_parser.add_argument(
        "--evidence-status",
        choices=["structural_check", "measured", "reproduced", "released"],
        required=True,
    )
    finish_parser.add_argument("--summary", required=True)
    finish_parser.set_defaults(func=finish_run)
    download_parser = sub.add_parser("data-download", help="download fixed dataset artifacts")
    download_parser.add_argument(
        "--data-root", type=Path, default=Path("/mnt/sda1/sol_data/vision-jev")
    )
    download_parser.add_argument("--catalog", type=Path, default=Path("configs/data/sources.json"))
    download_parser.add_argument("--source", action="append")
    download_parser.add_argument("--no-extract", action="store_true")
    download_parser.set_defaults(func=data_download)
    inventory_parser = sub.add_parser("data-inventory", help="show downloaded dataset state")
    inventory_parser.add_argument(
        "--data-root", type=Path, default=Path("/mnt/sda1/sol_data/vision-jev")
    )
    inventory_parser.set_defaults(func=data_inventory)
    weblinx_parser = sub.add_parser(
        "data-download-weblinx-subset",
        help="materialize a deterministic train-only WebLINX screenshot subset",
    )
    weblinx_parser.add_argument(
        "--data-root", type=Path, default=Path("/mnt/sda1/sol_data/vision-jev")
    )
    weblinx_parser.add_argument("--target-rows", type=int, default=7000)
    weblinx_parser.add_argument("--seed", default="vision-jev-sft-v2")
    weblinx_parser.set_defaults(func=data_download_weblinx_subset)
    gui_parser = sub.add_parser(
        "data-download-gui-odyssey-subset",
        help="materialize a deterministic train-only GUI-Odyssey screenshot subset",
    )
    gui_parser.add_argument("--data-root", type=Path, default=Path("/mnt/sda1/sol_data/vision-jev"))
    gui_parser.add_argument("--target-rows", type=int, default=8500)
    gui_parser.add_argument("--seed", default="vision-jev-sft")
    gui_parser.add_argument("--max-workers", type=int, default=8)
    gui_parser.set_defaults(func=data_download_gui_odyssey_subset)
    normalize_parser = sub.add_parser("data-normalize", help="convert raw data to canonical JSONL")
    normalize_parser.add_argument("source", choices=sorted(ADAPTERS))
    normalize_parser.add_argument(
        "--data-root", type=Path, default=Path("/mnt/sda1/sol_data/vision-jev")
    )
    normalize_parser.add_argument("--output", type=Path)
    normalize_parser.set_defaults(func=data_normalize)
    build_parser = sub.add_parser(
        "data-build-public", help="build the configured public-data manifest or report shortages"
    )
    build_parser.add_argument(
        "--data-root", type=Path, default=Path("/mnt/sda1/sol_data/vision-jev")
    )
    build_parser.add_argument("--mixture", type=Path, default=Path("configs/data/sft_120k.json"))
    build_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/sda1/sol_data/vision-jev/manifests/public-117k.jsonl"),
    )
    build_parser.add_argument("--seed", default="vision-jev-sft")
    build_parser.set_defaults(func=data_build_public)
    rewrite_parser = sub.add_parser(
        "data-rewrite-api", help="create checked same-language API question rewrites"
    )
    rewrite_parser.add_argument("--input", type=Path, required=True)
    rewrite_parser.add_argument("--output", type=Path, required=True)
    rewrite_parser.add_argument("--choice", type=int, required=True)
    rewrite_parser.add_argument("--noul", type=int, required=True)
    rewrite_parser.add_argument("--minimum-choice", type=int)
    rewrite_parser.add_argument("--minimum-noul", type=int)
    rewrite_parser.add_argument("--provider", choices=["kimi"], default="kimi")
    rewrite_parser.add_argument("--max-attempts", type=int, default=5)
    rewrite_parser.add_argument("--max-workers", type=int, choices=[1], default=1)
    rewrite_parser.add_argument("--group-size", type=int, default=60)
    rewrite_parser.add_argument("--min-interval-seconds", type=float, default=21.0)
    rewrite_parser.add_argument("--backoff-base-seconds", type=float, default=2.0)
    rewrite_parser.add_argument("--backoff-cap-seconds", type=float, default=60.0)
    rewrite_parser.add_argument("--max-runtime-hours", type=float, default=4.75)
    rewrite_parser.add_argument("--seed", default="vision-jev-api-rewrite")
    rewrite_parser.add_argument("--api-keys", type=Path, default=DEFAULT_API_KEYS_PATH)
    rewrite_parser.set_defaults(func=data_rewrite_api)
    final_parser = sub.add_parser(
        "data-build-final", help="join exact public and API-assisted quotas"
    )
    final_parser.add_argument("--public", type=Path, required=True)
    final_parser.add_argument("--api-candidates", type=Path, required=True)
    final_parser.add_argument("--mixture", type=Path, default=Path("configs/data/sft_120k.json"))
    final_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/sda1/sol_data/vision-jev/manifests/sft-120k.jsonl"),
    )
    final_parser.add_argument("--seed", default="vision-jev-sft")
    final_parser.set_defaults(func=data_build_final)
    audit_parser = sub.add_parser(
        "data-audit-rewrites", help="verify every API rewrite against its public parent"
    )
    audit_parser.add_argument("--candidates", type=Path, required=True)
    audit_parser.add_argument("--parents", type=Path, required=True)
    audit_parser.set_defaults(func=data_audit_rewrites)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
