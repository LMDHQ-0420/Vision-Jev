"""Command line entry points for repository governance and data checks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from vision_jev import __version__
from vision_jev.data import DataValidationError, validate_jsonl
from vision_jev.data.build import build_public_manifest
from vision_jev.data.download import (
    download_source,
    inventory,
    load_catalog,
    materialize_weblinx_subset,
)
from vision_jev.data.pipeline import ADAPTERS, normalize_source
from vision_jev.tracking import create_run, finalize_run


def _root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "vision_jev").exists():
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
    print(
        json.dumps(
            {"version": __version__, "checks": checks, "environment": environment_note}, indent=2
        )
    )
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
    normalize_parser = sub.add_parser("data-normalize", help="convert raw data to canonical JSONL")
    normalize_parser.add_argument("source", choices=sorted(ADAPTERS))
    normalize_parser.add_argument(
        "--data-root", type=Path, default=Path("/mnt/sda1/sol_data/vision-jev")
    )
    normalize_parser.add_argument("--output", type=Path)
    normalize_parser.set_defaults(func=data_normalize)
    build_parser = sub.add_parser(
        "data-build-public", help="build the exact v2 public-data manifest or report shortages"
    )
    build_parser.add_argument(
        "--data-root", type=Path, default=Path("/mnt/sda1/sol_data/vision-jev")
    )
    build_parser.add_argument("--mixture", type=Path, default=Path("configs/data/sft_120k.json"))
    build_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/sda1/sol_data/vision-jev/manifests/public-90k-v2.jsonl"),
    )
    build_parser.add_argument("--seed", default="vision-jev-sft-v2")
    build_parser.set_defaults(func=data_build_public)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
