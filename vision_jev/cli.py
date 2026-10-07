"""Command line entry points for repository governance and data checks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from vision_jev.data import DataValidationError, validate_jsonl
from vision_jev.data.boxoban_oracle import generate_boxoban
from vision_jev.data.build import build_public_manifest
from vision_jev.data.download import (
    download_source,
    inventory,
    load_catalog,
    materialize_gui_odyssey_subset,
    materialize_weblinx_subset,
)
from vision_jev.data.dynamic_obstacles import generate_dynamic_obstacles
from vision_jev.data.interactive import prepare_interactive_assets
from vision_jev.data.interactive_finalize import finalize_interactive
from vision_jev.data.interactive_generate import generate_minigrid_navigation
from vision_jev.data.minigrid_oracles import STAGE_SPECS, generate_minigrid_stage
from vision_jev.data.pilot import build_pilot_manifest, build_training_manifest
from vision_jev.data.pipeline import ADAPTERS, normalize_source
from vision_jev.data.rlcd import build_rlcd_manifest, build_rlcd_training_views
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


def data_prepare_interactive(args: argparse.Namespace) -> int:
    root = _root()
    report = prepare_interactive_assets(
        args.data_root,
        root / args.catalog,
        args.output or args.data_root / "processed" / "interactive",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def data_generate_minigrid_navigation(args: argparse.Namespace) -> int:
    report = generate_minigrid_navigation(
        destination=args.output,
        image_root=args.image_root,
        questions=args.questions,
    )
    validation = validate_jsonl(args.output, check_assets=True)
    if validation.questions != report["questions"]:
        raise RuntimeError("interactive generator and validator counts disagree")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def data_generate_minigrid_stage(args: argparse.Namespace) -> int:
    report = generate_minigrid_stage(
        stage=args.stage,
        destination=args.output,
        image_root=args.image_root,
        max_workers=args.max_workers,
    )
    validation = validate_jsonl(args.output, check_assets=True)
    if validation.questions != report["questions"]:
        raise RuntimeError("interactive generator and validator counts disagree")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def data_generate_boxoban(args: argparse.Namespace) -> int:
    report = generate_boxoban(
        level_index=args.level_index,
        destination=args.output,
        image_root=args.image_root,
        max_workers=args.max_workers,
    )
    validation = validate_jsonl(args.output, check_assets=True)
    if validation.questions != report["questions"]:
        raise RuntimeError("Boxoban generator and validator counts disagree")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def data_generate_dynamic_obstacles(args: argparse.Namespace) -> int:
    report = generate_dynamic_obstacles(
        destination=args.output,
        image_root=args.image_root,
        max_workers=args.max_workers,
    )
    validation = validate_jsonl(args.output, check_assets=True)
    if validation.questions != report["questions"]:
        raise RuntimeError("dynamic generator and validator counts disagree")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def data_finalize_interactive(args: argparse.Namespace) -> int:
    stage_root = args.data_root / "manifests" / "rlcd-interactive" / "stages"
    report = finalize_interactive(
        stage_paths={
            "minigrid_navigation": stage_root / "minigrid-navigation-4k.jsonl",
            "minigrid_tools": stage_root / "minigrid-tools-3k.jsonl",
            "minigrid_hazards_static": stage_root / "minigrid-hazards-static-2250.jsonl",
            "minigrid_dynamic_obstacles": stage_root / "minigrid-dynamic-obstacles-250.jsonl",
            "babyai_grounded": stage_root / "babyai-grounded-2500.jsonl",
            "procgen_maze": stage_root / "procgen-maze-2k.jsonl",
            "boxoban": stage_root / "boxoban-2k.jsonl",
        },
        destination=args.output
        or args.data_root / "manifests" / "rlcd-interactive" / "base-16k.jsonl",
        report_root=args.data_root / "manifests" / "rlcd-interactive",
    )
    validation = validate_jsonl(report["manifest"], check_assets=True)
    if validation.questions != report["questions"]:
        raise RuntimeError("interactive finalizer and validator counts disagree")
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


def data_build_pilot(args: argparse.Namespace) -> int:
    report = build_pilot_manifest(
        args.input,
        args.output,
        questions=args.questions,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def data_build_training(args: argparse.Namespace) -> int:
    report = build_training_manifest(
        args.input,
        args.output,
        seed=args.seed,
        eval_percent=args.eval_percent,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def data_build_rlcd(args: argparse.Namespace) -> int:
    report = build_rlcd_manifest(
        data_root=args.data_root,
        config_path=args.config,
        sft_manifests=args.sft_manifest,
        destination=args.output,
    )
    views = build_rlcd_training_views(
        args.output,
        args.views_output,
        seed=str(report["seed"]),
    )
    print(json.dumps({"base": report, "training_views": views}, ensure_ascii=False, indent=2))
    return 0


def model_prepare(args: argparse.Namespace) -> int:
    from vision_jev.model.qwen35 import prepare_snapshot

    destination = prepare_snapshot(args.config, args.model_root)
    print(destination)
    return 0


def train_sft_command(args: argparse.Namespace) -> int:
    from vision_jev.train.sft import train_sft

    summary = train_sft(
        args.config,
        args.data,
        args.output,
        model_root=args.model_root,
        resume_from=args.resume_from,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def eval_sft_command(args: argparse.Namespace) -> int:
    from vision_jev.train.sft import evaluate_checkpoint

    report = evaluate_checkpoint(
        args.config,
        args.data,
        args.checkpoint,
        args.output,
        model_root=args.model_root,
        maximum=args.maximum or None,
        progress_every=args.progress_every,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def train_rlcd_command(args: argparse.Namespace) -> int:
    from vision_jev.train.rlcd import train_rlcd

    summary = train_rlcd(
        args.config,
        args.output,
        model_root=args.model_root,
        sft_checkpoint=args.sft_checkpoint,
        train_manifest=args.train_manifest,
        role_manifest=args.role_manifest,
    )
    if summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def eval_rlcd_command(args: argparse.Namespace) -> int:
    from vision_jev.train.rlcd import evaluate_rlcd_checkpoint

    report = evaluate_rlcd_checkpoint(
        args.config,
        args.checkpoint,
        args.role,
        args.output,
        model_root=args.model_root,
        role_manifest=args.role_manifest,
        sft_checkpoint=args.sft_checkpoint,
        maximum=args.maximum or None,
        progress_every=args.progress_every,
        thresholds_path=args.thresholds,
        select_thresholds_path=args.select_thresholds,
        target_accuracy=args.target_accuracy,
        minimum_accepted=args.minimum_accepted,
    )
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
    download_parser.add_argument("--data-root", type=Path, default=Path("/data/vision-jev"))
    download_parser.add_argument("--catalog", type=Path, default=Path("configs/data/sources.json"))
    download_parser.add_argument("--source", action="append")
    download_parser.add_argument("--no-extract", action="store_true")
    download_parser.set_defaults(func=data_download)
    inventory_parser = sub.add_parser("data-inventory", help="show downloaded dataset state")
    inventory_parser.add_argument("--data-root", type=Path, default=Path("/data/vision-jev"))
    inventory_parser.set_defaults(func=data_inventory)
    weblinx_parser = sub.add_parser(
        "data-download-weblinx-subset",
        help="materialize a deterministic train-only WebLINX screenshot subset",
    )
    weblinx_parser.add_argument("--data-root", type=Path, default=Path("/data/vision-jev"))
    weblinx_parser.add_argument("--target-rows", type=int, default=7000)
    weblinx_parser.add_argument("--seed", default="vision-jev-sft-v2")
    weblinx_parser.set_defaults(func=data_download_weblinx_subset)
    gui_parser = sub.add_parser(
        "data-download-gui-odyssey-subset",
        help="materialize a deterministic train-only GUI-Odyssey screenshot subset",
    )
    gui_parser.add_argument("--data-root", type=Path, default=Path("/data/vision-jev"))
    gui_parser.add_argument("--target-rows", type=int, default=8500)
    gui_parser.add_argument("--seed", default="vision-jev-sft")
    gui_parser.add_argument("--max-workers", type=int, default=8)
    gui_parser.set_defaults(func=data_download_gui_odyssey_subset)
    interactive_parser = sub.add_parser(
        "data-prepare-interactive",
        help="validate pinned environments and index interactive level assets",
    )
    interactive_parser.add_argument("--data-root", type=Path, default=Path("/data/vision-jev"))
    interactive_parser.add_argument(
        "--catalog", type=Path, default=Path("configs/data/sources.json")
    )
    interactive_parser.add_argument("--output", type=Path)
    interactive_parser.set_defaults(func=data_prepare_interactive)
    navigation_parser = sub.add_parser(
        "data-generate-minigrid-navigation",
        help="generate exact-BFS MiniGrid navigation decisions without a model",
    )
    navigation_parser.add_argument("--output", type=Path, required=True)
    navigation_parser.add_argument("--image-root", type=Path, required=True)
    navigation_parser.add_argument("--questions", type=int, default=4000)
    navigation_parser.set_defaults(func=data_generate_minigrid_navigation)
    stage_parser = sub.add_parser(
        "data-generate-minigrid-stage",
        help="generate a pinned MiniGrid/BabyAI stage with exact state-search labels",
    )
    stage_parser.add_argument("stage", choices=sorted(STAGE_SPECS))
    stage_parser.add_argument("--output", type=Path, required=True)
    stage_parser.add_argument("--image-root", type=Path, required=True)
    stage_parser.add_argument("--max-workers", type=int, default=32)
    stage_parser.set_defaults(func=data_generate_minigrid_stage)
    boxoban_parser = sub.add_parser(
        "data-generate-boxoban", help="generate exact-A* decisions for the pinned Boxoban levels"
    )
    boxoban_parser.add_argument("--level-index", type=Path, required=True)
    boxoban_parser.add_argument("--output", type=Path, required=True)
    boxoban_parser.add_argument("--image-root", type=Path, required=True)
    boxoban_parser.add_argument("--max-workers", type=int, default=32)
    boxoban_parser.set_defaults(func=data_generate_boxoban)
    dynamic_parser = sub.add_parser(
        "data-generate-dynamic-obstacles",
        help="generate Monte Carlo policy decisions for stochastic dynamic obstacles",
    )
    dynamic_parser.add_argument("--output", type=Path, required=True)
    dynamic_parser.add_argument("--image-root", type=Path, required=True)
    dynamic_parser.add_argument("--max-workers", type=int, default=24)
    dynamic_parser.set_defaults(func=data_generate_dynamic_obstacles)
    finalize_parser = sub.add_parser(
        "data-finalize-interactive", help="merge and audit all pinned interactive stages"
    )
    finalize_parser.add_argument("--data-root", type=Path, default=Path("/data/vision-jev"))
    finalize_parser.add_argument("--output", type=Path)
    finalize_parser.set_defaults(func=data_finalize_interactive)
    normalize_parser = sub.add_parser("data-normalize", help="convert raw data to canonical JSONL")
    normalize_parser.add_argument("source", choices=sorted(ADAPTERS))
    normalize_parser.add_argument("--data-root", type=Path, default=Path("/data/vision-jev"))
    normalize_parser.add_argument("--output", type=Path)
    normalize_parser.set_defaults(func=data_normalize)
    build_parser = sub.add_parser(
        "data-build-public", help="build the configured public-data manifest or report shortages"
    )
    build_parser.add_argument("--data-root", type=Path, default=Path("/data/vision-jev"))
    build_parser.add_argument("--mixture", type=Path, default=Path("configs/data/sft_117k.json"))
    build_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/vision-jev/manifests/public-117k.jsonl"),
    )
    build_parser.add_argument("--seed", default="vision-jev-sft")
    build_parser.set_defaults(func=data_build_public)
    pilot_parser = sub.add_parser(
        "data-build-pilot", help="build a deterministic source/task-stratified pilot manifest"
    )
    pilot_parser.add_argument("--input", type=Path, required=True)
    pilot_parser.add_argument("--output", type=Path, required=True)
    pilot_parser.add_argument("--questions", type=int, default=12_000)
    pilot_parser.add_argument("--seed", default="vision-jev-pilot-12k")
    pilot_parser.set_defaults(func=data_build_pilot)
    training_data_parser = sub.add_parser(
        "data-build-training",
        help="assign a complete manifest to deterministic group-safe train/eval roles",
    )
    training_data_parser.add_argument("--input", type=Path, required=True)
    training_data_parser.add_argument("--output", type=Path, required=True)
    training_data_parser.add_argument("--eval-percent", type=int, default=5)
    training_data_parser.add_argument("--seed", default="vision-jev-main-117k")
    training_data_parser.set_defaults(func=data_build_training)
    rlcd_parser = sub.add_parser(
        "data-build-rlcd",
        help="freeze group-safe RLCD roots and deterministic train views",
    )
    rlcd_parser.add_argument("--data-root", type=Path, default=Path("/data/vision-jev"))
    rlcd_parser.add_argument("--config", type=Path, default=Path("configs/data/rlcd_72k.json"))
    rlcd_parser.add_argument("--sft-manifest", type=Path, action="append", required=True)
    rlcd_parser.add_argument("--output", type=Path, required=True)
    rlcd_parser.add_argument("--views-output", type=Path, required=True)
    rlcd_parser.set_defaults(func=data_build_rlcd)
    model_parser = sub.add_parser(
        "model-prepare", help="download the pinned Qwen model snapshot explicitly"
    )
    model_parser.add_argument("--config", type=Path, default=Path("configs/model/qwen35_08b.json"))
    model_parser.add_argument("--model-root", type=Path, default=Path("/data/vision-jev/models"))
    model_parser.set_defaults(func=model_prepare)
    train_parser = sub.add_parser("train-sft", help="run answer-only multimodal Qwen SFT")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument(
        "--resume-from",
        type=Path,
        help="resume adapter, optimizer, scheduler, RNG and data cursor from a checkpoint",
    )
    train_parser.add_argument("--model-root", type=Path, default=Path("/data/vision-jev/models"))
    train_parser.set_defaults(func=train_sft_command)
    eval_parser = sub.add_parser(
        "eval-sft", help="evaluate structured answers from a trained SFT adapter"
    )
    eval_parser.add_argument("--config", type=Path, required=True)
    eval_parser.add_argument("--data", type=Path, required=True)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--maximum", type=int, default=0, help="0 evaluates every row")
    eval_parser.add_argument("--progress-every", type=int, default=25)
    eval_parser.add_argument("--model-root", type=Path, default=Path("/data/vision-jev/models"))
    eval_parser.set_defaults(func=eval_sft_command)
    rlcd_train_parser = sub.add_parser(
        "train-rlcd", help="train calibrated decision heads on frozen SFT features"
    )
    rlcd_train_parser.add_argument("--config", type=Path, required=True)
    rlcd_train_parser.add_argument("--output", type=Path, required=True)
    rlcd_train_parser.add_argument("--sft-checkpoint", type=Path)
    rlcd_train_parser.add_argument("--train-manifest", type=Path)
    rlcd_train_parser.add_argument("--role-manifest", type=Path)
    rlcd_train_parser.add_argument(
        "--model-root", type=Path, default=Path("/data/vision-jev/models")
    )
    rlcd_train_parser.set_defaults(func=train_rlcd_command)
    rlcd_eval_parser = sub.add_parser(
        "eval-rlcd", help="evaluate frozen calibrated decision heads by RLCD role"
    )
    rlcd_eval_parser.add_argument("--config", type=Path, required=True)
    rlcd_eval_parser.add_argument("--checkpoint", type=Path, required=True)
    rlcd_eval_parser.add_argument("--role", choices=["threshold", "audit", "test"], required=True)
    rlcd_eval_parser.add_argument("--output", type=Path, required=True)
    rlcd_eval_parser.add_argument("--role-manifest", type=Path)
    rlcd_eval_parser.add_argument("--sft-checkpoint", type=Path)
    rlcd_eval_parser.add_argument("--maximum", type=int, default=0, help="0 evaluates every row")
    rlcd_eval_parser.add_argument("--progress-every", type=int, default=100)
    rlcd_eval_parser.add_argument("--thresholds", type=Path)
    rlcd_eval_parser.add_argument("--select-thresholds", type=Path)
    rlcd_eval_parser.add_argument("--target-accuracy", type=float, default=0.95)
    rlcd_eval_parser.add_argument("--minimum-accepted", type=int, default=25)
    rlcd_eval_parser.add_argument(
        "--model-root", type=Path, default=Path("/data/vision-jev/models")
    )
    rlcd_eval_parser.set_defaults(func=eval_rlcd_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
