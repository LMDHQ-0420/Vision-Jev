"""Command line interface for static RLCD showcase comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from showcase.schema import ShowcaseConfig


def _config(args: argparse.Namespace) -> ShowcaseConfig:
    return ShowcaseConfig.load(args.examples, args.models)


def validate(args: argparse.Namespace) -> int:
    config = _config(args)
    from showcase.evaluation import verify_showcase_selection
    from showcase.static import load_test_samples, selected_sample, sha256_file

    if sha256_file(args.manifest) != config.manifest_sha256:
        raise ValueError("static showcase manifest hash does not match the frozen config")
    samples = load_test_samples(args.manifest)
    for example in config.examples:
        selected_sample(example, samples)
    selection = verify_showcase_selection(
        config.examples,
        args.manifest,
        {
            "0.8B": args.data_root / "runs/qwen35-08b-rlcd-main-72k/evaluation",
            "9B": args.data_root / "runs/qwen35-9b-rlcd-main-72k/evaluation",
        },
    )
    print(
        json.dumps(
            {
                "examples": [item.id for item in config.examples],
                "models": [item.id for item in config.models],
                "manifest_sha256": config.manifest_sha256,
                "selection_rule": config.selection_rule,
                "selection": selection,
            },
            indent=2,
        )
    )
    return 0


def record(args: argparse.Namespace) -> int:
    from showcase.models import load_adapter
    from showcase.static import load_test_samples, record_prediction, selected_sample

    config = _config(args)
    example = next((item for item in config.examples if item.id == args.example), None)
    model = next((item for item in config.models if item.id == args.model), None)
    if example is None:
        raise ValueError(f"unknown example: {args.example}")
    if model is None:
        raise ValueError(f"unknown model: {args.model}")
    sample = selected_sample(example, load_test_samples(args.manifest))
    adapter = load_adapter(model, args.model_root)
    try:
        path = record_prediction(example, model, sample, adapter, args.output)
    finally:
        adapter.close()
    print(path)
    return 0


def render(args: argparse.Namespace) -> int:
    from showcase.render import render_static_comparison

    report = render_static_comparison(
        args.baseline,
        args.trained,
        args.output,
        frame_duration_ms=args.frame_duration_ms,
    )
    print(json.dumps(report, indent=2))
    return 0


def build_report(args: argparse.Namespace) -> int:
    from showcase.evaluation import build_static_test_report

    report = build_static_test_report(
        args.manifest,
        {
            "0.8B": args.data_root / "runs/qwen35-08b-rlcd-main-72k/evaluation",
            "9B": args.data_root / "runs/qwen35-9b-rlcd-main-72k/evaluation",
        },
        args.output,
    )
    print(json.dumps(report, indent=2))
    return 0


def readme(args: argparse.Namespace) -> int:
    from showcase.readme import build_fragment

    build_fragment(_config(args), args.asset_root, args.test_report, args.output)
    print(args.output)
    return 0


def publish(args: argparse.Namespace) -> int:
    from showcase.readme import publish_readme

    publish_readme(_config(args), args.asset_root, args.test_report, args.readme)
    print(args.readme)
    return 0


def run_all(args: argparse.Namespace) -> int:
    from showcase.models import load_adapter
    from showcase.render import render_static_comparison
    from showcase.static import load_test_samples, record_prediction, selected_sample

    config = _config(args)
    examples = [
        example for example in config.examples if not args.example or example.id in args.example
    ]
    models = [
        model
        for model in config.models
        if not args.parameter_group or model.parameter_group in args.parameter_group
    ]
    if not examples:
        raise ValueError("no examples matched the requested filters")
    if not models:
        raise ValueError("no models matched the requested parameter groups")
    samples = load_test_samples(args.manifest)
    predictions: dict[tuple[str, str], Path] = {}
    for model in models:
        pending = []
        for example in examples:
            path = args.output / example.id / model.id / "prediction.json"
            predictions[(example.id, model.id)] = path
            if not path.is_file():
                pending.append(example)
        if not pending:
            print(f"skip {model.id}: all predictions already exist", flush=True)
            continue
        adapter = load_adapter(model, args.model_root)
        try:
            for example in pending:
                sample = selected_sample(example, samples)
                path = record_prediction(example, model, sample, adapter, args.output)
                print(f"recorded {path}", flush=True)
        finally:
            adapter.close()

    reports = []
    groups = sorted({model.parameter_group for model in models})
    for group in groups:
        members = [model for model in models if model.parameter_group == group]
        if {model.role for model in members} != {"baseline", "trained"}:
            raise ValueError(f"filtered parameter group {group!r} is incomplete")
        baseline = next(model for model in members if model.role == "baseline")
        trained = next(model for model in members if model.role == "trained")
        for example in examples:
            destination = args.asset_output / example.id / f"{group.lower()}.gif"
            report = render_static_comparison(
                predictions[(example.id, baseline.id)],
                predictions[(example.id, trained.id)],
                destination,
                frame_duration_ms=config.frame_duration_ms,
            )
            reports.append(report)
            print(f"rendered {destination}", flush=True)
    print(json.dumps(reports, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--examples", type=Path, default=Path("showcase/configs/examples.json")
    )
    result.add_argument(
        "--models", type=Path, default=Path("showcase/configs/models.example.json")
    )
    commands = result.add_subparsers(required=True)

    shared_manifest = Path("/data/vision-jev/manifests/rlcd/base-72k.jsonl")
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--manifest", type=Path, default=shared_manifest)
    validate_parser.add_argument("--data-root", type=Path, default=Path("/data/vision-jev"))
    validate_parser.set_defaults(func=validate)

    record_parser = commands.add_parser("record")
    record_parser.add_argument("--example", required=True)
    record_parser.add_argument("--model", required=True)
    record_parser.add_argument("--manifest", type=Path, default=shared_manifest)
    record_parser.add_argument("--model-root", type=Path, default=Path("/data/vision-jev/models"))
    record_parser.add_argument("--output", type=Path, default=Path("showcase/output"))
    record_parser.set_defaults(func=record)

    render_parser = commands.add_parser("render")
    render_parser.add_argument("--baseline", type=Path, required=True)
    render_parser.add_argument("--trained", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.add_argument("--frame-duration-ms", type=int, default=1100)
    render_parser.set_defaults(func=render)

    report_parser = commands.add_parser("build-test-report")
    report_parser.add_argument("--manifest", type=Path, default=shared_manifest)
    report_parser.add_argument("--data-root", type=Path, default=Path("/data/vision-jev"))
    report_parser.add_argument(
        "--output", type=Path, default=Path("showcase/output/static-test-results.json")
    )
    report_parser.set_defaults(func=build_report)

    readme_parser = commands.add_parser("readme")
    readme_parser.add_argument("--asset-root", type=Path, default=Path("asset/demos"))
    readme_parser.add_argument(
        "--test-report", type=Path, default=Path("showcase/output/static-test-results.json")
    )
    readme_parser.add_argument(
        "--output", type=Path, default=Path("showcase/output/README-demos.md")
    )
    readme_parser.set_defaults(func=readme)

    publish_parser = commands.add_parser("publish-readme")
    publish_parser.add_argument("--asset-root", type=Path, default=Path("asset/demos"))
    publish_parser.add_argument(
        "--test-report", type=Path, default=Path("showcase/output/static-test-results.json")
    )
    publish_parser.add_argument("--readme", type=Path, default=Path("README.md"))
    publish_parser.set_defaults(func=publish)

    all_parser = commands.add_parser("run-all")
    all_parser.add_argument("--example", action="append")
    all_parser.add_argument("--parameter-group", action="append")
    all_parser.add_argument("--manifest", type=Path, default=shared_manifest)
    all_parser.add_argument("--model-root", type=Path, default=Path("/data/vision-jev/models"))
    all_parser.add_argument("--output", type=Path, default=Path("showcase/output"))
    all_parser.add_argument("--asset-output", type=Path, default=Path("asset/demos"))
    all_parser.set_defaults(func=run_all)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
