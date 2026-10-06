"""Generate the README static RLCD comparison section."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from showcase.schema import ShowcaseConfig

START_MARKER = "<!-- showcase:start -->"
END_MARKER = "<!-- showcase:end -->"


def _percent(value: Any) -> str:
    return f"{100 * float(value):.1f}%"


def fragment_text(config: ShowcaseConfig, asset_root: Path, test_report: Path) -> str:
    report = json.loads(test_report.read_text(encoding="utf-8"))
    groups = sorted({model.parameter_group for model in config.models})
    lines = [
        "## Static RLCD results",
        "",
        "The released checkpoints are evaluated on the complete frozen 12,000-question "
        "static RLCD test split. These aggregate results include every success and failure.",
        "",
        "| Model | Choice accuracy | Noul accuracy | Score accuracy | "
        "Choice accepted accuracy | Noul accepted accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in groups:
        summary = report["models"][group]["summary"]
        tasks = summary["by_task"]
        policy = summary["threshold_policy"]
        lines.append(
            f"| Vision-Jev-{group} | {_percent(tasks['choice']['accuracy'])} | "
            f"{_percent(tasks['noul']['accuracy'])} | {_percent(tasks['score']['accuracy'])} | "
            f"{_percent(policy['choice']['accuracy'])} at "
            f"{_percent(policy['choice']['coverage'])} coverage | "
            f"{_percent(policy['noul']['accuracy'])} at "
            f"{_percent(policy['noul']['coverage'])} coverage |"
        )
    lines.extend(
        [
            "",
            "## Frozen test examples",
            "",
            "Each animation uses one identical frozen test sample and candidate set for the "
            "original Qwen3.5 checkpoint and Vision-Jev. Categories were declared first. "
            "Within each category, the sample is the minimum SHA-256 sample ID for which both "
            "Vision-Jev checkpoints are correct and pass their already-frozen confidence "
            "threshold; baseline predictions were not used for selection.",
            "",
        ]
    )
    for example in config.examples:
        lines.extend([f"### {example.title}", "", example.description, ""])
        lines.append("| " + " | ".join(groups) + " |")
        lines.append("| " + " | ".join("---" for _ in groups) + " |")
        cells = []
        for group in groups:
            relative = asset_root / example.id / f"{group.lower()}.gif"
            if not relative.is_file():
                raise FileNotFoundError(f"missing showcase GIF: {relative}")
            cells.append(
                f'<img src="{relative.as_posix()}" alt="{example.title}, {group}: '
                'original Qwen3.5 versus Vision-Jev" width="560">'
            )
        lines.extend(["| " + " | ".join(cells) + " |", ""])
    return "\n".join(lines).rstrip() + "\n"


def build_fragment(
    config: ShowcaseConfig, asset_root: Path, test_report: Path, destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(fragment_text(config, asset_root, test_report), encoding="utf-8")


def publish_readme(
    config: ShowcaseConfig, asset_root: Path, test_report: Path, readme_path: Path
) -> None:
    current = readme_path.read_text(encoding="utf-8")
    if current.count(START_MARKER) != 1 or current.count(END_MARKER) != 1:
        raise ValueError(f"{readme_path} must contain one showcase marker pair")
    before, remainder = current.split(START_MARKER, 1)
    _, after = remainder.split(END_MARKER, 1)
    updated = (
        before.rstrip()
        + "\n\n"
        + START_MARKER
        + "\n"
        + fragment_text(config, asset_root, test_report)
        + END_MARKER
        + after
    )
    readme_path.write_text(updated, encoding="utf-8")
