"""Generate the README demo section from completed showcase artifacts."""

from __future__ import annotations

from pathlib import Path

from showcase.schema import ShowcaseConfig

START_MARKER = "<!-- showcase:start -->"
END_MARKER = "<!-- showcase:end -->"


def fragment_text(config: ShowcaseConfig, asset_root: Path) -> str:
    groups = sorted({model.parameter_group for model in config.models})
    lines = [
        "## Interactive comparisons",
        "",
        "Each comparison uses the same environment, seed, action set, and step limit. "
        "The left panel is the original Qwen3.5 checkpoint; the right panel is Vision-Jev.",
        "",
    ]
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
                'original Qwen3.5 versus Vision-Jev" width="500">'
            )
        lines.extend(["| " + " | ".join(cells) + " |", ""])
    return "\n".join(lines).rstrip() + "\n"


def build_fragment(config: ShowcaseConfig, asset_root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(fragment_text(config, asset_root), encoding="utf-8")


def publish_readme(config: ShowcaseConfig, asset_root: Path, readme_path: Path) -> None:
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
        + fragment_text(config, asset_root)
        + END_MARKER
        + after
    )
    readme_path.write_text(updated, encoding="utf-8")
