from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from showcase.readme import END_MARKER, START_MARKER, publish_readme
from showcase.render import render_comparison
from showcase.schema import Example, ShowcaseConfig


def test_showcase_config_has_paired_parameter_groups(tmp_path: Path) -> None:
    examples = tmp_path / "examples.json"
    models = tmp_path / "nested" / "configs" / "models.json"
    examples.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "examples": [
                    {
                        "id": "maze",
                        "title": "Maze",
                        "description": "Navigate.",
                        "environment": {
                            "family": "minigrid",
                            "environment_id": "MiniGrid-Empty-5x5-v0",
                            "seed": 1,
                            "max_steps": 4,
                            "actions": ["move_forward"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    models.parent.mkdir(parents=True)
    models.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "base",
                        "label": "Base",
                        "parameter_group": "1B",
                        "role": "baseline",
                        "kind": "qwen_base",
                        "model_config": "base.json",
                    },
                    {
                        "id": "trained",
                        "label": "Trained",
                        "parameter_group": "1B",
                        "role": "trained",
                        "kind": "vision_jev_rlcd",
                        "model_config": "base.json",
                        "sft_checkpoint": "sft",
                        "rlcd_config": "rlcd.json",
                        "rlcd_checkpoint": "heads",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = ShowcaseConfig.load(examples, models)
    assert [item.id for item in config.examples] == ["maze"]
    assert {item.role for item in config.models} == {"baseline", "trained"}


def test_showcase_config_rejects_unpaired_group(tmp_path: Path) -> None:
    examples = tmp_path / "examples.json"
    models = tmp_path / "models.json"
    examples.write_text(
        '{"schema_version":1,"examples":[]}', encoding="utf-8"
    )
    models.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "base",
                        "label": "Base",
                        "parameter_group": "1B",
                        "role": "baseline",
                        "kind": "qwen_base",
                        "model_config": "base.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="one baseline and one trained"):
        ShowcaseConfig.load(examples, models)


def test_visual_example_requires_exactly_one_seed() -> None:
    with pytest.raises(ValueError, match="exactly one integer seed"):
        Example.from_dict(
            {
                "id": "maze",
                "title": "Maze",
                "description": "Navigate.",
                "environment": {
                    "family": "minigrid",
                    "environment_id": "MiniGrid-Empty-5x5-v0",
                    "seed": [1, 2],
                    "max_steps": 4,
                    "actions": ["move_forward"],
                },
            }
        )


def _trajectory(path: Path, frame: Path, *, role: str, label: str) -> None:
    value = {
        "schema_version": 1,
        "example": {
            "id": "maze",
            "title": "Maze",
            "environment_id": "MiniGrid-Empty-5x5-v0",
            "seed": 7,
            "max_steps": 3,
        },
        "model": {
            "id": role,
            "label": label,
            "parameter_group": "1B",
            "role": role,
            "kind": "qwen_base" if role == "baseline" else "vision_jev_rlcd",
        },
        "steps": [
            {
                "step": 0,
                "frame": str(frame),
                "action": "move_forward",
                "probabilities": {"move_forward": 0.9, "turn_left": 0.1},
                "latency_ms": 12.0,
            }
        ],
        "final_frame": str(frame),
        "outcome": {"success": True, "decisions": 1},
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def test_render_comparison_creates_animated_gif(tmp_path: Path) -> None:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (64, 64), "#31a354").save(frame)
    baseline = tmp_path / "baseline.json"
    trained = tmp_path / "trained.json"
    _trajectory(baseline, frame, role="baseline", label="Qwen")
    _trajectory(trained, frame, role="trained", label="Vision-Jev")
    output = tmp_path / "comparison.gif"
    report = render_comparison(baseline, trained, output, frame_duration_ms=200)
    assert report["frames"] == 2
    with Image.open(output) as gif:
        assert gif.n_frames == 2
        assert gif.size == (1000, 730)


def test_publish_readme_replaces_only_marker_section(tmp_path: Path) -> None:
    examples = tmp_path / "showcase" / "configs" / "examples.json"
    models = tmp_path / "showcase" / "configs" / "models.json"
    examples.parent.mkdir(parents=True)
    examples.write_text('{"schema_version":1,"examples":[]}', encoding="utf-8")
    models.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "base",
                        "label": "Base",
                        "parameter_group": "1B",
                        "role": "baseline",
                        "kind": "qwen_base",
                        "model_config": "base.json",
                    },
                    {
                        "id": "trained",
                        "label": "Trained",
                        "parameter_group": "1B",
                        "role": "trained",
                        "kind": "vision_jev_rlcd",
                        "model_config": "base.json",
                        "sft_checkpoint": "sft",
                        "rlcd_config": "rlcd.json",
                        "rlcd_checkpoint": "heads",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# Before\n\n{START_MARKER}\nold\n{END_MARKER}\n\n## After\n",
        encoding="utf-8",
    )
    config = ShowcaseConfig.load(examples, models)
    publish_readme(config, tmp_path / "asset" / "demos", readme)
    updated = readme.read_text(encoding="utf-8")
    assert updated.startswith("# Before")
    assert "## Interactive comparisons" in updated
    assert updated.endswith("## After\n")
