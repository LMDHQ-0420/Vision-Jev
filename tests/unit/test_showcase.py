from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from showcase.models import Prediction, _normalize_generated
from showcase.readme import END_MARKER, START_MARKER, publish_readme
from showcase.render import _option_label, render_static_comparison, render_static_suite
from showcase.schema import ModelSpec, ShowcaseConfig, StaticExample
from showcase.static import is_correct, record_prediction, selected_sample


def _write_config(root: Path) -> tuple[Path, Path]:
    examples = root / "showcase/configs/examples.json"
    models = root / "showcase/configs/models.json"
    examples.parent.mkdir(parents=True)
    examples.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "manifest_sha256": "a" * 64,
                "selection_rule": "fixed test sample",
                "examples": [
                    {
                        "id": "vqa",
                        "title": "VQA",
                        "description": "Answer the question.",
                        "source": "vqav2",
                        "task_type": "choice",
                        "sample_id": "sample-1",
                    }
                ],
            }
        ),
        encoding="utf-8",
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
    return examples, models


def test_showcase_config_has_static_samples_and_paired_groups(tmp_path: Path) -> None:
    examples, models = _write_config(tmp_path)
    config = ShowcaseConfig.load(examples, models)
    assert [item.sample_id for item in config.examples] == ["sample-1"]
    assert {item.role for item in config.models} == {"baseline", "trained"}


def test_static_example_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="unsupported showcase task"):
        StaticExample.from_dict(
            {
                "id": "bad",
                "title": "Bad",
                "description": "Bad task.",
                "source": "test",
                "task_type": "policy",
                "sample_id": "sample",
            }
        )


def test_selected_sample_requires_frozen_test_match(tmp_path: Path) -> None:
    example = StaticExample("vqa", "VQA", "Answer.", "vqav2", "choice", "sample-1")
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image)
    sample = {
        "sample_id": "sample-1",
        "source": "vqav2",
        "task_type": "choice",
        "image": str(image),
    }
    assert selected_sample(example, {"sample-1": sample}) == sample
    sample["source"] = "gqa"
    with pytest.raises(ValueError, match="source mismatch"):
        selected_sample(example, {"sample-1": sample})


def test_generated_values_are_task_normalized() -> None:
    sample = {"options": [{"id": "a"}, {"id": "b"}]}
    assert _normalize_generated("choice", "a", sample) == "a"
    assert _normalize_generated("choice", "missing", sample) is None
    assert _normalize_generated("noul", True, sample) is True
    assert _normalize_generated("score", 4, sample) == "score_4"
    assert is_correct("a", {"target": ["a", "b"]})


def test_choice_labels_hide_internal_option_ids() -> None:
    sample = {
        "options": [
            {"id": "choice_1", "text": "Solution B"},
            {"id": "choice_0", "text": "Neither"},
        ]
    }
    assert _option_label(sample, "choice_1", "choice") == "A. Solution B"
    assert _option_label(sample, "choice_0", "choice") == "B. Neither"
    assert _option_label(sample, "choice_1", "score") == "Solution B"


def test_record_prediction_uses_repeatable_median_latency(tmp_path: Path) -> None:
    class Adapter:
        def __init__(self) -> None:
            self.predictions = iter(
                [
                    Prediction("a", {"a": 1.0}, 30.0),
                    Prediction("a", {"a": 1.0}, 10.0),
                    Prediction("a", {"a": 1.0}, 20.0),
                ]
            )

        def predict(self, sample: dict[str, object]) -> Prediction:
            return next(self.predictions)

        def close(self) -> None:
            pass

    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image)
    sample = {
        "sample_id": "sample-1",
        "root_id": "root-1",
        "decision_role": "test",
        "image": str(image),
        "question": "What is shown?",
        "options": [{"id": "a", "text": "green"}],
        "target": "a",
    }
    example = StaticExample("vqa", "VQA", "Answer.", "vqav2", "choice", "sample-1")
    model = ModelSpec("base", "Base", "1B", "baseline", "qwen_base", tmp_path)
    output = record_prediction(
        example, model, sample, Adapter(), tmp_path / "output", latency_repetitions=3
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["prediction"]["latency_samples_ms"] == [30.0, 10.0, 20.0]
    assert record["prediction"]["latency_ms"] == 20.0


def _prediction(
    path: Path, image: Path, *, role: str, label: str, latency_ms: float = 12.0
) -> None:
    value = {
        "schema_version": 2,
        "example": {
            "id": "vqa",
            "title": "Visual question",
            "description": "Answer.",
            "source": "vqav2",
            "task_type": "choice",
        },
        "sample": {
            "sample_id": "sample-1",
            "root_id": "root-1",
            "decision_role": "test",
            "image": str(image),
            "image_sha256": "digest",
            "state_text": "",
            "question": "What is shown?",
            "options": [{"id": "a", "text": "green"}, {"id": "b", "text": "blue"}],
            "target": ["a"],
        },
        "model": {
            "id": role,
            "label": label,
            "parameter_group": "1B",
            "role": role,
            "kind": "qwen_base" if role == "baseline" else "vision_jev_rlcd",
        },
        "prediction": {
            "value": "a",
            "probabilities": {"a": 0.9, "b": 0.1} if role == "trained" else {},
            "latency_ms": latency_ms,
            "raw_output": '{"choice":"a"}' if role == "baseline" else None,
            "valid": True,
            "correct": True,
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def test_render_static_comparison_creates_three_frame_gif(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (64, 64), "#31a354").save(image)
    baseline = tmp_path / "baseline.json"
    trained = tmp_path / "trained.json"
    _prediction(baseline, image, role="baseline", label="Qwen")
    _prediction(trained, image, role="trained", label="Vision-Jev")
    output = tmp_path / "comparison.gif"
    report = render_static_comparison(baseline, trained, output, frame_duration_ms=200)
    assert report["trained"]["correct"] is True
    with Image.open(output) as gif:
        assert gif.n_frames == 3
        assert gif.size == (1200, 820)


def test_render_static_suite_uses_latency_progress_frames(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (64, 64), "#31a354").save(image)
    baseline = tmp_path / "baseline.json"
    trained = tmp_path / "trained.json"
    _prediction(baseline, image, role="baseline", label="Qwen", latency_ms=300)
    _prediction(trained, image, role="trained", label="Vision-Jev", latency_ms=100)
    output = tmp_path / "suite.gif"
    report = render_static_suite(
        [(baseline, trained)], output, frame_duration_ms=100, completed_hold_ms=500
    )
    assert report["frames"] == 4
    with Image.open(output) as gif:
        assert gif.n_frames == 4


def test_publish_readme_replaces_only_marker_section(tmp_path: Path) -> None:
    examples, models = _write_config(tmp_path)
    config = ShowcaseConfig.load(examples, models)
    asset = tmp_path / "asset/demos/1b.gif"
    asset.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(asset, format="GIF")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "models": {
                    "1B": {
                        "summary": {
                            "by_task": {
                                task: {"accuracy": 0.8} for task in ("choice", "noul", "score")
                            },
                            "threshold_policy": {
                                "choice": {"accuracy": 0.95, "coverage": 0.5},
                                "noul": {"accuracy": 0.96, "coverage": 0.6},
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# Before\n\n{START_MARKER}\nold\n{END_MARKER}\n\n## After\n", encoding="utf-8"
    )
    publish_readme(config, tmp_path / "asset/demos", report, readme)
    updated = readme.read_text(encoding="utf-8")
    assert updated.startswith("# Before")
    assert "## Static RLCD results" in updated
    assert "## Frozen test showcase" in updated
    assert updated.endswith("## After\n")
