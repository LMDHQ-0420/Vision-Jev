from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from vision_jev.train.sft import (
    EpochRandomSampler,
    ManifestDataset,
    NativeQwenCollator,
    answer_text,
    question_text,
    validate_training_config,
)


def test_sft_choice_format_uses_candidate_id() -> None:
    sample = {
        "task_type": "choice",
        "target": "option_1",
        "question": "Pick one",
        "state_text": "",
        "options": [
            {"id": "option_0", "text": "left"},
            {"id": "option_1", "text": "right"},
        ],
    }
    assert answer_text(sample) == '{"choice":"option_1"}'
    prompt = question_text(sample)
    assert "option_0: left" in prompt
    assert "option_1: right" in prompt


def test_prompt_includes_completed_history() -> None:
    sample = {
        "task_type": "choice",
        "state_text": "Open settings",
        "question": "What next?",
        "allowed_history": [{"action_type": "click"}, "scroll down"],
        "options": [{"id": "a", "text": "Settings"}],
        "target": "a",
    }
    prompt = question_text(sample)
    assert "Completed steps (oldest to newest):" in prompt
    assert '1. {"action_type": "click"}' in prompt
    assert '2. "scroll down"' in prompt


def test_sft_noul_format_uses_json_boolean() -> None:
    sample = {
        "task_type": "noul",
        "target": False,
        "question": "Is it supported?",
        "state_text": "visible",
        "options": [],
    }
    assert answer_text(sample) == '{"noul":false}'
    assert "Visible state: visible" in question_text(sample)


def test_sft_score_uses_ordered_integer_and_semantics() -> None:
    sample = {
        "task_type": "score",
        "target": "score_4",
        "question": "Rate quality",
        "state_text": "",
        "options": [{"id": f"score_{level}", "text": f"level {level}"} for level in range(1, 6)],
    }
    assert answer_text(sample) == '{"score":4}'
    prompt = question_text(sample)
    assert "1 = very poor" in prompt
    assert "5 = excellent" in prompt
    assert "blur, noise, exposure" in prompt


def test_sft_region_prompt_contains_normalized_boxes(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (200, 100)).save(image_path)
    sample = {
        "task_type": "choice",
        "target": "region:1",
        "question": "Which region?",
        "state_text": "",
        "image": str(image_path),
        "options": [
            {"id": "region:1", "text": "first", "box": [20.0, 10.0, 100.0, 50.0]},
            {"id": "region:2", "text": "second", "box": [0.0, 0.0, 200.0, 100.0]},
        ],
    }
    prompt = question_text(sample)
    assert "box_0_1000=[100,100,500,500]" in prompt
    assert "box_0_1000=[0,0,1000,1000]" in prompt
    collator = NativeQwenCollator(processor=None)
    assert collator.visual_budget([sample]) == 576


def test_manifest_filters_sources_and_repeats_score(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "sample_id": "score",
            "source": "koniq10k",
            "task_type": "score",
            "pilot_role": "train",
        },
        {
            "sample_id": "region",
            "source": "refcoco",
            "task_type": "choice",
            "pilot_role": "train",
        },
        {
            "sample_id": "excluded",
            "source": "gqa",
            "task_type": "choice",
            "pilot_role": "train",
        },
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    dataset = ManifestDataset(
        manifest,
        "train",
        task_repeat={"score": 3},
        include_sources=["koniq10k", "refcoco"],
    )
    assert len(dataset) == 4
    assert [row["sample_id"] for row in dataset.rows].count("score") == 3
    assert all(row["sample_id"] != "excluded" for row in dataset.rows)


def test_training_config_and_epoch_sampler_are_deterministic(tmp_path: Path) -> None:
    config = {
        "epochs": 2,
        "microbatch_questions_per_device": 1,
        "gradient_accumulation_steps": 32,
        "global_batch_questions": 32,
        "learning_rate": 3e-5,
        "expected_world_size": 1,
        "checkpoint_every_steps": 250,
    }
    validate_training_config(config, world_size=1)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": str(index),
                    "source": "source",
                    "task_type": "choice",
                    "pilot_role": "train",
                }
            )
            + "\n"
            for index in range(8)
        ),
        encoding="utf-8",
    )
    dataset = ManifestDataset(manifest, "train")
    sampler = EpochRandomSampler(dataset, seed=42)
    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)
