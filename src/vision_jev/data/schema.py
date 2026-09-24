"""Dependency-free validation for the canonical complete-question JSONL format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DataValidationError(ValueError):
    """A sample violates the data contract."""


@dataclass(frozen=True)
class ValidationReport:
    path: str
    questions: int
    groups: int
    sources: tuple[str, ...]
    task_counts: dict[str, int]
    split_counts: dict[str, int]


REQUIRED = {
    "schema_version",
    "sample_id",
    "group_id",
    "source",
    "source_version",
    "license",
    "split",
    "task_type",
    "state_text",
    "question",
    "options",
    "target_kind",
    "target",
    "label_origin",
    "language",
}
REQUIRED_V2 = {
    "root_id",
    "source_bucket",
    "image_metadata",
    "allowed_history",
    "input_track",
    "candidates",
    "origin_label_method",
    "generator_revision",
    "quality",
    "teacher_only",
}
TASKS = {"choice", "noul", "score"}
SPLITS = {"train", "dev", "calibration", "test"}


def _fail(line: int, sample_id: str, message: str) -> None:
    raise DataValidationError(f"line {line}, sample {sample_id}: {message}")


def validate_sample(sample: dict[str, Any], line: int) -> None:
    sample_id = str(sample.get("sample_id", "<missing>"))
    missing = sorted(REQUIRED - sample.keys())
    if missing:
        _fail(line, sample_id, f"missing fields: {', '.join(missing)}")
    if sample["schema_version"] not in {1, 2}:
        _fail(line, sample_id, "schema_version must be 1 or 2")
    if sample["schema_version"] == 2:
        missing_v2 = sorted(REQUIRED_V2 - sample.keys())
        if missing_v2:
            _fail(line, sample_id, f"missing v2 fields: {', '.join(missing_v2)}")
        if sample["candidates"] != sample["options"]:
            _fail(line, sample_id, "v2 candidates/options compatibility views must match")
        if sample["teacher_only"] is True:
            _fail(line, sample_id, "v2 main manifests forbid teacher-only labels")
    if sample["task_type"] not in TASKS:
        _fail(line, sample_id, f"unknown task_type {sample['task_type']!r}")
    if sample["split"] not in SPLITS:
        _fail(line, sample_id, f"unknown split {sample['split']!r}")
    if not isinstance(sample["options"], list):
        _fail(line, sample_id, "options must be a list")

    option_ids: list[str] = []
    for index, option in enumerate(sample["options"]):
        if not isinstance(option, dict) or not option.get("id") or not option.get("text"):
            _fail(line, sample_id, f"options[{index}] requires non-empty id and text")
        option_ids.append(str(option["id"]))
        if "box" in option:
            box = option["box"]
            if not isinstance(box, list) or len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
                _fail(
                    line,
                    sample_id,
                    f"options[{index}].box must be [x1,y1,x2,y2] with positive area",
                )
    if len(option_ids) != len(set(option_ids)):
        _fail(line, sample_id, "option ids must be unique within a question")

    task = sample["task_type"]
    target = sample["target"]
    if task in {"choice", "score"}:
        if len(option_ids) < 2:
            _fail(line, sample_id, f"{task} requires at least two options")
        targets = target if isinstance(target, list) else [target]
        unknown = sorted(str(item) for item in targets if str(item) not in option_ids)
        if unknown:
            _fail(line, sample_id, f"target ids absent from options: {unknown}")
    elif task == "noul":
        if option_ids:
            _fail(line, sample_id, "noul must not encode yes/no as options")
        if not isinstance(target, bool):
            _fail(line, sample_id, "noul target must be boolean")

    if sample["label_origin"] == "teacher_only" and sample["split"] != "train":
        _fail(line, sample_id, "teacher_only labels are permitted only in train")


def validate_jsonl(path: str | Path, *, check_assets: bool = False) -> ValidationReport:
    resolved = Path(path)
    ids: set[str] = set()
    groups_by_split: dict[str, str] = {}
    sources: set[str] = set()
    groups: set[str] = set()
    task_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    count = 0
    with resolved.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                sample = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"line {line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(sample, dict):
                raise DataValidationError(f"line {line_no}: sample must be an object")
            validate_sample(sample, line_no)
            image = sample.get("image")
            if check_assets and image is not None and not Path(image).is_file():
                _fail(
                    line_no, str(sample.get("sample_id", "<missing>")), f"image not found: {image}"
                )
            sample_id = sample["sample_id"]
            if sample_id in ids:
                _fail(line_no, sample_id, "duplicate sample_id")
            ids.add(sample_id)
            group = sample["group_id"]
            split = sample["split"]
            previous = groups_by_split.setdefault(group, split)
            if previous != split:
                _fail(line_no, sample_id, f"group leaks across splits: {previous} and {split}")
            groups.add(group)
            sources.add(sample["source"])
            task_counts[sample["task_type"]] = task_counts.get(sample["task_type"], 0) + 1
            split_counts[split] = split_counts.get(split, 0) + 1
            count += 1
    if count == 0:
        raise DataValidationError(f"{resolved}: no samples")
    return ValidationReport(
        path=str(resolved),
        questions=count,
        groups=len(groups),
        sources=tuple(sorted(sources)),
        task_counts=dict(sorted(task_counts.items())),
        split_counts=dict(sorted(split_counts.items())),
    )
