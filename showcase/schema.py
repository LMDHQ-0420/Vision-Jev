"""Validated configuration and records for static RLCD showcases."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _required(value: dict[str, Any], key: str) -> Any:
    if key not in value:
        raise ValueError(f"missing required field: {key}")
    return value[key]


def _expand_path(value: str, *, root: Path) -> Path:
    expanded = Path(os.path.expandvars(value)).expanduser()
    return expanded if expanded.is_absolute() else root / expanded


@dataclass(frozen=True)
class StaticExample:
    id: str
    title: str
    description: str
    source: str
    task_type: str
    sample_id: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StaticExample:
        task_type = str(_required(value, "task_type"))
        if task_type not in {"choice", "noul", "score"}:
            raise ValueError(f"unsupported showcase task: {task_type}")
        return cls(
            id=str(_required(value, "id")),
            title=str(_required(value, "title")),
            description=str(_required(value, "description")),
            source=str(_required(value, "source")),
            task_type=task_type,
            sample_id=str(_required(value, "sample_id")),
        )


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    parameter_group: str
    role: str
    kind: str
    model_config: Path
    sft_checkpoint: Path | None = None
    rlcd_config: Path | None = None
    rlcd_checkpoint: Path | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, root: Path) -> ModelSpec:
        kind = str(_required(value, "kind"))
        if kind not in {"qwen_base", "vision_jev_rlcd"}:
            raise ValueError(f"unsupported model kind: {kind}")

        def optional_path(key: str) -> Path | None:
            raw = value.get(key)
            return None if raw is None else _expand_path(str(raw), root=root)

        spec = cls(
            id=str(_required(value, "id")),
            label=str(_required(value, "label")),
            parameter_group=str(_required(value, "parameter_group")),
            role=str(_required(value, "role")),
            kind=kind,
            model_config=_expand_path(str(_required(value, "model_config")), root=root),
            sft_checkpoint=optional_path("sft_checkpoint"),
            rlcd_config=optional_path("rlcd_config"),
            rlcd_checkpoint=optional_path("rlcd_checkpoint"),
        )
        if spec.role not in {"baseline", "trained"}:
            raise ValueError(f"unsupported comparison role: {spec.role}")
        if kind == "vision_jev_rlcd" and any(
            path is None for path in (spec.sft_checkpoint, spec.rlcd_config, spec.rlcd_checkpoint)
        ):
            raise ValueError(f"RLCD model {spec.id} is missing checkpoint configuration")
        return spec


@dataclass(frozen=True)
class ShowcaseConfig:
    progress_frame_duration_ms: int
    completed_hold_ms: int
    latency_repetitions: int
    manifest_sha256: str
    selection_rule: str
    examples: tuple[StaticExample, ...]
    models: tuple[ModelSpec, ...]

    @classmethod
    def load(cls, examples_path: Path, models_path: Path) -> ShowcaseConfig:
        example_data = json.loads(examples_path.read_text(encoding="utf-8"))
        model_data = json.loads(models_path.read_text(encoding="utf-8"))
        if example_data.get("schema_version") != 2:
            raise ValueError("static showcase examples require schema_version=2")
        if model_data.get("schema_version") != 1:
            raise ValueError("showcase models require schema_version=1")
        examples = tuple(StaticExample.from_dict(item) for item in example_data["examples"])
        root = models_path.resolve().parents[2]
        models = tuple(ModelSpec.from_dict(item, root=root) for item in model_data["models"])
        _validate_unique("example", [item.id for item in examples])
        _validate_unique("sample", [item.sample_id for item in examples])
        _validate_unique("model", [item.id for item in models])
        _validate_groups(models)
        progress_frame_duration_ms = int(example_data.get("progress_frame_duration_ms", 100))
        completed_hold_ms = int(example_data.get("completed_hold_ms", 1000))
        latency_repetitions = int(example_data.get("latency_repetitions", 3))
        if progress_frame_duration_ms < 50:
            raise ValueError("progress_frame_duration_ms must be at least 50")
        if completed_hold_ms < 250:
            raise ValueError("completed_hold_ms must be at least 250")
        if latency_repetitions < 1:
            raise ValueError("latency_repetitions must be positive")
        digest = str(_required(example_data, "manifest_sha256"))
        if len(digest) != 64:
            raise ValueError("manifest_sha256 must be a SHA-256 digest")
        return cls(
            progress_frame_duration_ms=progress_frame_duration_ms,
            completed_hold_ms=completed_hold_ms,
            latency_repetitions=latency_repetitions,
            manifest_sha256=digest,
            selection_rule=str(_required(example_data, "selection_rule")),
            examples=examples,
            models=models,
        )


def _validate_unique(label: str, values: list[str]) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {label} ids: {duplicates}")


def _validate_groups(models: tuple[ModelSpec, ...]) -> None:
    groups = sorted({model.parameter_group for model in models})
    for group in groups:
        members = [model for model in models if model.parameter_group == group]
        roles = sorted(model.role for model in members)
        if roles != ["baseline", "trained"]:
            raise ValueError(
                f"parameter group {group!r} must contain one baseline and one trained model"
            )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
