"""Configuration and trajectory contracts for showcase generation."""

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
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class Example:
    id: str
    title: str
    description: str
    environment_id: str
    seed: int
    max_steps: int
    actions: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Example:
        environment = dict(_required(value, "environment"))
        family = str(_required(environment, "family"))
        if family != "minigrid":
            raise ValueError(f"unsupported showcase environment family: {family}")
        actions = tuple(str(action) for action in _required(environment, "actions"))
        if not actions or len(actions) != len(set(actions)):
            raise ValueError(f"example {value.get('id')} needs unique actions")
        seed = _required(environment, "seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"example {value.get('id')} requires exactly one integer seed")
        max_steps = int(_required(environment, "max_steps"))
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        return cls(
            id=str(_required(value, "id")),
            title=str(_required(value, "title")),
            description=str(_required(value, "description")),
            environment_id=str(_required(environment, "environment_id")),
            seed=seed,
            max_steps=max_steps,
            actions=actions,
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
            path is None
            for path in (spec.sft_checkpoint, spec.rlcd_config, spec.rlcd_checkpoint)
        ):
            raise ValueError(f"RLCD model {spec.id} is missing checkpoint configuration")
        return spec


@dataclass(frozen=True)
class ShowcaseConfig:
    frame_duration_ms: int
    examples: tuple[Example, ...]
    models: tuple[ModelSpec, ...]

    @classmethod
    def load(cls, examples_path: Path, models_path: Path) -> ShowcaseConfig:
        example_data = json.loads(examples_path.read_text(encoding="utf-8"))
        model_data = json.loads(models_path.read_text(encoding="utf-8"))
        if example_data.get("schema_version") != 1 or model_data.get("schema_version") != 1:
            raise ValueError("showcase configs require schema_version=1")
        examples = tuple(Example.from_dict(item) for item in example_data["examples"])
        root = models_path.resolve().parents[2]
        models = tuple(ModelSpec.from_dict(item, root=root) for item in model_data["models"])
        _validate_unique("example", [item.id for item in examples])
        _validate_unique("model", [item.id for item in models])
        _validate_groups(models)
        frame_duration_ms = int(example_data.get("frame_duration_ms", 700))
        if frame_duration_ms < 100:
            raise ValueError("frame_duration_ms must be at least 100")
        return cls(frame_duration_ms, examples, models)


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
