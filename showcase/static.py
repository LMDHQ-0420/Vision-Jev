"""Load frozen RLCD test samples and record auditable static predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from showcase.models import ModelAdapter, PredictionValue
from showcase.schema import ModelSpec, StaticExample, write_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_test_samples(manifest: Path) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            if sample.get("decision_role") == "test":
                samples[str(sample["sample_id"])] = sample
    return samples


def selected_sample(
    example: StaticExample, samples: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    try:
        sample = samples[example.sample_id]
    except KeyError as exc:
        raise ValueError(
            f"showcase sample is absent from frozen test: {example.sample_id}"
        ) from exc
    if sample.get("source") != example.source:
        raise ValueError(f"source mismatch for showcase example {example.id}")
    if sample.get("task_type") != example.task_type:
        raise ValueError(f"task mismatch for showcase example {example.id}")
    image = Path(str(sample.get("image", "")))
    if not image.is_file():
        raise FileNotFoundError(f"showcase image is missing: {image}")
    return sample


def normalized_target(sample: dict[str, Any]) -> list[bool | int | str]:
    target = sample["target"]
    values = target if isinstance(target, list) else [target]
    return [value for value in values]


def is_correct(value: PredictionValue, sample: dict[str, Any]) -> bool:
    return value is not None and value in normalized_target(sample)


def record_prediction(
    example: StaticExample,
    model: ModelSpec,
    sample: dict[str, Any],
    adapter: ModelAdapter,
    output_root: Path,
) -> Path:
    destination = output_root / example.id / model.id / "prediction.json"
    if destination.exists():
        raise FileExistsError(f"showcase prediction already exists: {destination}")
    prediction = adapter.predict(sample)
    image = Path(str(sample["image"]))
    record = {
        "schema_version": 2,
        "example": {
            "id": example.id,
            "title": example.title,
            "description": example.description,
            "source": example.source,
            "task_type": example.task_type,
        },
        "sample": {
            "sample_id": sample["sample_id"],
            "root_id": sample["root_id"],
            "decision_role": sample["decision_role"],
            "image": str(image.resolve()),
            "image_sha256": sha256_file(image),
            "state_text": sample.get("state_text", ""),
            "question": sample["question"],
            "options": sample.get("options", []),
            "target": normalized_target(sample),
        },
        "model": {
            "id": model.id,
            "label": model.label,
            "parameter_group": model.parameter_group,
            "role": model.role,
            "kind": model.kind,
        },
        "prediction": {
            "value": prediction.value,
            "probabilities": prediction.probabilities,
            "latency_ms": prediction.latency_ms,
            "raw_output": prediction.raw_output,
            "valid": prediction.value is not None,
            "correct": is_correct(prediction.value, sample),
        },
    }
    write_json(destination, record)
    return destination
