"""Resumable frozen-test evaluation for generative Qwen comparison stages."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from showcase.models import ModelAdapter, load_adapter
from showcase.schema import ModelSpec, write_json
from showcase.static import is_correct, load_test_samples, normalized_target, sha256_file
from vision_jev.eval.timing import summarize_latencies

TASKS = ("choice", "noul", "score")


def _latency_summary(records: list[dict[str, Any]]) -> dict[str, float | int] | None:
    return summarize_latencies(float(record["latency_ms"]) for record in records)


def _read_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid baseline record at {path}:{line_number}") from exc
            sample_id = str(record["sample_id"])
            if sample_id in records:
                raise ValueError(f"duplicate baseline prediction: {sample_id}")
            records[sample_id] = record
    return records


def _metric_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    questions = len(records)
    if not questions:
        return {"questions": 0, "accuracy": None, "valid_rate": None}
    return {
        "questions": questions,
        "accuracy": sum(bool(record["correct"]) for record in records) / questions,
        "valid_rate": sum(bool(record["valid"]) for record in records) / questions,
        "latency_ms": _latency_summary(records),
    }


def summarize_baseline(
    model: ModelSpec,
    samples: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    manifest: Path,
    predictions_path: Path,
) -> dict[str, Any]:
    expected_ids = [str(sample["sample_id"]) for sample in samples]
    missing = [sample_id for sample_id in expected_ids if sample_id not in records]
    extras = sorted(set(records).difference(expected_ids))
    if missing or extras:
        raise ValueError(
            f"baseline prediction coverage mismatch: missing={len(missing)}, extras={len(extras)}"
        )
    ordered = [records[sample_id] for sample_id in expected_ids]
    by_task = {
        task: _metric_group([record for record in ordered if str(record["task_type"]) == task])
        for task in TASKS
    }
    sources = sorted({str(record["source"]) for record in ordered})
    by_source: dict[str, Any] = {}
    for source in sources:
        source_records = [record for record in ordered if str(record["source"]) == source]
        source_summary = _metric_group(source_records)
        source_summary["by_task"] = {
            task: _metric_group(
                [record for record in source_records if str(record["task_type"]) == task]
            )
            for task in TASKS
            if any(str(record["task_type"]) == task for record in source_records)
        }
        by_source[source] = source_summary
    return {
        "schema_version": 1,
        "protocol": (
            "frozen_static_rlcd_test_original_qwen"
            if model.kind == "qwen_base"
            else "frozen_static_rlcd_test_sft"
        ),
        "model_id": model.id,
        "model_label": model.label,
        "parameter_group": model.parameter_group,
        "questions": len(ordered),
        "valid_rate": sum(bool(record["valid"]) for record in ordered) / len(ordered),
        "accuracy": sum(bool(record["correct"]) for record in ordered) / len(ordered),
        "latency_scope": "model_generate",
        "latency_warmup_policy": "one_prediction_before_each_inference_session",
        "latency_ms": _latency_summary(ordered),
        "by_task": by_task,
        "by_source": by_source,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "predictions": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
    }


def evaluate_with_adapter(
    model: ModelSpec,
    samples: list[dict[str, Any]],
    adapter: ModelAdapter,
    output_path: Path,
    *,
    progress_every: int = 100,
) -> dict[str, dict[str, Any]]:
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    sample_ids = {str(sample["sample_id"]) for sample in samples}
    records = _read_records(output_path)
    extras = sorted(set(records).difference(sample_ids))
    if extras:
        raise ValueError(f"baseline output contains {len(extras)} unexpected samples")
    for record in records.values():
        if record.get("model_id") != model.id:
            raise ValueError("baseline output belongs to a different model")

    pending = [sample for sample in samples if str(sample["sample_id"]) not in records]
    if not pending:
        return records
    adapter.predict(pending[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with output_path.open("a", encoding="utf-8") as output:
        for index, sample in enumerate(pending, 1):
            prediction = adapter.predict(sample)
            record = {
                "model_id": model.id,
                "sample_id": sample["sample_id"],
                "root_id": sample["root_id"],
                "source": sample["source"],
                "task_type": sample["task_type"],
                "target": normalized_target(sample),
                "prediction": prediction.value,
                "valid": prediction.value is not None,
                "correct": is_correct(prediction.value, sample),
                "latency_ms": prediction.latency_ms,
                "raw_output": prediction.raw_output,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            records[str(sample["sample_id"])] = record
            if index % progress_every == 0 or index == len(pending):
                output.flush()
                completed = len(records)
                counts = Counter(
                    str(value["task_type"]) for value in records.values() if bool(value["correct"])
                )
                print(
                    json.dumps(
                        {
                            "model": model.id,
                            "completed": completed,
                            "total": len(samples),
                            "elapsed_seconds": time.monotonic() - started,
                            "correct": dict(sorted(counts.items())),
                        }
                    ),
                    flush=True,
                )
    return records


def evaluate_baseline(
    model: ModelSpec,
    manifest: Path,
    output_path: Path,
    *,
    model_root: Path,
    maximum: int | None = None,
    progress_every: int = 100,
) -> dict[str, Any]:
    if model.kind not in {"qwen_base", "qwen_sft"}:
        raise ValueError(f"generation evaluation does not support {model.kind}")
    samples = list(load_test_samples(manifest).values())
    if maximum is not None:
        if maximum < 1:
            raise ValueError("maximum must be positive")
        samples = samples[:maximum]
    if not samples:
        raise ValueError("manifest contains no baseline evaluation samples")
    records = _read_records(output_path)
    expected_ids = {str(sample["sample_id"]) for sample in samples}
    complete = len(records) == len(samples) and set(records) == expected_ids
    if not complete:
        adapter = load_adapter(model, model_root)
        try:
            records = evaluate_with_adapter(
                model,
                samples,
                adapter,
                output_path,
                progress_every=progress_every,
            )
        finally:
            adapter.close()
    summary = summarize_baseline(model, samples, records, manifest, output_path)
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary
