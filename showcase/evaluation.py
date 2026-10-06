"""Build a source-aware report from immutable static RLCD test artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from showcase.schema import StaticExample, write_json
from showcase.static import sha256_file


def verify_showcase_selection(
    examples: tuple[StaticExample, ...],
    manifest: Path,
    evaluations: dict[str, Path],
) -> dict[str, Any]:
    samples: dict[str, dict[str, Any]] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            if sample.get("decision_role") == "test":
                samples[str(sample["sample_id"])] = sample

    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    thresholds: dict[str, dict[str, float]] = {}
    for group, root in evaluations.items():
        with (root / "test.jsonl").open(encoding="utf-8") as handle:
            predictions[group] = {
                row["sample_id"]: row for row in (json.loads(line) for line in handle)
            }
        threshold_data = json.loads(
            (root / "confidence-thresholds.json").read_text(encoding="utf-8")
        )
        thresholds[group] = {
            task: float(values["threshold"])
            for task, values in threshold_data["by_task"].items()
        }

    report: dict[str, Any] = {}
    for example in examples:
        eligible = []
        for sample_id, sample in samples.items():
            if sample.get("source") != example.source:
                continue
            if sample.get("task_type") != example.task_type:
                continue
            if all(
                predictions[group][sample_id]["correct"]
                and float(predictions[group][sample_id]["confidence"])
                >= thresholds[group][example.task_type]
                for group in evaluations
            ):
                eligible.append(sample_id)
        if not eligible:
            raise ValueError(f"no accepted showcase samples for {example.id}")
        selected = min(
            eligible, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        )
        if selected != example.sample_id:
            raise ValueError(
                f"showcase selection mismatch for {example.id}: expected {selected}"
            )
        report[example.id] = {"sample_id": selected, "eligible_samples": len(eligible)}
    return report


def build_static_test_report(
    manifest: Path,
    evaluations: dict[str, Path],
    destination: Path,
) -> dict[str, Any]:
    sample_sources: dict[str, str] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            if sample.get("decision_role") == "test":
                sample_sources[str(sample["sample_id"])] = str(sample["source"])

    models: dict[str, Any] = {}
    for group, root in evaluations.items():
        predictions_path = root / "test.jsonl"
        summary_path = root / "test.summary.json"
        thresholds_path = root / "confidence-thresholds.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
        source_totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        questions = 0
        with predictions_path.open(encoding="utf-8") as handle:
            for line in handle:
                prediction = json.loads(line)
                sample_id = str(prediction["sample_id"])
                try:
                    source = sample_sources[sample_id]
                except KeyError as exc:
                    raise ValueError(
                        f"test prediction is absent from manifest: {sample_id}"
                    ) from exc
                task = str(prediction["task_type"])
                totals = source_totals[(source, task)]
                totals[0] += 1
                totals[1] += int(bool(prediction["correct"]))
                questions += 1
        if questions != int(summary["questions"]):
            raise ValueError(f"test question count mismatch for {group}")
        by_source: dict[str, dict[str, Any]] = {}
        for (source, task), (count, correct) in sorted(source_totals.items()):
            by_source.setdefault(source, {})[task] = {
                "questions": count,
                "accuracy": correct / count,
            }
        models[group] = {
            "summary": summary,
            "thresholds": thresholds,
            "by_source": by_source,
            "artifacts": {
                "predictions": str(predictions_path),
                "predictions_sha256": sha256_file(predictions_path),
                "summary": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
            },
        }
    report = {
        "schema_version": 1,
        "protocol": "frozen_static_rlcd_test",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "models": models,
    }
    write_json(destination, report)
    return report
