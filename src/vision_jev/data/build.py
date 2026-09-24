"""Deterministic manifest assembly with explicit quota failure reports."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _rank(sample: dict[str, Any], seed: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{sample['sample_id']}".encode()).digest()


def build_public_manifest(
    *,
    data_root: Path,
    mixture_config: Path,
    destination: Path,
    seed: str = "vision-jev-sft-v2",
) -> dict[str, Any]:
    """Select exact source quotas or fail with a machine-readable shortage report."""
    config = json.loads(mixture_config.read_text(encoding="utf-8"))
    quotas: dict[str, int] = config["public_source_quotas"]
    task_quotas: dict[str, dict[str, int]] = config.get("public_source_task_quotas", {})
    selected: list[dict[str, Any]] = []
    availability: dict[str, int] = {}
    shortages: dict[str, int] = {}
    for source, quota in quotas.items():
        path = data_root / "processed" / source / "canonical.jsonl"
        rows: list[dict[str, Any]] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    if not raw.strip():
                        continue
                    sample = json.loads(raw)
                    if (
                        sample["split"] != "train"
                        or sample.get("teacher_only", False)
                        or sample.get("quality", {}).get("release_blocked", False)
                    ):
                        continue
                    rows.append(sample)
        availability[source] = len(rows)
        rows.sort(key=lambda sample: _rank(sample, f"{seed}:{source}"))
        requested_tasks = task_quotas.get(source)
        if requested_tasks:
            for task, task_quota in requested_tasks.items():
                task_rows = [row for row in rows if row["task_type"] == task]
                if len(task_rows) < task_quota:
                    shortages[f"{source}:{task}"] = task_quota - len(task_rows)
                selected.extend(task_rows[:task_quota])
        else:
            if len(rows) < quota:
                shortages[source] = quota - len(rows)
            selected.extend(rows[:quota])

    report = {
        "schema_version": 1,
        "mixture_id": config["mixture_id"],
        "seed": seed,
        "requested": quotas,
        "available_train": availability,
        "shortages": shortages,
        "selected_questions": len(selected),
        "task_counts": dict(Counter(sample["task_type"] for sample in selected)),
        "language_counts": dict(Counter(sample["language"] for sample in selected)),
    }
    report_path = destination.with_suffix(".build-report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if shortages:
        raise ValueError(f"public manifest quota shortages; see {report_path}")

    selected.sort(key=lambda sample: _rank(sample, seed))
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample in selected:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)
    manifest_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    report["manifest_sha256"] = manifest_sha256
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
