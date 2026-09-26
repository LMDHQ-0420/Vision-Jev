"""Deterministic, stratified pilot-manifest construction."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _rank(seed: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode()).digest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_training_manifest(
    source: Path,
    destination: Path,
    *,
    seed: str = "vision-jev-main-120k",
    eval_percent: int = 5,
) -> dict[str, Any]:
    """Assign every source row to a deterministic, group-safe train/eval role."""
    if not 1 <= eval_percent <= 50:
        raise ValueError("eval_percent must be between 1 and 50")

    groups: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                groups.add(str(json.loads(raw)["group_id"]))
    if len(groups) < 2:
        raise ValueError("training manifest requires at least two groups")
    eval_groups = {
        group
        for group in groups
        if int.from_bytes(_rank(f"{seed}:eval", group)[:8], "big") % 100 < eval_percent
    }
    if not eval_groups:
        eval_groups.add(min(groups, key=lambda group: _rank(f"{seed}:eval", group)))
    if eval_groups == groups:
        eval_groups.remove(max(groups, key=lambda group: _rank(f"{seed}:train", group)))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    role_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    role_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_counts: Counter[str] = Counter()
    with source.open(encoding="utf-8") as input_handle, temporary.open(
        "w", encoding="utf-8"
    ) as output:
        for raw in input_handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            role = "eval" if str(row["group_id"]) in eval_groups else "train"
            row["pilot_role"] = role
            role_counts[role] += 1
            task = str(row["task_type"])
            task_counts[task] += 1
            role_task_counts[role][task] += 1
            source_counts[str(row["source"])] += 1
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)

    report = {
        "schema_version": 1,
        "source": str(source),
        "destination": str(destination),
        "seed": seed,
        "eval_percent": eval_percent,
        "questions": sum(role_counts.values()),
        "groups": len(groups),
        "eval_groups": len(eval_groups),
        "roles": dict(sorted(role_counts.items())),
        "tasks": dict(sorted(task_counts.items())),
        "role_tasks": {
            role: dict(sorted(counts.items())) for role, counts in sorted(role_task_counts.items())
        },
        "sources": dict(sorted(source_counts.items())),
        "sha256": _sha256(destination),
    }
    destination.with_suffix(".build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_pilot_manifest(
    source: Path,
    destination: Path,
    *,
    questions: int = 12_000,
    seed: str = "vision-jev-pilot-12k",
) -> dict[str, Any]:
    """Select a proportional source/task pilot and assign group-safe train/eval roles."""
    if questions < 2:
        raise ValueError("questions must be at least 2")
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    total = 0
    with source.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            sample = json.loads(raw)
            strata[(str(sample["source"]), str(sample["task_type"]))].append(sample)
            total += 1
    if questions > total:
        raise ValueError(f"requested {questions} questions from a {total}-question manifest")

    exact = {key: questions * len(rows) / total for key, rows in strata.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remainder = questions - sum(quotas.values())
    fractional = sorted(
        strata,
        key=lambda key: (exact[key] - quotas[key], key),
        reverse=True,
    )
    for key in fractional[:remainder]:
        quotas[key] += 1

    selected: list[dict[str, Any]] = []
    for key, rows in strata.items():
        rows.sort(key=lambda row: _rank(f"{seed}:select", str(row["sample_id"])))
        selected.extend(rows[: quotas[key]])
    selected.sort(key=lambda row: _rank(f"{seed}:order", str(row["sample_id"])))

    groups = {str(row["group_id"]) for row in selected}
    eval_groups = {
        group
        for group in groups
        if int.from_bytes(_rank(f"{seed}:eval", group)[:8], "big") % 20 == 0
    }
    if not eval_groups:
        eval_groups.add(min(groups, key=lambda group: _rank(f"{seed}:eval", group)))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    role_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    with temporary.open("w", encoding="utf-8") as output:
        for sample in selected:
            row = dict(sample)
            row["pilot_role"] = "eval" if str(row["group_id"]) in eval_groups else "train"
            role_counts[row["pilot_role"]] += 1
            task_counts[str(row["task_type"])] += 1
            source_counts[str(row["source"])] += 1
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)
    report = {
        "schema_version": 1,
        "source": str(source),
        "destination": str(destination),
        "seed": seed,
        "questions": len(selected),
        "roles": dict(sorted(role_counts.items())),
        "tasks": dict(sorted(task_counts.items())),
        "sources": dict(sorted(source_counts.items())),
        "sha256": _sha256(destination),
    }
    destination.with_suffix(".build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
