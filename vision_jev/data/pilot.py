"""Deterministic, stratified pilot-manifest construction."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _rank(seed: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode()).digest()


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
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    destination.with_suffix(".build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
