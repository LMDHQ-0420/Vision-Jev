"""Deterministic manifest assembly with explicit quota failure reports."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path
from typing import Any

from vision_jev.data.schema import validate_jsonl


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
    excluded: dict[str, dict[str, int]] = {}
    shortages: dict[str, int] = {}
    for source, quota in quotas.items():
        path = data_root / "processed" / source / "canonical.jsonl"
        requested_tasks = task_quotas.get(source)
        requested = requested_tasks or {"*": quota}
        heaps: dict[str, list[tuple[int, str, int, dict[str, Any]]]] = {
            task: [] for task in requested
        }
        task_availability: Counter[str] = Counter()
        source_excluded: Counter[str] = Counter()
        available = 0
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line_index, raw in enumerate(handle, start=1):
                    if not raw.strip():
                        continue
                    sample = json.loads(raw)
                    if sample["split"] != "train":
                        source_excluded["non_train"] += 1
                        continue
                    if sample.get("teacher_only", False):
                        source_excluded["teacher_only"] += 1
                        continue
                    if sample.get("quality", {}).get("release_blocked", False):
                        source_excluded["release_blocked"] += 1
                        continue
                    available += 1
                    task = str(sample["task_type"]) if requested_tasks else "*"
                    task_availability[task] += 1
                    if task not in heaps:
                        continue
                    rank = int.from_bytes(_rank(sample, f"{seed}:{source}"), "big")
                    entry = (-rank, str(sample["sample_id"]), line_index, sample)
                    task_quota = requested[task]
                    if len(heaps[task]) < task_quota:
                        heapq.heappush(heaps[task], entry)
                    elif entry > heaps[task][0]:
                        heapq.heapreplace(heaps[task], entry)
        availability[source] = available
        excluded[source] = dict(source_excluded)
        for task, task_quota in requested.items():
            if task_availability[task] < task_quota:
                key = f"{source}:{task}" if requested_tasks else source
                shortages[key] = task_quota - task_availability[task]
            selected.extend(entry[3] for entry in heaps[task])

    report = {
        "schema_version": 1,
        "mixture_id": config["mixture_id"],
        "seed": seed,
        "requested": quotas,
        "available_train": availability,
        "excluded": excluded,
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


def build_final_manifest(
    *,
    public_manifest: Path,
    api_candidates: Path,
    mixture_config: Path,
    destination: Path,
    seed: str = "vision-jev-sft",
) -> dict[str, Any]:
    """Select exact API-assisted task quotas and join them with the public manifest."""
    config = json.loads(mixture_config.read_text(encoding="utf-8"))
    requested = {
        task: int(config["blocks"]["api_assisted"][task]) for task in ("choice", "noul")
    }
    candidates: dict[str, list[dict[str, Any]]] = {task: [] for task in requested}
    with api_candidates.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            sample = json.loads(raw)
            task = str(sample["task_type"])
            if task in candidates:
                candidates[task].append(sample)
    shortages = {
        task: count - len(candidates[task])
        for task, count in requested.items()
        if len(candidates[task]) < count
    }
    if shortages:
        raise ValueError(f"API rewrite quota shortages: {shortages}")
    selected: list[dict[str, Any]] = []
    for task, count in requested.items():
        ranked = sorted(candidates[task], key=lambda sample: _rank(sample, f"{seed}:api:{task}"))
        selected.extend(ranked[:count])
    selected.sort(key=lambda sample: _rank(sample, f"{seed}:api-final"))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    public_count = 0
    with temporary.open("w", encoding="utf-8") as output, public_manifest.open(
        encoding="utf-8"
    ) as public:
        for raw in public:
            if raw.strip():
                output.write(raw if raw.endswith("\n") else raw + "\n")
                public_count += 1
        for sample in selected:
            output.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)
    validation = validate_jsonl(destination, check_assets=True)
    expected_total = int(config["total_questions"])
    if validation.questions != expected_total:
        raise ValueError(
            f"final manifest has {validation.questions} questions; expected {expected_total}"
        )
    report = {
        "schema_version": 1,
        "mixture_id": config["mixture_id"],
        "public_questions": public_count,
        "api_questions": len(selected),
        "api_task_counts": requested,
        "total_questions": validation.questions,
        "groups": validation.groups,
        "manifest_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    destination.with_suffix(".build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
