"""Deterministic, group-safe RLCD root and training-view manifests."""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vision_jev.data.schema import validate_jsonl


def _rank(seed: str, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{value}".encode()).digest(), "big")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eligible(sample: dict[str, Any]) -> bool:
    return not sample.get("teacher_only", False) and not sample.get("quality", {}).get(
        "release_blocked", False
    )


def _identity(sample: dict[str, Any]) -> tuple[str, str, str]:
    sample_id = str(sample["sample_id"])
    return sample_id, str(sample.get("root_id", sample_id)), str(sample["group_id"])


def _push_candidate(
    heap: list[tuple[int, str, int, dict[str, Any]]],
    sample: dict[str, Any],
    *,
    limit: int,
    seed: str,
    line_index: int,
) -> None:
    sample_id = str(sample["sample_id"])
    entry = (-_rank(seed, sample_id), sample_id, line_index, sample)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def _load_sft_state(
    manifests: list[Path],
) -> tuple[set[str], set[str], set[str], list[dict[str, Any]]]:
    sample_ids: set[str] = set()
    root_ids: set[str] = set()
    group_ids: set[str] = set()
    train_rows: list[dict[str, Any]] = []
    train_seen: set[str] = set()
    for manifest in manifests:
        if not manifest.is_file():
            raise FileNotFoundError(f"missing SFT manifest: {manifest}")
        with manifest.open(encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                sample = json.loads(raw)
                sample_id, root_id, group_id = _identity(sample)
                sample_ids.add(sample_id)
                root_ids.add(root_id)
                group_ids.add(group_id)
                if (
                    sample_id not in train_seen
                    and sample.get("pilot_role", "train") == "train"
                    and sample.get("source_bucket", "public") == "public"
                    and _eligible(sample)
                ):
                    train_rows.append(sample)
                    train_seen.add(sample_id)
    return sample_ids, root_ids, group_ids, train_rows


def _candidate_lists(
    heaps: dict[str, dict[str, dict[str, list[tuple[int, str, int, dict[str, Any]]]]]],
) -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    result: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for role, task_heaps in heaps.items():
        result[role] = {}
        for task, source_heaps in task_heaps.items():
            result[role][task] = {}
            for source, heap in source_heaps.items():
                rows = [entry[3] for entry in heap]
                rows.sort(key=lambda row: _rank(f"rlcd:{role}:{task}", str(row["sample_id"])))
                result[role][task][source] = rows
    return result


def _select_balanced_task(
    *,
    candidates: dict[str, list[dict[str, Any]]],
    count: int,
    role: str,
    task: str,
    assigned_groups: dict[str, str],
    selected_ids: set[str],
    source_counts: Counter[str],
    source_cap: int | None,
) -> list[dict[str, Any]]:
    sources = sorted(candidates, key=lambda source: _rank(f"rlcd:{role}:{task}:sources", source))
    positions = {source: 0 for source in sources}
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        made_progress = False
        for source in sources:
            if len(selected) >= count:
                break
            if source_cap is not None and source_counts[source] >= source_cap:
                continue
            rows = candidates[source]
            while positions[source] < len(rows):
                sample = rows[positions[source]]
                positions[source] += 1
                sample_id, _, group_id = _identity(sample)
                assigned_role = assigned_groups.get(group_id)
                if sample_id in selected_ids or (
                    assigned_role is not None and assigned_role != role
                ):
                    continue
                selected.append(sample)
                selected_ids.add(sample_id)
                assigned_groups[group_id] = role
                source_counts[source] += 1
                made_progress = True
                break
        if not made_progress:
            raise ValueError(
                f"RLCD shortage for {role}:{task}: selected {len(selected)} of {count}"
            )
    return selected


def _as_rlcd_row(
    sample: dict[str, Any], *, role: str, schema_split: str, plan_id: str
) -> dict[str, Any]:
    row = copy.deepcopy(sample)
    row["upstream_split"] = row["split"]
    row["split"] = schema_split
    row["decision_role"] = role
    row["rlcd_plan_id"] = plan_id
    return row


def build_rlcd_manifest(
    *,
    data_root: Path,
    config_path: Path,
    sft_manifests: list[Path],
    destination: Path,
) -> dict[str, Any]:
    """Freeze exact RLCD roles while excluding every SFT group from evaluation."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    roles: dict[str, dict[str, Any]] = config["roles"]
    plan_id = str(config["plan_id"])
    seed = str(config["seed"])
    sft_ids, sft_roots, sft_groups, sft_train_rows = _load_sft_state(sft_manifests)

    evaluation_roles = [role for role in roles if role != "train"]
    evaluation_task_totals = {
        task: sum(int(roles[role][task]) for role in evaluation_roles)
        for task in ("choice", "noul", "score")
    }
    heaps: dict[
        str, dict[str, dict[str, list[tuple[int, str, int, dict[str, Any]]]]]
    ] = {
        role: {task: defaultdict(list) for task in ("choice", "noul", "score")}
        for role in evaluation_roles
    }
    availability: Counter[str] = Counter()
    canonical_paths = sorted((data_root / "processed").glob("*/canonical.jsonl"))
    if not canonical_paths:
        raise FileNotFoundError(f"no canonical data below {data_root / 'processed'}")
    line_index = 0
    for path in canonical_paths:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                line_index += 1
                sample = json.loads(raw)
                if not _eligible(sample):
                    continue
                sample_id, root_id, group_id = _identity(sample)
                if sample_id in sft_ids or root_id in sft_roots or group_id in sft_groups:
                    continue
                task = str(sample["task_type"])
                source = str(sample["source"])
                availability[f"{source}:{task}"] += 1
                for role in evaluation_roles:
                    # Earlier roles may claim groups that rank highly for later roles.
                    # Retain enough per-source headroom for every evaluation role.
                    limit = max(evaluation_task_totals[task], 1)
                    _push_candidate(
                        heaps[role][task][source],
                        sample,
                        limit=limit,
                        seed=f"{seed}:{role}:{task}:{source}",
                        line_index=line_index,
                    )

    train_heaps: dict[str, dict[str, list[tuple[int, str, int, dict[str, Any]]]]] = {
        task: defaultdict(list) for task in ("choice", "noul", "score")
    }
    for index, sample in enumerate(sft_train_rows, 1):
        task = str(sample["task_type"])
        source = str(sample["source"])
        _push_candidate(
            train_heaps[task][source],
            sample,
            limit=max(int(roles["train"][task]), 1),
            seed=f"{seed}:train:{task}:{source}",
            line_index=index,
        )
    heaps["train"] = train_heaps
    candidates = _candidate_lists(heaps)

    assigned_groups: dict[str, str] = {}
    selected_ids: set[str] = set()
    selected_by_role: dict[str, list[dict[str, Any]]] = {}
    role_order = [*evaluation_roles, "train"]
    for role in role_order:
        role_config = roles[role]
        source_counts: Counter[str] = Counter()
        source_cap = None
        if role == "train":
            share = float(config["selection"]["maximum_train_share_per_source"])
            source_cap = math.ceil(int(role_config["questions"]) * share)
        rows: list[dict[str, Any]] = []
        for task in ("score", "noul", "choice"):
            rows.extend(
                _select_balanced_task(
                    candidates=candidates[role][task],
                    count=int(role_config[task]),
                    role=role,
                    task=task,
                    assigned_groups=assigned_groups,
                    selected_ids=selected_ids,
                    source_counts=source_counts,
                    source_cap=source_cap,
                )
            )
        selected_by_role[role] = rows

    output_rows = [
        _as_rlcd_row(
            sample,
            role=role,
            schema_split=str(roles[role]["schema_split"]),
            plan_id=plan_id,
        )
        for role in roles
        for sample in selected_by_role[role]
    ]
    output_rows.sort(key=lambda row: _rank(f"{seed}:output", str(row["sample_id"])))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)
    validation = validate_jsonl(destination, check_assets=True)

    role_counts = Counter(str(row["decision_role"]) for row in output_rows)
    role_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    role_source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in output_rows:
        role = str(row["decision_role"])
        role_task_counts[role][str(row["task_type"])] += 1
        role_source_counts[role][str(row["source"])] += 1
    evaluation_rows = [row for row in output_rows if row["decision_role"] != "train"]
    evaluation_roots = {str(row.get("root_id", row["sample_id"])) for row in evaluation_rows}
    evaluation_groups = {str(row["group_id"]) for row in evaluation_rows}
    train_roots = {
        str(row.get("root_id", row["sample_id"]))
        for row in output_rows
        if row["decision_role"] == "train"
    }
    sft_train_roots = {
        str(row.get("root_id", row["sample_id"])) for row in sft_train_rows
    }
    roles_by_group: dict[str, set[str]] = defaultdict(set)
    for row in output_rows:
        roles_by_group[str(row["group_id"])].add(str(row["decision_role"]))
    leakage = {
        "evaluation_root_overlap_with_sft": len(evaluation_roots & sft_roots),
        "evaluation_group_overlap_with_sft": len(evaluation_groups & sft_groups),
        "train_roots_absent_from_sft_train": len(train_roots - sft_train_roots),
        "groups_with_multiple_roles": sum(
            len(role_set) > 1 for role_set in roles_by_group.values()
        ),
    }
    if any(leakage.values()):
        destination.unlink(missing_ok=True)
        raise ValueError(f"RLCD leakage checks failed: {leakage}")
    expected_total = int(config["total_root_questions"])
    if validation.questions != expected_total:
        destination.unlink(missing_ok=True)
        raise ValueError(f"RLCD manifest has {validation.questions}; expected {expected_total}")

    report = {
        "schema_version": 1,
        "plan_id": plan_id,
        "seed": seed,
        "questions": validation.questions,
        "groups": validation.groups,
        "roles": dict(sorted(role_counts.items())),
        "role_tasks": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(role_task_counts.items())
        },
        "role_sources": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(role_source_counts.items())
        },
        "available_evaluation_candidates": dict(sorted(availability.items())),
        "sha256": _sha256(destination),
    }
    destination.with_name("split-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    destination.with_name("leakage-report.json").write_text(
        json.dumps({"schema_version": 1, **leakage}, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _candidate_orders(options: list[dict[str, Any]], seed: str) -> list[list[dict[str, Any]]]:
    if len(options) < 2:
        return []
    canonical = tuple(str(option["id"]) for option in options)
    seen = {canonical}
    orders: list[list[dict[str, Any]]] = []
    for attempt in range(32):
        ordered = sorted(
            options,
            key=lambda option: _rank(f"{seed}:{attempt}", str(option["id"])),
        )
        identity = tuple(str(option["id"]) for option in ordered)
        if identity in seen:
            continue
        seen.add(identity)
        orders.append(copy.deepcopy(ordered))
        if len(orders) == 3:
            break
    return orders


def build_rlcd_training_views(
    base_manifest: Path, destination: Path, *, seed: str
) -> dict[str, Any]:
    """Create label-preserving canonical and candidate-order train views."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    roots = 0
    view_counts: Counter[str] = Counter()
    with base_manifest.open(encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as output:
        for raw in source:
            if not raw.strip():
                continue
            sample = json.loads(raw)
            if sample["decision_role"] != "train":
                continue
            roots += 1
            root_sample_id = str(sample["sample_id"])
            views: list[tuple[str, list[dict[str, Any]]]] = [
                ("canonical", copy.deepcopy(sample["options"]))
            ]
            for index, options in enumerate(
                _candidate_orders(sample["options"], f"{seed}:{root_sample_id}"), 1
            ):
                views.append((f"candidate_permutation_{index}", options))
            for view_name, options in views:
                row = copy.deepcopy(sample)
                row["sample_id"] = f"{root_sample_id}::rlcd-view:{view_name}"
                row["options"] = options
                row["candidates"] = copy.deepcopy(options)
                row["rlcd_root_sample_id"] = root_sample_id
                row["rlcd_view_id"] = view_name
                row["rlcd_transform"] = (
                    {"type": "identity"}
                    if view_name == "canonical"
                    else {"type": "candidate_permutation", "label_preserving": True}
                )
                view_counts[view_name] += 1
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)
    validation = validate_jsonl(destination, check_assets=True)
    report = {
        "schema_version": 1,
        "base_manifest": str(base_manifest),
        "roots": roots,
        "views": validation.questions,
        "view_counts": dict(sorted(view_counts.items())),
        "sha256": _sha256(destination),
    }
    destination.with_suffix(".build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
