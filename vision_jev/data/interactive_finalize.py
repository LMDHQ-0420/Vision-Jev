"""Merge staged interactive data and enforce the pinned 16k quota contract."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

ROLE_TASK_TARGETS = {
    "train": {"choice": 7750, "noul": 2250},
    "dev": {"choice": 1160, "noul": 340},
    "calibration": {"choice": 1160, "noul": 340},
    "threshold": {"choice": 580, "noul": 170},
    "audit": {"choice": 580, "noul": 170},
    "test": {"choice": 1170, "noul": 330},
}
STAGE_TARGETS = {
    "minigrid_navigation": (4000, 3200),
    "minigrid_tools": (3000, 2400),
    "minigrid_hazards_static": (2250, 1350),
    "minigrid_dynamic_obstacles": (250, 250),
    "babyai_grounded": (2500, 2000),
    "procgen_maze": (2000, 1600),
    "boxoban": (2000, 1600),
}


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _rank(stage: str, sample_id: str) -> bytes:
    return hashlib.sha256(f"interactive-finalize-v1\0{stage}\0{sample_id}".encode()).digest()


def _choice_matrix(
    rows_by_stage: dict[str, list[dict[str, Any]]]
) -> dict[tuple[str, str], int]:
    fixed_stage = "minigrid_dynamic_obstacles"
    fixed_counts = Counter(str(row["decision_role"]) for row in rows_by_stage[fixed_stage])
    stages = [stage for stage in STAGE_TARGETS if stage != fixed_stage]
    roles = list(ROLE_TASK_TARGETS)
    source, sink = "source", "sink"
    capacity: dict[tuple[str, str], int] = {}
    adjacency: dict[str, list[str]] = defaultdict(list)

    def edge(left: str, right: str, value: int) -> None:
        capacity[(left, right)] = value
        capacity[(right, left)] = 0
        adjacency[left].append(right)
        adjacency[right].append(left)

    for stage in stages:
        edge(source, f"stage:{stage}", STAGE_TARGETS[stage][1])
        available = Counter(str(row["decision_role"]) for row in rows_by_stage[stage])
        for role in roles:
            edge(f"stage:{stage}", f"role:{role}", available[role])
    for role in roles:
        remaining = ROLE_TASK_TARGETS[role]["choice"] - fixed_counts[role]
        edge(f"role:{role}", sink, remaining)

    flow: dict[tuple[str, str], int] = defaultdict(int)
    while True:
        parent: dict[str, str | None] = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in parent and flow[(node, neighbor)] < capacity[(node, neighbor)]:
                    parent[neighbor] = node
                    queue.append(neighbor)
        if sink not in parent:
            break
        amount = 10**9
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            amount = min(amount, capacity[(previous, node)] - flow[(previous, node)])
            node = previous
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            flow[(previous, node)] += amount
            flow[(node, previous)] -= amount
            node = previous

    expected = sum(STAGE_TARGETS[stage][1] for stage in stages)
    actual = sum(flow[(source, f"stage:{stage}")] for stage in stages)
    if actual != expected:
        raise RuntimeError(f"role/task quota flow shortfall: {actual}/{expected}")
    matrix = {
        (stage, role): flow[(f"stage:{stage}", f"role:{role}")]
        for stage in stages
        for role in roles
    }
    matrix.update({(fixed_stage, role): fixed_counts[role] for role in roles})
    return matrix


def _optimal_actions(row: dict[str, Any]) -> list[str]:
    quality = row["quality"]
    values = quality.get("all_optimal_actions", quality.get("all_best_measured_actions"))
    if not values:
        raise ValueError(f"missing optimal action set: {row['sample_id']}")
    return [str(value) for value in values]


def _action_ids(row: dict[str, Any]) -> list[str]:
    quality = row["quality"]
    mapping = quality.get("oracle_q", quality.get("policy_success_rates"))
    if not mapping:
        raise ValueError(f"missing action universe: {row['sample_id']}")
    return [str(value) for value in mapping]


def _render_task(row: dict[str, Any], *, task: str, ordinal: int) -> None:
    optimal = _optimal_actions(row)
    action_ids = _action_ids(row)
    source = str(row["source"])
    dynamic = bool(row["quality"].get("stochastic_policy_label"))
    if task == "choice":
        options = [{"id": action, "text": action.replace("_", " ")} for action in action_ids]
        row["task_type"] = "choice"
        row["options"] = options
        row["candidates"] = options
        row["target"] = optimal
        row["target_kind"] = "set_empirical_policy" if dynamic else "set"
        if dynamic:
            row["question"] = (
                "Under the fixed reactive policy, which first action had the highest measured "
                "mean return across randomized obstacle rollouts?"
            )
        elif source == "boxoban":
            row["question"] = "Which move begins a shortest solution without entering a deadlock?"
        elif source == "procgen_maze":
            row["question"] = "Which move begins a shortest path to the cheese?"
        else:
            row["question"] = "Which action is on a shortest successful path?"
        row["quality"]["candidate_action"] = None
        return

    negatives = [action for action in action_ids if action not in optimal]
    if not negatives:
        raise ValueError(f"Noul row has no negative action: {row['sample_id']}")
    candidate = (
        optimal[ordinal % len(optimal)]
        if ordinal % 2 == 0
        else negatives[ordinal % len(negatives)]
    )
    row["task_type"] = "noul"
    row["options"] = []
    row["candidates"] = []
    row["target"] = candidate in optimal
    row["target_kind"] = "binary"
    row["question"] = f"Does action '{candidate.replace('_', ' ')}' preserve a shortest solution?"
    row["quality"]["candidate_action"] = candidate


def finalize_interactive(
    *, stage_paths: dict[str, Path], destination: Path, report_root: Path
) -> dict[str, Any]:
    if set(stage_paths) != set(STAGE_TARGETS):
        raise ValueError(
            f"stage mismatch: expected {sorted(STAGE_TARGETS)}, got {sorted(stage_paths)}"
        )
    rows_by_stage = {stage: _load(path) for stage, path in stage_paths.items()}
    for stage, rows in rows_by_stage.items():
        expected = STAGE_TARGETS[stage][0]
        if len(rows) != expected:
            raise ValueError(f"{stage} count mismatch: {len(rows)} != {expected}")

    matrix = _choice_matrix(rows_by_stage)
    output_rows: list[dict[str, Any]] = []
    for stage, rows in rows_by_stage.items():
        by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_role[str(row["decision_role"])].append(row)
        for role, role_rows in by_role.items():
            role_rows.sort(key=lambda row: _rank(stage, str(row["sample_id"])))
            choice_count = matrix[(stage, role)]
            forced_choice = [
                row
                for row in role_rows
                if not [
                    action
                    for action in _action_ids(row)
                    if action not in _optimal_actions(row)
                ]
            ]
            if len(forced_choice) > choice_count:
                raise RuntimeError(
                    f"{stage}/{role} has {len(forced_choice)} rows that cannot form Noul, "
                    f"above its {choice_count} Choice quota"
                )
            forced_ids = {str(row["sample_id"]) for row in forced_choice}
            flexible = [row for row in role_rows if str(row["sample_id"]) not in forced_ids]
            choice_ids = forced_ids | {
                str(row["sample_id"]) for row in flexible[: choice_count - len(forced_ids)]
            }
            for ordinal, row in enumerate(role_rows):
                _render_task(
                    row,
                    task="choice" if str(row["sample_id"]) in choice_ids else "noul",
                    ordinal=ordinal,
                )
                row["quality"]["interactive_stage"] = stage
                if stage == "minigrid_tools":
                    row["origin_label_method"] = "exact_bfs_validated_minigrid_transition_model"
                    row["generator_revision"] = "minigrid-tool-symbolic-oracle-v1"
                row["quality"].setdefault("episode_id", row["group_id"])
                row["quality"].setdefault("step", 0)
                if row["quality"].get("stochastic_policy_label"):
                    row["quality"].setdefault(
                        "terminal_conditions", ["goal_reached", "collision", "timeout"]
                    )
                else:
                    row["quality"].setdefault("terminal_condition", "successful_completion")
                output_rows.append(row)

    output_rows.sort(key=lambda row: str(row["sample_id"]))
    ids = [str(row["sample_id"]) for row in output_rows]
    groups_by_role: dict[str, set[str]] = defaultdict(set)
    image_hashes: set[bytes] = set()
    decision_states: set[tuple[bytes, str]] = set()
    for row in output_rows:
        groups_by_role[str(row["group_id"])].add(str(row["decision_role"]))
        image_hash = hashlib.sha256(Path(row["image"]).read_bytes()).digest()
        image_hashes.add(image_hash)
        decision_states.add((image_hash, str(row["state_text"])))
    role_counts = Counter(str(row["decision_role"]) for row in output_rows)
    task_counts = Counter(str(row["task_type"]) for row in output_rows)
    role_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in output_rows:
        role_task_counts[str(row["decision_role"])][str(row["task_type"])] += 1
    expected_roles = {role: sum(tasks.values()) for role, tasks in ROLE_TASK_TARGETS.items()}
    if len(output_rows) != 16000 or len(set(ids)) != 16000:
        raise RuntimeError("interactive manifest must contain 16000 unique sample IDs")
    if dict(role_counts) != expected_roles:
        raise RuntimeError(f"role quota mismatch: {dict(role_counts)}")
    if dict(task_counts) != {"choice": 12400, "noul": 3600}:
        raise RuntimeError(f"task quota mismatch: {dict(task_counts)}")
    if any(dict(role_task_counts[role]) != targets for role, targets in ROLE_TASK_TARGETS.items()):
        raise RuntimeError("role/task cross quotas do not match the pinned config")
    leaks = {group: sorted(roles) for group, roles in groups_by_role.items() if len(roles) > 1}
    if leaks:
        raise RuntimeError(f"groups leak across roles: {list(leaks.items())[:5]}")
    if len(decision_states) != 16000:
        raise RuntimeError(f"decision-state duplication: {16000 - len(decision_states)} duplicates")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in output_rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)
    report_root.mkdir(parents=True, exist_ok=True)
    exact_rows = sum(
        not row["quality"].get("stochastic_policy_label", False) for row in output_rows
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "questions": 16000,
        "roles": dict(sorted(role_counts.items())),
        "tasks": dict(sorted(task_counts.items())),
        "role_tasks": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(role_task_counts.items())
        },
        "exact_oracle_rows": exact_rows,
        "stochastic_policy_rows": 16000 - exact_rows,
        "unique_visible_images": len(image_hashes),
        "unique_image_and_instruction_states": len(decision_states),
        "manifest": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    (report_root / "oracle-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    leakage = {
        "schema_version": 1,
        "groups": len(groups_by_role),
        "cross_role_group_leaks": 0,
        "duplicate_sample_ids": 0,
        "duplicate_image_and_instruction_states": 0,
    }
    (report_root / "seed-leakage-report.json").write_text(
        json.dumps(leakage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    status = {
        "schema_version": 1,
        "status": "complete_16000_of_16000",
        "planned_decisions": 16000,
        "generated_decisions": 16000,
        "completed_stages": {
            "minigrid_navigation": 4000,
            "minigrid_tools": 3000,
            "minigrid_hazards": 2500,
            "babyai_grounded": 2500,
            "procgen_maze": 2000,
            "boxoban": 2000,
        },
        "pending_stages": {},
        "manifest": str(destination),
        "oracle_report": str(report_root / "oracle-report.json"),
        "leakage_report": str(report_root / "seed-leakage-report.json"),
    }
    (report_root / "generation-status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    data_root = report_root.parents[1]
    preprocessing_path = data_root / "_state" / "preprocessing.json"
    if preprocessing_path.is_file():
        preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
        preprocessing.update(
            {
                "status": "complete",
                "interactive_manifest": str(destination),
                "interactive_oracle_report": str(report_root / "oracle-report.json"),
                "interactive_leakage_report": str(report_root / "seed-leakage-report.json"),
                "interactive_generation_status": str(report_root / "generation-status.json"),
                "interactive_generated_decisions": 16000,
                "interactive_planned_decisions": 16000,
            }
        )
        preprocessing_path.write_text(
            json.dumps(preprocessing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report
