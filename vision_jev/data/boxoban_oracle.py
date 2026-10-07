"""Exact A* oracle and renderer for the pinned Boxoban subset."""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from vision_jev.data.interactive import BOXOBAN_REVISION
from vision_jev.data.interactive_generate import ROLE_SPLITS, _rank

DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
BOXOBAN_ROLES = {
    "train": 1250,
    "dev": 187,
    "calibration": 188,
    "threshold": 94,
    "audit": 94,
    "test": 187,
}


@dataclass(frozen=True)
class BoxobanSolution:
    level: dict[str, Any]
    shortest_distance: int
    optimal_actions: list[str]
    explored_states: int
    png: bytes


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] - b[0], a[1] - b[1]


def _reachable(
    player: tuple[int, int], floors: frozenset[tuple[int, int]], boxes: frozenset[tuple[int, int]]
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], frozenset[str]]]:
    distances = {player: 0}
    first_actions: dict[tuple[int, int], frozenset[str]] = {player: frozenset()}
    queue = deque([player])
    while queue:
        current = queue.popleft()
        for action, vector in DIRECTIONS.items():
            successor = _add(current, vector)
            if successor not in floors or successor in boxes:
                continue
            candidate_distance = distances[current] + 1
            candidate_first = frozenset({action}) if current == player else first_actions[current]
            previous = distances.get(successor)
            if previous is None:
                distances[successor] = candidate_distance
                first_actions[successor] = candidate_first
                queue.append(successor)
            elif previous == candidate_distance:
                first_actions[successor] = first_actions[successor] | candidate_first
    return distances, first_actions


def _goal_push_distances(
    floors: frozenset[tuple[int, int]], goals: frozenset[tuple[int, int]]
) -> dict[tuple[int, int], dict[tuple[int, int], int]]:
    result: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
    for goal in goals:
        distances = {goal: 0}
        queue = deque([goal])
        while queue:
            current = queue.popleft()
            for vector in DIRECTIONS.values():
                previous = _sub(current, vector)
                player_support = _sub(previous, vector)
                if previous in floors and player_support in floors and previous not in distances:
                    distances[previous] = distances[current] + 1
                    queue.append(previous)
        result[goal] = distances
    return result


def _assignment_heuristic(
    boxes: frozenset[tuple[int, int]],
    goals: frozenset[tuple[int, int]],
    push_distances: dict[tuple[int, int], dict[tuple[int, int], int]],
) -> int:
    box_list = tuple(boxes)
    goal_list = tuple(goals)
    candidates = [
        sum(push_distances[goal].get(box, 10**9) for box, goal in zip(box_list, order, strict=True))
        for order in itertools.permutations(goal_list)
    ]
    return min(candidates)


def solve_boxoban(level: dict[str, Any], *, node_limit: int = 2_000_000) -> BoxobanSolution:
    grid = [str(row) for row in level["grid"]]
    floors = frozenset(
        (x, y) for y, row in enumerate(grid) for x, tile in enumerate(row) if tile != "#"
    )
    goals = frozenset(tuple(map(int, item)) for item in level["goals"])
    initial_boxes = frozenset(tuple(map(int, item)) for item in level["boxes"])
    initial_player = tuple(map(int, level["player"]))
    push_distances = _goal_push_distances(floors, goals)
    live_cells = frozenset(cell for distances in push_distances.values() for cell in distances)

    serial = itertools.count()
    heap: list[tuple[int, int, int, tuple[int, int], frozenset[tuple[int, int]]]] = []
    initial_h = _assignment_heuristic(initial_boxes, goals, push_distances)
    heapq.heappush(heap, (initial_h, 0, next(serial), initial_player, initial_boxes))
    initial_marker = (initial_player, initial_boxes)
    best_cost: dict[tuple[tuple[int, int], frozenset[tuple[int, int]]], int] = {initial_marker: 0}
    firsts_by_state: dict[tuple[tuple[int, int], frozenset[tuple[int, int]]], frozenset[str]] = {
        initial_marker: frozenset()
    }
    propagated_firsts: dict[tuple[tuple[int, int], frozenset[tuple[int, int]]], frozenset[str]] = {}
    solution_cost: int | None = None
    optimal_actions: set[str] = set()
    explored = 0
    while heap:
        estimate, cost, _, player, boxes = heapq.heappop(heap)
        if solution_cost is not None and estimate > solution_cost:
            break
        marker = (player, boxes)
        if cost != best_cost.get(marker):
            continue
        first_actions = firsts_by_state[marker]
        if propagated_firsts.get(marker) == first_actions:
            continue
        propagated_firsts[marker] = first_actions
        if boxes == goals:
            solution_cost = cost if solution_cost is None else solution_cost
            if cost == solution_cost:
                optimal_actions.update(first_actions)
            continue
        explored += 1
        if explored > node_limit:
            raise RuntimeError(f"A* exceeded {node_limit} states")
        walk_distance, walk_first = _reachable(player, floors, boxes)
        for box in boxes:
            for push_action, vector in DIRECTIONS.items():
                support = _sub(box, vector)
                destination = _add(box, vector)
                if (
                    support not in walk_distance
                    or destination not in floors
                    or destination in boxes
                ):
                    continue
                if destination not in live_cells and destination not in goals:
                    continue
                successor_boxes = frozenset((boxes - {box}) | {destination})
                successor_player = box
                edge_cost = walk_distance[support] + 1
                successor_cost = cost + edge_cost
                if not first_actions:
                    successor_firsts = (
                        frozenset({push_action}) if support == player else walk_first[support]
                    )
                else:
                    successor_firsts = first_actions
                heuristic = _assignment_heuristic(successor_boxes, goals, push_distances)
                if heuristic >= 10**9:
                    continue
                successor_marker = (successor_player, successor_boxes)
                previous_cost = best_cost.get(successor_marker)
                if previous_cost is not None and successor_cost > previous_cost:
                    continue
                if previous_cost is None or successor_cost < previous_cost:
                    best_cost[successor_marker] = successor_cost
                    firsts_by_state[successor_marker] = successor_firsts
                else:
                    combined = firsts_by_state[successor_marker] | successor_firsts
                    if combined == firsts_by_state[successor_marker]:
                        continue
                    firsts_by_state[successor_marker] = combined
                heapq.heappush(
                    heap,
                    (
                        successor_cost + heuristic,
                        successor_cost,
                        next(serial),
                        successor_player,
                        successor_boxes,
                    ),
                )
    if solution_cost is None or not optimal_actions:
        raise RuntimeError("level has no proven solution")
    return BoxobanSolution(
        level=level,
        shortest_distance=solution_cost,
        optimal_actions=sorted(optimal_actions),
        explored_states=explored,
        png=_render_level(grid),
    )


def _render_level(grid: list[str], tile_size: int = 32) -> bytes:
    from PIL import Image, ImageDraw

    colors = {
        "wall": (68, 74, 82),
        "floor": (228, 224, 212),
        "goal": (61, 153, 112),
        "box": (196, 127, 55),
        "player": (44, 103, 176),
    }
    image = Image.new("RGB", (len(grid[0]) * tile_size, len(grid) * tile_size), colors["wall"])
    draw = ImageDraw.Draw(image)
    margin = max(3, tile_size // 8)
    for y, row in enumerate(grid):
        for x, tile in enumerate(row):
            box = (x * tile_size, y * tile_size, (x + 1) * tile_size - 1, (y + 1) * tile_size - 1)
            if tile != "#":
                draw.rectangle(box, fill=colors["floor"], outline=(194, 191, 181))
            if tile in ".*+":
                cx, cy = x * tile_size + tile_size // 2, y * tile_size + tile_size // 2
                draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=colors["goal"])
            if tile in "$*":
                draw.rounded_rectangle(
                    (box[0] + margin, box[1] + margin, box[2] - margin, box[3] - margin),
                    radius=3,
                    fill=colors["box"],
                    outline=(126, 77, 29),
                    width=2,
                )
            if tile in "@+":
                cx, cy = x * tile_size + tile_size // 2, y * tile_size + tile_size // 2
                draw.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), fill=colors["player"])
                draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(238, 244, 250))
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _safe_solve(level: dict[str, Any]) -> BoxobanSolution | str:
    try:
        return solve_boxoban(level)
    except Exception as exc:
        return f"{level.get('level_id')}: {type(exc).__name__}: {exc}"


def _allocated(values: list[str], seed: str) -> list[str]:
    ranked = sorted(enumerate(values), key=lambda item: _rank(seed, f"{item[0]}:{item[1]}"))
    return [item[1] for item in ranked]


def generate_boxoban(
    *, level_index: Path, destination: Path, image_root: Path, max_workers: int = 32
) -> dict[str, Any]:
    levels = [json.loads(line) for line in level_index.open(encoding="utf-8") if line.strip()]
    if len(levels) != 2000:
        raise ValueError(f"pinned Boxoban index must contain 2000 levels, found {len(levels)}")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        solved_or_errors = list(executor.map(_safe_solve, levels, chunksize=1))
    errors = [item for item in solved_or_errors if isinstance(item, str)]
    if errors:
        raise RuntimeError(
            f"Boxoban oracle failed for {len(errors)} levels; first errors: {errors[:10]}"
        )
    solved = [item for item in solved_or_errors if isinstance(item, BoxobanSolution)]
    roles = _allocated(
        [role for role, count in BOXOBAN_ROLES.items() for _ in range(count)], "boxoban-roles"
    )
    tasks = _allocated(["choice"] * 1600 + ["noul"] * 400, "boxoban-tasks")
    image_root.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    role_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    with temporary.open("w", encoding="utf-8") as output:
        for index, solution in enumerate(solved):
            role, task = roles[index], tasks[index]
            image_path = image_root / f"boxoban-{index:05d}.png"
            image_path.write_bytes(solution.png)
            options = [{"id": action, "text": f"move {action}"} for action in DIRECTIONS]
            if task == "choice":
                row_options = options
                target: bool | list[str] = solution.optimal_actions
                target_kind = "set"
                candidate_action = None
                question = "Which move begins a shortest solution without entering a deadlock?"
            else:
                negatives = [
                    action for action in DIRECTIONS if action not in solution.optimal_actions
                ]
                candidate_action = (
                    solution.optimal_actions[index % len(solution.optimal_actions)]
                    if index % 2 == 0
                    else negatives[index % len(negatives)]
                )
                row_options = []
                target = candidate_action in solution.optimal_actions
                target_kind = "binary"
                question = f"Does moving {candidate_action} preserve a shortest solution?"
            level = solution.level
            sample_id = f"boxoban:{index:05d}"
            row = {
                "schema_version": 2,
                "sample_id": sample_id,
                "root_id": sample_id,
                "group_id": str(level["level_id"]),
                "source": "boxoban",
                "source_version": BOXOBAN_REVISION,
                "source_bucket": "interactive_oracle",
                "license": "Apache-2.0",
                "split": ROLE_SPLITS[role],
                "decision_role": role,
                "task_type": task,
                "state_text": "Push every crate onto a green goal. Crates cannot be pulled.",
                "question": question,
                "image": str(image_path),
                "image_metadata": {"width": 320, "height": 320, "observation": "full_board_rgb"},
                "allowed_history": [],
                "input_track": "interactive_fully_observed",
                "options": row_options,
                "candidates": row_options,
                "target_kind": target_kind,
                "target": target,
                "label_origin": "programmatic",
                "origin_label_method": "exact_astar_push_state_search",
                "language": "en",
                "generator_revision": "boxoban-astar-oracle-v1",
                "quality": {
                    "level_id": level["level_id"],
                    "episode_id": level["level_id"],
                    "step": 0,
                    "difficulty": level["difficulty"],
                    "upstream_split": level["upstream_split"],
                    "level_sha256": level["level_sha256"],
                    "shortest_distance": solution.shortest_distance,
                    "all_optimal_actions": solution.optimal_actions,
                    "oracle_q": {
                        action: (
                            -solution.shortest_distance
                            if action in solution.optimal_actions
                            else None
                        )
                        for action in DIRECTIONS
                    },
                    "oracle_q_coverage": (
                        "all_optimal_actions_exact_nonoptimal_actions_not_expanded"
                    ),
                    "candidate_action": candidate_action,
                    "remaining_horizon": solution.shortest_distance,
                    "value_target": 1.0 / (1.0 + solution.shortest_distance / 100.0),
                    "explored_states": solution.explored_states,
                    "oracle_verified": True,
                    "terminal_condition": "all_boxes_on_goals",
                },
                "teacher_only": False,
            }
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            role_counts[role] += 1
            task_counts[task] += 1
            difficulty_counts[str(level["difficulty"])] += 1
    temporary.replace(destination)
    report = {
        "schema_version": 1,
        "stage": "boxoban",
        "questions": len(solved),
        "roles": dict(sorted(role_counts.items())),
        "tasks": dict(sorted(task_counts.items())),
        "difficulties": dict(sorted(difficulty_counts.items())),
        "images": str(image_root),
        "manifest": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    destination.with_suffix(".build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
