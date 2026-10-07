"""CPU-only oracle generation for staged interactive RLCD datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MINIGRID_REVISION = "90928729376741a41222a257911343b97103b548"
NAVIGATION_ENVIRONMENTS = (
    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-Empty-16x16-v0",
    "MiniGrid-FourRooms-v0",
    "MiniGrid-MultiRoom-N2-S4-v0",
    "MiniGrid-MultiRoom-N4-S5-v1",
    "MiniGrid-MultiRoom-N6-v0",
)
RANDOM_NAVIGATION_ENVIRONMENTS = (
    "MiniGrid-FourRooms-v0",
    "MiniGrid-MultiRoom-N2-S4-v0",
    "MiniGrid-MultiRoom-N4-S5-v1",
    "MiniGrid-MultiRoom-N6-v0",
)
ACTION_NAMES = {0: "turn_left", 1: "turn_right", 2: "move_forward", 5: "toggle"}
DIRECTION_VECTORS = ((1, 0), (0, 1), (-1, 0), (0, -1))
DIRECTION_NAMES = ("east", "south", "west", "north")
ROLE_COUNTS = {
    "train": 2500,
    "dev": 375,
    "calibration": 375,
    "threshold": 188,
    "audit": 187,
    "test": 375,
}
ROLE_SPLITS = {
    "train": "train",
    "dev": "dev",
    "calibration": "calibration",
    "threshold": "calibration",
    "audit": "calibration",
    "test": "test",
}


@dataclass(frozen=True)
class NavigationMap:
    width: int
    height: int
    passable: frozenset[tuple[int, int]]
    goals: frozenset[tuple[int, int]]
    doors: dict[tuple[int, int], int]
    initial_open_doors: int


NavigationState = tuple[tuple[int, int], int, int]


def _rank(seed: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode()).digest()


def _crop_frame_to_content(frame: Any) -> Any:
    """Remove unused black canvas while preserving the complete rendered grid."""
    occupied = frame.any(axis=2)
    ys, xs = occupied.nonzero()
    if not len(xs):
        raise ValueError("environment renderer returned a blank frame")
    return frame[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _crop_frame_to_grid(frame: Any, env: Any, *, tile_size: int = 16) -> Any:
    """Crop RoomGrid's unused canvas using the bounding box of real objects/walls."""
    unwrapped = env.unwrapped
    occupied = [
        (x, y)
        for x in range(unwrapped.width)
        for y in range(unwrapped.height)
        if unwrapped.grid.get(x, y) is not None
    ]
    if not occupied:
        return _crop_frame_to_content(frame)
    xs, ys = zip(*occupied, strict=True)
    return frame[
        min(ys) * tile_size : (max(ys) + 1) * tile_size,
        min(xs) * tile_size : (max(xs) + 1) * tile_size,
    ]


def _navigation_map(env: Any) -> NavigationMap:
    unwrapped = env.unwrapped
    passable: set[tuple[int, int]] = set()
    goals: set[tuple[int, int]] = set()
    door_positions: list[tuple[int, int]] = []
    initially_open: set[tuple[int, int]] = set()
    for x in range(unwrapped.width):
        for y in range(unwrapped.height):
            cell = unwrapped.grid.get(x, y)
            if cell is None or cell.type in {"floor", "goal"}:
                passable.add((x, y))
            if cell is not None and cell.type == "goal":
                goals.add((x, y))
            if cell is not None and cell.type == "door":
                if cell.is_locked:
                    raise ValueError("navigation stage does not permit locked doors")
                door_positions.append((x, y))
                if cell.is_open:
                    initially_open.add((x, y))
    doors = {position: index for index, position in enumerate(sorted(door_positions))}
    open_mask = sum(1 << doors[position] for position in initially_open)
    return NavigationMap(
        width=int(unwrapped.width),
        height=int(unwrapped.height),
        passable=frozenset(passable),
        goals=frozenset(goals),
        doors=doors,
        initial_open_doors=open_mask,
    )


def _transition(
    navigation_map: NavigationMap, state: NavigationState, action: int
) -> tuple[NavigationState, bool]:
    position, direction, open_doors = state
    if action == 0:
        return (position, (direction - 1) % 4, open_doors), True
    if action == 1:
        return (position, (direction + 1) % 4, open_doors), True
    dx, dy = DIRECTION_VECTORS[direction]
    front = (position[0] + dx, position[1] + dy)
    if action == 5:
        door_index = navigation_map.doors.get(front)
        if door_index is None or (open_doors >> door_index) & 1:
            return state, False
        return (position, direction, open_doors | (1 << door_index)), True
    if action != 2:
        return state, False
    door_index = navigation_map.doors.get(front)
    can_enter = front in navigation_map.passable or (
        door_index is not None and bool((open_doors >> door_index) & 1)
    )
    if not can_enter:
        return state, False
    return (front, direction, open_doors), True


def _shortest_navigation(
    navigation_map: NavigationMap, initial: NavigationState
) -> tuple[int, set[int]]:
    if initial[0] in navigation_map.goals:
        return 0, set()
    queue: deque[tuple[NavigationState, int, int | None]] = deque([(initial, 0, None)])
    seen: set[tuple[NavigationState, int | None]] = {(initial, None)}
    best_distance: int | None = None
    optimal_first_actions: set[int] = set()
    while queue:
        state, distance, first_action = queue.popleft()
        if best_distance is not None and distance >= best_distance:
            continue
        for action in ACTION_NAMES:
            next_state, legal = _transition(navigation_map, state, action)
            if not legal:
                continue
            next_first = action if first_action is None else first_action
            next_distance = distance + 1
            if next_state[0] in navigation_map.goals:
                if best_distance is None:
                    best_distance = next_distance
                if next_distance == best_distance:
                    optimal_first_actions.add(next_first)
                continue
            marker = (next_state, next_first)
            if marker in seen:
                continue
            seen.add(marker)
            queue.append((next_state, next_distance, next_first))
    if best_distance is None:
        raise ValueError("navigation state has no path to a goal")
    return best_distance, optimal_first_actions


def generate_minigrid_navigation(
    *, destination: Path, image_root: Path, questions: int = 4000
) -> dict[str, Any]:
    """Generate fully observed navigation decisions with exact BFS labels."""
    if questions != 4000:
        raise ValueError("the pinned navigation stage requires exactly 4000 questions")
    import gymnasium as gym
    import minigrid  # noqa: F401
    from PIL import Image

    # Rank stable identities rather than repeated string values.
    roles = [role for role, count in ROLE_COUNTS.items() for _ in range(count)]
    roles = [
        item[1]
        for item in sorted(
            enumerate(roles),
            key=lambda item: _rank("minigrid-navigation-roles", f"{item[0]}:{item[1]}"),
        )
    ]
    tasks = ["choice"] * 3200 + ["noul"] * 800
    tasks = [
        item[1]
        for item in sorted(
            enumerate(tasks),
            key=lambda item: _rank("minigrid-navigation-tasks", f"{item[0]}:{item[1]}"),
        )
    ]
    image_root.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    role_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    environment_counts: Counter[str] = Counter()
    # The fixed Empty environments ignore their seed. Keep one canonical state
    # from each instead of silently duplicating it hundreds of times. The
    # procedural environments supply the remaining unique visual states.
    environments = list(NAVIGATION_ENVIRONMENTS[:3])
    environments.extend(
        RANDOM_NAVIGATION_ENVIRONMENTS[index % len(RANDOM_NAVIGATION_ENVIRONMENTS)]
        for index in range(questions - len(environments))
    )
    seen_visual_states: set[bytes] = set()
    with temporary.open("w", encoding="utf-8") as output:
        for index in range(questions):
            role = roles[index]
            task = tasks[index]
            environment_id = environments[index]
            seed = index + 10_000
            env = gym.make(environment_id, render_mode="rgb_array")
            observation, _ = env.reset(seed=seed)
            unwrapped = env.unwrapped
            navigation_map = _navigation_map(env)
            initial_state: NavigationState = (
                tuple(int(value) for value in unwrapped.agent_pos),
                int(unwrapped.agent_dir),
                navigation_map.initial_open_doors,
            )
            shortest_distance, optimal_actions = _shortest_navigation(navigation_map, initial_state)
            action_q: dict[str, int | None] = {}
            action_mask: dict[str, bool] = {}
            for action, action_name in ACTION_NAMES.items():
                next_state, legal = _transition(navigation_map, initial_state, action)
                action_mask[action_name] = legal
                if not legal:
                    action_q[action_name] = None
                    continue
                if next_state[0] in navigation_map.goals:
                    action_q[action_name] = -1
                    continue
                remaining, _ = _shortest_navigation(navigation_map, next_state)
                action_q[action_name] = -(remaining + 1)

            image_path = image_root / f"navigation-{index:05d}.png"
            # RoomGrid's absolute placement inside the official canvas is part
            # of the state; cropping would collapse translated layouts.
            frame = unwrapped.get_frame(highlight=False, tile_size=16, agent_pov=False)
            visual_state = hashlib.sha256(frame.tobytes()).digest()
            if visual_state in seen_visual_states:
                raise RuntimeError(
                    f"duplicate visible state at navigation row {index}: "
                    f"{environment_id} seed {seed}"
                )
            seen_visual_states.add(visual_state)
            Image.fromarray(frame).save(image_path)
            action_options = [
                {"id": action_name, "text": action_name.replace("_", " ")}
                for action_name in ACTION_NAMES.values()
            ]
            optimal_ids = sorted(ACTION_NAMES[action] for action in optimal_actions)
            group_id = f"minigrid:{environment_id}:seed:{seed}"
            sample_id = f"minigrid-navigation:{index:05d}"
            mission = str(observation.get("mission", "reach the goal"))
            if task == "choice":
                question = "Which action is on a shortest successful path?"
                options = action_options
                target: bool | list[str] = optimal_ids
                target_kind = "set"
                candidate_action = None
            else:
                ordered_actions = sorted(ACTION_NAMES.values())
                if index % 2 == 0:
                    candidate_action = optimal_ids[0]
                else:
                    candidate_action = next(
                        action for action in ordered_actions if action not in optimal_ids
                    )
                question = (
                    f"Does action '{candidate_action.replace('_', ' ')}' preserve a shortest "
                    "successful path within the remaining horizon?"
                )
                options = []
                target = candidate_action in optimal_ids
                target_kind = "binary"
            row = {
                "schema_version": 2,
                "sample_id": sample_id,
                "root_id": sample_id,
                "group_id": group_id,
                "source": "minigrid",
                "source_version": MINIGRID_REVISION,
                "source_bucket": "interactive_oracle",
                "license": "Apache-2.0",
                "split": ROLE_SPLITS[role],
                "decision_role": role,
                "task_type": task,
                "state_text": (
                    f"Mission: {mission} Agent faces {DIRECTION_NAMES[initial_state[1]]}."
                ),
                "question": question,
                "image": str(image_path),
                "image_metadata": {
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "observation": "fully_observed_rgb",
                },
                "allowed_history": [],
                "input_track": "interactive_fully_observed",
                "options": options,
                "candidates": options,
                "target_kind": target_kind,
                "target": target,
                "label_origin": "programmatic",
                "origin_label_method": "exact_bfs_environment_dynamics",
                "language": "en",
                "generator_revision": "minigrid-navigation-oracle-v1",
                "quality": {
                    "environment_id": environment_id,
                    "level_seed": seed,
                    "episode_id": group_id,
                    "step": 0,
                    "mission": mission,
                    "shortest_distance": shortest_distance,
                    "all_optimal_actions": optimal_ids,
                    "action_mask": action_mask,
                    "oracle_q": action_q,
                    "candidate_action": candidate_action,
                    "remaining_horizon": shortest_distance,
                    "value_target": max(
                        0.0, 1.0 - 0.9 * shortest_distance / int(unwrapped.max_steps)
                    ),
                    "oracle_verified": True,
                    "terminal_condition": "goal_reached",
                },
                "teacher_only": False,
            }
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            role_counts[role] += 1
            task_counts[task] += 1
            environment_counts[environment_id] += 1
            env.close()
            if (index + 1) % 250 == 0:
                print(f"MiniGrid navigation decisions: {index + 1}/{questions}", flush=True)
    temporary.replace(destination)
    report = {
        "schema_version": 1,
        "stage": "minigrid_navigation",
        "questions": questions,
        "roles": dict(sorted(role_counts.items())),
        "tasks": dict(sorted(task_counts.items())),
        "environments": dict(sorted(environment_counts.items())),
        "unique_visible_states": len(seen_visual_states),
        "images": str(image_root),
        "manifest": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    destination.with_suffix(".build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    status_path = destination.parent.parent / "generation-status.json"
    status = {
        "schema_version": 1,
        "status": "partial_oracle_decisions_4000_of_16000",
        "planned_decisions": 16000,
        "generated_decisions": 4000,
        "completed_stages": {"minigrid_navigation": report},
        "pending_stages": {
            "minigrid_tools": 3000,
            "minigrid_hazards": 2500,
            "babyai_grounded": 2500,
            "procgen_maze": 2000,
            "boxoban": 2000,
        },
    }
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
