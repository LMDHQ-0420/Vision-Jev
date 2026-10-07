"""Reproducible CPU oracles for MiniGrid and BabyAI interactive data."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_jev.data.interactive_generate import (
    ACTION_NAMES,
    DIRECTION_NAMES,
    DIRECTION_VECTORS,
    MINIGRID_REVISION,
    ROLE_SPLITS,
    _crop_frame_to_grid,
    _navigation_map,
    _rank,
    _shortest_navigation,
    _transition,
)

ALL_ACTION_NAMES = {
    0: "turn_left",
    1: "turn_right",
    2: "move_forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

STAGE_SPECS: dict[str, dict[str, Any]] = {
    "minigrid_tools": {
        "questions": 3000,
        "choice": 2400,
        "source": "minigrid",
        "environments": (
            ("MiniGrid-DoorKey-5x5-v0", 70),
            ("MiniGrid-DoorKey-6x6-v0", 500),
            ("MiniGrid-DoorKey-8x8-v0", 1630),
            ("MiniGrid-Unlock-v0", 500),
            ("MiniGrid-UnlockPickup-v0", 300),
        ),
        "roles": {
            "train": 1875,
            "dev": 281,
            "calibration": 281,
            "threshold": 141,
            "audit": 141,
            "test": 281,
        },
    },
    "babyai_grounded": {
        "questions": 2500,
        "choice": 2000,
        "source": "babyai",
        "environments": (
            ("BabyAI-GoToLocalS5N2-v0", 900),
            ("BabyAI-PickupLoc-v0", 800),
            ("BabyAI-OpenDoorLoc-v0", 800),
        ),
        "roles": {
            "train": 1563,
            "dev": 234,
            "calibration": 234,
            "threshold": 117,
            "audit": 117,
            "test": 235,
        },
    },
    "minigrid_hazards": {
        "questions": 2250,
        "choice": 1350,
        "source": "minigrid",
        "environments": (
            ("MiniGrid-LavaCrossingS9N1-v0", 650),
            ("MiniGrid-LavaCrossingS9N2-v0", 600),
            ("MiniGrid-LavaCrossingS9N3-v0", 550),
            ("MiniGrid-LavaCrossingS11N5-v0", 450),
        ),
        "roles": {
            "train": 1406,
            "dev": 211,
            "calibration": 211,
            "threshold": 105,
            "audit": 105,
            "test": 212,
        },
    },
}


@dataclass(frozen=True)
class OracleResult:
    environment_id: str
    seed: int
    mission: str
    direction: int
    max_steps: int
    shortest_distance: int
    action_mask: dict[str, bool]
    oracle_q: dict[str, int | None]
    optimal_actions: list[str]
    state_sha256: str
    png: bytes
    width: int
    height: int
    explored_states: int


ToolObject = tuple[int, int, str, str, int]
ToolState = tuple[tuple[int, int], int, tuple[str, str] | None, tuple[ToolObject, ...]]


@dataclass(frozen=True)
class ToolWorld:
    walls: frozenset[tuple[int, int]]
    goals: frozenset[tuple[int, int]]
    mission_kind: str


def _tool_world(env: Any) -> tuple[ToolWorld, ToolState]:
    unwrapped = env.unwrapped
    walls: set[tuple[int, int]] = set()
    goals: set[tuple[int, int]] = set()
    objects: list[ToolObject] = []
    for x in range(unwrapped.width):
        for y in range(unwrapped.height):
            cell = unwrapped.grid.get(x, y)
            if cell is None:
                continue
            if cell.type == "wall":
                walls.add((x, y))
            elif cell.type == "goal":
                goals.add((x, y))
            elif cell.type == "door":
                state = 0 if cell.is_open else 2 if cell.is_locked else 1
                objects.append((x, y, "door", str(cell.color), state))
            elif cell.type in {"key", "box", "ball"}:
                objects.append((x, y, str(cell.type), str(cell.color), 0))
    carrying = (
        None
        if unwrapped.carrying is None
        else (str(unwrapped.carrying.type), str(unwrapped.carrying.color))
    )
    mission = str(unwrapped.mission)
    mission_kind = (
        "goal" if "goal" in mission else "pickup_box" if "pick up" in mission else "open_door"
    )
    world = ToolWorld(frozenset(walls), frozenset(goals), mission_kind)
    state: ToolState = (
        tuple(int(value) for value in unwrapped.agent_pos),
        int(unwrapped.agent_dir),
        carrying,
        tuple(sorted(objects)),
    )
    return world, state


def _tool_transition(world: ToolWorld, state: ToolState, action: int) -> tuple[ToolState, bool]:
    position, direction, carrying, objects = state
    if action == 0:
        return (position, (direction - 1) % 4, carrying, objects), False
    if action == 1:
        return (position, (direction + 1) % 4, carrying, objects), False
    dx, dy = DIRECTION_VECTORS[direction]
    front = (position[0] + dx, position[1] + dy)
    by_position = {(item[0], item[1]): item for item in objects}
    front_object = by_position.get(front)
    if action == 2:
        blocked = front in world.walls or (
            front_object is not None and not (front_object[2] == "door" and front_object[4] == 0)
        )
        if blocked:
            return state, False
        successor = (front, direction, carrying, objects)
        return successor, world.mission_kind == "goal" and front in world.goals
    if action == 3:
        if (
            carrying is not None
            or front_object is None
            or front_object[2] not in {"key", "box", "ball"}
        ):
            return state, False
        successor_objects = tuple(item for item in objects if item != front_object)
        successor_carrying = (front_object[2], front_object[3])
        successor = (position, direction, successor_carrying, successor_objects)
        success = world.mission_kind == "pickup_box" and front_object[2] == "box"
        return successor, success
    if action == 4:
        if (
            carrying is None
            or front in world.walls
            or front in world.goals
            or front_object is not None
        ):
            return state, False
        successor_objects = tuple(
            sorted((*objects, (front[0], front[1], carrying[0], carrying[1], 0)))
        )
        return (position, direction, None, successor_objects), False
    if action == 5:
        if front_object is None or front_object[2] != "door":
            return state, False
        door_state = front_object[4]
        if door_state == 2:
            if carrying != ("key", front_object[3]):
                return state, False
            next_door_state = 0
        elif door_state == 1:
            next_door_state = 0
        else:
            next_door_state = 1
        successor_objects = tuple(
            sorted(
                (item if item != front_object else (*item[:4], next_door_state)) for item in objects
            )
        )
        successor = (position, direction, carrying, successor_objects)
        success = world.mission_kind == "open_door" and next_door_state == 0
        return successor, success
    return state, False


def _optimal_tool_actions(
    world: ToolWorld, initial: ToolState, *, node_limit: int = 200_000
) -> tuple[dict[int, int], dict[int, bool], int]:
    queue: deque[tuple[ToolState, int, int]] = deque()
    seen: set[tuple[ToolState, int]] = set()
    legal: dict[int, bool] = {}
    distances: dict[int, int] = {}
    best: int | None = None
    for action in ALL_ACTION_NAMES:
        successor, success = _tool_transition(world, initial, action)
        legal[action] = successor != initial or success
        if success:
            distances[action] = 1
            best = 1
        elif successor != initial:
            marker = (successor, action)
            seen.add(marker)
            queue.append((successor, 1, action))
    explored = 0
    while queue:
        state, distance, first_action = queue.popleft()
        if best is not None and distance >= best:
            break
        explored += 1
        if explored > node_limit:
            raise RuntimeError(f"symbolic oracle exceeded {node_limit} states")
        for action in ALL_ACTION_NAMES:
            successor, success = _tool_transition(world, state, action)
            if success:
                candidate = distance + 1
                if best is None:
                    best = candidate
                if candidate == best:
                    distances[first_action] = candidate
                continue
            if successor == state:
                continue
            marker = (successor, first_action)
            if marker in seen:
                continue
            seen.add(marker)
            queue.append((successor, distance + 1, first_action))
    if not distances:
        raise RuntimeError("symbolic tool state has no successful continuation")
    return distances, legal, explored


def _state_key(env: Any) -> tuple[Any, ...]:
    unwrapped = env.unwrapped
    carrying = (
        None
        if unwrapped.carrying is None
        else tuple(int(value) for value in unwrapped.carrying.encode())
    )
    return (
        unwrapped.grid.encode().tobytes(),
        tuple(int(value) for value in unwrapped.agent_pos),
        int(unwrapped.agent_dir),
        carrying,
    )


def _optimal_first_action_distances(
    env: Any, *, node_limit: int = 12_000
) -> tuple[dict[int, int], dict[int, bool], int]:
    """Find the global shortest distance and every tied optimal first action."""
    initial_key = _state_key(env)
    queue: deque[tuple[Any, int, int]] = deque()
    seen: set[tuple[tuple[Any, ...], int]] = set()
    distances: dict[int, int] = {}
    best_distance: int | None = None
    legal: dict[int, bool] = {}
    for action in ALL_ACTION_NAMES:
        successor = copy.deepcopy(env)
        _, reward, terminated, truncated, _ = successor.step(action)
        successor_key = _state_key(successor)
        legal[action] = successor_key != initial_key or reward != 0
        if reward > 0:
            distances[action] = 1
            best_distance = 1
        elif not terminated and not truncated:
            marker = (successor_key, action)
            if marker not in seen:
                seen.add(marker)
                queue.append((successor, 1, action))

    explored = 0
    while queue:
        current, distance, first_action = queue.popleft()
        if best_distance is not None and distance >= best_distance:
            break
        explored += 1
        if explored > node_limit:
            raise RuntimeError(f"oracle exceeded {node_limit} states")
        for action in ALL_ACTION_NAMES:
            successor = copy.deepcopy(current)
            _, reward, terminated, truncated, _ = successor.step(action)
            if reward > 0:
                candidate_distance = distance + 1
                if best_distance is None:
                    best_distance = candidate_distance
                if candidate_distance == best_distance:
                    distances[first_action] = candidate_distance
                break
            if terminated or truncated:
                continue
            marker = (_state_key(successor), first_action)
            if marker in seen:
                continue
            seen.add(marker)
            queue.append((successor, distance + 1, first_action))
    if not distances:
        raise RuntimeError("environment state has no successful continuation")
    return distances, legal, explored


def _encode_frame(frame: Any, env: Any) -> tuple[bytes, int, int]:
    from PIL import Image

    cropped = _crop_frame_to_grid(frame, env)
    output = io.BytesIO()
    Image.fromarray(cropped).save(output, format="PNG", optimize=True)
    return output.getvalue(), int(cropped.shape[1]), int(cropped.shape[0])


def _solve_seed(job: tuple[str, int, str]) -> OracleResult:
    environment_id, seed, mode = job
    import gymnasium as gym
    import minigrid  # noqa: F401

    env = gym.make(environment_id, render_mode="rgb_array", disable_env_checker=True)
    observation, _ = env.reset(seed=seed)
    unwrapped = env.unwrapped
    if mode in {"state_search_varied", "tool_symbolic_varied"}:
        walk_steps = _rank_int("state-walk-length", environment_id, seed) % 25
        for step in range(walk_steps):
            initial_key = _state_key(env)
            actions = sorted(
                (0, 1, 2, 3, 5),
                key=lambda action: _rank_int(
                    "state-walk-action", environment_id, seed, step, action
                ),
            )
            for action in actions:
                successor = copy.deepcopy(env)
                _, reward, terminated, truncated, _ = successor.step(action)
                if reward > 0 or terminated or truncated or _state_key(successor) == initial_key:
                    continue
                env = successor
                unwrapped = env.unwrapped
                break
    if mode == "navigation_varied":
        navigation_map = _navigation_map(env)
        candidates = [
            (position, direction, navigation_map.initial_open_doors)
            for position in sorted(navigation_map.passable - navigation_map.goals)
            for direction in range(4)
        ]
        selected = candidates[_rank_int(environment_id, seed) % len(candidates)]
        unwrapped.agent_pos = selected[0]
        unwrapped.agent_dir = selected[1]
    state_key = _state_key(env)
    if mode in {"navigation", "navigation_varied"}:
        navigation_map = _navigation_map(env)
        state = (
            tuple(int(value) for value in unwrapped.agent_pos),
            int(unwrapped.agent_dir),
            navigation_map.initial_open_doors,
        )
        shortest, optimal = _shortest_navigation(navigation_map, state)
        distances: dict[int, int] = {}
        legal: dict[int, bool] = {}
        for action in ACTION_NAMES:
            successor, is_legal = _transition(navigation_map, state, action)
            legal[action] = is_legal
            if not is_legal:
                continue
            distances[action] = (
                1
                if successor[0] in navigation_map.goals
                else 1 + _shortest_navigation(navigation_map, successor)[0]
            )
        explored = 0
    elif mode in {"tool_symbolic", "tool_symbolic_varied"}:
        tool_world, tool_state = _tool_world(env)
        distances, legal, explored = _optimal_tool_actions(tool_world, tool_state)
        shortest = min(distances.values())
        optimal = {action for action, distance in distances.items() if distance == shortest}
    else:
        distances, legal, explored = _optimal_first_action_distances(env)
        shortest = min(distances.values())
        optimal = {action for action, distance in distances.items() if distance == shortest}
    frame = unwrapped.get_frame(highlight=False, tile_size=16, agent_pov=False)
    png, width, height = _encode_frame(frame, env)
    result = OracleResult(
        environment_id=environment_id,
        seed=seed,
        mission=str(observation.get("mission", "complete the mission")),
        direction=int(unwrapped.agent_dir),
        max_steps=int(unwrapped.max_steps),
        shortest_distance=int(shortest),
        action_mask={ALL_ACTION_NAMES[a]: bool(legal.get(a, False)) for a in ALL_ACTION_NAMES},
        oracle_q={
            ALL_ACTION_NAMES[a]: (-int(distances[a]) if a in distances else None)
            for a in ALL_ACTION_NAMES
        },
        optimal_actions=sorted(ALL_ACTION_NAMES[action] for action in optimal),
        state_sha256=hashlib.sha256(repr(state_key).encode()).hexdigest(),
        png=png,
        width=width,
        height=height,
        explored_states=explored,
    )
    env.close()
    return result


def _allocated(values: list[str], seed: str) -> list[str]:
    return [
        item[1]
        for item in sorted(enumerate(values), key=lambda item: _rank(seed, f"{item[0]}:{item[1]}"))
    ]


def _stage_jobs(stage: str) -> list[tuple[str, int, str]]:
    spec = STAGE_SPECS[stage]
    default_mode = {
        "minigrid_hazards": "navigation_varied",
        "minigrid_tools": "tool_symbolic_varied",
    }.get(stage, "state_search")
    jobs: list[tuple[str, int, str]] = []
    offset = {
        "minigrid_tools": 100_000,
        "minigrid_hazards": 200_000,
        "babyai_grounded": 300_000,
    }[stage]
    for environment_index, (environment_id, quota) in enumerate(spec["environments"]):
        mode = (
            "tool_symbolic"
            if stage == "minigrid_tools" and environment_id == "MiniGrid-DoorKey-8x8-v0"
            else default_mode
        )
        if stage == "minigrid_hazards":
            candidate_count = int(quota) * 8
        elif stage == "minigrid_tools":
            candidate_count = int(quota) * 2
        else:
            candidate_count = int(quota) + max(50, int(quota) // 10)
        for index in range(candidate_count):
            seed = offset + environment_index * 1_000_000 + index
            jobs.append((str(environment_id), seed, mode))
    return jobs


def _row_from_result(
    result: OracleResult,
    *,
    stage: str,
    index: int,
    role: str,
    task: str,
    image_path: Path,
) -> dict[str, Any]:
    action_options = [
        {"id": action_name, "text": action_name.replace("_", " ")}
        for action_name in ALL_ACTION_NAMES.values()
    ]
    if task == "choice":
        options = action_options
        target: bool | list[str] = result.optimal_actions
        target_kind = "set"
        candidate_action = None
        question = "Which action is on a shortest successful path?"
    else:
        ordered = list(ALL_ACTION_NAMES.values())
        if index % 2 == 0:
            candidate_action = result.optimal_actions[index % len(result.optimal_actions)]
        else:
            negatives = [action for action in ordered if action not in result.optimal_actions]
            candidate_action = negatives[index % len(negatives)]
        options = []
        target = candidate_action in result.optimal_actions
        target_kind = "binary"
        question = (
            f"Does action '{candidate_action.replace('_', ' ')}' preserve a shortest successful "
            "path within the remaining horizon?"
        )
    sample_id = f"{stage}:{index:05d}"
    return {
        "schema_version": 2,
        "sample_id": sample_id,
        "root_id": sample_id,
        "group_id": f"{result.environment_id}:seed:{result.seed}",
        "source": str(STAGE_SPECS[stage]["source"]),
        "source_version": MINIGRID_REVISION,
        "source_bucket": "interactive_oracle",
        "license": "Apache-2.0",
        "split": ROLE_SPLITS[role],
        "decision_role": role,
        "task_type": task,
        "state_text": f"Mission: {result.mission} Agent faces {DIRECTION_NAMES[result.direction]}.",
        "question": question,
        "image": str(image_path),
        "image_metadata": {
            "width": result.width,
            "height": result.height,
            "observation": "fully_observed_rgb",
        },
        "allowed_history": [],
        "input_track": "interactive_fully_observed",
        "options": options,
        "candidates": options,
        "target_kind": target_kind,
        "target": target,
        "label_origin": "programmatic",
        "origin_label_method": (
            "exact_bfs_validated_minigrid_transition_model"
            if stage == "minigrid_tools"
            else "exact_bfs_environment_dynamics"
        ),
        "language": "en",
        "generator_revision": (
            "minigrid-tool-symbolic-oracle-v1"
            if stage == "minigrid_tools"
            else "minigrid-state-search-oracle-v1"
        ),
        "quality": {
            "environment_id": result.environment_id,
            "level_seed": result.seed,
            "episode_id": f"{result.environment_id}:seed:{result.seed}",
            "step": 0,
            "mission": result.mission,
            "state_sha256": result.state_sha256,
            "shortest_distance": result.shortest_distance,
            "all_optimal_actions": result.optimal_actions,
            "action_mask": result.action_mask,
            "oracle_q": result.oracle_q,
            "oracle_q_coverage": "all_optimal_actions_exact_nonoptimal_actions_not_expanded",
            "candidate_action": candidate_action,
            "remaining_horizon": result.shortest_distance,
            "value_target": max(0.0, 1.0 - 0.9 * result.shortest_distance / result.max_steps),
            "explored_states": result.explored_states,
            "oracle_verified": True,
            "terminal_condition": "positive_environment_reward",
        },
        "teacher_only": False,
    }


def generate_minigrid_stage(
    *, stage: str, destination: Path, image_root: Path, max_workers: int = 32
) -> dict[str, Any]:
    """Generate one pinned stage, rejecting duplicate visible states."""
    if stage not in STAGE_SPECS:
        raise ValueError(f"unknown MiniGrid stage: {stage}")
    spec = STAGE_SPECS[stage]
    questions = int(spec["questions"])
    roles = _allocated(
        [role for role, count in spec["roles"].items() for _ in range(int(count))],
        f"{stage}-roles",
    )
    choice = int(spec["choice"])
    tasks = _allocated(["choice"] * choice + ["noul"] * (questions - choice), f"{stage}-tasks")
    required_by_environment = {str(env): int(count) for env, count in spec["environments"]}
    accepted_by_environment: Counter[str] = Counter()
    accepted: list[OracleResult] = []
    seen_png: set[bytes] = set()
    failures: Counter[str] = Counter()
    jobs = _stage_jobs(stage)
    executor = ProcessPoolExecutor(max_workers=max_workers)
    try:
        for result_or_error in executor.map(_safe_solve_seed, jobs, chunksize=1):
            if isinstance(result_or_error, str):
                failures[result_or_error] += 1
                continue
            result = result_or_error
            env_id = result.environment_id
            if accepted_by_environment[env_id] >= required_by_environment[env_id]:
                continue
            png_hash = hashlib.sha256(result.png).digest()
            if png_hash in seen_png:
                failures["duplicate_visible_state"] += 1
                continue
            seen_png.add(png_hash)
            accepted.append(result)
            accepted_by_environment[env_id] += 1
            if len(accepted) % 100 == 0:
                print(f"{stage} decisions: {len(accepted)}/{questions}", flush=True)
            if len(accepted) == questions:
                break
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    shortfalls = {
        env: required_by_environment[env] - accepted_by_environment[env]
        for env in required_by_environment
        if accepted_by_environment[env] < required_by_environment[env]
    }
    if shortfalls:
        raise RuntimeError(f"{stage} generation shortfall {shortfalls}; failures={dict(failures)}")

    accepted.sort(key=lambda item: (item.environment_id, item.seed))
    image_root.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    role_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    with temporary.open("w", encoding="utf-8") as output:
        for index, result in enumerate(accepted):
            image_path = image_root / f"{stage}-{index:05d}.png"
            image_path.write_bytes(result.png)
            row = _row_from_result(
                result,
                stage=stage,
                index=index,
                role=roles[index],
                task=tasks[index],
                image_path=image_path,
            )
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            role_counts[roles[index]] += 1
            task_counts[tasks[index]] += 1
    temporary.replace(destination)
    report = {
        "schema_version": 1,
        "stage": stage,
        "questions": questions,
        "roles": dict(sorted(role_counts.items())),
        "tasks": dict(sorted(task_counts.items())),
        "environments": dict(sorted(accepted_by_environment.items())),
        "unique_visible_states": len(seen_png),
        "rejected": dict(sorted(failures.items())),
        "images": str(image_root),
        "manifest": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    destination.with_suffix(".build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _safe_solve_seed(job: tuple[str, int, str]) -> OracleResult | str:
    try:
        return _solve_seed(job)
    except Exception as exc:  # worker failures are counted and replaced by later candidates
        return f"{type(exc).__name__}: {exc}"


def _rank_int(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("\0".join(map(str, parts)).encode()).digest()[:8], "big")
