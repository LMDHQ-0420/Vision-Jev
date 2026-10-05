#!/usr/bin/env python3
"""Generate the pinned Procgen Maze decision set in an isolated Python 3.10 env."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

PROCGEN_VERSION = "0.10.7"
PROCGEN_REVISION = "5e1dbf341d291eff40d1f9e0c0a0d5003643aebf"
ACTIONS = {1: "left", 3: "down", 5: "up", 7: "right"}
ROLE_COUNTS = {
    "train": 1250,
    "dev": 188,
    "calibration": 187,
    "threshold": 93,
    "audit": 94,
    "test": 188,
}
ROLE_SPLITS = {
    "train": "train",
    "dev": "dev",
    "calibration": "calibration",
    "threshold": "calibration",
    "audit": "calibration",
    "test": "test",
}


def _rank(seed: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode()).digest()


def _allocated(values: list[str], seed: str) -> list[str]:
    return [
        item[1]
        for item in sorted(
            enumerate(values), key=lambda item: _rank(seed, f"{item[0]}:{item[1]}")
        )
    ]


def _solve(job: tuple[int, str]) -> dict[str, Any]:
    import numpy as np
    from PIL import Image
    from procgen import ProcgenGym3Env

    level_seed, distribution_mode = job
    env = ProcgenGym3Env(
        num=1,
        env_name="maze",
        start_level=level_seed,
        num_levels=1,
        num_threads=0,
        render_mode="rgb_array",
        distribution_mode=distribution_mode,
    )
    _, observation, _ = env.observe()
    initial_frame = observation["rgb"][0].copy()
    initial_state = env.callmethod("get_state")[0]
    queue: deque[tuple[bytes, int, str | None]] = deque([(initial_state, 0, None)])
    seen: set[tuple[bytes, str]] = set()
    best_distance: int | None = None
    optimal_actions: set[str] = set()
    explored = 0
    while queue:
        state, distance, first_action = queue.popleft()
        if best_distance is not None and distance >= best_distance:
            break
        for action, action_name in ACTIONS.items():
            env.callmethod("set_state", [state])
            env.act(np.asarray([action], dtype=np.int32))
            reward, next_observation, _ = env.observe()
            next_first = action_name if first_action is None else first_action
            if float(reward[0]) > 0:
                candidate = distance + 1
                if best_distance is None:
                    best_distance = candidate
                if candidate == best_distance:
                    optimal_actions.add(next_first)
                continue
            next_state = env.callmethod("get_state")[0]
            # Time counters do not affect Maze transitions before the fixed 500-step
            # timeout. Visual-state deduplication removes no-op/cycle expansion while
            # retaining the sprite orientation visible to the model.
            visual_key = hashlib.sha256(next_observation["rgb"][0].tobytes()).digest()
            marker = (visual_key, next_first)
            if marker in seen:
                continue
            seen.add(marker)
            queue.append((next_state, distance + 1, next_first))
        explored += 1
        if explored > 20_000:
            env.close()
            raise RuntimeError(f"seed {level_seed} exceeded state-search budget")
    env.close()
    if best_distance is None or not optimal_actions:
        raise RuntimeError(f"seed {level_seed} has no proven solution")
    encoded = io.BytesIO()
    Image.fromarray(initial_frame).save(encoded, format="PNG", optimize=True)
    return {
        "level_seed": level_seed,
        "distribution_mode": distribution_mode,
        "shortest_distance": best_distance,
        "optimal_actions": sorted(optimal_actions),
        "explored_states": explored,
        "state_sha256": hashlib.sha256(initial_state).hexdigest(),
        "png": encoded.getvalue(),
    }


def _safe_solve(job: tuple[int, str]) -> dict[str, Any]:
    try:
        return {"ok": True, "result": _solve(job)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "job": job}


def generate(output: Path, image_root: Path, max_workers: int) -> dict[str, Any]:
    jobs = [
        (500_000 + index, "easy" if index % 2 == 0 else "hard")
        for index in range(2300)
    ]
    accepted: list[dict[str, Any]] = []
    seen_images: set[bytes] = set()
    failures: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for item in executor.map(_safe_solve, jobs, chunksize=1):
            if not item["ok"]:
                failures[str(item["error"])] += 1
                continue
            result = item["result"]
            image_hash = hashlib.sha256(result["png"]).digest()
            if image_hash in seen_images:
                failures["duplicate_visible_state"] += 1
                continue
            seen_images.add(image_hash)
            accepted.append(result)
            if len(accepted) % 100 == 0:
                print(f"Procgen Maze decisions: {len(accepted)}/2000", flush=True)
            if len(accepted) == 2000:
                break
    if len(accepted) != 2000:
        raise RuntimeError(f"Procgen shortfall: {len(accepted)}/2000; failures={dict(failures)}")

    roles = _allocated(
        [role for role, count in ROLE_COUNTS.items() for _ in range(count)], "procgen-roles"
    )
    tasks = _allocated(["choice"] * 1600 + ["noul"] * 400, "procgen-tasks")
    output.parent.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    role_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    with temporary.open("w", encoding="utf-8") as handle:
        for index, result in enumerate(accepted):
            role, task = roles[index], tasks[index]
            image_path = image_root / f"procgen-maze-{index:05d}.png"
            image_path.write_bytes(result["png"])
            options = [{"id": name, "text": f"move {name}"} for name in ACTIONS.values()]
            optimal = result["optimal_actions"]
            if task == "choice":
                row_options = options
                target: bool | list[str] = optimal
                target_kind = "set"
                candidate_action = None
                question = "Which move begins a shortest path to the cheese?"
            else:
                negatives = [name for name in ACTIONS.values() if name not in optimal]
                candidate_action = (
                    optimal[index % len(optimal)]
                    if index % 2 == 0
                    else negatives[index % len(negatives)]
                )
                row_options = []
                target = candidate_action in optimal
                target_kind = "binary"
                question = f"Does moving {candidate_action} preserve a shortest path to the cheese?"
            sample_id = f"procgen-maze:{index:05d}"
            seed = int(result["level_seed"])
            row = {
                "schema_version": 2,
                "sample_id": sample_id,
                "root_id": sample_id,
                "group_id": f"procgen:maze:seed:{seed}",
                "source": "procgen_maze",
                "source_version": PROCGEN_REVISION,
                "source_bucket": "interactive_oracle",
                "license": "MIT",
                "split": ROLE_SPLITS[role],
                "decision_role": role,
                "task_type": task,
                "state_text": "Navigate the mouse through the maze to the cheese.",
                "question": question,
                "image": str(image_path),
                "image_metadata": {"width": 64, "height": 64, "observation": "procgen_rgb"},
                "allowed_history": [],
                "input_track": "interactive_fully_observed",
                "options": row_options,
                "candidates": row_options,
                "target_kind": target_kind,
                "target": target,
                "label_origin": "programmatic",
                "origin_label_method": "exact_bfs_procgen_state_restore",
                "language": "en",
                "generator_revision": "procgen-maze-oracle-v1",
                "quality": {
                    "environment_id": "procgen-maze",
                    "procgen_version": PROCGEN_VERSION,
                    "level_seed": seed,
                    "episode_id": f"procgen:maze:seed:{seed}",
                    "step": 0,
                    "distribution_mode": result["distribution_mode"],
                    "state_sha256": result["state_sha256"],
                    "shortest_distance": result["shortest_distance"],
                    "all_optimal_actions": optimal,
                    "oracle_q": {
                        action: (-result["shortest_distance"] if action in optimal else None)
                        for action in ACTIONS.values()
                    },
                    "oracle_q_coverage": (
                        "all_optimal_actions_exact_nonoptimal_actions_not_expanded"
                    ),
                    "candidate_action": candidate_action,
                    "remaining_horizon": result["shortest_distance"],
                    "value_target": max(0.0, 1.0 - result["shortest_distance"] / 500.0),
                    "explored_states": result["explored_states"],
                    "oracle_verified": True,
                    "terminal_condition": "cheese_collected",
                },
                "teacher_only": False,
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            role_counts[role] += 1
            task_counts[task] += 1
            mode_counts[str(result["distribution_mode"])] += 1
    temporary.replace(output)
    report = {
        "schema_version": 1,
        "stage": "procgen_maze",
        "questions": 2000,
        "roles": dict(sorted(role_counts.items())),
        "tasks": dict(sorted(task_counts.items())),
        "distribution_modes": dict(sorted(mode_counts.items())),
        "unique_visible_states": len(seen_images),
        "rejected": dict(sorted(failures.items())),
        "images": str(image_root),
        "manifest": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.image_root, args.max_workers), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
