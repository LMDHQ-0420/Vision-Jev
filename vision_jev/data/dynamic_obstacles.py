"""Monte Carlo policy labels for MiniGrid Dynamic Obstacles."""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from vision_jev.data.interactive_generate import (
    DIRECTION_NAMES,
    MINIGRID_REVISION,
    ROLE_SPLITS,
    _crop_frame_to_grid,
    _navigation_map,
    _rank,
    _shortest_navigation,
)

DYNAMIC_ACTIONS = {0: "turn_left", 1: "turn_right", 2: "move_forward"}
DYNAMIC_ENVS = (
    "MiniGrid-Dynamic-Obstacles-Random-5x5-v0",
    "MiniGrid-Dynamic-Obstacles-Random-6x6-v0",
    "MiniGrid-Dynamic-Obstacles-8x8-v0",
)
DYNAMIC_ROLES = {
    "train": 156,
    "dev": 24,
    "calibration": 24,
    "threshold": 12,
    "audit": 12,
    "test": 22,
}
TRIALS = 64


def _rank_int(*parts: Any) -> int:
    digest = hashlib.sha256("\0".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _rollout(
    env: Any, environment_id: str, seed: int, first_action: int, trial: int
) -> tuple[str, int, float]:
    import numpy as np

    env.reset(seed=seed)
    env.unwrapped.np_random = np.random.default_rng(
        _rank_int("dynamic-rollout", environment_id, seed, first_action, trial)
    )
    _, reward, terminated, truncated, _ = env.step(first_action)
    if reward > 0:
        return "success", 1, float(reward)
    if terminated and reward < 0:
        return "collision", 1, float(reward)
    horizon = min(int(env.unwrapped.max_steps), 100)
    for step in range(1, horizon):
        unwrapped = env.unwrapped
        navigation_map = _navigation_map(env)
        state = (
            tuple(int(value) for value in unwrapped.agent_pos),
            int(unwrapped.agent_dir),
            navigation_map.initial_open_doors,
        )
        try:
            _, optimal = _shortest_navigation(navigation_map, state)
            candidates = sorted(action for action in optimal if action in DYNAMIC_ACTIONS)
        except ValueError:
            candidates = []
        if candidates:
            action = candidates[_rank_int(seed, first_action, trial, step) % len(candidates)]
        else:
            action = (0, 1)[_rank_int(seed, first_action, trial, step) % 2]
        _, reward, terminated, truncated, _ = env.step(action)
        if reward > 0:
            return "success", step + 1, float(reward)
        if terminated and reward < 0:
            return "collision", step + 1, float(reward)
        if terminated or truncated:
            return "timeout", step + 1, 0.0
    return "timeout", horizon, 0.0


def _evaluate(job: tuple[str, int]) -> dict[str, Any]:
    import gymnasium as gym
    import minigrid  # noqa: F401
    from PIL import Image

    environment_id, seed = job
    env = gym.make(environment_id, render_mode="rgb_array", disable_env_checker=True)
    observation, _ = env.reset(seed=seed)
    unwrapped = env.unwrapped
    frame = _crop_frame_to_grid(
        unwrapped.get_frame(highlight=False, tile_size=16, agent_pov=False), env
    )
    encoded = io.BytesIO()
    Image.fromarray(frame).save(encoded, format="PNG", optimize=True)
    direction = int(unwrapped.agent_dir)
    state_hash = hashlib.sha256(
        unwrapped.grid.encode().tobytes()
        + bytes(tuple(int(value) for value in unwrapped.agent_pos))
        + bytes([direction])
    ).hexdigest()
    mission = str(observation["mission"])
    outcomes: dict[str, dict[str, int]] = {}
    mean_returns: dict[str, float] = {}
    for action, name in DYNAMIC_ACTIONS.items():
        rollouts = [_rollout(env, environment_id, seed, action, trial) for trial in range(TRIALS)]
        counts = Counter(item[0] for item in rollouts)
        outcomes[name] = {
            "successes": counts["success"],
            "collisions": counts["collision"],
            "timeouts": counts["timeout"],
            "trials": TRIALS,
        }
        mean_returns[name] = sum(item[2] for item in rollouts) / TRIALS
    env.close()
    success_rates = {name: value["successes"] / TRIALS for name, value in outcomes.items()}
    best = max(mean_returns.values())
    optimal = sorted(name for name, value in mean_returns.items() if math.isclose(value, best))
    weights = {name: math.exp(5.0 * value) for name, value in mean_returns.items()}
    total = sum(weights.values())
    policy_probs = {name: weight / total for name, weight in weights.items()}
    return {
        "environment_id": environment_id,
        "seed": seed,
        "mission": mission,
        "direction": direction,
        "state_sha256": state_hash,
        "png": encoded.getvalue(),
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "outcomes": outcomes,
        "success_rates": success_rates,
        "mean_returns": mean_returns,
        "policy_probs": policy_probs,
        "optimal_actions": optimal,
    }


def _safe_evaluate(job: tuple[str, int]) -> dict[str, Any]:
    try:
        return {"ok": True, "result": _evaluate(job)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _allocated(values: list[str], seed: str) -> list[str]:
    ranked = sorted(enumerate(values), key=lambda item: _rank(seed, f"{item[0]}:{item[1]}"))
    return [item[1] for item in ranked]


def generate_dynamic_obstacles(
    *, destination: Path, image_root: Path, max_workers: int = 24
) -> dict[str, Any]:
    jobs = [(DYNAMIC_ENVS[index % len(DYNAMIC_ENVS)], 4_000_000 + index) for index in range(350)]
    accepted: list[dict[str, Any]] = []
    seen_images: set[bytes] = set()
    failures: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for item in executor.map(_safe_evaluate, jobs, chunksize=1):
            if not item["ok"]:
                failures[str(item["error"])] += 1
                continue
            result = item["result"]
            digest = hashlib.sha256(result["png"]).digest()
            if digest in seen_images:
                failures["duplicate_visible_state"] += 1
                continue
            seen_images.add(digest)
            accepted.append(result)
            if len(accepted) % 25 == 0:
                print(f"Dynamic Obstacles decisions: {len(accepted)}/250", flush=True)
            if len(accepted) == 250:
                break
    if len(accepted) != 250:
        raise RuntimeError(f"Dynamic Obstacles shortfall {len(accepted)}/250: {dict(failures)}")
    roles = _allocated(
        [role for role, count in DYNAMIC_ROLES.items() for _ in range(count)], "dynamic-roles"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    role_counts: Counter[str] = Counter()
    environment_counts: Counter[str] = Counter()
    with temporary.open("w", encoding="utf-8") as output:
        for index, result in enumerate(accepted):
            role = roles[index]
            image_path = image_root / f"dynamic-obstacles-{index:05d}.png"
            image_path.write_bytes(result["png"])
            options = [
                {"id": name, "text": name.replace("_", " ")} for name in DYNAMIC_ACTIONS.values()
            ]
            sample_id = f"minigrid-dynamic-obstacles:{index:05d}"
            row = {
                "schema_version": 2,
                "sample_id": sample_id,
                "root_id": sample_id,
                "group_id": f"{result['environment_id']}:seed:{result['seed']}",
                "source": "minigrid",
                "source_version": MINIGRID_REVISION,
                "source_bucket": "interactive_policy_rollout",
                "license": "Apache-2.0",
                "split": ROLE_SPLITS[role],
                "decision_role": role,
                "task_type": "choice",
                "state_text": (
                    f"Mission: {result['mission']} Agent faces "
                    f"{DIRECTION_NAMES[result['direction']]}."
                ),
                "question": (
                    "Under the fixed reactive policy, which first action had the highest measured "
                    "mean return across randomized obstacle rollouts?"
                ),
                "image": str(image_path),
                "image_metadata": {
                    "width": result["width"],
                    "height": result["height"],
                    "observation": "fully_observed_rgb",
                },
                "allowed_history": [],
                "input_track": "interactive_fully_observed",
                "options": options,
                "candidates": options,
                "target_kind": "set_empirical_policy",
                "target": result["optimal_actions"],
                "label_origin": "programmatic",
                "origin_label_method": "monte_carlo_fixed_reactive_policy",
                "language": "en",
                "generator_revision": "dynamic-obstacles-rollout-v1",
                "quality": {
                    "environment_id": result["environment_id"],
                    "level_seed": result["seed"],
                    "episode_id": f"{result['environment_id']}:seed:{result['seed']}",
                    "step": 0,
                    "mission": result["mission"],
                    "state_sha256": result["state_sha256"],
                    "all_best_measured_actions": result["optimal_actions"],
                    "rollout_outcomes": result["outcomes"],
                    "policy_success_rates": result["success_rates"],
                    "policy_mean_returns": result["mean_returns"],
                    "policy_probs": result["policy_probs"],
                    "trials_per_action": TRIALS,
                    "oracle_verified": False,
                    "stochastic_policy_label": True,
                    "terminal_conditions": ["goal_reached", "collision", "timeout"],
                    "value_target": max(result["mean_returns"].values()),
                },
                "teacher_only": False,
            }
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            role_counts[role] += 1
            environment_counts[str(result["environment_id"])] += 1
    temporary.replace(destination)
    report = {
        "schema_version": 1,
        "stage": "minigrid_dynamic_obstacles",
        "questions": 250,
        "roles": dict(sorted(role_counts.items())),
        "tasks": {"choice": 250},
        "environments": dict(sorted(environment_counts.items())),
        "trials_per_action": TRIALS,
        "unique_visible_states": len(seen_images),
        "rejected": dict(sorted(failures.items())),
        "images": str(image_root),
        "manifest": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    destination.with_suffix(".build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
