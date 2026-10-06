"""Run deterministic episodes and persist auditable trajectories."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from showcase.environments import MiniGridShowcase
from showcase.models import ModelAdapter
from showcase.schema import Example, ModelSpec, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_episode(
    example: Example,
    model: ModelSpec,
    adapter: ModelAdapter,
    output_root: Path,
) -> Path:
    run_root = output_root / example.id / model.id
    frames_root = run_root / "frames"
    if run_root.exists():
        raise FileExistsError(f"showcase run already exists: {run_root}")
    environment = MiniGridShowcase(example)
    steps: list[dict[str, Any]] = []
    failure: str | None = None
    try:
        for step in range(example.max_steps):
            frame_path = frames_root / f"{step:03d}.png"
            environment.save_frame(frame_path)
            sample = environment.sample(frame_path, step)
            decision = adapter.decide(sample)
            steps.append(
                {
                    "step": step,
                    "frame": str(frame_path.resolve()),
                    "frame_sha256": _sha256(frame_path),
                    "action": decision.action,
                    "probabilities": decision.probabilities,
                    "latency_ms": decision.latency_ms,
                    "raw_output": decision.raw_output,
                }
            )
            if decision.action is None:
                failure = "invalid_model_output"
                break
            environment.step(decision.action)
            if environment.finished:
                break
        else:
            failure = "maximum_steps_reached"

        final_frame = frames_root / f"{len(steps):03d}-final.png"
        environment.save_frame(final_frame)
        record = {
            "schema_version": 1,
            "example": {
                "id": example.id,
                "title": example.title,
                "environment_id": example.environment_id,
                "seed": example.seed,
                "max_steps": example.max_steps,
            },
            "model": {
                "id": model.id,
                "label": model.label,
                "parameter_group": model.parameter_group,
                "role": model.role,
                "kind": model.kind,
            },
            "steps": steps,
            "final_frame": str(final_frame.resolve()),
            "outcome": {
                "success": environment.success,
                "finished": environment.finished,
                "failure": None if environment.success else failure or "environment_terminated",
                "decisions": len(steps),
                "total_reward": environment.total_reward,
                "cumulative_latency_ms": sum(float(item["latency_ms"]) for item in steps),
            },
        }
        destination = run_root / "trajectory.json"
        write_json(destination, record)
        return destination
    finally:
        environment.close()
