"""Closed-loop environments used by the showcase recorder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from showcase.schema import Example
from vision_jev.data.interactive_generate import DIRECTION_NAMES

ACTION_IDS = {
    "turn_left": 0,
    "turn_right": 1,
    "move_forward": 2,
    "pickup": 3,
    "drop": 4,
    "toggle": 5,
    "done": 6,
}


class MiniGridShowcase:
    """A deterministic, fully observed MiniGrid episode."""

    def __init__(self, example: Example) -> None:
        import gymnasium as gym
        import minigrid  # noqa: F401

        self.example = example
        self.env = gym.make(example.environment_id, render_mode="rgb_array")
        observation, _ = self.env.reset(seed=example.seed)
        self.mission = str(observation.get("mission", "complete the mission"))
        self.history: list[dict[str, Any]] = []
        self.finished = False
        self.success = False
        self.total_reward = 0.0

    def save_frame(self, path: Path) -> None:
        unwrapped = self.env.unwrapped
        frame = unwrapped.get_frame(highlight=False, tile_size=32, agent_pov=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(path)

    def sample(self, image: Path, step: int) -> dict[str, Any]:
        unwrapped = self.env.unwrapped
        direction = DIRECTION_NAMES[int(unwrapped.agent_dir)]
        options = [
            {"id": action, "text": action.replace("_", " ")}
            for action in self.example.actions
        ]
        return {
            "schema_version": 2,
            "sample_id": f"showcase:{self.example.id}:{step:03d}",
            "root_id": f"showcase:{self.example.id}",
            "group_id": f"showcase:{self.example.id}:{self.example.seed}",
            "source": "showcase",
            "task_type": "choice",
            "state_text": f"Mission: {self.mission} Agent faces {direction}.",
            "question": "Which action should the agent take next to complete the mission?",
            "image": str(image.resolve()),
            "allowed_history": list(self.history),
            "options": options,
            "candidates": options,
            "target": [],
        }

    def step(self, action: str) -> None:
        if action not in self.example.actions:
            raise ValueError(f"action {action!r} is not enabled for {self.example.id}")
        _, reward, terminated, truncated, _ = self.env.step(ACTION_IDS[action])
        self.total_reward += float(reward)
        self.history.append(
            {"step": len(self.history), "action": action, "reward": float(reward)}
        )
        self.finished = bool(terminated or truncated)
        self.success = bool(terminated and reward > 0)

    def close(self) -> None:
        self.env.close()
