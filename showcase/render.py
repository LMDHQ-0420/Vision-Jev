"""Render paired trajectories as compact GitHub-friendly GIFs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

CANVAS_WIDTH = 1000
PANEL_WIDTH = 476
FRAME_BOX = (448, 448)
BACKGROUND = "#f4f6f8"
INK = "#17212b"
MUTED = "#66717d"
BASELINE = "#66717d"
TRAINED = "#008c82"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _step(record: dict[str, Any], index: int) -> dict[str, Any]:
    steps = record["steps"]
    if index < len(steps):
        return dict(steps[index])
    return {
        "step": len(steps),
        "frame": record["final_frame"],
        "action": None,
        "probabilities": {},
        "latency_ms": 0.0,
    }


def _fit_frame(path: str) -> Image.Image:
    with Image.open(path) as source:
        frame = source.convert("RGB")
    frame.thumbnail(FRAME_BOX, Image.Resampling.NEAREST)
    canvas = Image.new("RGB", FRAME_BOX, "#ffffff")
    x = (FRAME_BOX[0] - frame.width) // 2
    y = (FRAME_BOX[1] - frame.height) // 2
    canvas.paste(frame, (x, y))
    return canvas


def _draw_probabilities(
    draw: ImageDraw.ImageDraw,
    probabilities: dict[str, float],
    *,
    x: int,
    y: int,
    width: int,
    color: str,
) -> None:
    if not probabilities:
        draw.text(
            (x, y),
            "Generated action (no calibrated probabilities)",
            fill=MUTED,
            font=_font(15),
        )
        return
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:4]
    for offset, (label, probability) in enumerate(ordered):
        row_y = y + offset * 24
        draw.text((x, row_y), label.replace("_", " "), fill=INK, font=_font(14))
        bar_x = x + 122
        draw.rounded_rectangle(
            (bar_x, row_y + 3, bar_x + width - 122, row_y + 17),
            3,
            fill="#dde2e6",
        )
        draw.rounded_rectangle(
            (bar_x, row_y + 3, bar_x + max(2, int((width - 122) * probability)), row_y + 17),
            3,
            fill=color,
        )
        draw.text((x + width - 48, row_y), f"{probability:.0%}", fill=INK, font=_font(14))


def render_comparison(
    baseline_path: Path,
    trained_path: Path,
    destination: Path,
    *,
    frame_duration_ms: int = 700,
) -> dict[str, Any]:
    baseline = _load(baseline_path)
    trained = _load(trained_path)
    for field in ("id", "seed", "environment_id"):
        if baseline["example"][field] != trained["example"][field]:
            raise ValueError(f"trajectory mismatch for example.{field}")
    if baseline["model"]["parameter_group"] != trained["model"]["parameter_group"]:
        raise ValueError("trajectories must use the same parameter group")
    if baseline["model"]["role"] != "baseline" or trained["model"]["role"] != "trained":
        raise ValueError("comparison requires baseline then trained trajectories")

    count = max(len(baseline["steps"]), len(trained["steps"])) + 1
    height = 730
    frames: list[Image.Image] = []
    for index in range(count):
        canvas = Image.new("RGB", (CANVAS_WIDTH, height), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        title = baseline["example"]["title"]
        group = baseline["model"]["parameter_group"]
        draw.text((24, 16), f"{title} | {group}", fill=INK, font=_font(23, bold=True))
        draw.text(
            (24, 47),
            f"Same environment seed: {baseline['example']['seed']}",
            fill=MUTED,
            font=_font(15),
        )
        for column, record in enumerate((baseline, trained)):
            x = 16 + column * 492
            accent = BASELINE if column == 0 else TRAINED
            step = _step(record, index)
            draw.rounded_rectangle(
                (x, 78, x + PANEL_WIDTH, 716),
                6,
                fill="#ffffff",
                outline="#d7dce0",
            )
            draw.rectangle((x, 78, x + PANEL_WIDTH, 83), fill=accent)
            draw.text((x + 14, 94), record["model"]["label"], fill=INK, font=_font(18, bold=True))
            frame = _fit_frame(str(step["frame"]))
            canvas.paste(frame, (x + 14, 126))
            action = step.get("action")
            action_text = "final state" if action is None else str(action).replace("_", " ")
            draw.text(
                (x + 14, 584),
                f"Step {min(index + 1, len(record['steps']))} | Action: {action_text}",
                fill=INK,
                font=_font(16, bold=True),
            )
            latency = float(step.get("latency_ms", 0.0))
            if action is not None:
                draw.text((x + 330, 586), f"{latency:.0f} ms", fill=MUTED, font=_font(14))
            _draw_probabilities(
                draw,
                dict(step.get("probabilities", {})),
                x=x + 14,
                y=611,
                width=440,
                color=accent,
            )
        frames.append(canvas)

    destination.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return {
        "schema_version": 1,
        "example_id": baseline["example"]["id"],
        "parameter_group": baseline["model"]["parameter_group"],
        "frames": count,
        "gif": str(destination),
        "baseline": baseline["outcome"],
        "trained": trained["outcome"],
    }
