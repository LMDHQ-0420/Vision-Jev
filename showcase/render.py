"""Render paired static predictions as compact GitHub-friendly GIFs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

CANVAS = (1200, 820)
BACKGROUND = "#f4f6f8"
INK = "#17212b"
MUTED = "#66717d"
BASELINE = "#66717d"
TRAINED = "#008c82"
CORRECT = "#17804b"
INCORRECT = "#b42318"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fit_image(path: str, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#ffffff")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _wrap(draw: ImageDraw.ImageDraw, text: str, width: int, font: Any) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines or [""]


def _text_block(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x: int,
    y: int,
    width: int,
    font: Any,
    fill: str = INK,
    max_lines: int = 4,
    spacing: int = 5,
) -> int:
    lines = _wrap(draw, text, width, font)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .") + "..."
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height + spacing
    return y


def _option_label(sample: dict[str, Any], value: Any, task_type: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    for index, item in enumerate(sample["options"]):
        if str(item["id"]) != str(value):
            continue
        text = str(item["text"])
        if task_type == "choice":
            return f"{chr(ord('A') + index)}. {text}"
        return text
    return str(value)


def _draw_choice_option(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    index: int,
    text: str,
) -> int:
    marker = chr(ord("A") + index)
    marker_font = _font(14, bold=True)
    draw.ellipse((x, y, x + 25, y + 25), fill="#e4e9ed")
    marker_box = draw.textbbox((0, 0), marker, font=marker_font)
    marker_width = marker_box[2] - marker_box[0]
    marker_height = marker_box[3] - marker_box[1]
    draw.text(
        (x + (25 - marker_width) / 2, y + (25 - marker_height) / 2 - marker_box[1]),
        marker,
        fill=INK,
        font=marker_font,
    )
    text_end = _text_block(
        draw,
        text,
        x=x + 38,
        y=y + 2,
        width=582,
        font=_font(16),
        max_lines=2,
        spacing=3,
    )
    return max(y + 34, text_end + 7)


def _draw_context(canvas: Image.Image, draw: ImageDraw.ImageDraw, record: dict[str, Any]) -> None:
    example = record["example"]
    sample = record["sample"]
    draw.text((24, 18), example["title"], fill=INK, font=_font(25, bold=True))
    draw.text(
        (24, 52),
        f"Frozen RLCD test | {example['source']} | {example['task_type']}",
        fill=MUTED,
        font=_font(15),
    )
    image = _fit_image(sample["image"], (500, 450))
    canvas_x, canvas_y = 24, 88
    draw.rectangle((canvas_x - 1, canvas_y - 1, canvas_x + 500, canvas_y + 450), outline="#d7dce0")
    canvas.paste(image, (canvas_x, canvas_y))

    x = 552
    y = 92
    state = str(sample.get("state_text", "")).strip()
    if state:
        y = (
            _text_block(draw, state, x=x, y=y, width=620, font=_font(15), fill=MUTED, max_lines=4)
            + 8
        )
    y = (
        _text_block(
            draw,
            str(sample["question"]),
            x=x,
            y=y,
            width=620,
            font=_font(22, bold=True),
            max_lines=5,
            spacing=7,
        )
        + 16
    )
    task_type = str(example["task_type"])
    options = sample["options"]
    if options:
        for index, option in enumerate(options[:8]):
            if task_type == "choice":
                y = _draw_choice_option(draw, x=x, y=y, index=index, text=str(option["text"]))
            else:
                text = f"{option['id']}: {option['text']}"
                y = (
                    _text_block(
                        draw,
                        text,
                        x=x,
                        y=y,
                        width=620,
                        font=_font(16),
                        max_lines=2,
                        spacing=3,
                    )
                    + 5
                )
    target = ", ".join(_option_label(sample, value, task_type) for value in sample["target"])
    draw.text((x, 507), f"Reference: {target}", fill=CORRECT, font=_font(17, bold=True))


def _draw_prediction_panel(
    draw: ImageDraw.ImageDraw,
    record: dict[str, Any],
    *,
    x: int,
    progress: float,
) -> None:
    model = record["model"]
    prediction = record["prediction"]
    sample = record["sample"]
    task_type = str(record["example"]["task_type"])
    accent = BASELINE if model["role"] == "baseline" else TRAINED
    draw.rounded_rectangle((x, 570, x + 564, 798), 6, fill="#ffffff", outline="#d7dce0")
    draw.rectangle((x, 570, x + 564, 576), fill=accent)
    draw.text((x + 16, 590), model["label"], fill=INK, font=_font(19, bold=True))
    progress = max(0.0, min(1.0, progress))
    draw.rounded_rectangle((x + 16, 625, x + 548, 641), 3, fill="#dde2e6")
    if progress:
        draw.rounded_rectangle(
            (x + 16, 625, x + 16 + max(3, int(532 * progress)), 641),
            3,
            fill=accent,
        )
    if progress < 1.0:
        draw.text(
            (x + 16, 650),
            f"Running inference... {progress:.0%}",
            fill=MUTED,
            font=_font(16),
        )
        return
    valid = bool(prediction["valid"])
    correct = bool(prediction["correct"])
    value = prediction["value"]
    answer = _option_label(sample, value, task_type) if valid else "No valid option"
    color = CORRECT if correct else INCORRECT
    status = "correct" if correct else "incorrect"
    draw.text((x + 16, 650), f"Answer: {answer}", fill=INK, font=_font(18, bold=True))
    draw.text((x + 16, 680), status, fill=color, font=_font(16, bold=True))
    draw.text(
        (x + 430, 594),
        f"{float(prediction['latency_ms']):.0f} ms total",
        fill=MUTED,
        font=_font(14),
    )
    probabilities = dict(prediction.get("probabilities", {}))
    if not probabilities:
        if not valid:
            draw.text(
                (x + 16, 714),
                "The response did not select one of the listed options.",
                fill=MUTED,
                font=_font(14),
            )
        return
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
    for index, (label, probability) in enumerate(ordered):
        y = 714 + index * 27
        display = _option_label(sample, label, task_type)
        draw.text((x + 16, y), display[:28], fill=INK, font=_font(14))
        draw.rectangle((x + 230, y + 4, x + 500, y + 18), fill="#dde2e6")
        draw.rectangle(
            (x + 230, y + 4, x + 230 + max(2, int(270 * probability)), y + 18),
            fill=accent,
        )
        draw.text((x + 507, y), f"{probability:.0%}", fill=INK, font=_font(14))


def render_static_comparison(
    baseline_path: Path,
    trained_path: Path,
    destination: Path,
    *,
    frame_duration_ms: int = 1100,
) -> dict[str, Any]:
    baseline = _load(baseline_path)
    trained = _load(trained_path)
    for field in ("sample_id", "image_sha256", "options", "target"):
        if baseline["sample"][field] != trained["sample"][field]:
            raise ValueError(f"static comparison mismatch for sample.{field}")
    if baseline["model"]["parameter_group"] != trained["model"]["parameter_group"]:
        raise ValueError("predictions must use the same parameter group")
    if baseline["model"]["role"] != "baseline" or trained["model"]["role"] != "trained":
        raise ValueError("comparison requires baseline then trained predictions")

    frames: list[Image.Image] = []
    for baseline_progress, trained_progress in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)):
        canvas = Image.new("RGB", CANVAS, BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        _draw_context(canvas, draw, baseline)
        _draw_prediction_panel(draw, baseline, x=24, progress=baseline_progress)
        _draw_prediction_panel(draw, trained, x=612, progress=trained_progress)
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
        "schema_version": 2,
        "example_id": baseline["example"]["id"],
        "sample_id": baseline["sample"]["sample_id"],
        "parameter_group": baseline["model"]["parameter_group"],
        "gif": str(destination),
        "baseline": baseline["prediction"],
        "trained": trained["prediction"],
    }


def render_static_suite(
    prediction_pairs: list[tuple[Path, Path]],
    destination: Path,
    *,
    frame_duration_ms: int = 100,
    completed_hold_ms: int = 1000,
) -> dict[str, Any]:
    if not prediction_pairs:
        raise ValueError("static suite requires at least one prediction pair")
    frames: list[Image.Image] = []
    durations: list[int] = []
    examples: list[dict[str, Any]] = []
    parameter_group: str | None = None
    for baseline_path, trained_path in prediction_pairs:
        baseline = _load(baseline_path)
        trained = _load(trained_path)
        for field in ("sample_id", "image_sha256", "options", "target"):
            if baseline["sample"][field] != trained["sample"][field]:
                raise ValueError(f"static suite mismatch for sample.{field}")
        group = str(baseline["model"]["parameter_group"])
        if trained["model"]["parameter_group"] != group:
            raise ValueError("static suite pair uses different parameter groups")
        if parameter_group is None:
            parameter_group = group
        elif parameter_group != group:
            raise ValueError("static suite cannot mix parameter groups")
        baseline_latency = max(1.0, float(baseline["prediction"]["latency_ms"]))
        trained_latency = max(1.0, float(trained["prediction"]["latency_ms"]))
        maximum_latency = max(baseline_latency, trained_latency)
        steps = max(1, math.ceil(maximum_latency / frame_duration_ms))
        for step in range(steps + 1):
            elapsed = min(maximum_latency, step * frame_duration_ms)
            canvas = Image.new("RGB", CANVAS, BACKGROUND)
            draw = ImageDraw.Draw(canvas)
            _draw_context(canvas, draw, baseline)
            _draw_prediction_panel(draw, baseline, x=24, progress=elapsed / baseline_latency)
            _draw_prediction_panel(draw, trained, x=612, progress=elapsed / trained_latency)
            frames.append(canvas)
            durations.append(frame_duration_ms)
        durations[-1] = completed_hold_ms
        examples.append(
            {
                "example_id": baseline["example"]["id"],
                "sample_id": baseline["sample"]["sample_id"],
                "baseline": baseline["prediction"],
                "trained": trained["prediction"],
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return {
        "schema_version": 2,
        "parameter_group": parameter_group,
        "gif": str(destination),
        "frames": len(frames),
        "examples": examples,
    }
