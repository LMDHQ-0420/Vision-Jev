"""Render paired static predictions as compact GitHub-friendly GIFs."""

from __future__ import annotations

import json
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


def _option_label(sample: dict[str, Any], value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    lookup = {str(item["id"]): str(item["text"]) for item in sample["options"]}
    return lookup.get(str(value), str(value))


def _draw_context(
    canvas: Image.Image, draw: ImageDraw.ImageDraw, record: dict[str, Any]
) -> None:
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
        y = _text_block(
            draw, state, x=x, y=y, width=620, font=_font(15), fill=MUTED, max_lines=4
        ) + 8
    y = _text_block(
        draw,
        str(sample["question"]),
        x=x,
        y=y,
        width=620,
        font=_font(22, bold=True),
        max_lines=5,
        spacing=7,
    ) + 16
    options = sample["options"]
    if options:
        for option in options[:8]:
            text = f"{option['id']}: {option['text']}"
            y = _text_block(
                draw, text, x=x, y=y, width=620, font=_font(16), max_lines=2, spacing=3
            ) + 5
    target = ", ".join(_option_label(sample, value) for value in sample["target"])
    draw.text((x, 507), f"Reference: {target}", fill=CORRECT, font=_font(17, bold=True))


def _draw_prediction_panel(
    draw: ImageDraw.ImageDraw,
    record: dict[str, Any],
    *,
    x: int,
    reveal: bool,
) -> None:
    model = record["model"]
    prediction = record["prediction"]
    sample = record["sample"]
    accent = BASELINE if model["role"] == "baseline" else TRAINED
    draw.rounded_rectangle((x, 570, x + 564, 798), 6, fill="#ffffff", outline="#d7dce0")
    draw.rectangle((x, 570, x + 564, 576), fill=accent)
    draw.text((x + 16, 590), model["label"], fill=INK, font=_font(19, bold=True))
    if not reveal:
        draw.text((x + 16, 640), "Prediction hidden", fill=MUTED, font=_font(17))
        return
    valid = bool(prediction["valid"])
    correct = bool(prediction["correct"])
    value = prediction["value"]
    answer = _option_label(sample, value) if valid else "invalid structured output"
    color = CORRECT if correct else INCORRECT
    status = "correct" if correct else "incorrect"
    draw.text((x + 16, 630), f"Answer: {answer}", fill=INK, font=_font(18, bold=True))
    draw.text((x + 16, 662), status, fill=color, font=_font(16, bold=True))
    draw.text(
        (x + 445, 594),
        f"{float(prediction['latency_ms']):.0f} ms",
        fill=MUTED,
        font=_font(14),
    )
    probabilities = dict(prediction.get("probabilities", {}))
    if not probabilities:
        raw = str(prediction.get("raw_output") or "")
        _text_block(
            draw,
            f"Generated: {raw}",
            x=x + 16,
            y=700,
            width=530,
            font=_font(14),
            fill=MUTED,
            max_lines=3,
        )
        return
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
    for index, (label, probability) in enumerate(ordered):
        y = 700 + index * 27
        display = _option_label(sample, label)
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
    for baseline_visible, trained_visible in ((False, False), (True, False), (True, True)):
        canvas = Image.new("RGB", CANVAS, BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        _draw_context(canvas, draw, baseline)
        _draw_prediction_panel(draw, baseline, x=24, reveal=baseline_visible)
        _draw_prediction_panel(draw, trained, x=612, reveal=trained_visible)
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
