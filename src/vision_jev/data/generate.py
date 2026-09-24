"""Deterministic local visual-data generation for the v2.2 training mixture."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_jev.data.schema import validate_jsonl

GENERATOR_REVISION = "local-27k-v2.2-r2"
CANVAS = (1024, 768)
COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#be123c",
)
SHAPES = ("circle", "square", "triangle", "diamond")


@dataclass(frozen=True)
class Card:
    index: int
    shape: str
    color: str
    price: int
    rating: int
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class CandidateScene:
    parent_id: str
    root_id: str
    group_id: str
    image: Path
    cards: tuple[Card, ...]
    required_shape: str
    minimum_rating: int
    options: tuple[dict[str, Any], ...]


def _seed(seed: str, *parts: object) -> int:
    value = "\0".join([seed, *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def _rng(seed: str, *parts: object) -> random.Random:
    return random.Random(_seed(seed, *parts))


def _pil() -> tuple[Any, Any, Any]:
    from PIL import Image, ImageDraw, ImageFont

    return Image, ImageDraw, ImageFont


def _font(size: int, *, bold: bool = False) -> Any:
    _, _, image_font = _pil()
    family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return image_font.truetype(family, size)
    except OSError:
        return image_font.load_default()


def _canvas(title: str) -> tuple[Any, Any]:
    image_module, image_draw, _ = _pil()
    image = image_module.new("RGB", CANVAS, "#f8fafc")
    draw = image_draw.Draw(image)
    draw.rectangle((0, 0, CANVAS[0], 72), fill="#0f172a")
    draw.text((32, 20), title, fill="white", font=_font(28, bold=True))
    return image, draw


def _save_image(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG", optimize=True)
    temporary.replace(path)


def _candidate(
    candidate_id: str,
    text: str,
    *,
    box: tuple[int, int, int, int] | None = None,
    action_type: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {"id": candidate_id, "text": text}
    if box is not None:
        item["box"] = list(box)
    if action_type is not None:
        item["action_spec"] = {"type": action_type}
    return item


def _sample(
    *,
    sample_id: str,
    root_id: str,
    group_id: str,
    source: str,
    source_bucket: str,
    task_type: str,
    image: Path,
    question: str,
    options: Iterable[dict[str, Any]],
    target: str | list[str] | bool,
    target_kind: str,
    input_track: str,
    proposal_kind: str,
    supervision_semantics: str,
    evidence_reference: str,
    parent_id: str | None = None,
) -> dict[str, Any]:
    candidates = list(options)
    sample: dict[str, Any] = {
        "schema_version": 2,
        "sample_id": sample_id,
        "root_id": root_id,
        "group_id": group_id,
        "source": source,
        "source_version": "v2.2",
        "source_bucket": source_bucket,
        "license": "CC0-1.0",
        "split": "train",
        "task_type": task_type,
        "image": str(image),
        "image_metadata": {
            "width": CANVAS[0],
            "height": CANVAS[1],
            "transform": "none",
            "visual_budget": "source_native",
            "synthetic": True,
        },
        "state_text": "",
        "question": question,
        "allowed_history": [],
        "input_track": input_track,
        "options": candidates,
        "candidates": candidates,
        "target_kind": target_kind,
        "target": target,
        "label_origin": "programmatic",
        "origin_label_method": "deterministic_solver",
        "language": "en",
        "evidence_reference": evidence_reference,
        "generator_version": GENERATOR_REVISION,
        "generator_revision": GENERATOR_REVISION,
        "candidate_generator_revision": GENERATOR_REVISION,
        "proposal_kind": proposal_kind,
        "supervision_semantics": supervision_semantics,
        "quality": {
            "solver_verified": True,
            "visible_evidence_verified": True,
            "api_used": False,
            "release_blocked": False,
        },
        "teacher_only": False,
    }
    if parent_id is not None:
        sample["parent_id"] = parent_id
        sample["verification_method"] = "deterministic_solver"
    return sample


def _k_values(bins: Iterable[tuple[int, int, int]], *, seed: str, namespace: str) -> list[int]:
    values: list[int] = []
    for low, high, count in bins:
        values.extend(low + (index % (high - low + 1)) for index in range(count))
    _rng(seed, "k", namespace).shuffle(values)
    return values


def _draw_shape(draw: Any, shape: str, box: tuple[int, int, int, int], color: str) -> None:
    x1, y1, x2, y2 = box
    if shape == "circle":
        draw.ellipse(box, fill=color)
    elif shape == "square":
        draw.rectangle(box, fill=color)
    elif shape == "triangle":
        draw.polygon(((x1 + x2) // 2, y1, x2, y2, x1, y2), fill=color)
    else:
        draw.polygon(
            (
                ((x1 + x2) // 2, y1),
                (x2, (y1 + y2) // 2),
                ((x1 + x2) // 2, y2),
                (x1, (y1 + y2) // 2),
            ),
            fill=color,
        )


def _grid_sample(index: int, image_root: Path, seed: str) -> dict[str, Any]:
    rng = _rng(seed, "grid", index)
    size = 8
    start = (rng.randrange(size), rng.randrange(size))
    goal = (rng.randrange(size), rng.randrange(size))
    while goal == start:
        goal = (rng.randrange(size), rng.randrange(size))

    path_cells = {start, goal}
    cursor = start
    while cursor[0] != goal[0]:
        cursor = (cursor[0] + (1 if goal[0] > cursor[0] else -1), cursor[1])
        path_cells.add(cursor)
    while cursor[1] != goal[1]:
        cursor = (cursor[0], cursor[1] + (1 if goal[1] > cursor[1] else -1))
        path_cells.add(cursor)
    obstacles = {
        (row, col)
        for row in range(size)
        for col in range(size)
        if (row, col) not in path_cells and rng.random() < 0.22
    }

    directions = {
        "move_up": (-1, 0),
        "move_down": (1, 0),
        "move_left": (0, -1),
        "move_right": (0, 1),
    }
    distances = {goal: 0}
    queue: deque[tuple[int, int]] = deque([goal])
    while queue:
        row, col = queue.popleft()
        for delta_row, delta_col in directions.values():
            neighbor = (row + delta_row, col + delta_col)
            if (
                0 <= neighbor[0] < size
                and 0 <= neighbor[1] < size
                and neighbor not in obstacles
                and neighbor not in distances
            ):
                distances[neighbor] = distances[(row, col)] + 1
                queue.append(neighbor)
    valid: list[str] = []
    for action, (delta_row, delta_col) in directions.items():
        neighbor = (start[0] + delta_row, start[1] + delta_col)
        if neighbor in distances and distances[neighbor] == distances[start] - 1:
            valid.append(action)
    if not valid:
        raise RuntimeError("grid solver produced no shortest-path action")

    image, draw = _canvas("Fully observable grid navigation")
    cell = 72
    left, top = 224, 112
    for row in range(size):
        for col in range(size):
            box = (
                left + col * cell,
                top + row * cell,
                left + (col + 1) * cell,
                top + (row + 1) * cell,
            )
            fill = "#334155" if (row, col) in obstacles else "white"
            draw.rectangle(box, fill=fill, outline="#94a3b8", width=2)
    goal_box = (
        left + goal[1] * cell + 12,
        top + goal[0] * cell + 12,
        left + (goal[1] + 1) * cell - 12,
        top + (goal[0] + 1) * cell - 12,
    )
    draw.rectangle(goal_box, fill="#22c55e", outline="#166534", width=3)
    start_box = (
        left + start[1] * cell + 14,
        top + start[0] * cell + 14,
        left + (start[1] + 1) * cell - 14,
        top + (start[0] + 1) * cell - 14,
    )
    draw.ellipse(start_box, fill="#2563eb", outline="#1e3a8a", width=3)
    draw.text((36, 118), "BLUE = agent", fill="#1e3a8a", font=_font(20, bold=True))
    draw.text((36, 154), "GREEN = goal", fill="#166534", font=_font(20, bold=True))
    draw.text((36, 190), "DARK = blocked", fill="#334155", font=_font(20, bold=True))

    image_path = image_root / "grid" / f"grid-{index:05d}.png"
    _save_image(image, image_path)
    options = [_candidate(action, action.replace("_", " ")) for action in directions]
    target: str | list[str] = valid[0] if len(valid) == 1 else valid
    return _sample(
        sample_id=f"local-grid:{index:05d}",
        root_id=f"local-grid:{index:05d}",
        group_id=f"local-grid:{index:05d}",
        source="local_grid_navigation",
        source_bucket="programmatic",
        task_type="choice",
        image=image_path,
        question="Choose a shortest-path next action from the blue agent to the green goal.",
        options=options,
        target=target,
        target_kind="single" if isinstance(target, str) else "multiple",
        input_track="fully_observable_grid",
        proposal_kind="environment",
        supervision_semantics="verified_action_set",
        evidence_reference=f"bfs:grid-{index:05d}",
    )


def _gui_sample(index: int, k: int, image_root: Path, seed: str) -> dict[str, Any]:
    rng = _rng(seed, "gui", index)
    scenario = ("form", "filter", "sort", "dialog", "pagination", "workflow")[index % 6]
    labels = {
        "form": ("Save draft", "Submit form", "Reset", "Preview", "Attach file", "Cancel"),
        "filter": (
            "Apply filters",
            "Clear filters",
            "Newest",
            "Lowest price",
            "In stock",
            "All items",
        ),
        "sort": ("Sort ascending", "Sort descending", "Name", "Date", "Priority", "Status"),
        "dialog": ("Confirm", "Cancel", "Go back", "Review", "Continue", "Close"),
        "pagination": ("Previous", "Next", "First page", "Last page", "Page 2", "Page 3"),
        "workflow": ("Start", "Pause", "Resume", "Approve", "Reject", "Finish"),
    }[scenario]
    rendered = [labels[position % len(labels)] for position in range(k)]
    rendered = [
        f"{label} {position // len(labels) + 1}" if position >= len(labels) else label
        for position, label in enumerate(rendered)
    ]
    target_index = rng.randrange(k)
    target_label = rendered[target_index]

    image, draw = _canvas(f"Local GUI sandbox · {scenario}")
    draw.rounded_rectangle((60, 104, 964, 708), radius=18, fill="white", outline="#cbd5e1", width=3)
    draw.text(
        (96, 132), f"Demo {scenario.title()} Panel", fill="#0f172a", font=_font(30, bold=True)
    )
    draw.text((96, 178), "Sandbox data only · no external action", fill="#64748b", font=_font(18))
    cols = 2 if k <= 4 else 3
    rows = math.ceil(k / cols)
    gap_x, gap_y = 24, 22
    usable_width, usable_height = 820, 430
    width = (usable_width - gap_x * (cols - 1)) // cols
    height = min(76, (usable_height - gap_y * (rows - 1)) // rows)
    options: list[dict[str, Any]] = []
    for position, label in enumerate(rendered):
        row, col = divmod(position, cols)
        x1 = 102 + col * (width + gap_x)
        y1 = 236 + row * (height + gap_y)
        box = (x1, y1, x1 + width, y1 + height)
        draw.rounded_rectangle(box, radius=12, fill="#e2e8f0", outline="#64748b", width=2)
        draw.text((x1 + 18, y1 + 22), label, fill="#0f172a", font=_font(18, bold=True))
        options.append(
            _candidate(
                f"widget:{position:02d}",
                f"visible widget {position + 1}",
                box=box,
                action_type="click",
            )
        )
    image_path = image_root / "gui" / f"gui-{index:05d}.png"
    _save_image(image, image_path)
    return _sample(
        sample_id=f"local-gui:{index:05d}",
        root_id=f"local-gui:{index:05d}",
        group_id=f"local-gui:{index:05d}",
        source="local_gui_sandbox",
        source_bucket="programmatic",
        task_type="choice",
        image=image_path,
        question=f"Click the visible control labeled '{target_label}'.",
        options=options,
        target=f"widget:{target_index:02d}",
        target_kind="single",
        input_track="synthetic_gui_regions",
        proposal_kind="environment",
        supervision_semantics="verified_action_set",
        evidence_reference=f"gui-layout:{index:05d}:widget:{target_index:02d}",
    )


def _candidate_scene(
    index: int, k: int, image_root: Path, seed: str
) -> tuple[dict[str, Any], CandidateScene]:
    rng = _rng(seed, "candidate", index)
    required_shape = SHAPES[index % len(SHAPES)]
    minimum_rating = 3 + (index % 2)
    cols = min(8, max(2, math.ceil(math.sqrt(k * 4 / 3))))
    rows = math.ceil(k / cols)
    gap = 10
    left, top = 34, 120
    card_width = (956 - gap * (cols - 1)) // cols
    card_height = min(142, (610 - gap * (rows - 1)) // rows)
    eligible_positions = rng.sample(range(k), min(3, k))
    target_prices = rng.sample(range(12, 70), len(eligible_positions))
    cards: list[Card] = []
    for position in range(k):
        row, col = divmod(position, cols)
        box = (
            left + col * (card_width + gap),
            top + row * (card_height + gap),
            left + col * (card_width + gap) + card_width,
            top + row * (card_height + gap) + card_height,
        )
        if position in eligible_positions:
            local_index = eligible_positions.index(position)
            shape = required_shape
            rating = minimum_rating + rng.randrange(6 - minimum_rating)
            price = target_prices[local_index]
        else:
            shape = rng.choice(SHAPES)
            rating = rng.randint(1, 5)
            price = rng.randint(10, 99)
            if shape == required_shape and rating >= minimum_rating:
                rating = minimum_rating - 1
        cards.append(
            Card(
                index=position,
                shape=shape,
                color=COLORS[position % len(COLORS)],
                price=price,
                rating=rating,
                box=box,
            )
        )
    eligible = [
        card for card in cards if card.shape == required_shape and card.rating >= minimum_rating
    ]
    eligible.sort(key=lambda card: (card.price, card.index))
    if len(eligible) < 2:
        raise RuntimeError("candidate solver requires at least two eligible cards")

    image, draw = _canvas("Visual candidate comparison")
    for card in cards:
        x1, y1, x2, y2 = card.box
        draw.rounded_rectangle(card.box, radius=10, fill="white", outline="#94a3b8", width=2)
        shape_size = min(42, max(20, (y2 - y1) // 3))
        shape_box = (x1 + 12, y1 + 12, x1 + 12 + shape_size, y1 + 12 + shape_size)
        _draw_shape(draw, card.shape, shape_box, card.color)
        draw.text((x1 + 10, y2 - 58), f"${card.price}", fill="#0f172a", font=_font(18, bold=True))
        draw.text((x1 + 10, y2 - 30), f"rating {card.rating}", fill="#475569", font=_font(14))
        draw.text(
            (x2 - 36, y1 + 12), str(card.index + 1), fill="#64748b", font=_font(14, bold=True)
        )
    image_path = image_root / "candidate" / f"candidate-{index:05d}.png"
    _save_image(image, image_path)
    options = tuple(
        _candidate(f"card:{card.index:02d}", f"visual card {card.index + 1}", box=card.box)
        for card in cards
    )
    sample_id = f"local-candidate:{index:05d}"
    sample = _sample(
        sample_id=sample_id,
        root_id=sample_id,
        group_id=sample_id,
        source="local_candidate_comparison",
        source_bucket="programmatic",
        task_type="choice",
        image=image_path,
        question=(
            f"Select the lowest-price {required_shape} card whose displayed rating is at least "
            f"{minimum_rating}."
        ),
        options=options,
        target=f"card:{eligible[0].index:02d}",
        target_kind="single",
        input_track="synthetic_card_regions",
        proposal_kind="environment",
        supervision_semantics="factual_answer",
        evidence_reference=f"set-solver:{index:05d}:rank:1",
    )
    scene = CandidateScene(
        parent_id=sample_id,
        root_id=sample_id,
        group_id=sample_id,
        image=image_path,
        cards=tuple(cards),
        required_shape=required_shape,
        minimum_rating=minimum_rating,
        options=options,
    )
    return sample, scene


def _score_sample(index: int, image_root: Path, seed: str) -> dict[str, Any]:
    rng = _rng(seed, "score", index)
    levels = (3, 5, 7)[index % 3]
    total = 14
    completed = rng.randrange(total + 1)
    intervals: list[tuple[int, int]] = []
    for level in range(levels):
        low = math.floor(level * (total + 1) / levels)
        high = math.floor((level + 1) * (total + 1) / levels) - 1
        intervals.append((low, high))
    target_level = next(
        level for level, (low, high) in enumerate(intervals) if low <= completed <= high
    )

    image, draw = _canvas("Visible completion dashboard")
    draw.rounded_rectangle((74, 110, 950, 690), radius=18, fill="white", outline="#cbd5e1", width=3)
    draw.text(
        (112, 140), f"Completed: {completed} / {total}", fill="#0f172a", font=_font(34, bold=True)
    )
    for item in range(total):
        row, col = divmod(item, 7)
        x1 = 114 + col * 112
        y1 = 230 + row * 128
        box = (x1, y1, x1 + 72, y1 + 72)
        done = item < completed
        draw.rounded_rectangle(
            box,
            radius=8,
            fill="#dcfce7" if done else "#f1f5f9",
            outline="#16a34a" if done else "#94a3b8",
            width=3,
        )
        draw.text(
            (x1 + 24, y1 + 18), "✓" if done else "·", fill="#166534", font=_font(30, bold=True)
        )
        draw.text((x1 + 18, y1 + 82), f"item {item + 1}", fill="#475569", font=_font(13))
    image_path = image_root / "score" / f"score-{index:05d}.png"
    _save_image(image, image_path)
    options = [
        _candidate(f"level:{level + 1}", f"Level {level + 1}: {low}-{high} completed")
        for level, (low, high) in enumerate(intervals)
    ]
    return _sample(
        sample_id=f"local-score:{index:05d}",
        root_id=f"local-score:{index:05d}",
        group_id=f"local-score:{index:05d}",
        source="local_visual_score",
        source_bucket="programmatic",
        task_type="score",
        image=image_path,
        question="Choose the level whose complete interval contains the visible completed count.",
        options=options,
        target=f"level:{target_level + 1}",
        target_kind="single",
        input_track="synthetic_dashboard",
        proposal_kind="none",
        supervision_semantics="ordinal_level",
        evidence_reference=f"interval-solver:{index:05d}:{completed}",
    )


def _uncertainty_choice_sample(index: int, image_root: Path, seed: str) -> dict[str, Any]:
    rng = _rng(seed, "uncertainty-choice", index)
    statuses = ["ready", "blocked", "review required"]
    rng.shuffle(statuses)
    k = 2 + (index % 3)
    shown = statuses[: k - 1]
    insert_at = rng.randrange(k)
    shown.insert(insert_at, "insufficient evidence")
    image, draw = _canvas("Evidence availability panel")
    draw.rounded_rectangle(
        (180, 150, 844, 620), radius=20, fill="white", outline="#94a3b8", width=3
    )
    draw.text((238, 210), "System status", fill="#0f172a", font=_font(32, bold=True))
    draw.text((238, 292), "Required status field", fill="#475569", font=_font(22))
    hidden_box = (238, 344, 786, 432)
    draw.rounded_rectangle(hidden_box, radius=10, fill="#cbd5e1", outline="#64748b", width=2)
    for offset in range(-60, 560, 24):
        draw.line((238 + offset, 430, 238 + offset + 90, 346), fill="#94a3b8", width=5)
    draw.text((238, 500), "Updated 09:42 · owner: test-user", fill="#64748b", font=_font(18))
    image_path = image_root / "uncertainty" / f"choice-{index:05d}.png"
    _save_image(image, image_path)
    options = [
        _candidate(
            "insufficient_evidence" if text == "insufficient evidence" else f"status:{position}",
            text,
        )
        for position, text in enumerate(shown)
    ]
    return _sample(
        sample_id=f"local-uncertainty-choice:{index:05d}",
        root_id=f"local-uncertainty-choice:{index:05d}",
        group_id=f"local-uncertainty-choice:{index:05d}",
        source="local_known_uncertainty",
        source_bucket="programmatic",
        task_type="choice",
        image=image_path,
        question="Which status is supported by the currently visible evidence?",
        options=options,
        target="insufficient_evidence",
        target_kind="single",
        input_track="synthetic_evidence_panel",
        proposal_kind="none",
        supervision_semantics="factual_answer",
        evidence_reference=f"visibility-rule:{index:05d}:status-hidden",
    )


def _uncertainty_noul_sample(index: int, image_root: Path, seed: str) -> dict[str, Any]:
    rng = _rng(seed, "uncertainty-noul", index)
    visible = index % 2 == 0
    threshold = rng.randint(40, 80)
    value = rng.randint(0, 100)
    image, draw = _canvas("Evidence sufficiency check")
    draw.rounded_rectangle(
        (180, 150, 844, 620), radius=20, fill="white", outline="#94a3b8", width=3
    )
    draw.text(
        (238, 206),
        f"Decision rule: metric ≥ {threshold}",
        fill="#0f172a",
        font=_font(30, bold=True),
    )
    draw.text((238, 304), "Observed metric", fill="#475569", font=_font(22))
    if visible:
        draw.rounded_rectangle(
            (238, 356, 786, 444), radius=10, fill="#e0f2fe", outline="#0284c7", width=3
        )
        draw.text((480, 378), str(value), fill="#075985", font=_font(30, bold=True))
    else:
        hidden_box = (238, 356, 786, 444)
        draw.rounded_rectangle(hidden_box, radius=10, fill="#cbd5e1", outline="#64748b", width=2)
        for offset in range(-60, 560, 24):
            draw.line((238 + offset, 442, 238 + offset + 90, 358), fill="#94a3b8", width=5)
    draw.text((238, 510), "Snapshot timestamp: 09:42", fill="#64748b", font=_font(18))
    image_path = image_root / "uncertainty" / f"noul-{index:05d}.png"
    _save_image(image, image_path)
    return _sample(
        sample_id=f"local-uncertainty-noul:{index:05d}",
        root_id=f"local-uncertainty-noul:{index:05d}",
        group_id=f"local-uncertainty-noul:{index:05d}",
        source="local_known_uncertainty",
        source_bucket="programmatic",
        task_type="noul",
        image=image_path,
        question=(
            "Is there enough visible evidence to determine whether the decision rule is satisfied?"
        ),
        options=[],
        target=visible,
        target_kind="binary",
        input_track="synthetic_evidence_panel",
        proposal_kind="none",
        supervision_semantics="factual_answer",
        evidence_reference=f"visibility-rule:{index:05d}:{'visible' if visible else 'hidden'}",
    )


def _hard_candidate_sample(index: int, scene: CandidateScene) -> dict[str, Any]:
    eligible = sorted(
        (
            card
            for card in scene.cards
            if card.shape == scene.required_shape and card.rating >= scene.minimum_rating
        ),
        key=lambda card: (card.price, card.index),
    )
    target = eligible[1]
    return _sample(
        sample_id=f"local-hard-candidate:{index:05d}",
        root_id=scene.root_id,
        group_id=scene.group_id,
        source="programmatic_hard_candidates",
        source_bucket="verified_augmentation",
        task_type="choice",
        image=scene.image,
        question=(
            f"Select the second-lowest-price {scene.required_shape} card whose displayed rating "
            f"is at least {scene.minimum_rating}."
        ),
        options=scene.options,
        target=f"card:{target.index:02d}",
        target_kind="single",
        input_track="synthetic_card_regions",
        proposal_kind="environment",
        supervision_semantics="factual_answer",
        evidence_reference=f"set-solver:{index:05d}:rank:2",
        parent_id=scene.parent_id,
    )


def _counterfactual_sample(index: int, scene: CandidateScene) -> dict[str, Any]:
    first = scene.cards[index % len(scene.cards)]
    second = scene.cards[(index * 3 + 1) % len(scene.cards)]
    if first.index == second.index:
        second = scene.cards[(second.index + 1) % len(scene.cards)]
    relation = first.price < second.price
    if index % 2:
        relation = first.price > second.price
        operator = "higher"
    else:
        operator = "lower"
    return _sample(
        sample_id=f"local-counterfactual:{index:05d}",
        root_id=scene.root_id,
        group_id=scene.group_id,
        source="programmatic_counterfactual",
        source_bucket="verified_augmentation",
        task_type="noul",
        image=scene.image,
        question=(
            f"Is the displayed price of visual card {first.index + 1} {operator} than the "
            f"displayed price of visual card {second.index + 1}?"
        ),
        options=[],
        target=relation,
        target_kind="binary",
        input_track="synthetic_card_regions",
        proposal_kind="none",
        supervision_semantics="factual_answer",
        evidence_reference=f"counterfactual-solver:{index:05d}:{first.index}:{second.index}",
        parent_id=scene.parent_id,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _k_bucket(k: int) -> str:
    if k <= 4:
        return "2-4"
    if k <= 8:
        return "5-8"
    if k <= 16:
        return "9-16"
    return "17-32"


def generate_local_dataset(
    *,
    data_root: Path,
    mixture_config: Path,
    destination: Path,
    seed: str = "vision-jev-local-v2.2",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate the exact 27k API-free local block and validate every sample."""
    config = json.loads(mixture_config.read_text(encoding="utf-8"))
    programmatic = config["programmatic_quotas"]
    augmentation = config["augmentation_quotas"]
    expected = {
        "environment_action_demo": 8000,
        "candidate_set_comparison": 6000,
        "visual_score": 6000,
        "known_uncertainty": 4000,
        "programmatic_hard_candidates": 2000,
        "programmatic_counterfactual": 1000,
    }
    actual = {
        **{key: int(programmatic[key]) for key in tuple(programmatic)},
        "programmatic_hard_candidates": int(augmentation["programmatic_hard_candidates"]),
        "programmatic_counterfactual": int(augmentation["programmatic_counterfactual"]),
    }
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"local v2.2 generator requires {key}={value}, got {actual.get(key)}")

    image_root = data_root / "generated" / GENERATOR_REVISION / "images"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    source_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    image_paths: set[str] = set()

    def emit(handle: Any, sample: dict[str, Any]) -> None:
        nonlocal count
        handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
        count += 1
        source_counts[str(sample["source"])] += 1
        task_counts[str(sample["task_type"])] += 1
        if sample["task_type"] == "choice":
            bucket_counts[_k_bucket(len(sample["options"]))] += 1
        image_paths.add(str(sample["image"]))
        if progress is not None and count % 1000 == 0:
            progress(f"local generation: {count}/27000")

    gui_k = _k_values(((2, 4, 4000), (5, 8, 2000)), seed=seed, namespace="gui")
    candidate_k = _k_values(
        ((2, 4, 1400), (5, 8, 2400), (9, 16, 1300), (17, 32, 900)),
        seed=seed,
        namespace="candidate",
    )
    candidate_scenes: list[CandidateScene] = []
    with temporary.open("w", encoding="utf-8") as handle:
        for index in range(2000):
            emit(handle, _grid_sample(index, image_root, seed))
        for index, k in enumerate(gui_k):
            emit(handle, _gui_sample(index, k, image_root, seed))
        for index, k in enumerate(candidate_k):
            sample, scene = _candidate_scene(index, k, image_root, seed)
            emit(handle, sample)
            candidate_scenes.append(scene)
        for index in range(6000):
            emit(handle, _score_sample(index, image_root, seed))
        for index in range(2000):
            emit(handle, _uncertainty_choice_sample(index, image_root, seed))
        for index in range(2000):
            emit(handle, _uncertainty_noul_sample(index, image_root, seed))
        for index in range(2000):
            emit(handle, _hard_candidate_sample(index, candidate_scenes[index]))
        for index in range(1000):
            emit(handle, _counterfactual_sample(index, candidate_scenes[2000 + index]))
    temporary.replace(destination)

    validation = validate_jsonl(destination, check_assets=True)
    if count != 27000 or validation.questions != count:
        raise RuntimeError(f"expected 27000 local questions, wrote {count}")
    expected_tasks = {"choice": 18000, "noul": 3000, "score": 6000}
    if dict(task_counts) != expected_tasks:
        raise RuntimeError(f"unexpected task counts: {dict(task_counts)}")
    expected_buckets = {"2-4": 9900, "5-8": 5400, "9-16": 1800, "17-32": 900}
    if dict(bucket_counts) != expected_buckets:
        raise RuntimeError(f"unexpected Choice K distribution: {dict(bucket_counts)}")

    report = {
        "schema_version": 1,
        "mixture_id": config["mixture_id"],
        "generator_revision": GENERATOR_REVISION,
        "seed": seed,
        "questions": count,
        "programmatic_questions": 24000,
        "verified_augmentation_questions": 3000,
        "api_calls": 0,
        "project_human_annotations": 0,
        "source_counts": dict(source_counts),
        "task_counts": dict(task_counts),
        "language_counts": {"en": count},
        "choice_k_counts": dict(bucket_counts),
        "unique_images": len(image_paths),
        "output": str(destination),
        "asset_root": str(image_root),
        "manifest_sha256": _sha256(destination),
    }
    report_path = destination.with_suffix(".generation-report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def generate_local_pilot(
    *,
    data_root: Path,
    mixture_config: Path,
    destination: Path,
    samples_per_family: int = 8,
    seed: str = "vision-jev-local-v2.2",
) -> dict[str, Any]:
    """Generate a small visual pilot covering every local-data family."""
    if samples_per_family < 2:
        raise ValueError("samples_per_family must be at least 2")
    config = json.loads(mixture_config.read_text(encoding="utf-8"))
    image_root = data_root / "generated" / f"{GENERATOR_REVISION}-pilot" / "images"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    source_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    image_paths: set[str] = set()
    samples: list[dict[str, Any]] = []

    for index in range(samples_per_family):
        samples.append(_grid_sample(index, image_root, seed))
    gui_sizes = (2, 3, 4, 5, 6, 7, 8)
    for index in range(samples_per_family):
        samples.append(_gui_sample(index, gui_sizes[index % len(gui_sizes)], image_root, seed))
    candidate_sizes = (2, 4, 6, 8, 10, 16, 20, 32)
    scenes: list[CandidateScene] = []
    for index in range(samples_per_family * 2):
        sample, scene = _candidate_scene(
            index, candidate_sizes[index % len(candidate_sizes)], image_root, seed
        )
        samples.append(sample)
        scenes.append(scene)
    for index in range(samples_per_family):
        samples.append(_score_sample(index, image_root, seed))
        samples.append(_uncertainty_choice_sample(index, image_root, seed))
        samples.append(_uncertainty_noul_sample(index, image_root, seed))
        samples.append(_hard_candidate_sample(index, scenes[index]))
        samples.append(_counterfactual_sample(index, scenes[samples_per_family + index]))

    with temporary.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            source_counts[str(sample["source"])] += 1
            task_counts[str(sample["task_type"])] += 1
            if sample["task_type"] == "choice":
                bucket_counts[_k_bucket(len(sample["options"]))] += 1
            image_paths.add(str(sample["image"]))
    temporary.replace(destination)
    validation = validate_jsonl(destination, check_assets=True)
    if validation.questions != len(samples):
        raise RuntimeError("pilot validation count mismatch")
    report = {
        "schema_version": 1,
        "mixture_id": config["mixture_id"],
        "pilot": True,
        "generator_revision": GENERATOR_REVISION,
        "seed": seed,
        "samples_per_family": samples_per_family,
        "questions": len(samples),
        "api_calls": 0,
        "project_human_annotations": 0,
        "source_counts": dict(source_counts),
        "task_counts": dict(task_counts),
        "language_counts": {"en": len(samples)},
        "choice_k_counts": dict(bucket_counts),
        "unique_images": len(image_paths),
        "output": str(destination),
        "asset_root": str(image_root),
        "manifest_sha256": _sha256(destination),
    }
    destination.with_suffix(".generation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
