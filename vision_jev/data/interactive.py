"""Preprocessing for reproducible interactive-environment assets."""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

BOXOBAN_REVISION = "78011d8c86ad423bfb2e1005012393b4c3c24409"
BOXOBAN_TILES = frozenset("# @$.*+")
LEVEL_HEADER = re.compile(r"^;\s*(\d+)\s*$")
BOXOBAN_SELECTION_QUOTAS = {
    "hard:hard": 400,
    "medium:train": 700,
    "medium:valid": 100,
    "unfiltered:train": 600,
    "unfiltered:valid": 100,
    "unfiltered:test": 100,
}


def _single_directory(path: Path) -> Path:
    roots = sorted(item for item in path.iterdir() if item.is_dir()) if path.is_dir() else []
    if len(roots) != 1:
        raise FileNotFoundError(
            f"expected one extracted directory below {path}, found {len(roots)}"
        )
    return roots[0]


def _boxoban_coordinates(grid: list[str]) -> dict[str, Any]:
    players: list[list[int]] = []
    boxes: list[list[int]] = []
    goals: list[list[int]] = []
    for y, row in enumerate(grid):
        for x, tile in enumerate(row):
            if tile in "@+":
                players.append([x, y])
            if tile in "$*":
                boxes.append([x, y])
            if tile in ".*+":
                goals.append([x, y])
    if len(players) != 1:
        raise ValueError(f"Boxoban level needs one player, found {len(players)}")
    if not boxes or len(boxes) != len(goals):
        raise ValueError(f"Boxoban boxes/goals mismatch: {len(boxes)} != {len(goals)}")
    return {"player": players[0], "boxes": boxes, "goals": goals}


def _parse_boxoban_file(path: Path, root: Path) -> Iterator[dict[str, Any]]:
    relative = path.relative_to(root)
    parts = relative.parts
    difficulty = parts[0]
    upstream_split = parts[1] if len(parts) == 3 else "hard"
    role_hint = {"train": "train", "valid": "dev", "test": "test", "hard": "test"}[upstream_split]
    current_number: int | None = None
    grid: list[str] = []

    def emit() -> dict[str, Any] | None:
        if current_number is None:
            return None
        if not grid or len({len(row) for row in grid}) != 1:
            raise ValueError(f"non-rectangular Boxoban level in {relative}:{current_number}")
        invalid = sorted(set("".join(grid)) - BOXOBAN_TILES)
        if invalid:
            raise ValueError(f"unsupported Boxoban tiles in {relative}:{current_number}: {invalid}")
        coordinates = _boxoban_coordinates(grid)
        identity = f"{difficulty}:{upstream_split}:{path.stem}:{current_number}"
        digest = hashlib.sha256("\n".join(grid).encode()).hexdigest()
        return {
            "schema_version": 1,
            "level_id": f"boxoban:{identity}",
            "source": "boxoban",
            "source_revision": BOXOBAN_REVISION,
            "difficulty": difficulty,
            "upstream_split": upstream_split,
            "role_hint": role_hint,
            "source_file": relative.as_posix(),
            "source_level_number": current_number,
            "width": len(grid[0]),
            "height": len(grid),
            "grid": list(grid),
            **coordinates,
            "box_count": len(coordinates["boxes"]),
            "goal_count": len(coordinates["goals"]),
            "level_sha256": digest,
        }

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n\r")
            match = LEVEL_HEADER.fullmatch(line)
            if match:
                item = emit()
                if item is not None:
                    yield item
                current_number = int(match.group(1))
                grid = []
            elif current_number is not None and line:
                grid.append(line)
        item = emit()
        if item is not None:
            yield item


def build_boxoban_index(
    data_root: Path,
    output: Path,
    selection_quotas: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Validate every level and deterministically select the planned 2k subset."""
    extracted = data_root / "raw" / "boxoban" / "extracted" / "repository"
    root = _single_directory(extracted)
    files = sorted(
        (
            *root.glob("hard/*.txt"),
            *root.glob("medium/*/*.txt"),
            *root.glob("unfiltered/*/*.txt"),
        )
    )
    if not files:
        raise FileNotFoundError(f"no Boxoban level files below {root}")

    available_counts: Counter[str] = Counter()
    seen: set[str] = set()
    quotas = selection_quotas or BOXOBAN_SELECTION_QUOTAS
    selected: dict[str, list[tuple[int, str, dict[str, Any]]]] = {key: [] for key in quotas}
    for path in files:
        for level in _parse_boxoban_file(path, root):
            level_id = str(level["level_id"])
            if level_id in seen:
                raise ValueError(f"duplicate Boxoban level ID: {level_id}")
            seen.add(level_id)
            stratum = f"{level['difficulty']}:{level['upstream_split']}"
            available_counts[stratum] += 1
            quota = quotas.get(stratum)
            if quota is None:
                raise ValueError(f"unplanned Boxoban stratum: {stratum}")
            rank = int.from_bytes(hashlib.sha256(level_id.encode()).digest(), "big")
            candidate = (-rank, level_id, level)
            heap = selected[stratum]
            if len(heap) < quota:
                heapq.heappush(heap, candidate)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, candidate)

    shortfalls = {
        key: quotas[key] - len(selected[key]) for key in quotas if len(selected[key]) < quotas[key]
    }
    if shortfalls:
        raise ValueError(f"Boxoban selection shortfalls: {shortfalls}")
    selected_levels = [item[2] for heap in selected.values() for item in heap]
    selected_levels.sort(key=lambda item: str(item["level_id"]))
    selected_counts = Counter(
        f"{item['difficulty']}:{item['upstream_split']}" for item in selected_levels
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for level in selected_levels:
            handle.write(json.dumps(level, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)
    return {
        "source": "boxoban",
        "revision": BOXOBAN_REVISION,
        "available_levels": len(seen),
        "available_counts": dict(sorted(available_counts.items())),
        "levels": len(selected_levels),
        "counts": dict(sorted(selected_counts.items())),
        "output": str(output),
    }


def _license_files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name.lower() in {"license", "license.txt", "asset_licenses.md"}
    )


def prepare_interactive_assets(
    data_root: Path, catalog_path: Path, output_root: Path
) -> dict[str, Any]:
    """Validate environment archives and build deterministic preprocessing indexes."""
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    sources = {
        item["id"]: item for item in catalog["sources"] if item.get("track") == "interactive"
    }
    expected = {"minigrid", "procgen_maze", "boxoban"}
    if set(sources) != expected:
        raise ValueError(
            f"interactive catalog mismatch: expected {sorted(expected)}, got {sorted(sources)}"
        )

    roots = {
        source_id: _single_directory(data_root / "raw" / source_id / "extracted" / "repository")
        for source_id in sorted(expected)
    }
    required = {
        "minigrid": roots["minigrid"] / "minigrid" / "envs",
        "procgen_maze": roots["procgen_maze"] / "procgen" / "src" / "games" / "maze.cpp",
        "boxoban": roots["boxoban"] / "unfiltered",
    }
    missing = [f"{source}: {path}" for source, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing interactive assets: " + "; ".join(missing))

    output_root.mkdir(parents=True, exist_ok=True)
    boxoban_report = build_boxoban_index(data_root, output_root / "boxoban-levels.jsonl")
    records = []
    for source_id, source in sorted(sources.items()):
        state_path = data_root / "_state" / "downloads" / f"{source_id}.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"missing download state: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        records.append(
            {
                "source": source_id,
                "homepage": source["homepage"],
                "license_status": source["license_status"],
                "root": str(roots[source_id]),
                "license_files": _license_files(roots[source_id]),
                "artifacts": state["artifacts"],
            }
        )
    generation_status_path = data_root / "manifests" / "rlcd-interactive" / "generation-status.json"
    generation = (
        json.loads(generation_status_path.read_text(encoding="utf-8"))
        if generation_status_path.is_file()
        else None
    )
    report = {
        "schema_version": 1,
        "sources": records,
        "boxoban": boxoban_report,
        "generation_status": (
            generation["status"] if generation else "assets_ready_oracle_decisions_pending"
        ),
        "generation_progress": generation,
    }
    index_path = output_root / "environment-index.json"
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(index_path)
    return {**report, "environment_index": str(index_path)}
