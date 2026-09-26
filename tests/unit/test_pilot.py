from __future__ import annotations

import json
from pathlib import Path

from vision_jev.data.pilot import build_pilot_manifest, build_training_manifest


def test_build_pilot_is_deterministic_and_group_safe(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = []
    for index in range(40):
        rows.append(
            {
                "sample_id": f"s{index}",
                "group_id": f"g{index // 2}",
                "source": "a" if index < 20 else "b",
                "task_type": "choice" if index % 3 else "noul",
            }
        )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    report = build_pilot_manifest(source, first, questions=20, seed="test")
    build_pilot_manifest(source, second, questions=20, seed="test")
    assert first.read_bytes() == second.read_bytes()
    assert report["questions"] == 20
    roles: dict[str, set[str]] = {}
    for line in first.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        roles.setdefault(row["group_id"], set()).add(row["pilot_role"])
    assert all(len(values) == 1 for values in roles.values())


def test_build_training_manifest_keeps_every_row_and_groups_safe(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        {
            "sample_id": f"s{index}",
            "group_id": f"g{index // 2}",
            "source": "source",
            "task_type": "score" if index % 5 == 0 else "choice",
        }
        for index in range(200)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    destination = tmp_path / "training.jsonl"
    report = build_training_manifest(source, destination, seed="test", eval_percent=10)
    assert report["questions"] == 200
    assert sum(report["roles"].values()) == 200
    assert set(report["roles"]) == {"eval", "train"}
    roles: dict[str, set[str]] = {}
    for line in destination.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        roles.setdefault(row["group_id"], set()).add(row["pilot_role"])
    assert all(len(values) == 1 for values in roles.values())
