"""Check repository documentation and configuration invariants."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "README_CN.md",
    "STATUS.md",
    "LICENSE",
    "asset/Vision-Jev.svg",
    "environment.yml",
    "docs/index.md",
    "docs/architecture/system.md",
    "docs/data/sources.md",
    "docs/data/mixtures.md",
    "docs/training/sft.md",
    "docs/training/ppo.md",
    "docs/evaluation/protocol.md",
    "docs/experiments/README.md",
    "docs/release/checklist.md",
    "data/schemas/sample.schema.json",
]


def main() -> int:
    failures: list[str] = []
    for relative in REQUIRED:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty: {relative}")
    for path in sorted(ROOT.glob("configs/**/*.json")) + sorted(ROOT.glob("data/**/*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"repository check passed: {len(REQUIRED)} required files and JSON configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
