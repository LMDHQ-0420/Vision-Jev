#!/usr/bin/env python3
"""Apply the frozen SFT-to-RLCD quality gate to a completed evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def check_gate(summary: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "syntax_valid_rate": summary["syntax_valid_rate"]
        >= gate["minimum_syntax_valid_rate"],
        "exact_match": summary["exact_match"] >= gate["minimum_exact_match"],
        "score_within_one_accuracy": summary["by_task"]["score"]["within_one_accuracy"]
        >= gate["minimum_score_within_one_accuracy"],
    }
    for task, minimum in gate["minimum_task_exact_match"].items():
        checks[f"{task}_exact_match"] = summary["by_task"][task]["exact_match"] >= minimum
    return {"passed": all(checks.values()), "checks": checks, "thresholds": gate}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gate", type=Path, default=Path("configs/eval/sft_gate.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = check_gate(summary, gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
