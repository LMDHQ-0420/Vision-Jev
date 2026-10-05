from __future__ import annotations

from scripts.check_sft_gate import check_gate


def _summary() -> dict[str, object]:
    return {
        "syntax_valid_rate": 1.0,
        "exact_match": 0.85,
        "by_task": {
            "choice": {"exact_match": 0.86},
            "noul": {"exact_match": 0.84},
            "score": {"within_one_accuracy": 0.90},
        },
    }


def test_gate_passes_all_thresholds() -> None:
    gate = {
        "minimum_syntax_valid_rate": 0.995,
        "minimum_exact_match": 0.80,
        "minimum_task_exact_match": {"choice": 0.80, "noul": 0.80},
        "minimum_score_within_one_accuracy": 0.80,
    }
    assert check_gate(_summary(), gate)["passed"] is True


def test_gate_reports_failed_slice() -> None:
    summary = _summary()
    summary["by_task"]["noul"]["exact_match"] = 0.70  # type: ignore[index]
    gate = {
        "minimum_syntax_valid_rate": 0.995,
        "minimum_exact_match": 0.80,
        "minimum_task_exact_match": {"choice": 0.80, "noul": 0.80},
        "minimum_score_within_one_accuracy": 0.80,
    }
    report = check_gate(summary, gate)
    assert report["passed"] is False
    assert report["checks"]["noul_exact_match"] is False
