from __future__ import annotations

import torch

from vision_jev.train.rlcd import (
    DecisionHeads,
    _attach_latency,
    _source_threshold_reports,
    calibrated_probabilities,
    decision_loss,
    select_confidence_thresholds,
)


def test_decision_heads_cover_all_static_tasks() -> None:
    heads = DecisionHeads(hidden_size=16, width=8, attention_heads=2)
    question = torch.randn(1, 16)
    candidates = torch.randn(1, 3, 16)
    mask = torch.ones(1, 3, dtype=torch.bool)
    for task in ("choice", "score"):
        logits, probabilities = heads(task, question, candidates, mask)
        assert logits.shape == (1, 3)
        assert torch.allclose(probabilities.sum(dim=-1), torch.ones(1))
    logits, probabilities = heads("noul", question)
    assert logits.shape == probabilities.shape == (1,)


def test_choice_loss_accepts_multiple_valid_targets() -> None:
    sample = {
        "sample_id": "interactive",
        "target": ["left", "forward"],
        "options": [
            {"id": "left", "text": "turn left"},
            {"id": "right", "text": "turn right"},
            {"id": "forward", "text": "move forward"},
        ],
    }
    probabilities = torch.tensor([[0.2, 0.1, 0.7]])
    logits = probabilities.log()
    loss = decision_loss("choice", logits, probabilities, sample)
    assert torch.allclose(loss, -torch.tensor(0.9).log())


def test_calibrated_probabilities_respect_task_domain() -> None:
    multiclass = calibrated_probabilities("choice", torch.tensor([[1.0, 2.0]]), 2.0)
    binary = calibrated_probabilities("noul", torch.tensor([1.0]), 2.0)
    assert torch.allclose(multiclass.sum(dim=-1), torch.ones(1))
    assert torch.allclose(binary, torch.sigmoid(torch.tensor([0.5])))


def test_threshold_selection_maximizes_eligible_coverage() -> None:
    records = []
    for task in ("choice", "noul", "score"):
        records.extend(
            [
                {"task_type": task, "confidence": 0.99, "correct": True},
                {"task_type": task, "confidence": 0.90, "correct": True},
                {"task_type": task, "confidence": 0.80, "correct": False},
            ]
        )
    report = select_confidence_thresholds(
        records, target_accuracy=1.0, minimum_accepted=2
    )
    for task in ("choice", "noul", "score"):
        assert report["by_task"][task]["threshold"] == 0.90
        assert report["by_task"][task]["coverage"] == 2 / 3


def test_source_threshold_reports_keep_sources_separate() -> None:
    records = [
        {
            "source": "minigrid",
            "task_type": "choice",
            "confidence": 0.9,
            "correct": True,
        },
        {
            "source": "boxoban",
            "task_type": "choice",
            "confidence": 0.9,
            "correct": False,
        },
    ]
    thresholds = {
        "by_task": {
            task: {"threshold": 0.8} for task in ("choice", "noul", "score")
        }
    }
    report = _source_threshold_reports(records, thresholds)
    assert report["minigrid"]["choice"]["accuracy"] == 1.0
    assert report["boxoban"]["choice"]["accuracy"] == 0.0


def test_latency_is_attached_to_overall_and_task_reports() -> None:
    report = {"by_task": {"choice": {}, "noul": {}}}
    records = [
        {"task_type": "choice", "latency_ms": 10.0},
        {"task_type": "choice", "latency_ms": 30.0},
        {"task_type": "noul", "latency_ms": 5.0},
    ]
    _attach_latency(report, records)
    assert report["latency_ms"]["total_ms"] == 45.0
    assert report["by_task"]["choice"]["latency_ms"]["mean_ms"] == 20.0
    assert report["by_task"]["noul"]["latency_ms"]["p99_ms"] == 5.0
