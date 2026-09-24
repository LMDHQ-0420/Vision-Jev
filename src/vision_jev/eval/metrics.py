"""Small reference metrics with no third-party dependency."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _check(probs: Sequence[Sequence[float]], targets: Sequence[int]) -> None:
    if not probs or len(probs) != len(targets):
        raise ValueError("probabilities and targets need the same non-zero length")
    for row, target in zip(probs, targets, strict=True):
        if not row or target < 0 or target >= len(row):
            raise ValueError("target must index a non-empty probability row")
        if any(value < 0 for value in row) or not math.isclose(sum(row), 1.0, abs_tol=1e-6):
            raise ValueError("each probability row must be non-negative and sum to one")


def accuracy(probs: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    _check(probs, targets)
    correct = sum(
        max(range(len(row)), key=row.__getitem__) == target
        for row, target in zip(probs, targets, strict=True)
    )
    return correct / len(targets)


def negative_log_likelihood(probs: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    _check(probs, targets)
    return -sum(
        math.log(max(row[target], 1e-12)) for row, target in zip(probs, targets, strict=True)
    ) / len(targets)


def brier_multiclass(probs: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    _check(probs, targets)
    total = 0.0
    for row, target in zip(probs, targets, strict=True):
        total += sum((value - float(index == target)) ** 2 for index, value in enumerate(row))
    return total / len(targets)
