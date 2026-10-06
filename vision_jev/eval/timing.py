"""Latency aggregation shared by evaluation and showcase reporting."""

from __future__ import annotations

import statistics
from collections.abc import Iterable


def summarize_latencies(values: Iterable[float]) -> dict[str, float | int] | None:
    samples = [float(value) for value in values]
    if not samples:
        return None
    if len(samples) == 1:
        p50 = p95 = p99 = samples[0]
    else:
        percentiles = statistics.quantiles(samples, n=100, method="inclusive")
        p50, p95, p99 = percentiles[49], percentiles[94], percentiles[98]
    return {
        "samples": len(samples),
        "total_ms": sum(samples),
        "mean_ms": statistics.fmean(samples),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "min_ms": min(samples),
        "max_ms": max(samples),
    }
