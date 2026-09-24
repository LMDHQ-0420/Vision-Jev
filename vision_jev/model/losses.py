"""Question-normalized supervised objectives."""

from __future__ import annotations

import torch
from torch import Tensor


def choice_set_loss(probs: Tensor, valid_targets: Tensor) -> Tensor:
    """Negative log probability mass over one or more valid candidates."""
    mass = (probs.float() * valid_targets.float()).sum(dim=-1)
    if (valid_targets.sum(dim=-1) == 0).any():
        raise ValueError("each question needs at least one valid target")
    return -mass.clamp_min(1e-12).log().mean()


def ranked_probability_score(probs: Tensor, target_index: Tensor, mask: Tensor) -> Tensor:
    """Mean squared CDF error over valid thresholds, excluding the trivial final CDF."""
    cdf = probs.float().cumsum(dim=-1)
    positions = torch.arange(probs.shape[-1], device=probs.device).unsqueeze(0)
    target_cdf = (positions >= target_index.unsqueeze(1)).float()
    threshold_mask = mask.bool().clone()
    last = threshold_mask.long().sum(dim=-1) - 1
    threshold_mask.scatter_(1, last.unsqueeze(1), False)
    error = (cdf - target_cdf).square() * threshold_mask
    return (error.sum(dim=-1) / threshold_mask.sum(dim=-1).clamp_min(1)).mean()
