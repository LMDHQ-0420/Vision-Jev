"""Hidden-state decision heads. Requires the optional train dependencies."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


def masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    if logits.shape != mask.shape:
        raise ValueError(f"logits {logits.shape} and mask {mask.shape} must match")
    if (~mask.bool()).all(dim=-1).any():
        raise ValueError("every question needs at least one valid candidate")
    safe = logits.float().masked_fill(~mask.bool(), torch.finfo(torch.float32).min)
    return torch.softmax(safe, dim=-1).masked_fill(~mask.bool(), 0.0)


class ChoiceHead(nn.Module):
    """Independent evidence plus a small permutation-equivariant set correction."""

    def __init__(self, hidden_size: int = 1024, width: int = 256, heads: int = 4) -> None:
        super().__init__()
        self.question = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
        self.candidate = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
        self.independent = nn.Linear(width, 1)
        self.set_norm = nn.LayerNorm(width)
        self.set_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.set_ffn = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )
        self.correction = nn.Linear(width, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, q: Tensor, h: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if h.ndim != 3 or q.ndim != 2 or h.shape[0] != q.shape[0]:
            raise ValueError("expected q[B,D] and h[B,K,D]")
        q_proj = torch.tanh(self.question(q)).unsqueeze(1)
        x = self.candidate(h)
        independent = self.independent(x * q_proj).squeeze(-1)
        normed = self.set_norm(x + q_proj)
        attended, _ = self.set_attention(
            normed, normed, normed, key_padding_mask=~mask.bool(), need_weights=False
        )
        set_x = x + attended
        set_x = set_x + self.set_ffn(set_x)
        logits = independent + self.residual_scale * self.correction(set_x).squeeze(-1)
        return logits, masked_softmax(logits, mask)


class NoulHead(nn.Module):
    def __init__(self, hidden_size: int = 1024, width: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(self, q: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.net(q).squeeze(-1)
        return logits, torch.sigmoid(logits.float())


class RegionEvidence(nn.Module):
    """Fuse pooled native-grid evidence; missing regions contribute exactly zero."""

    def __init__(self, hidden_size: int = 1024, width: int = 256) -> None:
        super().__init__()
        self.region = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
        self.geometry = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width))
        self.gate = nn.Linear(width * 3, width)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(
        self, candidate: Tensor, question: Tensor, region: Tensor, geometry: Tensor
    ) -> Tensor:
        if geometry.shape[-1] != 6:
            raise ValueError("geometry must contain x1,y1,x2,y2,area,valid")
        valid = geometry[..., 5:6].to(candidate.dtype)
        region_x = (self.region(region) + self.geometry(geometry)) * valid
        gate = torch.sigmoid(
            self.gate(
                torch.cat([candidate, question.unsqueeze(1).expand_as(candidate), region_x], dim=-1)
            )
        )
        return cast(Tensor, candidate + gate * region_x * valid)


class ScoreHead(nn.Module):
    def __init__(self, hidden_size: int = 1024, width: int = 256) -> None:
        super().__init__()
        self.question = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
        self.level = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
        self.score = nn.Linear(width, 1)

    def forward(self, q: Tensor, levels: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        fused = self.level(levels) * torch.tanh(self.question(q)).unsqueeze(1)
        logits = self.score(fused).squeeze(-1)
        return logits, masked_softmax(logits, mask)


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int = 1024, width: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, q: Tensor, h: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(h.dtype).unsqueeze(-1)
        pooled = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return cast(Tensor, self.net(torch.cat([q, pooled], dim=-1)).squeeze(-1))
