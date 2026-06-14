"""Expert modules: the compute units of the MoE layer.

Each expert is an independent FFN (MLP). In the multi-process simulation,
rank `r` owns expert `r`. The all-to-all dispatch sends tokens to the
rank whose expert they were assigned to.

Start tiny (single linear layer) and grow toward realistic sizes:
  Stage 1: Linear(hidden_dim, hidden_dim)           — minimal
  Stage 2: Linear → GELU → Linear                    — standard FFN
  Stage N: Multi-head attention expert (future)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TinyExpert(nn.Module):
    """Minimal expert: single linear projection (Stage 1)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FFNExpert(nn.Module):
    """Standard 2-layer FFN with GELU activation (Stage 2+)."""

    def __init__(self, hidden_dim: int, expand_mult: int = 4):
        super().__init__()
        inner_dim = hidden_dim * expand_mult
        self.fc1 = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.fc2 = nn.Linear(inner_dim, hidden_dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
