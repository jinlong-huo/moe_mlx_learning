"""Backbone: non-MoE computation to overlap with communication.

In a real MoE model, non-expert layers (attention, layer norm, embeddings)
run between MoE layers. These provide "free" computation to hide
communication behind. This module provides synthetic non-MoE work.

For Stage 1, this is just a configurable sleep/projection to represent
the compute budget available for overlap.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DenseProjection(nn.Module):
    """A simple dense projection to simulate non-MoE compute."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.proj(x))
