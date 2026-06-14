"""Training step: forward pass through one MoE layer.

Stage 1 only does forward pass (no backward, no optimizer).
This keeps the mechanism verification clean.
"""
from __future__ import annotations

import torch

from src.model.moe_layer import MoELayer
from src.comm.transport import Transport
from src.utils.timer import Timer


def forward_step(
    tokens: torch.Tensor,
    moe: MoELayer,
    transport: Transport,
    timer: Timer,
    step: int = 0,
) -> torch.Tensor:
    """Single serial forward pass through the MoE layer (used by serial scheduler)."""
    timer.start(f"step/{step}/forward")
    output = moe.forward_serial(tokens, transport)
    timer.stop(f"step/{step}/forward")
    return output
