"""Micro-batching utilities: split batches and manage indices."""
from __future__ import annotations

from typing import List

import torch


def split_microbatches(tensor: torch.Tensor, num_microbatches: int) -> List[torch.Tensor]:
    """Split a batch tensor into equal-sized micro-batches."""
    return list(torch.chunk(tensor, num_microbatches, dim=0))
