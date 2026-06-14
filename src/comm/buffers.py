"""Double-buffering for async communication-computation overlap.

The key idea: while communication is in-flight for micro-batch K,
we compute expert outputs for micro-batch K-1 using a second buffer.
This eliminates the round-trip serialization penalty.

Buffer layout (per micro-batch slot):
  - tokens_buf: input token embeddings awaiting dispatch
  - expert_out_buf: expert outputs after gather, ready for combine
  - comm_handle: async handle for in-flight all-to-all
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class MicrobatchBuffer:
    tokens: Optional[torch.Tensor] = None
    expert_ids: Optional[torch.Tensor] = None
    expert_output: Optional[torch.Tensor] = None
    scatter_handle: Optional[object] = None
    gather_handle: Optional[object] = None
    step: int = -1


class DoubleBuffer:
    """Two-slot buffer pool for pipeline-style overlap."""

    def __init__(self):
        self.buf_a = MicrobatchBuffer()
        self.buf_b = MicrobatchBuffer()

    def front(self) -> MicrobatchBuffer:
        """The buffer currently being computed."""
        return self.buf_a

    def back(self) -> MicrobatchBuffer:
        """The buffer currently being communicated."""
        return self.buf_b

    def swap(self) -> None:
        """Swap front and back buffers after a pipeline stage completes."""
        self.buf_a, self.buf_b = self.buf_b, self.buf_a

    def reset(self) -> None:
        self.buf_a = MicrobatchBuffer()
        self.buf_b = MicrobatchBuffer()
