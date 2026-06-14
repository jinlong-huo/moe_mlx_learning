"""Abstract transport interface wrapping torch.distributed.

All communication passes through this module so we can:
  - swap backends (Gloo / NCCL) without touching model code
  - inject synthetic delay for network simulation
  - instrument every collective with timeline events
  - model hierarchical topology (NVLink / IB / cross-pod) delays
"""
from __future__ import annotations

import random
import time
from typing import Optional

import torch
import torch.distributed as dist

from src.utils.timer import Timer


class Transport:
    """Wraps torch.distributed collective ops with optional delay injection.

    Supports two delay modes:
      1. Flat delay: comm_delay_us +- jitter applied uniformly to all collectives
      2. Topology-aware delay: hierarchical network model with per-tier latency
         and bandwidth (requires a Topology instance)

    When topology is set, the flat delay parameters are ignored in favor
    of topology-computed delays.
    """

    def __init__(
        self,
        timer: Optional[Timer] = None,
        comm_delay_us: float = 0.0,
        comm_delay_jitter_us: float = 0.0,
        topology=None,   # Topology instance (optional, avoids circular import)
        rank: int = 0,
        world_size: Optional[int] = None,
    ):
        self.timer = timer
        self.comm_delay_us = comm_delay_us
        self.comm_delay_jitter_us = comm_delay_jitter_us
        self.topology = topology
        self.rank = rank
        self._world_size = world_size

    def set_world_size(self, world_size: int) -> None:
        """Set the world size (call after init_process_group if not passed)."""
        self._world_size = world_size

    # -- delay injection --------------------------------------------------

    def _inject_delay(self, tensor_bytes: int = 0) -> None:
        """Inject synthetic communication delay.

        If topology is configured, uses topology-aware delay based on
        link tier and tensor size. Otherwise falls back to flat delay + jitter.

        Args:
            tensor_bytes: total bytes in the tensor (for bandwidth modeling)
        """
        if self.topology is not None and self._world_size is not None:
            # Topology-aware delay
            total = self.topology.get_delay(self.rank, self._world_size, tensor_bytes)
            if total > 0:
                time.sleep(total / 1_000_000.0)
            return

        # Flat delay mode (backward compatible)
        if self.comm_delay_us <= 0 and self.comm_delay_jitter_us <= 0:
            return
        jitter = random.uniform(-self.comm_delay_jitter_us, self.comm_delay_jitter_us)
        total = max(0.0, self.comm_delay_us + jitter)
        if total > 0:
            time.sleep(total / 1_000_000.0)

    # -- collective ops ----------------------------------------------------

    def all_to_all(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, async_op: bool = False
    ):
        """All-to-all collective.  Optionally async for overlap mode.

        Uses all_to_all_single which splits a single tensor evenly across
        ranks along dim 0 -- the natural fit for MoE dispatch where
        each rank handles one or more experts.
        """
        if self.timer:
            self.timer.start("comm/all_to_all", async_op=async_op)

        # Compute tensor bytes for bandwidth-aware delay
        tensor_bytes = input_tensor.numel() * input_tensor.element_size()
        self._inject_delay(tensor_bytes=tensor_bytes)

        handle = dist.all_to_all_single(output_tensor, input_tensor, async_op=async_op)
        if self.timer and not async_op:
            self.timer.stop("comm/all_to_all")
        return handle

    def all_gather(self, tensor: torch.Tensor, async_op: bool = False):
        """Gather tensors from all ranks into a list."""
        world_size = dist.get_world_size()
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
        if self.timer:
            self.timer.start("comm/all_gather", async_op=async_op)

        tensor_bytes = tensor.numel() * tensor.element_size()
        self._inject_delay(tensor_bytes=tensor_bytes)

        handle = dist.all_gather(gather_list, tensor, async_op=async_op)
        if self.timer and not async_op:
            self.timer.stop("comm/all_gather")
        return gather_list, handle

    def barrier(self) -> None:
        if self.timer:
            self.timer.start("comm/barrier")
        dist.barrier()
        if self.timer:
            self.timer.stop("comm/barrier")

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> None:
        if self.timer:
            self.timer.start("comm/broadcast", src=src)
        dist.broadcast(tensor, src=src)
        if self.timer:
            self.timer.stop("comm/broadcast")

    def wait(self, handle) -> None:
        """Wait on an async handle and record completion."""
        handle.wait()
        # Timer stop happens at the call-site so caller controls the label
