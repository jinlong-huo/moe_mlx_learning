"""Initialize the distributed process group.

On macOS, torch.distributed requires:
  - backend='gloo' (NCCL is CUDA-only)
  - start_method='spawn' (fork is unsafe on macOS)
  - MASTER_ADDR / MASTER_PORT set in environment
"""
from __future__ import annotations

import os
from typing import Dict

import torch
import torch.distributed as dist


def init_process_group(
    rank: int,
    world_size: int,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    backend: str = "gloo",
) -> None:
    """Initialize torch.distributed for a single process.

    Must be called once per spawned process before any communication.
    """
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_addr}:{master_port}",
    )


def cleanup_process_group() -> None:
    """Destroy the process group. Call before process exit."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    return dist.get_rank()


def get_world_size() -> int:
    return dist.get_world_size()
