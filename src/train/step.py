"""Training step: forward + loss + backward + optimizer step.

Provides serial and overlap training steps. Each micro-batch runs
through the full MoE pipeline: route → scatter → compute → gather →
combine → loss → backward → optimizer.step().

In overlap mode, the backward pass for micro-batch K-1 is interleaved
with the forward scatter for micro-batch K.
"""
from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist

from src.model.moe_layer import MoELayer
from src.comm.transport import Transport
from src.train.loss import compute_loss
from src.utils.timer import Timer


def _sync_router_grads(moe: MoELayer) -> None:
    """All-reduce gradients for router parameters across all ranks."""
    world_size = dist.get_world_size()
    for param in moe.router.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size


def train_step_serial(
    step: int,
    microbatches: List[torch.Tensor],
    targets: List[torch.Tensor],
    moe: MoELayer,
    transport: Transport,
    optimizer: torch.optim.Optimizer,
    timer: Timer,
    alpha: float = 0.01,
) -> dict:
    """One serial training step: each micro-batch runs forward→loss→backward.

    Returns a dict of per-microbatch metrics.
    """
    mb_metrics = []

    for mb_idx, (tokens, tgt) in enumerate(zip(microbatches, targets)):
        timer.start(f"step/{step}/mb_{mb_idx}/forward")
        output, logits, send_counts, _dispatch = moe.forward_train(tokens, transport)
        timer.stop(f"step/{step}/mb_{mb_idx}/forward")

        timer.start(f"step/{step}/mb_{mb_idx}/loss")
        loss, loss_task, loss_aux = compute_loss(
            output, tgt, logits, send_counts, moe.experts_per_rank, alpha,
        )
        timer.stop(f"step/{step}/mb_{mb_idx}/loss")

        timer.start(f"step/{step}/mb_{mb_idx}/backward")
        optimizer.zero_grad()
        loss.backward()
        _sync_router_grads(moe)
        optimizer.step()
        timer.stop(f"step/{step}/mb_{mb_idx}/backward")

        mb_metrics.append({
            "mb_idx": mb_idx,
            "loss_total": loss.item(),
            "loss_task": loss_task.item(),
            "loss_aux": loss_aux.item(),
        })

    return {
        "step": step,
        "microbatches": mb_metrics,
        "loss_total": sum(m["loss_total"] for m in mb_metrics) / len(mb_metrics),
        "loss_task": sum(m["loss_task"] for m in mb_metrics) / len(mb_metrics),
        "loss_aux": sum(m["loss_aux"] for m in mb_metrics) / len(mb_metrics),
    }
