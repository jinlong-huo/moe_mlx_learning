"""Micro-batch scheduler: serial baseline vs async overlap.

Serial mode:
  For each micro-batch i:
    route -> scatter (blocking) -> compute -> gather (blocking) -> combine

Overlap mode:
  Pipeline across micro-batches using double-buffering:
    Fire scatter for batch K
    While scatter is in-flight, compute expert for batch K-1
    Wait for scatter K, fire gather K
    While gather K is in-flight, fire scatter K+1 and compute K

Top-K support:
  For top_k > 1, the router returns expert_ids [T, K] and gate_weights [T, K].
  scatter_tokens flattens to [T*K] internally. combine_expert_outputs()
  applies gate weights and sums after gather.
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from src.model.moe_layer import MoELayer
from src.comm.transport import Transport
from src.comm.all_to_all import (
    scatter_tokens, gather_tokens, combine_expert_outputs, DispatchResult,
)
from src.utils.timer import Timer


def run_serial(
    step: int,
    microbatches: List[torch.Tensor],
    moe: MoELayer,
    transport: Transport,
    timer: Timer,
) -> None:
    """Baseline: comm -> compute -> comm -> combine, no overlap."""
    for mb_idx, tokens in enumerate(microbatches):
        # Route: get expert assignments and gate weights
        timer.start(f"step/{step}/mb_{mb_idx}/route")
        expert_ids, gate_weights = moe.router(tokens)
        timer.stop(f"step/{step}/mb_{mb_idx}/route")

        # Dispatch (blocking all-to-all) -- top-K flattening handled internally
        timer.start(f"step/{step}/mb_{mb_idx}/scatter")
        dispatch = scatter_tokens(
            tokens, expert_ids, moe.num_experts,
            moe.experts_per_rank, transport, async_op=False,
        )
        timer.stop(f"step/{step}/mb_{mb_idx}/scatter")

        # Compute -- dispatch to correct local expert
        timer.start(f"step/{step}/mb_{mb_idx}/compute")
        expert_out = moe.compute_experts(dispatch.tokens, dispatch.local_expert_ids)
        timer.stop(f"step/{step}/mb_{mb_idx}/compute")

        # Gather (blocking all-to-all) -- returns [T*K, H] for top-K
        timer.start(f"step/{step}/mb_{mb_idx}/gather")
        gathered = gather_tokens(expert_out, dispatch, transport, async_op=False)
        timer.stop(f"step/{step}/mb_{mb_idx}/gather")

        # Combine multi-expert outputs (no-op for top_k=1)
        timer.start(f"step/{step}/mb_{mb_idx}/combine")
        combined = combine_expert_outputs(gathered, gate_weights)
        timer.stop(f"step/{step}/mb_{mb_idx}/combine")


def run_overlap(
    step: int,
    microbatches: List[torch.Tensor],
    moe: MoELayer,
    transport: Transport,
    timer: Timer,
) -> None:
    """Overlap: pipeline scatter/compute/gather across micro-batches.

    For top_k > 1: combine is a cheap local operation done after gather
    returns, outside the critical comm path.

    Timeline (2 micro-batches):
      mb_0: [scatter_0 fire] --------------------+
            (scatter_0 in flight)                |
      mb_1: [scatter_1 fire]                     |
            [scatter_0 wait] <-------------------+
            [compute_0]
            [gather_0 fire] ---------------------+
            (gather_0 in flight)                 |
      tail: [scatter_1 wait]                     |
            [compute_1]                          |
            [gather_1 fire+wait] <---------------+
            [combine_0] [combine_1]              (cheap local ops)
    """
    num_mbs = len(microbatches)

    # Pre-route all micro-batches (record route time per mb)
    routes = []        # List[Tuple[expert_ids, gate_weights]]
    for mb_idx, tokens in enumerate(microbatches):
        timer.start(f"step/{step}/mb_{mb_idx}/route")
        expert_ids, gate_weights = moe.router(tokens)
        timer.stop(f"step/{step}/mb_{mb_idx}/route")
        routes.append((expert_ids, gate_weights))

    # -- Pipeline ----------------------------------------------------
    prev_dispatch = None   # DispatchResult from the previous micro-batch
    prev_gate_weights = None
    prev_out = None        # expert outputs from the previous micro-batch
    scatter_handle = None
    prev_gather_handle = None   # gather from TWO iterations ago
    gathered_prev = None   # raw gather output from 2-iterations-ago (for tail combine)

    for mb_idx in range(num_mbs):
        tokens = microbatches[mb_idx]
        expert_ids, gate_weights = routes[mb_idx]

        # 1. Wait for previous scatter (so we can read dispatch result safely)
        if scatter_handle is not None:
            timer.start(f"step/{step}/mb_{mb_idx-1}/scatter_wait")
            scatter_handle.wait()
            timer.stop(f"step/{step}/mb_{mb_idx-1}/scatter_wait")
            prev_dispatch = scatter_result

        # 2. Fire scatter for current micro-batch (async)
        timer.start(f"step/{step}/mb_{mb_idx}/scatter")
        scatter_result, scatter_handle = scatter_tokens(
            tokens, expert_ids, moe.num_experts,
            moe.experts_per_rank, transport, async_op=True,
        )
        timer.stop(f"step/{step}/mb_{mb_idx}/scatter")

        # 3. Wait for gather from mb_{idx-2} (it should be done by now)
        if prev_gather_handle is not None:
            timer.start(f"step/{step}/mb_{mb_idx-2}/gather_wait")
            prev_gather_handle.wait()
            timer.stop(f"step/{step}/mb_{mb_idx-2}/gather_wait")

        # 4. While scatter is in-flight, compute mb_{idx-1}
        if prev_dispatch is not None:
            timer.start(f"step/{step}/mb_{mb_idx-1}/compute")
            prev_out = moe.compute_experts(
                prev_dispatch.tokens, prev_dispatch.local_expert_ids,
            )
            timer.stop(f"step/{step}/mb_{mb_idx-1}/compute")

            # Fire gather for mb_{idx-1} (async)
            timer.start(f"step/{step}/mb_{mb_idx-1}/gather")
            gathered_prev, prev_gather_handle = gather_tokens(
                prev_out, prev_dispatch, transport, async_op=True,
            )
            timer.stop(f"step/{step}/mb_{mb_idx-1}/gather")

            # Combine for mb_{idx-1} (cheap local op, do it while comm is in-flight)
            timer.start(f"step/{step}/mb_{mb_idx-1}/combine")
            _combined = combine_expert_outputs(gathered_prev, prev_gate_weights)
            timer.stop(f"step/{step}/mb_{mb_idx-1}/combine")

        # Save gate_weights for next iteration's combine
        prev_gate_weights = gate_weights

    # -- Drain the pipeline tail ------------------------------------
    # Wait for final scatter (mb_{num_mbs-1})
    if scatter_handle is not None:
        last = num_mbs - 1
        timer.start(f"step/{step}/mb_{last}/scatter_wait")
        scatter_handle.wait()
        timer.stop(f"step/{step}/mb_{last}/scatter_wait")
        prev_dispatch = scatter_result

    # Wait for second-to-last gather (mb_{num_mbs-2})
    if prev_gather_handle is not None:
        second_last = num_mbs - 2
        timer.start(f"step/{step}/mb_{second_last}/gather_wait")
        prev_gather_handle.wait()
        timer.stop(f"step/{step}/mb_{second_last}/gather_wait")

    # Compute and gather last micro-batch
    if prev_dispatch is not None:
        last = num_mbs - 1
        timer.start(f"step/{step}/mb_{last}/compute")
        prev_out = moe.compute_experts(
            prev_dispatch.tokens, prev_dispatch.local_expert_ids,
        )
        timer.stop(f"step/{step}/mb_{last}/compute")

        # Gather last micro-batch (blocking -- no more work to overlap with)
        timer.start(f"step/{step}/mb_{last}/gather")
        gathered_last = gather_tokens(prev_out, prev_dispatch, transport, async_op=False)
        timer.stop(f"step/{step}/mb_{last}/gather")

        # Combine last micro-batch
        timer.start(f"step/{step}/mb_{last}/combine")
        combined = combine_expert_outputs(gathered_last, prev_gate_weights)
        timer.stop(f"step/{step}/mb_{last}/combine")
