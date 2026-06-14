"""Loss functions for MoE training.

Two losses combined:
  1. Task loss (MSE): how well the MoE output matches the target
  2. Load-balancing auxiliary loss (Switch Transformer): encourages
     uniform expert usage, prevents expert collapse

The load-balancing loss uses router logits and per-rank dispatch counts
to compute f_e (fraction of tokens to expert e) and P_e (mean router
probability for expert e), then: loss_aux = E * sum(f_e * P_e)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def task_loss(output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error between MoE output and target tokens."""
    return F.mse_loss(output, targets)


def load_balance_loss(
    logits: torch.Tensor,          # [T, num_experts]  raw gate logits
    send_counts: torch.Tensor,     # [world_size]  tokens sent to each rank
    experts_per_rank: int = 1,
) -> torch.Tensor:
    """Switch Transformer auxiliary loss: encourage uniform expert usage.

    f_e = fraction of tokens dispatched to expert e
    P_e = mean router softmax probability for expert e
    loss = num_experts * sum(f_e * P_e)

    The loss is minimized when routing is perfectly uniform (f_e = P_e = 1/E),
    giving a minimum value of 1.0. Values above 1.0 indicate imbalance.

    For top-K > 1, tokens are dispatched K times, so f_e accounts for
    the fact that each token can go to K experts.

    Args:
        logits: raw gate logits from the router [T, num_experts]
        send_counts: dispatch counts per rank from scatter_tokens [world_size]
        experts_per_rank: number of local experts per rank

    Returns:
        scalar auxiliary loss
    """
    T, E = logits.shape
    world_size = send_counts.shape[0]
    total_experts = E  # same as world_size * experts_per_rank

    if total_experts != world_size * experts_per_rank:
        raise ValueError(
            f"num_experts ({total_experts}) != world_size ({world_size}) * "
            f"experts_per_rank ({experts_per_rank})"
        )

    # P_e: mean router probability for each expert [E]
    probs = F.softmax(logits, dim=-1)   # [T, E]
    P = probs.mean(dim=0)               # [E]

    # f_e: fraction of tokens dispatched to each expert [E]
    # send_counts[r] tells how many tokens went to rank r
    # Each rank r hosts experts_per_rank experts
    total_sent = send_counts.sum().float()
    if total_sent == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    f_rank = send_counts.float() / total_sent       # [world_size]
    # Each expert on rank r gets an equal share of that rank's tokens
    f = f_rank.repeat_interleave(experts_per_rank) / experts_per_rank  # [E]

    # Switch Transformer loss: E * sum(f_e * P_e)
    # Minimum = 1.0 when perfectly uniform
    loss = total_experts * (f.detach() * P).sum()

    return loss


def compute_loss(
    output: torch.Tensor,
    targets: torch.Tensor,
    logits: torch.Tensor,
    send_counts: torch.Tensor,
    experts_per_rank: int = 1,
    alpha: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined loss: task + alpha * load_balance.

    Returns:
        total_loss, task_loss, aux_loss
    """
    loss_task = task_loss(output, targets)
    loss_aux = load_balance_loss(logits, send_counts, experts_per_rank)
    loss_total = loss_task + alpha * loss_aux
    return loss_total, loss_task.detach(), loss_aux.detach()
