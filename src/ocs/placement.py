"""Expert affinity tracking for OCS-aware expert placement.

Tracks which experts are frequently co-activated by the router, then
suggests expert-to-rank mappings that group co-activated experts together.
This minimizes OCS reconfiguration by keeping frequently communicating
expert pairs on the same rank (intra-rank transfer = no circuit needed)
or on ranks with stable circuits.

The tracker is sampling-based: it records routing decisions during warm-up
steps and produces placement suggestions offline. Online re-placement
(remapping experts mid-training) is deferred to future work.
"""
from __future__ import annotations

from typing import List

import torch


class ExpertAffinityTracker:
    """Tracks expert co-activation from router outputs.

    For top-K routing (K >= 2), records pairwise co-selection of experts.
    For top-1 routing, records per-expert usage frequency.

    The resulting affinity matrix can be used to:
      - Suggest expert-to-rank placement minimizing inter-rank communication
      - Estimate OCS circuit pressure (many distinct pairs = many circuits needed)
      - Evaluate whether the current placement matches actual routing patterns

    Usage:
        tracker = ExpertAffinityTracker(num_experts)
        for step in warmup_steps:
            expert_ids, gate_weights, _ = moe.router(tokens)
            tracker.record_routing(expert_ids, gate_weights)
        suggested_placement = tracker.suggest_placement(experts_per_rank, world_size)
    """

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        # co_activation[e_a, e_b] = how often expert a and b were selected together
        self.co_activation_counts = torch.zeros(num_experts, num_experts, dtype=torch.float64)
        # per-expert usage count (for top-1 or load-aware placement)
        self.expert_usage = torch.zeros(num_experts, dtype=torch.float64)
        self.total_samples = 0

    def record_routing(
        self,
        expert_ids: torch.Tensor,   # [T] or [T, K]
        gate_weights: torch.Tensor, # [T, K]
    ) -> None:
        """Record one routing event.

        Args:
            expert_ids: expert assignments. Shape [T] for top-1, [T, K] for top-K.
            gate_weights: routing weights, shape [T, K].
        """
        T = expert_ids.shape[0]

        if expert_ids.dim() == 1:
            # top-1: record per-expert usage
            for e in expert_ids.unique():
                count = (expert_ids == e).sum().item()
                self.co_activation_counts[e, e] += count
                self.expert_usage[e] += count
            self.total_samples += T
        else:
            # top-K: record pairwise co-selection
            K = expert_ids.shape[1]
            self.total_samples += T

            for i in range(T):
                for a in range(K):
                    ea = int(expert_ids[i, a].item())
                    self.expert_usage[ea] += 1.0
                    for b in range(K):
                        eb = int(expert_ids[i, b].item())
                        self.co_activation_counts[ea, eb] += 1.0

    def get_affinity_scores(self) -> torch.Tensor:
        """Return normalized co-activation matrix [num_experts, num_experts].

        Values are in [0, 1], representing the probability that expert e_b
        is co-selected when expert e_a is selected.
        """
        if self.total_samples == 0:
            return torch.zeros(self.num_experts, self.num_experts)

        # Normalize by per-expert usage so each row sums to K (top-K co-selection)
        normalized = self.co_activation_counts.clone()
        for e in range(self.num_experts):
            if self.expert_usage[e] > 0:
                normalized[e] /= self.expert_usage[e]

        return normalized

    def suggest_placement(
        self,
        experts_per_rank: int,
        world_size: int,
    ) -> List[List[int]]:
        """Suggest expert-to-rank placement based on co-activation affinity.

        Uses a greedy clustering heuristic: for each rank, pick a seed expert
        (highest total affinity to unplaced experts), then fill remaining slots
        with the experts most co-activated with the seed.

        Returns:
            List of length world_size, each element is a list of expert IDs
            assigned to that rank. Total experts across all ranks equals
            world_size * experts_per_rank = num_experts.
        """
        affinity = self.get_affinity_scores()
        total_experts = world_size * experts_per_rank

        # If no data or affinity is uniform, fall back to round-robin
        if self.total_samples == 0 or total_experts != self.num_experts:
            return [
                list(range(r * experts_per_rank, (r + 1) * experts_per_rank))
                for r in range(world_size)
            ]

        # Greedy clustering
        remaining = set(range(self.num_experts))
        placement: List[List[int]] = [[] for _ in range(world_size)]

        for rank in range(world_size):
            if not remaining:
                break
            slots = experts_per_rank

            # Pick seed: expert with highest total affinity to all other remaining experts
            if len(placement[rank]) == 0:
                best_seed = -1
                best_score = -1.0
                for cand in sorted(remaining):
                    score = affinity[cand, list(remaining)].sum().item()
                    if score > best_score:
                        best_score = score
                        best_seed = cand
                if best_seed >= 0:
                    placement[rank].append(best_seed)
                    remaining.remove(best_seed)
                    slots -= 1

            # Fill remaining slots with most co-activated experts
            while slots > 0 and remaining:
                best_expert = -1
                best_score = -1.0
                for cand in sorted(remaining):
                    score = sum(
                        affinity[cand, e].item() + affinity[e, cand].item()
                        for e in placement[rank]
                    )
                    if score > best_score:
                        best_score = score
                        best_expert = cand
                if best_expert >= 0:
                    placement[rank].append(best_expert)
                    remaining.remove(best_expert)
                    slots -= 1
                else:
                    break

        # Distribute any remaining experts round-robin
        for i, e in enumerate(sorted(remaining)):
            placement[i % world_size].append(e)

        return placement

    def get_expert_utilization(self) -> torch.Tensor:
        """Return normalized per-expert usage frequencies [num_experts]."""
        if self.total_samples == 0:
            return torch.zeros(self.num_experts)
        return self.expert_usage / self.expert_usage.sum()

    def reset(self) -> None:
        """Reset all counters. Useful for per-epoch or per-phase tracking."""
        self.co_activation_counts.zero_()
        self.expert_usage.zero_()
        self.total_samples = 0
