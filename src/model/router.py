"""Token-to-expert routing strategies.

Supported strategies (Stage 1 → Stage N):
  - fixed: round-robin assignment, deterministic
  - uniform_random: random assignment, uniform distribution
  - top1: pick the expert with highest gate score
  - top2: pick top-2 experts (for future expansion)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Maps token embeddings → expert assignments."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 1,
        strategy: str = "fixed",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.strategy = strategy

        # Gate network: linear projection → expert logits
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            tokens: [total_tokens, hidden_dim]

        Returns:
            expert_ids: [total_tokens]  expert index per token
            gate_weights: [total_tokens, top_k]  routing weights
        """
        total_tokens = tokens.shape[0]

        if self.strategy == "fixed":
            # Round-robin: token i → expert i % num_experts
            expert_ids = torch.arange(total_tokens, device=tokens.device) % self.num_experts
            gate_weights = torch.ones(total_tokens, 1, device=tokens.device)

        elif self.strategy == "uniform_random":
            expert_ids = torch.randint(0, self.num_experts, (total_tokens,), device=tokens.device)
            gate_weights = torch.ones(total_tokens, 1, device=tokens.device)

        elif self.strategy == "top1":
            logits = self.gate(tokens)  # [T, E]
            gate_weights, expert_ids = torch.topk(logits, k=1, dim=-1)
            expert_ids = expert_ids.squeeze(-1)

        elif self.strategy == "top2":
            logits = self.gate(tokens)
            gate_weights, expert_ids = torch.topk(logits, k=2, dim=-1)
            # expert_ids: [T, 2], gate_weights: [T, 2]
            # We return both; the MoE layer handles multi-expert aggregation

        else:
            raise ValueError(f"Unknown routing strategy: {self.strategy}")

        return expert_ids, gate_weights
