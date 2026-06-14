"""Synthetic mixture-of-modes dataset for MoE training.

Generates tokens drawn from K distinct Gaussian clusters ("modes"), each
associated with a different target linear transformation. The MoE router
never sees the mode label — it must discover clusters from token structure
and learn to dispatch mode-i tokens to the right expert.

This is the simplest possible dataset that demonstrates expert specialization:
  - Mode 0: tokens ~ N(μ₀, σ²) → target = tokens @ W₀
  - Mode 1: tokens ~ N(μ₁, σ²) → target = tokens @ W₁
  - ...

With K = num_experts, the ideal outcome is a 1:1 mode-to-expert mapping
where each expert learns its mode's transform and the router learns to
separate the clusters.
"""
from __future__ import annotations

import torch


class SyntheticMixtureDataset:
    """Synthetic dataset: K Gaussian modes, each with a different target W.

    Each mode is a cluster centered at a random direction μ_k scaled by a
    large magnitude so modes are well-separated in token space. Each mode
    has a distinct target linear transform W_k. The task is to predict
    `tokens @ W_k` given `tokens` without knowing k.

    Args:
        hidden_dim: token embedding dimension
        num_modes: number of distinct clusters (ideally = num_experts)
        num_samples: total number of tokens to generate
        seed: random seed for reproducibility
        mode_separation: magnitude of cluster centers (higher = easier routing)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_modes: int = 8,
        num_samples: int = 1024,
        seed: int = 42,
        mode_separation: float = 5.0,
    ):
        gen = torch.Generator()
        gen.manual_seed(seed)

        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.num_samples = num_samples

        # -- Per-mode target transforms W_k: [num_modes, hidden_dim, hidden_dim] --
        # Fully random matrices so each mode requires a genuinely different transform.
        # Scaled to keep output magnitude similar to input.
        self.W = torch.zeros(num_modes, hidden_dim, hidden_dim)
        for k in range(num_modes):
            wk = torch.randn(hidden_dim, hidden_dim, generator=gen) / (hidden_dim ** 0.5)
            self.W[k] = wk

        # -- Per-mode cluster centers μ_k: [num_modes, hidden_dim] --
        # Random directions scaled by mode_separation for clean separation
        self.centers = torch.randn(num_modes, hidden_dim, generator=gen)
        self.centers = self.centers / self.centers.norm(dim=1, keepdim=True) * mode_separation

        # -- Generate all samples --
        samples_per_mode = num_samples // num_modes
        remainder = num_samples % num_modes

        tokens_list = []
        targets_list = []
        labels_list = []

        for k in range(num_modes):
            n = samples_per_mode + (1 if k < remainder else 0)
            # Tokens centered at μ_k with unit Gaussian noise
            noise = torch.randn(n, hidden_dim, generator=gen)
            t = self.centers[k].unsqueeze(0) + noise  # [n, H]
            # Target: transform by W_k
            y = t @ self.W[k]  # [n, H]
            tokens_list.append(t)
            targets_list.append(y)
            labels_list.append(torch.full((n,), k, dtype=torch.long))

        self.tokens = torch.cat(tokens_list, dim=0)       # [N, H]
        self.targets = torch.cat(targets_list, dim=0)     # [N, H]
        self.mode_labels = torch.cat(labels_list, dim=0)  # [N]

        # Shuffle
        perm = torch.randperm(num_samples, generator=gen)
        self.tokens = self.tokens[perm]
        self.targets = self.targets[perm]
        self.mode_labels = self.mode_labels[perm]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.targets[idx]

    def subset(self, size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a random subset as (tokens, targets, mode_labels)."""
        idxs = torch.randperm(self.num_samples)[:size]
        return self.tokens[idxs], self.targets[idxs], self.mode_labels[idxs]

    def full(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all data as (tokens, targets, mode_labels)."""
        return self.tokens, self.targets, self.mode_labels
