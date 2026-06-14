"""Training orchestrator: epoch loop, metrics, checkpointing.

Runs inside each worker process. Coordinates:
  - SyntheticMixtureDataset generation (same seed on all ranks → same data)
  - Epoch / step loop
  - Per-epoch metrics: loss, expert utilization, routing accuracy
  - Chrome Trace export per epoch
  - Model checkpointing

All ranks train on identical data (deterministic dataset with fixed seed).
In a real distributed MoE training run, each rank would process different
data — but for this mechanism verification testbed, identical data makes
it easy to verify correctness (all ranks should learn the same thing).
"""
from __future__ import annotations

import os
from typing import Dict, List

import torch

from src.data.synthetic import SyntheticMixtureDataset
from src.model.moe_layer import MoELayer
from src.comm.transport import Transport
from src.comm.timeline import export_chrome_trace
from src.train.step import train_step_serial
from src.utils.timer import Timer
from src.utils.logging import log


class Trainer:
    """Single-rank training orchestrator."""

    def __init__(
        self,
        moe: MoELayer,
        transport: Transport,
        timer: Timer,
        config: Dict,
        rank: int,
        world_size: int,
        trace_dir: str = "outputs/traces",
    ):
        self.moe = moe
        self.transport = transport
        self.timer = timer
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.trace_dir = trace_dir

        # Training hyperparams
        train_cfg = config.get("training", {})
        self.num_epochs = train_cfg.get("num_epochs", 20)
        self.learning_rate = train_cfg.get("learning_rate", 0.001)
        self.alpha = train_cfg.get("load_balance_alpha", 0.01)

        # Dataset
        data_cfg = train_cfg.get("dataset", {})
        model_cfg = config["model"]
        runtime_cfg = config["runtime"]
        data_cfg_runtime = config["data"]

        self.hidden_dim = model_cfg["hidden_dim"]
        self.num_experts = model_cfg["num_experts"]
        self.batch_size = data_cfg_runtime["batch_size"]
        self.seq_len = data_cfg_runtime["seq_len"]
        self.num_microbatches = data_cfg_runtime["num_microbatches"]
        self.tokens_per_mb = self.batch_size * self.seq_len
        self.num_steps = runtime_cfg.get("num_steps", 5)

        # Build dataset (same seed → all ranks generate identical data)
        dataset_seed = data_cfg.get("seed", 42)
        self.dataset = SyntheticMixtureDataset(
            hidden_dim=self.hidden_dim,
            num_modes=data_cfg.get("num_modes", self.num_experts),
            num_samples=data_cfg.get("num_samples", 1024),
            seed=dataset_seed,
            mode_separation=data_cfg.get("mode_separation", 5.0),
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            moe.parameters(), lr=self.learning_rate,
        )

        # Metrics history
        self.epoch_metrics: List[Dict] = []

    def train(self) -> List[Dict]:
        """Run full training loop. Returns epoch metrics history."""
        log(self.rank, f"Trainer: {self.num_epochs} epochs, {self.num_steps} steps/epoch, "
            f"lr={self.learning_rate}, alpha={self.alpha}")
        log(self.rank, f"Dataset: {len(self.dataset)} samples, "
            f"{self.dataset.num_modes} modes, dim={self.hidden_dim}")

        for epoch in range(self.num_epochs):
            epoch_metrics = self._train_epoch(epoch)
            self.epoch_metrics.append(epoch_metrics)

            # Log
            log(self.rank,
                f"Epoch {epoch:3d} | "
                f"loss={epoch_metrics['loss_total']:.6f} "
                f"task={epoch_metrics['loss_task']:.6f} "
                f"aux={epoch_metrics['loss_aux']:.6f} | "
                f"router_acc={epoch_metrics.get('router_accuracy', 0):.3f}"
            )

        return self.epoch_metrics

    def _train_epoch(self, epoch: int) -> Dict:
        """Run one epoch of training steps."""
        timer = self.timer
        timer.reset()

        step_losses = []
        step_tasks = []
        step_auxs = []

        for step in range(self.num_steps):
            # Sample a fresh batch each step
            tokens, targets, _mode_labels = self.dataset.subset(self.tokens_per_mb)
            microbatches = torch.chunk(tokens, self.num_microbatches, dim=0)
            target_mbs = torch.chunk(targets, self.num_microbatches, dim=0)

            step_result = train_step_serial(
                step=step,
                microbatches=microbatches,
                targets=target_mbs,
                moe=self.moe,
                transport=self.transport,
                optimizer=self.optimizer,
                timer=timer,
                alpha=self.alpha,
            )

            step_losses.append(step_result["loss_total"])
            step_tasks.append(step_result["loss_task"])
            step_auxs.append(step_result["loss_aux"])

        # -- Evaluate router accuracy on full dataset --
        router_acc = self._eval_router_accuracy()

        # -- Expert utilization --
        expert_usage = self._compute_expert_utilization()

        # -- Export trace for last epoch --
        if self.config.get("profiling", {}).get("export_trace", True):
            self._export_epoch_trace(epoch)

        return {
            "epoch": epoch,
            "loss_total": sum(step_losses) / len(step_losses),
            "loss_task": sum(step_tasks) / len(step_tasks),
            "loss_aux": sum(step_auxs) / len(step_auxs),
            "router_accuracy": router_acc,
            "expert_utilization": expert_usage,
            "num_steps": self.num_steps,
        }

    def _eval_router_accuracy(self) -> float:
        """Measure how well the router maps modes to the correct expert.

        Since we don't pre-define which expert should handle which mode,
        we measure clustering purity: for each mode, what fraction of its
        tokens go to the most-common expert. If the router perfectly separates
        modes, each mode routes 100% to a single expert → accuracy = 1.0.
        """
        tokens, _targets, mode_labels = self.dataset.full()
        with torch.no_grad():
            expert_ids, _weights, _logits = self.moe.router(tokens)

        if expert_ids.dim() == 2:
            expert_ids = expert_ids[:, 0]  # top-K: use first choice

        # For each mode, compute the fraction going to its most-popular expert
        accuracies = []
        for mode in range(self.dataset.num_modes):
            mode_mask = (mode_labels == mode)
            if mode_mask.sum() == 0:
                continue
            mode_experts = expert_ids[mode_mask]
            # Most-common expert for this mode
            best_expert_count = torch.bincount(mode_experts, minlength=self.num_experts).max()
            acc = best_expert_count.float() / mode_mask.sum().float()
            accuracies.append(acc.item())

        return sum(accuracies) / len(accuracies) if accuracies else 0.0

    def _compute_expert_utilization(self) -> List[float]:
        """Compute what fraction of tokens go to each expert (on this rank).

        Returns a per-rank-expert list [E_local] of token fractions.
        """
        tokens, _targets, _mode_labels = self.dataset.full()
        with torch.no_grad():
            expert_ids, _weights, _logits = self.moe.router(tokens)

        if expert_ids.dim() == 2:
            expert_ids = expert_ids[:, 0]

        total = expert_ids.numel()
        utilization = []
        for e in range(self.num_experts):
            frac = (expert_ids == e).sum().float().item() / total
            utilization.append(frac)

        return utilization

    def _export_epoch_trace(self, epoch: int) -> None:
        """Export Chrome Trace for this epoch."""
        os.makedirs(self.trace_dir, exist_ok=True)
        trace_path = os.path.join(
            self.trace_dir, f"rank_{self.rank:02d}_epoch_{epoch:03d}_trace.json"
        )
        ep_meta = {
            "world_size": self.world_size,
            "num_experts": self.num_experts,
            "experts_per_rank": self.moe.experts_per_rank,
            "mode": "train",
            "epoch": epoch,
        }
        export_chrome_trace(self.timer.events, trace_path, pid=self.rank, tid=0, metadata=ep_meta)

    def save_checkpoint(self, path: str) -> None:
        """Save model and optimizer state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state": self.moe.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch_metrics": self.epoch_metrics,
            "config": self.config,
        }
        torch.save(checkpoint, path)
