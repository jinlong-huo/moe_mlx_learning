"""Per-rank worker: the main execution loop for one process.

Each worker:
  1. Initializes its process group
  2. Builds the MoE layer (owning one expert)
  3. Generates synthetic input data
  4. Runs the scheduler (serial or overlap mode)
  5. Records timeline events
  6. Exports trace and metrics
"""
from __future__ import annotations

import os
from typing import Dict

import torch

from src.runtime.process_group import init_process_group, cleanup_process_group, get_rank
from src.runtime.scheduler import run_serial, run_overlap
from src.model.moe_layer import MoELayer
from src.comm.transport import Transport
from src.comm.topology import Topology, TopologyConfig
from src.utils.timer import Timer
from src.utils.logging import log, log_summary
from src.utils.seed import set_seed
from src.comm.timeline import export_chrome_trace


def worker(
    rank: int,
    world_size: int,
    config: Dict,
    trace_dir: str = "outputs/traces",
) -> None:
    """Entry point for a single spawned process."""
    # ── Init ────────────────────────────────────────────────────
    set_seed(42 + rank)
    init_process_group(
        rank=rank,
        world_size=world_size,
        master_addr=config.get("master_addr", "127.0.0.1"),
        master_port=config.get("master_port", 29500),
        backend=config.get("backend", "gloo"),
    )

    timer = Timer(rank)

    # ── Build model ─────────────────────────────────────────────
    model_cfg = config["model"]
    delay_cfg = config.get("delay", {})
    runtime_cfg = config["runtime"]
    data_cfg = config["data"]

    # -- Build topology (if enabled) ---------------------------------
    topo_cfg = config.get("topology", {})
    topology = None
    if topo_cfg.get("enabled", False):
        topology = Topology(TopologyConfig(
            num_pods=topo_cfg.get("num_pods", 1),
            nodes_per_pod=topo_cfg.get("nodes_per_pod", 1),
            ranks_per_node=topo_cfg.get("ranks_per_node", world_size),
            intra_node_latency_us=topo_cfg.get("intra_node_latency_us", 1.0),
            intra_pod_latency_us=topo_cfg.get("intra_pod_latency_us", 3.0),
            cross_pod_latency_us=topo_cfg.get("cross_pod_latency_us", 10.0),
            intra_node_bandwidth_gbps=topo_cfg.get("intra_node_bandwidth_gbps", 900.0),
            intra_pod_bandwidth_gbps=topo_cfg.get("intra_pod_bandwidth_gbps", 200.0),
            cross_pod_bandwidth_gbps=topo_cfg.get("cross_pod_bandwidth_gbps", 100.0),
            delay_multiplier=topo_cfg.get("delay_multiplier", 1.0),
        ))
        # Pre-assign ALL ranks (each spawned process has its own copy of topology)
        for r in range(world_size):
            topology.assign(r)
        loc = topology.get_location(rank)
        log(rank, f"Topology: pod={loc.pod_id} node={loc.node_id} local={loc.local_rank}")

    transport = Transport(
        timer=timer,
        comm_delay_us=delay_cfg.get("comm_delay_us", 0.0),
        comm_delay_jitter_us=delay_cfg.get("comm_delay_jitter_us", 0.0),
        topology=topology,
        rank=rank,
        world_size=world_size,
    )

    experts_per_rank = model_cfg.get("experts_per_rank", 1)

    moe = MoELayer(
        hidden_dim=model_cfg["hidden_dim"],
        num_experts=model_cfg["num_experts"],
        top_k=model_cfg.get("top_k", 1),
        expert_type=model_cfg.get("expert_type", "tiny"),
        expert_mult=model_cfg.get("expert_hidden_mult", 4),
        routing_strategy=config.get("routing", {}).get("strategy", "fixed"),
        experts_per_rank=experts_per_rank,
    )
    moe.set_rank(rank, world_size)

    # ── Synthetic data ──────────────────────────────────────────
    batch_size = data_cfg["batch_size"]
    seq_len = data_cfg["seq_len"]
    hidden_dim = model_cfg["hidden_dim"]
    num_microbatches = data_cfg["num_microbatches"]

    tokens_per_mb = batch_size * seq_len
    full_batch = torch.randn(tokens_per_mb, hidden_dim)

    # Split into micro-batches
    microbatches = torch.chunk(full_batch, num_microbatches, dim=0)

    log(rank, f"Worker ready — {num_microbatches} microbatches × {tokens_per_mb // num_microbatches} tokens each")

    # ── Run ─────────────────────────────────────────────────────
    mode = runtime_cfg.get("mode", "serial")
    num_steps = runtime_cfg.get("num_steps", 5)

    for step in range(num_steps):
        if mode == "serial":
            run_serial(
                step=step,
                microbatches=microbatches,
                moe=moe,
                transport=transport,
                timer=timer,
            )
        elif mode == "overlap":
            run_overlap(
                step=step,
                microbatches=microbatches,
                moe=moe,
                transport=transport,
                timer=timer,
            )
        else:
            raise ValueError(f"Unknown runtime mode: {mode}")

    # ── Barrier and summarize ───────────────────────────────────
    transport.barrier()

    summary = timer.summary()
    total_us = sum(summary.values())
    comm_us = summary.get("comm", 0.0)
    compute_us = summary.get("compute", 0.0)

    log_summary(rank, {
        "total_steps": num_steps,
        "num_events": len(timer.events),
        "total_us": total_us,
        "comm_us": comm_us,
        "compute_us": compute_us,
        "comm_pct": (comm_us / total_us * 100) if total_us > 0 else 0,
        "mode": mode,
    })

    # -- Export trace ----------------------------------------------------
    if config.get("profiling", {}).get("export_trace", True):
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, f"rank_{rank:02d}_trace.json")

        # Build EP metadata for the viewer
        ep_meta = {
            "world_size": world_size,
            "num_experts": model_cfg["num_experts"],
            "experts_per_rank": experts_per_rank,
            "top_k": model_cfg.get("top_k", 1),
            "routing_strategy": config.get("routing", {}).get("strategy", "fixed"),
            "mode": mode,
            "backend": config.get("backend", "gloo"),
        }
        # Add topology info if available
        if topology is not None:
            topo_cfg = config.get("topology", {})
            ep_meta["topology"] = {
                "num_pods": topo_cfg.get("num_pods", 1),
                "nodes_per_pod": topo_cfg.get("nodes_per_pod", 1),
                "ranks_per_node": topo_cfg.get("ranks_per_node", world_size),
            }
            # Per-rank location for this rank
            loc = topology.get_location(rank)
            ep_meta["rank_location"] = {
                "pod_id": loc.pod_id,
                "node_id": loc.node_id,
                "local_rank": loc.local_rank,
            }

        export_chrome_trace(timer.events, trace_path, pid=rank, tid=0, metadata=ep_meta)
        log(rank, f"Trace exported -> {trace_path}")

    cleanup_process_group()
