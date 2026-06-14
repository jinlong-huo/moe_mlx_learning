# MoE Communication Research Testbed

CPU-based MoE overlap-algorithm testbed. Real all-to-all over TCP via PyTorch Gloo with per-rank expert-parallelism, top-K gating, and hierarchical topology delay modeling. Verifies mechanism correctness *before* GPU cluster deployment.

## Quick Start

```bash
pip install torch pyyaml

# Small verification (4 ranks, 8 experts)
python3 -m src.launcher --config configs/synthetic_moe.yaml

# Realistic cluster (16 ranks, 64 experts, FFN, top-2, multi-pod topology)
python3 -m src.launcher --config configs/realistic_16gpu.yaml

# View results
open outputs/traces/trace_viewer.html    # interactive HTML (click "EP Layout")
```

## Architecture

```
tokens [B*S, H] --> Router --> expert_ids [T, K] + gate_weights [T, K]
                        |
                        v
              scatter_tokens()  -- groups by (expert_id // experts_per_rank),
                                   packs [token | local_expert_id | orig_idx],
                                   all_to_all_single (async in overlap mode)
                        |
                        v
              compute_experts() -- dispatches to correct local expert
                                   via per-expert masking
                        |
                        v
              gather_tokens()   -- all_to_all_single back,
                                   unpacks orig_idx to reconstruct order
                        |
                        v
              combine_expert_outputs() -- top-K: softmax gate weights, weighted sum
```

### Expert Parallelism (EP)

```
expert_id --> target_rank = expert_id // experts_per_rank
              local_expert = expert_id %  experts_per_rank

Constraint: num_experts = world_size * experts_per_rank
```

**4-rank example** (8 experts, 2/rank):

| Rank | Experts |
|------|---------|
| 0 | 0, 1 |
| 1 | 2, 3 |
| 2 | 4, 5 |
| 3 | 6, 7 |

**16-GPU cluster** (64 experts, 4/rank, `realistic_16gpu.yaml`):

| Pod | Node | Ranks | Experts |
|-----|------|-------|---------|
| 0 | 0 | 0-3 | 0-15 |
| 0 | 1 | 4-7 | 16-31 |
| 1 | 0 | 8-11 | 32-47 |
| 1 | 1 | 12-15 | 48-63 |

### Network Topology (3-tier)

| Tier | Link | Latency | Bandwidth |
|------|------|---------|-----------|
| INTRA_NODE | NVLink/NVSwitch | ~2 µs | 600 GB/s |
| INTRA_POD | InfiniBand | ~5 µs | 200 GB/s |
| CROSS_POD | IB fabric | ~15 µs | 100 GB/s |

Delay = `latency + tensor_bytes / (bandwidth_gbps * 1000)`. All tiers configurable in YAML.

### Overlap Pipeline

```
mb_0: [scatter_0 fire] -------+
      (in flight)              |
mb_1: [scatter_1 fire]        | overlap zone
      [scatter_0 wait]        |
      [compute_0]             |
      [gather_0 fire] --------+
tail: [scatter_1 wait] [compute_1] [gather_1]
```

## Project Structure

```
src/
├── main.py, launcher.py         # CLI + multiprocessing spawner
├── model/
│   ├── moe_layer.py             # MoELayer: ModuleList[experts], set_rank(), expert_id_to_local()
│   ├── router.py                # fixed | top1 | top2 | uniform_random
│   └── experts.py               # TinyExpert (Linear) | FFNExpert (Linear->GELU->Linear)
├── comm/
│   ├── all_to_all.py            # scatter_tokens, gather_tokens, combine_expert_outputs, DispatchResult
│   ├── transport.py             # Wraps dist ops, injects topology/flat delay
│   ├── topology.py              # Topology, TopologyConfig, LinkTier (3-tier hierarchy)
│   ├── timeline.py              # Chrome Trace JSON export (with EP metadata)
│   └── buffers.py               # Double-buffer dataclass
├── runtime/
│   ├── worker.py                # Per-rank: init PG, build MoE+Transport+Topology, run scheduler
│   ├── scheduler.py             # run_serial, run_overlap (async pipeline)
│   └── process_group.py         # dist.init_process_group / cleanup
├── eval/
│   ├── metrics.py               # compute_overlap_ratio, step_metrics
│   ├── profiler.py              # Multi-rank trace aggregation
│   └── plots.py                 # Gantt charts (matplotlib optional)
└── utils/                       # timer (ns-precision), logging, seed

configs/
├── base.yaml                    # 4 ranks, 4 experts, 1 expert/rank
├── synthetic_moe.yaml           # 4 ranks, 8 experts, 2/rank, tiny, fast iteration
├── realistic_16gpu.yaml         # 16 ranks, 64 experts, 4/rank, FFN, top-2, 2p×2n×4r topology
├── realistic_32gpu.yaml         # 32 ranks, 128 experts, 4/rank, FFN, top-2, 2p×4n×4r topology
└── mac_cpu.yaml                 # Overlap mode + 200µs flat delay

scripts/
├── run_synthetic.sh             # One-command: serial or overlap mode
├── trace_viz.py                 # Standalone HTML viewer (EP panel, topology groups, overlap stat)
└── merge_traces.py              # Merge per-rank traces for chrome://tracing
```

## Configuration Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `world_size` | Number of ranks | 4 |
| `model.num_experts` | Total experts (= world_size × experts_per_rank) | 4 |
| `model.experts_per_rank` | Experts per GPU | 1 |
| `model.hidden_dim` | Token embedding dimension | 256 |
| `model.expert_type` | `tiny` (Linear) or `ffn` (2-layer GELU) | `tiny` |
| `model.top_k` | Top-K gating | 1 |
| `routing.strategy` | `fixed`, `top1`, `top2`, `uniform_random` | `fixed` |
| `runtime.mode` | `serial` or `overlap` | `serial` |
| `delay.comm_delay_us` | Flat delay (ignored if topology enabled) | 0 |
| `topology.enabled` | Use hierarchical topology delays | false |

## Key Design Decisions

- **Metadata packing:** `local_expert_id` and `original_index` are packed as float columns alongside tokens — travels through `all_to_all_single` with zero extra communication rounds.
- **Padded equal-size all-to-all:** Counts exchanged via `all_gather`, max computed, padded to uniform chunks. Required by `all_to_all_single` on Gloo.
- **Top-K dispatch:** Tokens replicated K times, `index_add_` handles duplicate indices during gather. Gate weights applied via softmax-weighted sum.
- **Async overlap:** Scatter/gather issued with `async_op=True`, `handle.wait()` called when result needed. Identical pattern to NCCL CUDA-stream overlap.
- **Topology per-process:** Each spawned process pre-assigns *all* ranks to populate the topology lookup table (required by `spawn` start method).
- **No GPU:** Runs entirely on CPU with real TCP communication. Overlap algorithm verified here transfers directly to GPU/NCCL.

## Viewing Traces

```bash
# Interactive HTML (recommended) — click "EP Layout" for expert mapping
open outputs/traces/trace_viewer.html

# Chrome Trace Viewer
open chrome://tracing  →  load outputs/traces/merged_trace.json

# Perfetto
open https://ui.perfetto.dev  →  drag in merged_trace.json
```

**Event colors:** Green=Route, Blue=Scatter, Orange=Compute, Purple=Gather, Pink=Combine, Cyan=ScatterWait, Red=AllToAll

## What This IS and IS NOT

**IS:** Mechanism verification testbed. Answers: does the async overlap algorithm work correctly? Shows EP mapping, topology structure, comm/compute interleaving. Fast iteration (seconds, not hours).

**IS NOT:** Performance simulator. Do NOT extrapolate timings to GPU clusters. The comm/compute ratio is inverted here (comm dominates because experts are tiny). On real GPUs, compute dwarfs communication → overlap ratio much higher.

Progression: verify here → profile single-GPU NCCL → scale multi-node.

## License

MIT
