# MoE Routing Research Toolkit

Capture and analyze token-to-expert routing decisions in Qwen MoE models, with two backends sharing a canonical trace format.

## Quick Start

```bash
# HF backend (PyTorch — CPU / MPS / CUDA)
python run_research.py run --model Qwen/Qwen1.5-MoE-A2.7B-Chat --max-tokens 64

# MLX backend (Apple Silicon)
python moe_run.py --model ./models/Qwen1.5-MoE-A2.7B-Chat-4bit --max-tokens 64
```

Both produce `logs/routing.json` — identical schema, interchangeable across backends.

## Commands

| Command | Description |
|---|---|
| `python run_research.py run` | Generate text + capture routing (HF) |
| `python run_research.py analyze logs/routing.json` | Load balance, entropy, specialization, coactivation |
| `python run_research.py compare trace_a.json trace_b.json` | Side-by-side expert load comparison |
| `python run_research.py intervene --force-expert 12 5` | Force expert selection during generation |
| `python run_research.py ablate --ablate-expert 12 5` | Prevent expert selection |
| `python moe_run.py` | Generate text + capture routing (MLX, Apple Silicon) |

## Output Format

All backends produce identical JSON:

```json
{
  "meta": {
    "model_id": "Qwen1.5-MoE-A2.7B-Chat-4bit",
    "model_type": "qwen2_moe",
    "num_layers": 24,
    "num_moe_layers": 24,
    "num_experts": 60,
    "top_k": 4,
    "prompt_len": 42,
    "generated_len": 64,
    "total_tokens": 106,
    "backend": "mlx",
    "run_id": "a1b2c3d4e5f6"
  },
  "prompt_tokens": [...],
  "generated_tokens": [...],
  "routes": [
    {
      "token_pos": 0,
      "token_id": 151644,
      "token_str": "<|im_start|>",
      "phase": "prefill",
      "layers": {
        "0": {"experts": [12, 45, 3, 28], "weights": [0.31, 0.28, 0.22, 0.19]},
        "1": {"experts": [7, 33, 50, 19], "weights": [0.29, 0.26, 0.24, 0.21]}
      }
    }
  ]
}
```

## Project Structure

```
moe_framework/          # Shared HF toolkit
  schema.py             #   Canonical data types (RoutingTrace, TokenRoute, etc.)
  model_utils.py        #   Model introspection, MPS patches, router-logits enable
  capture.py            #   Non-invasive routing capture (output_router_logits)
  interventions.py      #   Router steering / biasing / ablation
run_research.py         # HF CLI (run, analyze, compare, intervene, ablate)
moe_run.py              # MLX CLI (Apple Silicon)
hook.py                 # MLX routing capture → RoutingTrace bridge
```

## Key Design

- **Token positions are absolute** (0-indexed full sequence), not per-step — cross-layer analysis works trivially
- **HF backend uses `output_router_logits=True`** — no monkey-patching for capture; only interventions patch the model
- **MLX backend monkey-patches `__call__`** on the MoE block class to intercept gate decisions
- **`RoutingTrace` is the single source of truth** — both backends produce it, all analysis tools consume it
