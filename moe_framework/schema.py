"""
schema.py — Unified routing data structures for MoE research.

Single canonical JSON format shared across all backends (HF, MLX, vLLM).
Indexed by absolute token position (not forward-pass step) so that
cross-token and cross-layer analysis is trivial.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class RunMeta:
    """Metadata recorded once per inference run."""

    model_id: str
    model_type: str  # "qwen2_moe" | "qwen3_5_moe" | ...
    num_layers: int
    num_moe_layers: int
    num_experts: int
    top_k: int
    prompt_len: int
    generated_len: int
    total_tokens: int
    backend: str  # "hf" | "mlx" | "vllm"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class LayerRoute:
    """Routing decision for a single layer at a single token position."""

    experts: list[int]
    weights: list[float]


@dataclass
class TokenRoute:
    """Routing decisions for one token position across all MoE layers."""

    token_pos: int  # absolute position in full sequence (0-indexed)
    token_id: int
    token_str: str
    phase: str  # "prefill" | "decode"
    layers: dict[str, LayerRoute]  # "0" → LayerRoute, "5" → LayerRoute, ...


@dataclass
class RoutingTrace:
    """Complete routing trace for a single inference run."""

    meta: RunMeta
    prompt_tokens: list[int]
    generated_tokens: list[int]
    routes: list[TokenRoute]

    # ── serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "meta": asdict(self.meta),
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "routes": [
                {
                    "token_pos": r.token_pos,
                    "token_id": r.token_id,
                    "token_str": r.token_str,
                    "phase": r.phase,
                    "layers": {
                        lid: asdict(lr) for lid, lr in r.layers.items()
                    },
                }
                for r in self.routes
            ],
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "RoutingTrace":
        with open(path) as f:
            raw = json.load(f)
        meta_raw = raw["meta"]
        meta = RunMeta(
            model_id=meta_raw["model_id"],
            model_type=meta_raw["model_type"],
            num_layers=meta_raw["num_layers"],
            num_moe_layers=meta_raw["num_moe_layers"],
            num_experts=meta_raw["num_experts"],
            top_k=meta_raw["top_k"],
            prompt_len=meta_raw["prompt_len"],
            generated_len=meta_raw["generated_len"],
            total_tokens=meta_raw["total_tokens"],
            backend=meta_raw["backend"],
            run_id=meta_raw.get("run_id", ""),
        )
        routes = [
            TokenRoute(
                token_pos=r["token_pos"],
                token_id=r["token_id"],
                token_str=r["token_str"],
                phase=r["phase"],
                layers={
                    lid: LayerRoute(**lr) for lid, lr in r["layers"].items()
                },
            )
            for r in raw["routes"]
        ]
        return cls(
            meta=meta,
            prompt_tokens=raw["prompt_tokens"],
            generated_tokens=raw["generated_tokens"],
            routes=routes,
        )

    # ── convenience ────────────────────────────────────────────────

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_tokens + self.generated_tokens

    def total_routing_events(self) -> int:
        """Total (token, layer) routing records."""
        return sum(len(r.layers) for r in self.routes)

    def expert_load(self) -> dict[int, int]:
        """Per-expert token count across all layers."""
        counts: dict[int, int] = {}
        for route in self.routes:
            for lr in route.layers.values():
                for e in lr.experts:
                    counts[e] = counts.get(e, 0) + 1
        return dict(sorted(counts.items()))

    def per_layer_expert_load(self) -> dict[str, dict[int, int]]:
        """Per-expert token count, grouped by layer."""
        result: dict[str, dict[int, int]] = {}
        for route in self.routes:
            for lid, lr in route.layers.items():
                if lid not in result:
                    result[lid] = {}
                for e in lr.experts:
                    result[lid][e] = result[lid].get(e, 0) + 1
        return result
