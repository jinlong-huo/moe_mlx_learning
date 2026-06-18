"""
interventions.py — Router steering for MoE research.

Provides ``RouterSteering``, a context manager that temporarily patches
gate forward methods to apply forced routing, logit biasing, or expert
ablation — without modifying model weights.

Usage::

    from moe_framework.model_utils import ModelLayout
    from moe_framework.interventions import RouterSteering

    layout = ModelLayout.from_model(model, model.config)
    steering = RouterSteering(model, layout)

    steering.force_expert(layer=12, expert_id=5)
    steering.bias_expert(layer=15, expert_id=3, bias=2.5)
    steering.ablate_expert(layer=18, expert_id=7)

    with steering:
        outputs = model(input_ids, ..., output_router_logits=True)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .model_utils import ModelLayout, _find_decoder_layers


def _make_steered_forward(
    orig_forward,
    gate_module: nn.Module,
    biases: dict[int, float],
    force: dict[int, bool],
    ablate: set[int],
):
    """Return a replacement ``forward`` that applies interventions to logits.

    Intercepts the gate's computation at the logit level (before softmax +
    top-k), so biases / force / ablation affect which experts are selected.
    """

    weight = gate_module.weight
    hidden_dim = gate_module.hidden_dim
    top_k = gate_module.top_k
    norm_topk_prob = getattr(gate_module, "norm_topk_prob", False)

    def steered_forward(hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        router_logits = F.linear(hidden_states, weight)

        # 1. Bias — additive offset before softmax
        for eid, bias_val in biases.items():
            router_logits[:, eid] = router_logits[:, eid] + bias_val

        # 2. Ablate — prevent selection
        for eid in ablate:
            router_logits[:, eid] = float("-inf")

        # 3. Force — ensure expert is always in top-k
        if force:
            router_logits = router_logits.float()
            for eid, exclusive in force.items():
                if exclusive:
                    router_logits[:, :] = float("-inf")
                router_logits[:, eid] = 100.0
            router_logits = router_logits.to(weight.dtype)

        # Standard routing (identical to Qwen2MoeTopKRouter.forward)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(
            router_probs, top_k, dim=-1
        )
        if norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(
                dim=-1, keepdim=True
            )
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices

    return steered_forward


class RouterSteering:
    """Context manager for MoE routing interventions.

    Temporarily patches ``gate.forward`` on specified MoE layers.
    Original methods are restored on context exit.

    Parameters
    ----------
    model : nn.Module
        Loaded HF causal LM model.
    layout : ModelLayout
        Auto-detected model architecture metadata.

    Examples
    --------

    Force expert 5 in layer 12 (alongside normal top-3)::

        steering = RouterSteering(model, layout)
        steering.force_expert(layer=12, expert_id=5)
        with steering:
            outputs = model(input_ids, ..., output_router_logits=True)

    Force *only* expert 5 (suppress all others)::

        steering.force_expert(layer=12, expert_id=5, exclusive=True)

    Bias toward expert 3::

        steering.bias_expert(layer=15, expert_id=3, bias=3.0)

    Ablate (prevent) expert 7::

        steering.ablate_expert(layer=18, expert_id=7)
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        layout: Optional[ModelLayout] = None,
    ):
        self.model = model
        self.layout = layout
        self._interventions: dict[int, dict] = {}  # layer_idx → {biases, force, ablate}
        self._originals: dict[int, object] = {}    # layer_idx → original gate.forward

    # ── configuration (call before __enter__) ───────────────────────

    def force_expert(
        self, layer: int, expert_id: int, exclusive: bool = False
    ) -> None:
        """Force *expert_id* to always be selected in *layer*.

        Parameters
        ----------
        layer : int
            Absolute decoder-layer index (0-indexed).
        expert_id : int
            Expert to force.
        exclusive : bool
            If True, suppress all other experts so *only* this one fires.
        """
        cfg = self._interventions.setdefault(layer, {})
        cfg.setdefault("force", {})[expert_id] = exclusive

    def bias_expert(self, layer: int, expert_id: int, bias: float) -> None:
        """Add *bias* to the router logit of *expert_id* in *layer*.

        Positive bias makes the expert more likely to be selected;
        negative bias makes it less likely.
        """
        cfg = self._interventions.setdefault(layer, {})
        cfg.setdefault("biases", {})[expert_id] = (
            cfg.setdefault("biases", {}).get(expert_id, 0.0) + bias
        )

    def ablate_expert(self, layer: int, expert_id: int) -> None:
        """Prevent *expert_id* from being selected in *layer*."""
        cfg = self._interventions.setdefault(layer, {})
        cfg.setdefault("ablate", set()).add(expert_id)

    # ── context manager ─────────────────────────────────────────────

    def __enter__(self) -> "RouterSteering":
        decoder_layers = _find_decoder_layers(self.model)

        for layer_idx, config in self._interventions.items():
            if layer_idx < 0 or layer_idx >= len(decoder_layers):
                print(
                    f"[steering] WARNING: layer {layer_idx} out of range "
                    f"(0-{len(decoder_layers) - 1}), skipping"
                )
                continue

            layer = decoder_layers[layer_idx]
            mlp = getattr(layer, "mlp", None)
            if mlp is None or not hasattr(mlp, "gate"):
                print(
                    f"[steering] WARNING: layer {layer_idx} has no MoE gate, skipping"
                )
                continue

            gate = mlp.gate
            self._originals[layer_idx] = gate.forward

            steered = _make_steered_forward(
                orig_forward=gate.forward,
                gate_module=gate,
                biases=config.get("biases", {}),
                force=config.get("force", {}),
                ablate=config.get("ablate", set()),
            )
            gate.forward = steered

        return self

    def __exit__(self, *args) -> bool:
        decoder_layers = _find_decoder_layers(self.model)
        for layer_idx, orig_forward in self._originals.items():
            if layer_idx < len(decoder_layers):
                decoder_layers[layer_idx].mlp.gate.forward = orig_forward
        self._originals.clear()
        return False

    @property
    def active_layers(self) -> list[int]:
        """Layers that have at least one intervention configured."""
        return sorted(self._interventions.keys())
