"""
model_utils.py — Architecture-agnostic model introspection + MPS compatibility.

Key responsibilities:
  - ``_find_decoder_layers`` — locate transformer layers regardless of nesting
  - ``ModelLayout`` — auto-detect model structure (experts, top_k, layer indices)
  - ``enable_router_logits`` — flip the config flag so forward() returns router_logits
  - ``apply_mps_patches`` — work around known MPS backend gaps
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# Model introspection
# ═══════════════════════════════════════════════════════════════════

def _find_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """
    Find the flat list of decoder layers regardless of architectural nesting.

    Tries, in order:
      1. model.model.language_model.layers     (Qwen3.5: Qwen3_5MoeTextModel)
      2. model.model.layers                     (Qwen1.5: Qwen2MoeModel)
      3. model.transformer.h                    (GPT-2 style)

    Uses structural probing (attribute names), not class-name checks,
    so it's resistant to class renames across transformers versions.
    """
    # Strip top-level wrapper (QwenXxxForCausalLM → QwenXxxModel)
    base = getattr(model, "model", model)

    # Qwen3.5 / Qwen3 pattern — nested language_model
    language_model = getattr(base, "language_model", None)
    if language_model is not None:
        layers = getattr(language_model, "layers", None)
        if layers is not None:
            return layers

    # Standard pattern: model.model.layers (Qwen1.5, LLaMA, Mistral, ...)
    layers = getattr(base, "layers", None)
    if layers is not None:
        return layers

    # GPT-2 / older HF pattern
    h = getattr(base, "h", None)
    if h is not None:
        return h

    raise ValueError(
        "Cannot locate decoder layers in model. "
        "Supported patterns: model.model.language_model.layers, "
        "model.model.layers, model.transformer.h"
    )


@dataclass
class ModelLayout:
    """Auto-detected structure of a MoE model.

    Hides architectural differences between model families so the rest
    of the framework can work with a uniform interface.
    """

    model_type: str
    hidden_size: int
    num_layers: int
    num_experts: int
    top_k: int
    moe_layer_indices: list[int]  # which decoder layers contain MoE blocks

    @classmethod
    def from_model(cls, model: nn.Module, config) -> "ModelLayout":
        """
        Auto-detect architecture and build a layout.

        Parameters
        ----------
        model : nn.Module
            Loaded HF model (AutoModelForCausalLM).
        config
            HF config object (model.config or model.config.text_config).
        """
        model_type = getattr(config, "model_type", "unknown")
        hidden_size = getattr(config, "hidden_size", -1)
        num_experts = getattr(config, "num_experts", 0)
        top_k = getattr(config, "num_experts_per_tok", 4)

        decoder_layers = _find_decoder_layers(model)
        num_layers = len(decoder_layers)

        # Detect which layers are MoE
        moe_indices: list[int] = []
        for idx, layer in enumerate(decoder_layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            if not hasattr(mlp, "gate"):
                continue
            # Heuristic: MoE blocks have a gate (router) AND either
            # .experts or .switch_mlp (the expert FFNs).
            if hasattr(mlp, "experts") or hasattr(mlp, "switch_mlp"):
                moe_indices.append(idx)

        # If top_k wasn't in config, try to read it from the first MoE block
        if moe_indices:
            first_moe = getattr(decoder_layers[moe_indices[0]], "mlp", None)
            block_top_k = getattr(first_moe, "top_k", None)
            if block_top_k is not None:
                top_k = block_top_k

        return cls(
            model_type=model_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            moe_layer_indices=moe_indices,
        )

    @property
    def num_moe_layers(self) -> int:
        return len(self.moe_layer_indices)

    def get_decoder_layers(self, model: nn.Module) -> list[nn.Module]:
        """Return the flat list of decoder layers."""
        return _find_decoder_layers(model)


# ═══════════════════════════════════════════════════════════════════
# Router logits
# ═══════════════════════════════════════════════════════════════════

def enable_router_logits(model: nn.Module) -> bool:
    """
    Set ``output_router_logits = True`` on the model config so that
    ``model.forward(..., output_router_logits=True)`` returns a
    ``router_logits`` tuple.

    For Qwen3.5 the text config may be nested under ``config.text_config``.
    This function tries both.
    """
    config = model.config
    patched = False
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is not None and hasattr(cfg, "output_router_logits"):
            cfg.output_router_logits = True
            patched = True
    return patched


# ═══════════════════════════════════════════════════════════════════
# MPS compatibility
# ═══════════════════════════════════════════════════════════════════

def apply_mps_patches() -> bool:
    """
    Apply known MPS-backend workarounds.

    Currently patches ``torch.histc`` to handle integer tensors on MPS
    (the MPS histogram kernel only supports float inputs, but HF's
    grouped-MoE expert dispatch calls it on int tensors).

    Returns True if any patches were applied.
    """
    if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return False

    _orig_histc = torch.histc

    def _mps_safe_histc(input, bins=100, min=0, max=0):
        if input.device.type == "mps" and not torch.is_floating_point(input):
            input = input.to(torch.float32)
        return _orig_histc(input, bins=bins, min=min, max=max)

    torch.histc = _mps_safe_histc  # type: ignore[assignment]
    return True
