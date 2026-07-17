"""BitNet b1.58: ternary-weight Linear layer with int8 activation.

Why
---
A vanilla 1.13B-param Linear stack costs:

    weights        : 1.13B * 2 B (fp16)   = 2.26 GB
    AdamW state    : 2 * weights          = 4.52 GB
    activations    : ~1-2 GB              = ~1.5 GB
    -----------------------------------------------------
    total           : ~8 GB minimum -- breaks 6 GB GeForce 1660.

BitNet b1.58 (Microsoft Research, 2024) replaces every Linear with a
*ternary* layer: weights ∈ {-1, 0, +1} packed at 1.58 bits per weight
(log2(3)). The forward becomes integer addition / subtraction; no
multiplications. Memory drops by ~10x; FLOPs drop by ~5x with
custom kernels (we use a generic torch path here -- a CUDA kernel
is the next-day work).

Training works because we keep a fp16 *latent* copy of the weights
that the optimizer updates; the forward quantizes on every pass; the
backward uses a straight-through estimator (gradient flows through
the quantizer as if it were identity) so the optimizer sees usable
gradients.

Layout
------
    BitLinear:                                 (drop-in nn.Linear replacement)
       fp16 latent W           (m, n)           ← optimizer updates this
       quantize-on-forward → ternary W         {-1, 0, +1}
       absmax-quant activation → int8 a
       y = a @ W * (β_a * β_w)                 ← scales restore magnitude

Failure modes (honest)
----------------------
* The straight-through estimator is biased; convergence is slower
  than fp16 by ~1.5-2x in steps, recovered by the BitNet paper's
  larger learning rate (typically 2-3x baseline).
* For very small models (<100M) the quantization noise dominates
  and it doesn't converge. 1.13B is comfortably in the "works" zone
  per the BitNet paper's ablation.
* Numerical instability without RMSNorm + careful warmup. We use
  RMSNorm everywhere (already in `ai/model/normalization.py`) and
  the trainer's existing CosineSchedulerWithWarmup.

References
----------
* "BitNet: Scaling 1-bit Transformers for LLMs" (Microsoft, 2023)
* "The Era of 1-bit LLMs: All LLMs are in 1.58 Bits" (Ma et al., 2024)
"""

from __future__ import annotations

import math
from typing import Optional

# Lazy torch import so this module loads even on the CPU-only Pluginfer
# dev box. Tests gate on torch availability.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
    _BASE_MODULE = nn.Module
except Exception:                                                # pragma: no cover
    torch = None
    nn = None
    F = None
    _HAS_TORCH = False
    _BASE_MODULE = object


# ---------------------------------------------------------------------------
# Quantization primitives (all operate on torch tensors when torch is up)
# ---------------------------------------------------------------------------


def quant_weight_158(w):
    """Ternary {-1, 0, +1} quantization with per-tensor absmean scale.

    Returns (w_quant, scale). The original `w` is recoverable as
    `w_quant * scale` only approximately; the gradient through this
    operation uses the straight-through estimator (`weight_ste` below).
    """
    # Per-tensor absmean -- NOT the L1 mean. The paper uses absmean
    # because for symmetric ternary weights, the optimal scale is
    # E[|w|]: minimises L2 quantization error.
    scale = w.abs().mean().clamp_min_(1e-5)
    # Round to nearest of {-1, 0, +1} after rescaling.
    w_norm = w / scale
    w_q = w_norm.round().clamp_(-1, 1)
    return w_q, scale


def quant_activation_int8(a):
    """Per-token absmax quantization to int8 ({-127..127} range).

    Activation rows are scaled independently so a layer's per-token
    spike doesn't crush other tokens. Returns (a_q, scale_per_token).
    """
    # `a` shape: (..., features). Compute scale along the last dim.
    scale = a.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-5) / 127.0
    a_q = (a / scale).round().clamp_(-127, 127)
    return a_q, scale


# ---------------------------------------------------------------------------
# Straight-through estimator wrappers
# ---------------------------------------------------------------------------


if _HAS_TORCH:
    class _STEWeight(torch.autograd.Function):
        """Forward: quantize weight to ternary. Backward: identity
        gradient (straight-through estimator)."""
        @staticmethod
        def forward(ctx, w):
            w_q, scale = quant_weight_158(w)
            return w_q * scale

        @staticmethod
        def backward(ctx, grad):
            return grad


    class _STEActivation(torch.autograd.Function):
        """Forward: int8-quantize activation. Backward: identity."""
        @staticmethod
        def forward(ctx, a):
            a_q, scale = quant_activation_int8(a)
            return a_q * scale

        @staticmethod
        def backward(ctx, grad):
            return grad


    def weight_ste(w):
        return _STEWeight.apply(w)


    def activation_ste(a):
        return _STEActivation.apply(a)


# ---------------------------------------------------------------------------
# BitLinear: drop-in replacement for nn.Linear
# ---------------------------------------------------------------------------


class BitLinear(_BASE_MODULE):
    """Drop-in replacement for nn.Linear with ternary-quantized weights
    and int8-quantized activations. The fp16 latent weight is what the
    optimizer updates; the forward quantizes on every pass."""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, *, normalize_input: bool = True):
        if not _HAS_TORCH:
            raise RuntimeError("BitLinear requires torch")
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.normalize_input = normalize_input

        # Latent fp16 weight that the optimizer updates. Kept fp16 in
        # the on-disk checkpoint; promoted to fp32 inside the optimizer
        # for stable Adam moment accumulation.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float32),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # RMSNorm on the input is mandatory in the BitNet recipe -- the
        # int8 activation quantization is per-token absmax, and without
        # input normalization the absmax is dominated by occasional
        # spikes which crushes the resolution of the rest.
        self.input_rms = nn.Parameter(torch.ones(in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Same init as nn.Linear; the BitNet paper reports no benefit
        # from special init.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _rms_norm(self, x):
        # Standard RMSNorm: x / sqrt(mean(x^2) + eps) * gamma
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + 1e-6)
        return x * self.input_rms

    def forward(self, x):
        if self.normalize_input:
            x = self._rms_norm(x)
        # Quantize activation to int8 with STE
        x_q = activation_ste(x)
        # Quantize weight to ternary with STE
        w_q = weight_ste(self.weight)
        out = F.linear(x_q, w_q, self.bias)
        return out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bits=1.58, activation=int8")


# ---------------------------------------------------------------------------
# Conversion utility: swap every nn.Linear for BitLinear in a model
# ---------------------------------------------------------------------------


def convert_to_bitnet(module, *, exclude_names: tuple = ("lm_head",)):
    """Walk the module tree and replace every nn.Linear with BitLinear.

    `exclude_names` -- substrings of param names to leave as fp16 nn.Linear.
    The output projection (lm_head) is conventionally left full-precision
    because the final softmax is sensitive to quantization noise.

    Returns the (modified-in-place) module + a dict of statistics:
      {converted: N, skipped: M, params_quantized: P, params_total: T}
    """
    if not _HAS_TORCH:
        raise RuntimeError("convert_to_bitnet requires torch")

    converted = 0
    skipped = 0
    params_quantized = 0
    params_total = sum(p.numel() for p in module.parameters())

    def _convert(parent, name, child):
        nonlocal converted, skipped, params_quantized
        if isinstance(child, nn.Linear):
            full_name = name
            if any(ex in full_name for ex in exclude_names):
                skipped += 1
                return
            new = BitLinear(child.in_features, child.out_features,
                             bias=child.bias is not None)
            # Copy the fp32 weight as the latent.
            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None and new.bias is not None:
                    new.bias.copy_(child.bias)
            setattr(parent, name.split(".")[-1], new)
            converted += 1
            params_quantized += child.weight.numel()

    # Walk via named_modules so we have the fully-qualified path for
    # exclude_names matching.
    targets = []
    for name, child in module.named_modules():
        for sub_name, sub in child.named_children():
            full = f"{name}.{sub_name}" if name else sub_name
            if isinstance(sub, nn.Linear):
                targets.append((child, full, sub))

    for parent, full_name, child in targets:
        _convert(parent, full_name, child)

    return module, {
        "converted": converted,
        "skipped": skipped,
        "params_quantized": params_quantized,
        "params_total": params_total,
        "compression_ratio": (
            (params_total * 32) / max(1, (params_quantized * 1.58
                                          + (params_total - params_quantized) * 32))
        ),
    }


def memory_estimate(num_params: int, *,
                    bitnet: bool = True,
                    optimizer: str = "adamw_8bit",
                    activations_gb: float = 1.5) -> dict:
    """Estimate VRAM requirement for training the model. Returns a
    dict of components in GB. Use this to pick batch size."""
    if bitnet:
        weight_bits = 1.58 + 16    # ternary inference + fp16 latent
    else:
        weight_bits = 16
    weights_gb = num_params * weight_bits / 8 / 1e9

    if optimizer == "adamw_8bit":
        opt_bits = 16              # 8-bit Adam states (m + v)
    else:
        opt_bits = 64              # fp32 Adam states (m + v)
    opt_gb = num_params * opt_bits / 8 / 1e9

    grad_gb = num_params * 16 / 8 / 1e9    # fp16 gradients

    return {
        "weights_gb": round(weights_gb, 2),
        "optimizer_state_gb": round(opt_gb, 2),
        "gradient_buffer_gb": round(grad_gb, 2),
        "activations_gb": activations_gb,
        "total_gb": round(weights_gb + opt_gb + grad_gb + activations_gb, 2),
    }
