"""Post-training INT8 quantization (absmax, from scratch).

Reference: per-tensor absmax quantization. For each weight matrix W:
    scale = max(|W|) / 127
    W_int8 = round(W / scale).clamp(-127, 127).to(int8)
    W_approx = W_int8.float() * scale

Replaces every `nn.Linear` layer's weight with a quantized version. The
forward pass dequantises on the fly:
    out = F.linear(x, W_int8.float() * scale, bias)

This roughly halves the disk and memory footprint of the model without
touching activations. Accuracy impact is typically < 0.5 perplexity
points on well-trained models.

We do NOT quantise:
  - Embedding tables (sparse access -> int8 lookup is awkward and the
    embedding is on the residual stream's input/output, sensitive to
    rounding noise)
  - Norm parameters (small, no benefit, sensitive)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    """Replacement for nn.Linear. Stores weight as int8 + a single fp32 scale.

    The scale is per-tensor (not per-row). Per-row would give better
    accuracy at minor extra memory cost; pencil it in for CP-AI-5 follow-ups.
    """

    __constants__ = ("in_features", "out_features")

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        scale = float(weight.detach().abs().max().item() / 127.0)
        # Guard against all-zero matrices (would divide by zero)
        if scale == 0.0:
            scale = 1.0 / 127.0
        q = (weight.detach() / scale).round().clamp(-127, 127).to(torch.int8)
        # Buffers (not parameters): we don't continue training quantised tensors here.
        self.register_buffer("weight_q", q)
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantise on the fly. For CPU this is roughly the same cost as
        # an fp32 matmul (the int8->fp32 cast is cheap relative to the
        # matmul); the win is in disk + memory size. A real fast path
        # would use int8 GEMM kernels (CUDA cutlass / triton) - tracked
        # under CP-AI-5 follow-ups.
        w = self.weight_q.to(torch.float32) * self.scale
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int8=True"


class INT8Quantizer:
    """Convenience wrapper exposing the algorithmic primitives.

    The actual model rewrite is done by `quantize_module_in_place`.
    """

    def quantize_weight(
        self, w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = w.abs().max() / 127.0
        if float(scale) == 0.0:
            scale = torch.tensor(1.0 / 127.0)
        q = (w / scale).round().clamp(-127, 127).to(torch.int8)
        return q, scale.to(torch.float32) if scale.dim() == 0 else scale.float()

    def dequantize(
        self, w_int8: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return w_int8.to(torch.float32) * scale


def quantize_module_in_place(module: nn.Module) -> nn.Module:
    """Replace every nn.Linear inside `module` (recursively) with a
    QuantizedLinear. Embeddings + RMSNorm are left untouched."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            replacement = QuantizedLinear(child.weight.data, child.bias.data if child.bias is not None else None)
            setattr(module, name, replacement)
        else:
            quantize_module_in_place(child)
    return module


def measure_param_bytes(module: nn.Module) -> int:
    """Estimate live memory: parameters + non-persistent buffers (int8 weights live there)."""
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total
