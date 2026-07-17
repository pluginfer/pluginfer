"""RMSNorm - Root-Mean-Square layer normalization.

Reference: Zhang & Sennrich 2019 (arXiv:1910.07467).

Why RMSNorm and not LayerNorm:
  - Skips the mean-subtraction step (cheaper).
  - Empirically as stable as LayerNorm in transformer training, sometimes
    better. Used in Llama, Mistral, T5, Gemma.
  - One learnable parameter per feature dim (gamma).

Formula:
    rms(x) = sqrt(mean(x ** 2, dim=-1) + eps)
    out    = x / rms(x) * gamma

Compute is done in float32 internally, then cast back to the input dtype,
so bf16 / fp16 activations don't lose stability through the norm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        x32 = x32 * rms
        return (x32 * self.weight.float()).to(in_dtype)
