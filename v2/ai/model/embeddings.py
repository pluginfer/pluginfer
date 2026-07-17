"""Token embeddings + Rotary Positional Embeddings (RoPE).

TokenEmbedding is a thin wrapper around `F.embedding` that exposes the
weight matrix at `.weight` so the LM head can tie to it.

RotaryPositionalEmbedding implements the half-split convention used by
Llama / Mistral / Falcon (NOT the GPT-NeoX even-odd interleaved
convention). For a query/key tensor of shape (B, T, H, D) we:
  1. Split the last dimension in half: x1 = x[..., :D/2], x2 = x[..., D/2:]
  2. For each position p compute cos(p * theta_i), sin(p * theta_i)
     where theta_i = 1 / (rope_theta ** (2i / D)) for i in [0, D/2).
  3. Apply the rotation:
        out[..., :D/2] = x1 * cos - x2 * sin
        out[..., D/2:] = x1 * sin + x2 * cos
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig


class TokenEmbedding(nn.Module):
    """Standard learned embedding. `forward(ids) -> embeddings`."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # Llama-style normal init with std=0.02 (also the GPT-2 default).
        self.weight = nn.Parameter(
            torch.randn(config.vocab_size, config.d_model) * config.init_std
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.embedding(x, self.weight)


class RotaryPositionalEmbedding(nn.Module):
    """Half-split RoPE applied to query and key tensors before attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.max_seq_len = config.context_length

        # theta_i = 1 / (rope_theta ** (2i / head_dim)) for i in [0, head_dim/2)
        # Stored as a non-persistent buffer (recomputable from config; not
        # checkpointed).
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin tables for max_seq_len. Lazy build so we can
        # extend on the fly when callers pass longer contexts at inference.
        self._cached_seq_len: int = 0
        self.register_buffer("_cos", torch.empty(0), persistent=False)
        self.register_buffer("_sin", torch.empty(0), persistent=False)
        self._extend_cache(self.max_seq_len)

    def _extend_cache(self, seq_len: int) -> None:
        if seq_len <= self._cached_seq_len:
            return
        device = self.inv_freq.device
        t = (
            torch.arange(seq_len, device=device, dtype=torch.float32)
            / float(self.rope_scaling)
        )
        # freqs: (seq_len, head_dim/2)
        freqs = torch.outer(t, self.inv_freq)
        self._cos = freqs.cos()
        self._sin = freqs.sin()
        self._cached_seq_len = seq_len

    @staticmethod
    def _apply_rotation(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply RoPE to a tensor of shape (..., T, H, D).

        cos / sin: (T, D/2) - broadcast across the leading and head axes.
        """
        # Split the last dim into halves
        x1, x2 = x.chunk(2, dim=-1)  # each (..., T, H, D/2)
        # Reshape cos/sin so they broadcast over batch and head: (1, T, 1, D/2)
        cos_b = cos[None, :, None, :]
        sin_b = sin[None, :, None, :]
        rotated_x1 = x1 * cos_b - x2 * sin_b
        rotated_x2 = x1 * sin_b + x2 * cos_b
        return torch.cat([rotated_x1, rotated_x2], dim=-1)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        seq_len: int,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Rotate q and k. q: (B, T, n_heads, D); k: (B, T, n_kv_heads, D).

        `offset` is the position of the first token in q/k - used during
        autoregressive decoding to keep RoPE phase coherent across calls.
        """
        end = offset + seq_len
        if end > self._cached_seq_len:
            self._extend_cache(end)
        cos = self._cos[offset:end].to(q.dtype)
        sin = self._sin[offset:end].to(q.dtype)
        q_rot = self._apply_rotation(q, cos, sin)
        k_rot = self._apply_rotation(k, cos, sin)
        return q_rot, k_rot
