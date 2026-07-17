"""Grouped-Query Attention (GQA) with RoPE.

Reference: Ainslie et al. 2023 (arXiv:2305.13245).

Key idea: have fewer key/value heads than query heads, with each KV
head shared across `n_heads / n_kv_heads` query heads. Reduces KV-cache
memory by that factor (16 -> 4 = 4x reduction in our default config),
which is the bottleneck for long-context inference. Quality on
language modelling is statistically indistinguishable from full
multi-head attention.

This module is responsible for:
  1. Q/K/V projections (separate, no biases)
  2. RoPE rotation of Q and K
  3. KV-cache append (when `cache` is provided)
  4. Repeating KV heads to match Q-head count (GQA expansion)
  5. Scaled dot-product attention with causal mask (or none, when
     `cache` is provided - the cache stores past K/V so a single
     decode step doesn't need a per-step causal mask)
  6. Output projection

The KV cache is per-call: the caller passes `cache=None` for full-sequence
training and `cache={layer_id: (k, v)}` (handled by the parent block)
for autoregressive decoding. Keeping cache state out of the module's
state dict means a single model can serve many simultaneous inference
streams without state contamination.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .embeddings import RotaryPositionalEmbedding


class KVCache:
    """Per-call KV cache. One instance per generation stream.

    Shape contract:
      k_cache[layer]: (B, T_so_far, n_kv_heads, head_dim)
      v_cache[layer]: (B, T_so_far, n_kv_heads, head_dim)

    The cache grows by `T_step` each call; we use list-of-tensors with
    `torch.cat` per layer per step, which is O(total_len) per step and
    fine for typical batch=1, seq < 4k inference. A pre-allocated tensor
    + write-pointer would be faster; tracked under CP-AI-5 optimisations.
    """

    def __init__(self, n_layers: int) -> None:
        self.k: list[Optional[Tensor]] = [None] * n_layers
        self.v: list[Optional[Tensor]] = [None] * n_layers

    def get_pos(self, layer: int) -> int:
        if self.k[layer] is None:
            return 0
        return int(self.k[layer].shape[1])

    def update(self, layer: int, new_k: Tensor, new_v: Tensor) -> tuple[Tensor, Tensor]:
        if self.k[layer] is None:
            self.k[layer] = new_k
            self.v[layer] = new_v
        else:
            self.k[layer] = torch.cat([self.k[layer], new_k], dim=1)
            self.v[layer] = torch.cat([self.v[layer], new_v], dim=1)
        return self.k[layer], self.v[layer]


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(
            config.d_model, config.n_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.d_model, config.n_kv_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.d_model, config.n_kv_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.n_heads * config.head_dim, config.d_model, bias=False
        )

        self.rope = RotaryPositionalEmbedding(config)

    @staticmethod
    def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
        """Expand (B, T, n_kv, D) -> (B, T, n_kv * n_rep, D)."""
        if n_rep == 1:
            return x
        # repeat_interleave puts each kv-head's copies adjacent, which
        # matches how query heads are grouped: heads [0..n_rep-1] share
        # KV head 0, heads [n_rep..2*n_rep-1] share KV head 1, etc.
        return x.repeat_interleave(n_rep, dim=2)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        cache: Optional[KVCache] = None,
    ) -> Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # RoPE - applied after projection but before attention
        offset = cache.get_pos(self.layer_id) if cache is not None else 0
        q, k = self.rope(q, k, seq_len=T, offset=offset)

        # KV-cache update: append the new k/v, fetch the full history
        if cache is not None:
            k, v = cache.update(self.layer_id, k, v)

        # GQA: expand kv heads to match query heads
        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

        # (B, T or T_full, H, D) -> (B, H, T or T_full, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use SDPA when no custom mask + no cache -> causal kernel selects;
        # else fall back to manual computation so we can apply the additive
        # mask correctly.
        if cache is None and mask is None:
            # Training path: full causal mask over the sequence
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif cache is not None:
            # Inference path: q has length T_step (usually 1), k/v have
            # the full prefix. No mask needed because every query position
            # is the latest, allowed to see the entire prefix.
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # Custom additive mask path
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            scores = scores + mask
            weights = F.softmax(scores, dim=-1)
            attn = torch.matmul(weights, v)

        # (B, H, T, D) -> (B, T, H * D)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(attn)
