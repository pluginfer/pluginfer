"""TransformerBlock + PluginferLM - the model backbone.

Pre-norm architecture (Llama-style):

    x = x + Attention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))

This is more stable for training than post-norm: the residual stream is
preserved through every layer at full magnitude, while the norm
prevents activation drift inside attention/FFN.

`PluginferLM.generate()` implements autoregressive sampling with KV
cache (KVCache instance), nucleus + top-k filtering, temperature scaling,
and EOS early-stop. The KV cache is per-call so simultaneous inference
streams can't contaminate each other.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import GroupedQueryAttention, KVCache
from .config import ModelConfig
from .embeddings import TokenEmbedding
from .ffn import SwiGLUFFN
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(config, layer_id=layer_id)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        cache: Optional[KVCache] = None,
    ) -> Tensor:
        h = self.attn(self.attn_norm(x), mask=mask, cache=cache)
        x = x + self.dropout(h)
        h = self.ffn(self.ffn_norm(x))
        x = x + self.dropout(h)
        return x


class PluginferLM(nn.Module):
    """The Pluginfer brain language-model backbone.

    Five intelligence modules attach via task-specific heads (see
    `heads.py`). The backbone is shared and can be jointly trained
    multi-task (CP-AI-4) or fine-tuned per task.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = TokenEmbedding(config)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, layer_id=i) for i in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        # lm_head shares weights with the embedding when tie_word_embeddings.
        # We keep `lm_head` as None and use F.linear(x, self.embed.weight)
        # in the forward pass; this avoids any duplication or sync drift.
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        # Llama-style residual scaling for stability: scale init of o_proj
        # and ffn.w2 by 1 / sqrt(2 * n_layers). This prevents the residual
        # stream from drifting upward during early training.
        self._init_weights()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        std = self.config.init_std
        scale = (2 * self.config.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if p.dim() < 2:
                continue  # leave norms / biases at their init
            if name.endswith("attn.o_proj.weight") or name.endswith("ffn.w2.weight"):
                # Output of each residual sub-block: shrink so cumulative
                # variance stays bounded across layers.
                nn.init.normal_(p, mean=0.0, std=std * scale)
            elif name.endswith("embed.weight"):
                # Embedding init is set in TokenEmbedding constructor; skip.
                pass
            else:
                nn.init.normal_(p, mean=0.0, std=std)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        cache: Optional[KVCache] = None,
    ) -> Tensor:
        """Return logits of shape (B, T, vocab_size)."""
        x = self.embed(input_ids)  # (B, T, D)
        # SDPA's is_causal=True handles training masks; cache path passes
        # mask=None and lets SDPA see only past+current tokens.
        for block in self.blocks:
            x = block(x, mask=None, cache=cache)
        x = self.norm(x)
        if self.config.tie_word_embeddings:
            logits = F.linear(x, self.embed.weight)
        else:
            assert self.lm_head is not None
            logits = self.lm_head(x)
        return logits

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_human": f"{total / 1e9:.2f}B",
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """Autoregressive sampling with KV cache.

        Args:
            prompt_ids: (B, T) tensor of input ids.
            max_new_tokens: number of tokens to generate (per batch element).
            temperature: 0 = greedy (deterministic argmax). >0 scales logits.
            top_p: nucleus filter; keeps the smallest set of tokens whose
                cumulative probability >= top_p.
            top_k: keeps only the top-k highest-probability tokens.
            eos_token_id: if set, generation stops for the whole batch when
                every sequence has emitted at least one EOS. (Per-stream
                early stop without cross-stream coupling needs continuous
                batching - shipped in CP-AI-5 InferenceEngine.)

        Returns:
            (B, T + n_emitted) tensor where n_emitted <= max_new_tokens.
        """
        if prompt_ids.dim() != 2:
            raise ValueError(
                f"prompt_ids must be (B, T); got shape {tuple(prompt_ids.shape)}"
            )
        was_training = self.training
        self.eval()

        B = prompt_ids.shape[0]
        cache = KVCache(n_layers=self.config.n_layers)

        # Prefill: run the full prompt through the model once. This populates
        # cache.k/v for every layer over positions [0, T_prompt).
        logits = self.forward(prompt_ids, cache=cache)
        next_logits = logits[:, -1, :]  # (B, vocab)

        out_ids = prompt_ids
        finished = torch.zeros(B, dtype=torch.bool, device=prompt_ids.device)

        for _ in range(max_new_tokens):
            next_tok = self._sample_one(next_logits, temperature, top_p, top_k)
            next_tok = torch.where(finished, prompt_ids.new_zeros(()), next_tok)
            out_ids = torch.cat([out_ids, next_tok.unsqueeze(1)], dim=1)
            if eos_token_id is not None:
                finished = finished | (next_tok == eos_token_id)
                if bool(finished.all()):
                    break
            # Decode step: feed only the newest token; cache holds everything else.
            step_logits = self.forward(next_tok.unsqueeze(1), cache=cache)
            next_logits = step_logits[:, -1, :]

        if was_training:
            self.train()
        return out_ids

    @staticmethod
    def _sample_one(
        logits: Tensor, temperature: float, top_p: float, top_k: int
    ) -> Tensor:
        """Sample one token per batch row from `logits` of shape (B, vocab)."""
        if temperature <= 0.0:
            return logits.argmax(dim=-1)
        scaled = logits / max(temperature, 1e-6)
        # top-k filter: zero out everything outside the top k
        if top_k > 0 and top_k < scaled.size(-1):
            kth_vals, _ = torch.topk(scaled, top_k, dim=-1)
            min_keep = kth_vals[:, -1, None]  # (B, 1)
            scaled = torch.where(
                scaled < min_keep, torch.full_like(scaled, float("-inf")), scaled
            )
        # top-p (nucleus) filter
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cum = sorted_probs.cumsum(dim=-1)
            # Keep tokens until cumulative >= top_p; always keep the top one.
            keep = cum <= top_p
            keep[..., 0] = True
            sorted_logits = torch.where(
                keep, sorted_logits, torch.full_like(sorted_logits, float("-inf"))
            )
            # Scatter back to original order
            scaled = torch.full_like(scaled, float("-inf"))
            scaled.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
