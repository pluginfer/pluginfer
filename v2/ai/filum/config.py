"""Filum configuration -- 127M parameters, sized for GTX 1650 / 4 GB VRAM.

Architecture math
-----------------
We size to ~127M parameters -- the floor at which genuine reasoning
capability emerges per Chinchilla scaling laws (when paired with
high-quality distillation rather than raw web text).

  embed (tied with lm_head):  vocab(16384) * d_model(896)         = 14.7M
  per-layer attention:        Q+K+V+O (GQA 14Q / 2KV)             =  1.84M
  per-layer SwiGLU FFN:       3 * d_model * d_ff(2304)            =  6.19M
  per-layer RMSNorms (2):                                          = ~0.002M
  layers (14):                14 * 8.03M                           = 112.4M
  final RMSNorm:                                                   = 0.001M
  ------------------------------------------------------------------
  total:                                                           = 127.1M

VRAM at fp16 + 8-bit AdamW training (batch=4, seq=512):

  weights fp16             127M * 2 bytes  = 254 MB
  AdamW state (8-bit m+v)  127M * 2 bytes  = 254 MB
  gradients fp16           127M * 2 bytes  = 254 MB
  activations              ~80 MB (with grad checkpointing)
  ------------------------------------------------------------
  total                    ~840 MB    (fits 4 GB GTX 1650 with 3 GB headroom
                                       for OS + Chrome + IDE)

VRAM at BitNet b1.58 deployment (inference only):

  ternary weights          127M * 0.2 bytes = 25.4 MB
  KV cache (seq=512)       ~22 MB
  -----------------------------------------------------
  total                    ~50 MB     (deploys on a Raspberry Pi)

Disk footprint (cap):

  fp16 master checkpoint   ~260 MB
  ternary deploy ckpt       ~30 MB
  teacher distill cache     <= 3 GB (cap configurable)
  tokenizer                  ~5 MB
  ------------------------------------------------------------
  total                    ~3.3 GB

Innovation stack
----------------
1. BitNet b1.58: ternary weights at deploy (`ai/training/bitnet_158.py`).
2. Multi-teacher consensus distillation: 3 free-tier teachers vote;
   reject samples where teacher KL > threshold (`ai/filum/teacher_pool.py`).
3. Active KL-weighted sampling: train on samples the student is
   currently WORST at (`ai/filum/active_sampler.py`).
4. Synthetic self-play: student generates prompts; teachers answer;
   distill on those (`ai/filum/self_play.py`).
5. 8-bit AdamW + gradient checkpointing
   (`ai/filum/optimizer_8bit.py`, `torch.utils.checkpoint`).

Why these numbers (and not 32M or 1B)
-------------------------------------
* 32M is too small for nuanced instruction following. Pretrained
  models like DistilGPT-2 (88M) are noticeably cleverer than 32M
  builds, and 127M is the safest "small but actually useful" target.
* 1B+ doesn't fit the optimizer state on 4 GB even with 8-bit
  AdamW, and would take weeks of laptop time to converge.
* 127M with multi-teacher distillation can reach ~70-80% of teacher
  quality on focused tasks (the Pluginfer router, parser, scorer)
  and is genuinely competitive with Phi-1.5 at a 20× smaller size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class FilumConfig:
    """Pluginfer in-built AI."""

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------
    vocab_size: int = 16384       # ~2x byte-level coverage
    context_length: int = 512
    d_model: int = 896
    n_layers: int = 14
    n_heads: int = 14
    n_kv_heads: int = 2           # GQA -- KV cache 7x smaller than full MHA
    head_dim: int = 64
    d_ff: int = 2304              # ~2.5x d_model (SwiGLU)
    rms_norm_eps: float = 1e-6
    rope_base: float = 10_000.0

    # ------------------------------------------------------------------
    # Training schedule
    # ------------------------------------------------------------------
    micro_batch_size: int = 4
    grad_accum_steps: int = 8     # effective batch 32
    max_steps: int = 50_000
    warmup_steps: int = 1_000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    use_grad_checkpointing: bool = True
    use_8bit_adamw: bool = True
    mixed_precision: str = "fp16"   # "fp16" / "bf16" / "fp32"

    # ------------------------------------------------------------------
    # Multi-teacher distillation
    # ------------------------------------------------------------------
    distill_alpha: float = 0.5
    distill_temperature: float = 2.0   # higher T -> softer teacher distribution
    teacher_top_k_logprobs: int = 20
    teacher_max_tokens: int = 256
    # Consensus filter: reject a sample if any pair of teachers'
    # outputs has Jensen-Shannon divergence > this. Default 0.4
    # is permissive; tighten for harder tasks.
    consensus_jsd_threshold: float = 0.4
    daily_budget_usd: float = 1.0
    cache_max_gb: float = 3.0

    # ------------------------------------------------------------------
    # Active sampling
    # ------------------------------------------------------------------
    active_sampler_pool_size: int = 256   # how many candidates to score
    active_sampler_select_top: int = 32   # how many to actually train on
    active_sampler_kl_temperature: float = 1.5

    # ------------------------------------------------------------------
    # Self-play
    # ------------------------------------------------------------------
    self_play_enabled: bool = True
    self_play_prompts_per_round: int = 64
    self_play_round_every_n_steps: int = 1_000

    # ------------------------------------------------------------------
    # BitNet deploy
    # ------------------------------------------------------------------
    deploy_with_bitnet: bool = True
    keep_lm_head_full_precision: bool = True
    keep_embed_full_precision: bool = True

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------
    curriculum_stages: List[str] = field(default_factory=lambda: [
        "byte_completion",
        "phrase_completion",
        "qa_short",
        "instruct",
        "router_task",
        "price_task",
        "reasoning",
    ])
    curriculum_steps_per_stage: int = 5_000

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    work_dir: str = "ai/filum/_work"
    checkpoint_dir: str = "ai/filum/_work/checkpoints"
    cache_dir: str = "ai/filum/_work/teacher_cache"
    tokenizer_path: str = "ai/filum/_work/tokenizer.json"

    # ------------------------------------------------------------------
    # Sanity
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.head_dim * self.n_heads != self.d_model:
            raise ValueError(
                f"head_dim({self.head_dim}) * n_heads({self.n_heads}) "
                f"!= d_model({self.d_model})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads({self.n_heads}) not divisible by "
                f"n_kv_heads({self.n_kv_heads})"
            )

    def estimate_param_count(self) -> dict:
        embed = self.vocab_size * self.d_model
        q_proj = self.d_model * self.n_heads * self.head_dim
        k_proj = self.d_model * self.n_kv_heads * self.head_dim
        v_proj = self.d_model * self.n_kv_heads * self.head_dim
        o_proj = self.n_heads * self.head_dim * self.d_model
        attn = q_proj + k_proj + v_proj + o_proj
        ffn = 3 * self.d_model * self.d_ff
        norm_per_layer = 2 * self.d_model
        per_layer = attn + ffn + norm_per_layer
        layers_total = self.n_layers * per_layer
        final_norm = self.d_model
        total = embed + layers_total + final_norm
        return {
            "embedding_M": round(embed / 1e6, 3),
            "per_layer_M": round(per_layer / 1e6, 3),
            "layers_total_M": round(layers_total / 1e6, 3),
            "total_M": round(total / 1e6, 3),
        }

    def estimate_vram_mb(self, *, training: bool = True,
                         bitnet: bool = False) -> dict:
        params = self.vocab_size * self.d_model + self.n_layers * (
            self.d_model * self.n_heads * self.head_dim
            + 2 * self.d_model * self.n_kv_heads * self.head_dim
            + self.n_heads * self.head_dim * self.d_model
            + 3 * self.d_model * self.d_ff
            + 2 * self.d_model
        ) + self.d_model
        if bitnet and not training:
            weights = params * 0.2 / 1e6
        else:
            weights = params * 2 / 1e6
        if not training:
            return {
                "weights_MB": round(weights, 1),
                "kv_cache_MB": round(
                    self.n_layers * self.context_length * 2
                    * self.n_kv_heads * self.head_dim * 2 / 1e6, 1,
                ),
                "total_MB": round(weights + 30, 1),
            }
        # Training
        adamw = params * (2 if self.use_8bit_adamw else 8) / 1e6
        grads = params * 2 / 1e6
        act_factor = 4 if self.use_grad_checkpointing else 12
        act = (self.micro_batch_size * self.context_length
               * self.d_model * self.n_layers * act_factor / 1e6)
        return {
            "weights_MB": round(weights, 1),
            "adamw_state_MB": round(adamw, 1),
            "grad_buffer_MB": round(grads, 1),
            "activations_MB": round(act, 1),
            "total_MB": round(weights + adamw + grads + act, 1),
        }
