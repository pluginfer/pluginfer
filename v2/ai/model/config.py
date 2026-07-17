"""ModelConfig - all hyperparameters in one dataclass.

The default values target ~1.1B parameters (24 layers x d_model=2048,
GQA 16 Q / 4 KV heads, SwiGLU d_ff=5504). For unit tests use the
`debug()` factory which produces a ~10M-param model that constructs
and runs in seconds on CPU.

Parameter accounting (default config):
  Embedding:        32000 * 2048           = 65,536,000
  Per layer:
    q_proj          2048 * (16*128) = 2048*2048   = 4,194,304
    k_proj          2048 * (4*128)  = 2048*512    = 1,048,576
    v_proj          2048 * (4*128)  = 2048*512    = 1,048,576
    o_proj          (16*128) * 2048 = 2048*2048   = 4,194,304
    SwiGLU w1       2048 * 5504                   = 11,272,192
    SwiGLU w3       2048 * 5504                   = 11,272,192
    SwiGLU w2       5504 * 2048                   = 11,272,192
    RMSNorm x2      2 * 2048                      = 4,096
    --------------------------------------------------
                                                   44,306,432 / layer
  24 layers:        24 * 44,306,432               = 1,063,354,368
  Final RMSNorm:    2048
  --------------------------------------------------
  Total trainable:  ~1,128,892,416 ~= 1.13B (lm_head shares embed weights)

`count_parameters_for_config()` exposes this calculation so the CP-AI-2
parameter-count test can run without allocating the full model in
memory (default config = ~4.5 GB of fp32 weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Vocabulary - kept slightly larger than the tokenizer's actual
    # output range to leave headroom; embedding looks up by id directly.
    vocab_size: int = 32000

    # Sequence + hidden dimensions
    context_length: int = 2048
    d_model: int = 2048

    # Transformer depth + attention shape
    n_layers: int = 24
    n_heads: int = 16     # number of query heads
    n_kv_heads: int = 4   # number of key/value heads (GQA)
    head_dim: int = 128   # d_model / n_heads

    # FFN inner dim chosen so SwiGLU's 3-matrix block matches a 4x ReLU
    # FFN's parameter count: d_ff ~ (8/3) * d_model rounded.
    d_ff: int = 5504

    # Positional encoding
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0  # >1 stretches positions for long-context fine-tuning

    # Regularisation
    dropout: float = 0.0  # set to 0.1 during pretraining

    # Initialisation: scale of normal init for weights. 0.02 is the
    # GPT-2/Llama default; works well with weight tying.
    init_std: float = 0.02

    # Inference batching upper bound (only used for KV cache pre-allocation
    # in `InferenceEngine`; the model itself does not eagerly allocate this).
    max_batch_size: int = 32

    # When True, the embedding matrix is reused as the output projection
    # weight (saves vocab_size * d_model parameters; the standard Llama
    # default).
    tie_word_embeddings: bool = True

    # Reserved for cache-friendly fp32 vs bf16 on the host device.
    # The model itself uses whatever dtype the caller passes; this is
    # advisory metadata for downstream tools.
    preferred_dtype: str = "float32"

    # Sanity check at construction time.
    def __post_init__(self) -> None:
        if self.head_dim * self.n_heads != self.d_model:
            raise ValueError(
                f"head_dim * n_heads must equal d_model: "
                f"{self.head_dim} * {self.n_heads} != {self.d_model}"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads must be divisible by n_kv_heads: "
                f"{self.n_heads} % {self.n_kv_heads} != 0"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even (RoPE pairs adjacent dims): {self.head_dim}"
            )

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def debug(cls) -> "ModelConfig":
        """Tiny ~10M-param config used by unit tests.

        Trains 100 steps on CPU in seconds and exercises every code path
        (RoPE, GQA expansion, weight tying, KV cache) with the same
        invariants as the full config.
        """
        return cls(
            vocab_size=512,
            context_length=128,
            d_model=128,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            head_dim=32,
            d_ff=256,
            init_std=0.02,
            max_batch_size=8,
        )

    @classmethod
    def small(cls) -> "ModelConfig":
        """~125M-param config - useful for local fine-tuning runs."""
        return cls(
            vocab_size=32000,
            context_length=2048,
            d_model=768,
            n_layers=12,
            n_heads=12,
            n_kv_heads=4,
            head_dim=64,
            d_ff=2048,
        )


def count_parameters_for_config(config: ModelConfig) -> dict:
    """Compute exact parameter count without allocating any tensors.

    Used by tests on the default 1.1B config which would otherwise
    require ~4.5 GB of host RAM just for the float32 weights.
    """
    # Embedding (and tied lm_head)
    embed = config.vocab_size * config.d_model

    # Per-layer attention
    q_proj = config.d_model * config.n_heads * config.head_dim
    k_proj = config.d_model * config.n_kv_heads * config.head_dim
    v_proj = config.d_model * config.n_kv_heads * config.head_dim
    o_proj = config.n_heads * config.head_dim * config.d_model
    attn = q_proj + k_proj + v_proj + o_proj

    # Per-layer SwiGLU FFN
    ffn = 3 * config.d_model * config.d_ff

    # Per-layer RMSNorms (2 per block) + final RMSNorm
    norm_per_layer = 2 * config.d_model
    layer = attn + ffn + norm_per_layer

    final_norm = config.d_model

    # When `tie_word_embeddings=False` we'd add another vocab_size * d_model
    # for the output projection.
    head = 0 if config.tie_word_embeddings else config.vocab_size * config.d_model

    total = embed + config.n_layers * layer + final_norm + head
    return {
        "embedding": embed,
        "per_layer": layer,
        "n_layers": config.n_layers,
        "final_norm": final_norm,
        "lm_head": head,
        "total": total,
        "total_human": f"{total / 1e9:.2f}B",
    }
