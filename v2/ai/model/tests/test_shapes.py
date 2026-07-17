"""Shape tests for every model component (CP-AI-2 part 1).

We use ModelConfig.debug() (~10M params) so each test runs in seconds
on CPU. The default-config parameter-count check is a separate test
that uses count_parameters_for_config() (no allocation) -- the full
config eats ~4.5GB of fp32 weights which is impractical for unit tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from ai.model.attention import GroupedQueryAttention, KVCache  # noqa: E402
from ai.model.config import ModelConfig, count_parameters_for_config  # noqa: E402
from ai.model.embeddings import RotaryPositionalEmbedding, TokenEmbedding  # noqa: E402
from ai.model.ffn import SwiGLUFFN  # noqa: E402
from ai.model.heads import (  # noqa: E402
    AnomalyDetectorAutoencoder,
    JobRouterHead,
    PriceEngineHead,
    ProviderQualityScorerHead,
)
from ai.model.normalization import RMSNorm  # noqa: E402
from ai.model.transformer import PluginferLM, TransformerBlock  # noqa: E402


@pytest.fixture(scope="module")
def cfg() -> ModelConfig:
    return ModelConfig.debug()


def test_default_config_param_count_is_in_range() -> None:
    """The 1.1B default config should land in [0.9B, 1.5B] without allocation."""
    counts = count_parameters_for_config(ModelConfig())
    assert 900_000_000 < counts["total"] < 1_500_000_000
    # Sanity: the human-readable string is non-empty
    assert counts["total_human"].endswith("B")


def test_token_embedding_shape(cfg: ModelConfig) -> None:
    e = TokenEmbedding(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = e(ids)
    assert out.shape == (2, 16, cfg.d_model)


def test_rope_shapes(cfg: ModelConfig) -> None:
    rope = RotaryPositionalEmbedding(cfg)
    q = torch.randn(2, 16, cfg.n_heads, cfg.head_dim)
    k = torch.randn(2, 16, cfg.n_kv_heads, cfg.head_dim)
    q_rot, k_rot = rope(q, k, seq_len=16)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    # Rotation must change at least some values (sanity)
    assert not torch.allclose(q_rot, q)
    assert not torch.allclose(k_rot, k)


def test_rmsnorm_shape_and_dtype() -> None:
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 64)
    out = norm(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_swiglu_shape(cfg: ModelConfig) -> None:
    ffn = SwiGLUFFN(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    out = ffn(x)
    assert out.shape == x.shape


def test_attention_shape_no_cache(cfg: ModelConfig) -> None:
    attn = GroupedQueryAttention(cfg, layer_id=0)
    x = torch.randn(2, 16, cfg.d_model)
    out = attn(x)
    assert out.shape == x.shape


def test_attention_kv_cache_extends(cfg: ModelConfig) -> None:
    attn = GroupedQueryAttention(cfg, layer_id=0)
    cache = KVCache(n_layers=1)
    # Prefill 5 tokens
    x1 = torch.randn(1, 5, cfg.d_model)
    out1 = attn(x1, cache=cache)
    assert out1.shape == (1, 5, cfg.d_model)
    assert cache.get_pos(0) == 5
    # Decode 1 token
    x2 = torch.randn(1, 1, cfg.d_model)
    out2 = attn(x2, cache=cache)
    assert out2.shape == (1, 1, cfg.d_model)
    assert cache.get_pos(0) == 6


def test_transformer_block_shape(cfg: ModelConfig) -> None:
    block = TransformerBlock(cfg, layer_id=0)
    x = torch.randn(2, 16, cfg.d_model)
    out = block(x)
    assert out.shape == x.shape


def test_full_lm_forward_shape(cfg: ModelConfig) -> None:
    model = PluginferLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_lm_param_count_debug(cfg: ModelConfig) -> None:
    """count_parameters_for_config and an actual instantiation should agree."""
    model = PluginferLM(cfg)
    instance = sum(p.numel() for p in model.parameters())
    static = count_parameters_for_config(cfg)["total"]
    assert instance == static, f"static={static} instance={instance}"


def test_job_router_head_shape(cfg: ModelConfig) -> None:
    head = JobRouterHead(cfg, n_gpu_classes=12)
    h = torch.randn(4, cfg.d_model)
    out = head(h)
    assert out["gpu_logits"].shape == (4, 12)
    assert out["vram_gb"].shape == (4,)
    assert out["runtime_ms_log"].shape == (4,)
    assert out["confidence"].shape == (4,)
    # confidence should be in [0, 1]
    assert (out["confidence"] >= 0).all() and (out["confidence"] <= 1).all()


def test_provider_quality_head_shape(cfg: ModelConfig) -> None:
    head = ProviderQualityScorerHead(cfg)
    h = torch.randn(4, cfg.d_model)
    out = head(h)
    assert out["quality"].shape == (4,)
    assert out["reliability_24h"].shape == (4,)
    assert out["anomaly_logit"].shape == (4,)


def test_price_engine_head_shape(cfg: ModelConfig) -> None:
    head = PriceEngineHead(cfg)
    h = torch.randn(4, cfg.d_model)
    out = head(h)
    for k in ("floor", "ceiling", "demand_1hr", "supply_1hr", "surge_factor"):
        assert out[k].shape == (4,)
        assert (out[k] >= 0).all()


def test_anomaly_autoencoder_shape() -> None:
    ae = AnomalyDetectorAutoencoder(input_dim=64, bottleneck_dim=8)
    x = torch.randn(4, 64)
    x_hat, err = ae(x)
    assert x_hat.shape == x.shape
    assert err.shape == (4,)
    # Reconstruction error is non-negative
    assert (err >= 0).all()
