"""PluginferLM model package - transformer + task heads, all from scratch."""

from .config import ModelConfig, count_parameters_for_config
from .embeddings import TokenEmbedding, RotaryPositionalEmbedding
from .normalization import RMSNorm
from .ffn import SwiGLUFFN
from .attention import GroupedQueryAttention
from .transformer import TransformerBlock, PluginferLM
from .heads import (
    JobRouterHead,
    ProviderQualityScorerHead,
    PriceEngineHead,
    AnomalyDetectorAutoencoder,
)

__all__ = [
    "ModelConfig",
    "count_parameters_for_config",
    "TokenEmbedding",
    "RotaryPositionalEmbedding",
    "RMSNorm",
    "SwiGLUFFN",
    "GroupedQueryAttention",
    "TransformerBlock",
    "PluginferLM",
    "JobRouterHead",
    "ProviderQualityScorerHead",
    "PriceEngineHead",
    "AnomalyDetectorAutoencoder",
]
