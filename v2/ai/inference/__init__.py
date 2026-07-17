"""Inference engine for PluginferLM (KV cache + INT8 quant + FastAPI)."""

from .engine import InferenceEngine, GenerationParams
from .quantization import INT8Quantizer, quantize_module_in_place
from .batching import GenerationRequest, ContinuousBatcher

__all__ = [
    "InferenceEngine",
    "GenerationParams",
    "INT8Quantizer",
    "quantize_module_in_place",
    "GenerationRequest",
    "ContinuousBatcher",
]
