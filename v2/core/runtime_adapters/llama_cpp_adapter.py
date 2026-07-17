"""llama-cpp-python runtime adapter.

Lightest-weight real-model option. Single Python dep
(`pip install llama-cpp-python`), no separate process, runs on
CPU + CUDA + Metal + Vulkan + ROCm depending on how it was built.

Operator drops a `.gguf` quantised model file (Q4_K_M etc) into
`~/.pluginfer/models/` and the adapter loads it on first use.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .base import RunnerFn, RuntimeAdapterUnavailable, opt_num, register_adapter

logger = logging.getLogger(__name__)

# Per-process cache of loaded models — `Llama` instances are heavy
# (multi-GB weight tensors); we keep one per (model_id, n_ctx) tuple.
_LLAMA_CACHE: Dict[str, Any] = {}

DEFAULT_MODELS_DIR = Path(
    os.environ.get("PLUGINFER_GGUF_DIR",
                   str(Path.home() / ".pluginfer" / "models"))
)


def _resolve_gguf_path(model_id: str) -> Optional[Path]:
    """Map an HF-style model id to a local .gguf file. Operator
    points at the same name they used with llama-cpp's bundled
    converter; we don't auto-download (operator decides licence)."""
    safe = model_id.replace("/", "_")
    candidates = [
        DEFAULT_MODELS_DIR / f"{safe}.gguf",
        DEFAULT_MODELS_DIR / f"{safe}.Q4_K_M.gguf",
        DEFAULT_MODELS_DIR / f"{safe}.q4_k_m.gguf",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Operator may have dropped any .gguf in the dir; pick first match
    # for the model basename.
    if DEFAULT_MODELS_DIR.exists():
        for p in DEFAULT_MODELS_DIR.glob("*.gguf"):
            if safe.lower() in p.name.lower():
                return p
    return None


@register_adapter("llama-cpp")
def make_llama_cpp_runner(
    *,
    model_id: str,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,        # -1 = use all available
    _probe: bool = False,
) -> RunnerFn:
    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError as e:
        raise RuntimeAdapterUnavailable(
            f"llama_cpp not installed: {e}. Run "
            f"`pip install llama-cpp-python` (or the CUDA/Metal/ROCm "
            f"variant for hardware acceleration)."
        ) from e

    if _probe:
        # Probe path: only check the import + at least one .gguf
        # exists in the models dir.
        if not DEFAULT_MODELS_DIR.exists() or not any(
            DEFAULT_MODELS_DIR.glob("*.gguf")
        ):
            raise RuntimeAdapterUnavailable(
                f"No .gguf files in {DEFAULT_MODELS_DIR}. Drop a "
                f"quantised GGUF there or set PLUGINFER_GGUF_DIR."
            )
        # Don't actually load the model in probe mode — that's an
        # expensive call we save for first real use.
        return lambda prompt, payload: b""

    gguf_path = _resolve_gguf_path(model_id)
    if gguf_path is None:
        raise RuntimeAdapterUnavailable(
            f"No GGUF found for model_id={model_id!r} in "
            f"{DEFAULT_MODELS_DIR}. Download a quantised version "
            f"(e.g., Q4_K_M) from HuggingFace and drop it there."
        )

    cache_key = f"{gguf_path}|{n_ctx}|{n_gpu_layers}"
    llm = _LLAMA_CACHE.get(cache_key)
    if llm is None:
        logger.info("llama_cpp: loading %s (ctx=%d, gpu_layers=%d)",
                    gguf_path, n_ctx, n_gpu_layers)
        llm = Llama(
            model_path=str(gguf_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        _LLAMA_CACHE[cache_key] = llm

    def _run(prompt: str, payload: Dict[str, Any]) -> bytes:
        max_tokens = int(opt_num(payload, "max_tokens", 256))
        temperature = float(opt_num(payload, "temperature", 0.7))
        out = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=payload.get("stop") or [],
        )
        text = ""
        try:
            text = out["choices"][0]["text"]
        except (KeyError, IndexError, TypeError):
            text = str(out)
        return text.encode("utf-8")

    return _run
