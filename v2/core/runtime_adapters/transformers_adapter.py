"""HuggingFace `transformers` runtime adapter.

The heaviest but most flexible adapter. Pulls model + tokenizer
from HuggingFace on first use, runs inference via `pipeline()` on
the best available device (CUDA / MPS / CPU). Requires torch.

For operators who want auto-download from HuggingFace and
deterministic versioning. Production deployments with strict
isolation should prefer `llama_cpp_adapter` (offline by default,
no network call at runtime)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .base import RunnerFn, RuntimeAdapterUnavailable, opt_num, register_adapter

logger = logging.getLogger(__name__)

_PIPELINE_CACHE: Dict[str, Any] = {}


@register_adapter("transformers")
def make_transformers_runner(
    *,
    model_id: str,
    device: Optional[str] = None,
    _probe: bool = False,
) -> RunnerFn:
    try:
        import torch                # type: ignore
        from transformers import pipeline  # type: ignore
    except ImportError as e:
        raise RuntimeAdapterUnavailable(
            f"transformers / torch not installed: {e}. Run "
            f"`pip install transformers accelerate torch`."
        ) from e

    if _probe:
        # Probe MUST establish the runtime is actually ready, not
        # just "torch is importable" — otherwise autodetect builds
        # a runner that then triggers a 1-3GB HF download on its
        # first prompt, which surprise-stalls the mesh.
        #
        # Ready iff (a) the model is already in the HF cache, or
        # (b) the operator has explicitly opted into auto-download
        # via PLUGINFER_ALLOW_HF_DOWNLOAD=1.
        if os.environ.get("PLUGINFER_ALLOW_HF_DOWNLOAD") == "1":
            return lambda prompt, payload: b""
        try:
            from huggingface_hub import scan_cache_dir  # type: ignore
            cache = scan_cache_dir()
            cached_ids = {repo.repo_id for repo in cache.repos}
            if model_id in cached_ids:
                return lambda prompt, payload: b""
        except Exception:
            pass
        raise RuntimeAdapterUnavailable(
            f"model {model_id!r} not in HuggingFace cache and "
            f"PLUGINFER_ALLOW_HF_DOWNLOAD is not set. Either "
            f"`huggingface-cli download {model_id}` first, or set "
            f"PLUGINFER_ALLOW_HF_DOWNLOAD=1 to permit auto-download."
        )

    # Pick the best device for THIS host. transformers' pipeline()
    # takes -1 for CPU, an int for the CUDA index, or 'mps' for
    # Apple Silicon.
    if device is None:
        if torch.cuda.is_available():
            device = 0
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = -1

    cache_key = f"{model_id}|{device}"
    pipe = _PIPELINE_CACHE.get(cache_key)
    if pipe is None:
        logger.info("transformers: loading %s on device=%s", model_id, device)
        pipe = pipeline(
            "text-generation",
            model=model_id,
            device=device,
            torch_dtype="auto",
        )
        _PIPELINE_CACHE[cache_key] = pipe

    def _run(prompt: str, payload: Dict[str, Any]) -> bytes:
        max_new = int(opt_num(payload, "max_tokens", 256))
        temperature = float(opt_num(payload, "temperature", 0.7))
        out = pipe(
            prompt,
            max_new_tokens=max_new,
            temperature=max(0.01, temperature),    # transformers rejects 0
            return_full_text=False,
            do_sample=temperature > 0.0,
        )
        text = ""
        if isinstance(out, list) and out:
            text = out[0].get("generated_text", "")
        return str(text).encode("utf-8")

    return _run
