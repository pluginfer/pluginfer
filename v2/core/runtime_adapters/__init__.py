"""Pluginfer runtime adapters — bridge from `Provider.execute()` to
real LLM runtimes. The operator picks one (or installs several and
the auto-detect picks the best available at runtime).

Each adapter exports `make_runner(model_id: str, **kwargs)` returning
a callable `runner(prompt: str, payload: dict) -> bytes` compatible
with `core.flagship.FlagshipProvider`'s `runner_fn` slot. The bytes
returned are UTF-8-encoded model output text; the flagship layer
signs the SHA-256 of those bytes into the PNIS receipt.

Public API:

    from core.runtime_adapters import autodetect_runner, list_available
    runner = autodetect_runner(model_id="Qwen/Qwen2.5-1.5B-Instruct")
    text_bytes = runner("Hello", {"max_tokens": 128})

The autodetect tries adapters in priority order (NVIDIA CUDA via
transformers -> Apple MPS via transformers -> llama-cpp -> Ollama
HTTP -> CPU-only stub) and returns the first one whose dependency
chain imports cleanly. Operators on heterogeneous hardware get the
right adapter without manual config.

This is the seam between the protocol (mesh + auction + receipts)
and the actual model inference. Until an operator installs at least
one adapter's deps, the alpha-tier flagship runs as the deterministic
echo `_alpha_runner` in `tools.auto_mesh` — clearly tagged as
NOT-REAL in the receipts so audit reviewers don't get fooled.
"""

from .base import (
    RunnerFn,
    RuntimeAdapterUnavailable,
    autodetect_runner,
    list_available_adapters,
)
# api_proxy is imported LAST so it registers at the bottom of the
# autodetect ladder: a node that has its own local model always
# prefers serving that (real mesh compute) over bridging an external
# endpoint. The proxy adapter self-excludes unless explicitly
# configured, so its mere presence never changes a default node.
from . import (
    ollama_adapter,
    llama_cpp_adapter,
    transformers_adapter,
    api_proxy_adapter,
)

__all__ = [
    "RunnerFn",
    "RuntimeAdapterUnavailable",
    "autodetect_runner",
    "list_available_adapters",
    "ollama_adapter",
    "llama_cpp_adapter",
    "transformers_adapter",
    "api_proxy_adapter",
]
