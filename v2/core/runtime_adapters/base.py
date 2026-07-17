"""Runtime-adapter base types + the autodetect resolver."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RunnerFn = Callable[[str, Dict[str, Any]], bytes]


class RuntimeAdapterUnavailable(RuntimeError):
    """Raised when an adapter's dependencies aren't satisfied. The
    autodetect catches this and falls through to the next adapter."""


def opt_num(payload: Dict[str, Any], key: str, default):
    """payload.get with None-coalescing. OpenAI-style clients send
    explicit nulls ("temperature": null); dict.get's default does NOT
    apply to a present-but-None key, so int()/float() crash the runner
    mid-job. Every adapter must use this for numeric options."""
    v = payload.get(key)
    return default if v is None else v


# Registered adapters, in priority order. Each entry is
# (name, factory_callable) where factory_callable returns RunnerFn
# or raises RuntimeAdapterUnavailable.
_REGISTRY: List[Tuple[str, Callable[..., RunnerFn]]] = []


def register_adapter(
    name: str,
) -> Callable[[Callable[..., RunnerFn]], Callable[..., RunnerFn]]:
    """Decorator. The adapter module calls this at import time so
    `list_available_adapters` + `autodetect_runner` see it."""

    def _wrap(factory: Callable[..., RunnerFn]) -> Callable[..., RunnerFn]:
        _REGISTRY.append((name, factory))
        return factory

    return _wrap


def list_available_adapters() -> List[str]:
    """Return the names of every adapter whose factory imports its
    dependencies cleanly. Useful for CLI status output."""
    available = []
    for name, factory in _REGISTRY:
        try:
            # Probe by calling the factory with no model — adapters
            # MUST handle the probe path without doing real work.
            factory(model_id="__probe__", _probe=True)
        except RuntimeAdapterUnavailable:
            continue
        except Exception as e:
            logger.debug("adapter %s probe raised %r", name, e)
            continue
        available.append(name)
    return available


def autodetect_runner(
    *,
    model_id: str,
    prefer: Optional[List[str]] = None,
    **kwargs: Any,
) -> RunnerFn:
    """Try every registered adapter in priority order; return the
    first one that comes up. `prefer` reorders the trial list — pass
    `["ollama"]` to skip the transformers path on a machine that has
    Ollama running but no torch installed.

    Raises `RuntimeAdapterUnavailable` if every adapter fails."""
    order: List[Tuple[str, Callable[..., RunnerFn]]] = list(_REGISTRY)
    if prefer:
        prefer_set = {p.lower() for p in prefer}
        # Put preferred adapters first, preserve relative order
        # among non-preferred ones.
        head = [t for t in order if t[0].lower() in prefer_set]
        tail = [t for t in order if t[0].lower() not in prefer_set]
        order = head + tail

    last_err: Optional[Exception] = None
    for name, factory in order:
        # Phase 1: probe — cheap, asks the adapter whether its
        # runtime dependencies are actually present + reachable
        # (Ollama listening, GGUF on disk, etc.). If the probe
        # raises RuntimeAdapterUnavailable we skip without ever
        # building a broken runner. This is the difference between
        # "torch is importable" and "we can actually serve a forward
        # pass right now" — the latter is what callers need.
        try:
            factory(model_id=model_id, _probe=True, **kwargs)
        except RuntimeAdapterUnavailable as e:
            last_err = e
            logger.debug("adapter %s probe failed: %s", name, e)
            continue
        except Exception as e:
            last_err = e
            logger.debug("adapter %s probe raised %r", name, e)
            continue
        # Phase 2: build the real runner. The probe established
        # the runtime is ready; this call wires it up for prod use.
        try:
            runner = factory(model_id=model_id, **kwargs)
        except RuntimeAdapterUnavailable as e:
            last_err = e
            logger.debug("adapter %s unavailable on build: %s", name, e)
            continue
        except Exception as e:
            last_err = e
            logger.warning("adapter %s factory raised on build: %s", name, e)
            continue
        logger.info("autodetect_runner: using %s for %s", name, model_id)
        return runner

    raise RuntimeAdapterUnavailable(
        f"No runtime adapter is available. Install one of: "
        f"`pip install ollama` (+ `ollama serve` running), "
        f"`pip install llama-cpp-python`, or "
        f"`pip install transformers accelerate torch`. "
        f"Last error: {last_err}"
    )


__all__ = [
    "RunnerFn",
    "RuntimeAdapterUnavailable",
    "autodetect_runner",
    "list_available_adapters",
    "register_adapter",
]
