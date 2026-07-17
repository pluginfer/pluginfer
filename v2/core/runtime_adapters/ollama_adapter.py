"""Ollama HTTP runtime adapter.

The single easiest path to "real LLM on the mesh." Operator installs
Ollama (cross-platform binary, no Python deps), pulls a model
(`ollama pull qwen2.5:1.5b`), runs `ollama serve`. This adapter
talks to the local Ollama HTTP API on `localhost:11434` (which
collides with Pluginfer's devserver default port — operator should
move one or the other; documented in the devserver README).

Why this is the most defensive default:
  * Zero Python ML deps — works on machines without torch.
  * Cross-vendor — Ollama itself uses CUDA on NVIDIA, MPS on
    Apple Silicon, ROCm on AMD, CPU otherwise.
  * Honest fallback — adapter probes the server; if Ollama isn't
    running, raises RuntimeAdapterUnavailable instead of pretending.

For production deployments preferring an in-process model, swap to
`transformers_adapter` (heavier but no separate process) or
`llama_cpp_adapter` (lighter; single-binary).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .base import RunnerFn, RuntimeAdapterUnavailable, opt_num, register_adapter

logger = logging.getLogger(__name__)

# Candidate hosts, most-authoritative first. A real Ollama install
# serves on 11434 (its universal default) — probing only a relocated
# port silently downgraded every standard machine to the echo runner.
# The Pluginfer devserver shim also defaults to 11434, but it has no
# /api/tags route, so the probe's GET distinguishes them for free
# (devserver 404s -> candidate skipped). 11435 stays as a fallback for
# operators who relocated Ollama per the earlier docs.
def _candidate_hosts() -> list:
    hosts = []
    env = os.environ.get("OLLAMA_HOST", "").strip()
    if env:
        if "://" not in env:
            env = "http://" + env
        hosts.append(env.rstrip("/"))
    for h in ("http://127.0.0.1:11434", "http://127.0.0.1:11435"):
        if h not in hosts:
            hosts.append(h)
    return hosts


DEFAULT_OLLAMA_HOST = _candidate_hosts()[0]


# Cold model load on consumer GPUs (or CPU offload) can take minutes;
# a 60s cap made the FIRST request every new user sends time out.
# Steady-state requests are seconds — the generous cap only matters
# for cold starts, which node boot pre-warms away anyway.
DEFAULT_RUNNER_TIMEOUT_S = float(os.environ.get(
    "PLUGINFER_RUNNER_TIMEOUT_S", "300"))


def _normalize_requested(model_id: str) -> str:
    """`Qwen/Qwen2.5-1.5B-Instruct` -> `qwen2.5-1.5b-instruct`; Ollama's
    native `family:tag` form passes through lowercased."""
    if "/" in model_id and ":" not in model_id:
        model_id = model_id.split("/")[-1]
    return model_id.lower()


def _negotiate_model(requested: str, pulled: list) -> Optional[str]:
    """Pick the pulled tag this endpoint will actually serve.

    An endpoint that is up but has never pulled the requested model
    404s every job — availability means "can serve", not "is
    listening". Ladder: exact tag (with or without `:tag` suffix) ->
    same family (pulled base is a prefix of the requested name) ->
    first pulled tag (/api/tags is most-recently-modified first).
    Whatever wins is stamped into receipts via `served_model_id`, so
    remapping is never a provenance lie. None means nothing pulled."""
    if not pulled:
        return None
    req = _normalize_requested(requested)
    for tag in pulled:
        if tag.lower() == req or tag.lower().split(":")[0] == req:
            return tag
    for tag in pulled:
        if req.startswith(tag.lower().split(":")[0]):
            return tag
    return pulled[0]


def _filter_by_headroom(pulled: list, sizes: Dict[str, int]) -> list:
    """Drop pulled tags whose estimated runtime footprint exceeds the
    host's free-RAM headroom (host_guard).

    This is the gate that stops a 14B model from swap-freezing a
    16 GB laptop: negotiation's `pulled[0]` fallback used to win
    purely by being the most-recently-modified tag, and Ollama then
    held it resident for keep_alive=60m. /api/tags reports each
    model's on-disk size, so the fit check is data-driven, not a
    name-parsing heuristic. Unknown sizes (0) pass — we can't judge
    them. When host_guard isn't importable, every tag passes (old
    behavior, no new failure mode)."""
    try:
        import host_guard
    except ImportError:
        return pulled
    keep, dropped = [], []
    for tag in pulled:
        (keep if host_guard.fits_model(sizes.get(tag, 0))
         else dropped).append(tag)
    if dropped:
        logger.warning(
            "ollama models %s exceed host RAM headroom (%.1f GB free) — "
            "refusing to load them (host_guard). Pull a smaller quant "
            "or free memory to serve them.",
            dropped, host_guard.headroom_bytes() / 1e9,
        )
    return keep


def _resolve_endpoint(
    model_id: str, candidates: list,
) -> "tuple[str, str]":
    """Return (host, served_model_tag) for the first candidate that is
    reachable AND has at least one model pulled that FITS the host's
    memory headroom; raise RuntimeAdapterUnavailable otherwise."""
    errors = []
    for cand in candidates:
        try:
            req = urllib.request.Request(f"{cand}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as r:
                doc = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as e:
            errors.append(f"{cand}: {e}")
            continue
        models = [m for m in doc.get("models", []) if m.get("name")]
        pulled = [m["name"] for m in models]
        sizes = {m["name"]: int(m.get("size") or 0) for m in models}
        fitting = _filter_by_headroom(pulled, sizes)
        if pulled and not fitting:
            errors.append(
                f"{cand}: models pulled but none fit host RAM headroom"
            )
            continue
        served = _negotiate_model(model_id, fitting)
        if served is None:
            errors.append(f"{cand}: reachable but zero models pulled")
            continue
        if served.lower() != _normalize_requested(model_id):
            logger.info(
                "ollama at %s does not have `%s` pulled; serving `%s` "
                "instead (receipts stamp the served model).",
                cand, model_id, served,
            )
        return cand, served
    raise RuntimeAdapterUnavailable(
        "No Ollama endpoint can serve: tried "
        f"[{', '.join(candidates)}] ({'; '.join(errors)}). Install "
        "Ollama, run `ollama serve`, and `ollama pull <model>`."
    )


@register_adapter("ollama")
def make_ollama_runner(
    *,
    model_id: str,
    host: str = "",
    timeout_s: float = DEFAULT_RUNNER_TIMEOUT_S,
    _probe: bool = False,
) -> RunnerFn:
    """Return a `RunnerFn` that POSTs to Ollama's /api/generate.

    Probe AND build both resolve through `_resolve_endpoint`: the
    endpoint must be reachable and the served model is negotiated
    against what is actually pulled (autodetect_runner calls the
    factory twice — once per phase — so both must agree). The runner
    carries `served_model_id` so callers stamp receipts with the model
    that really answers, per the refuse-rather-than-lie discipline."""
    candidates = [host] if host else _candidate_hosts()
    cand, served = _resolve_endpoint(model_id, candidates)
    return _make_ollama_runner_unchecked(
        model_id=served, host=cand, timeout_s=timeout_s,
    )


def _make_ollama_runner_unchecked(
    *, model_id: str, host: str, timeout_s: float,
) -> RunnerFn:
    # Strip a leading HF-style prefix so callers can pass either
    # `Qwen/Qwen2.5-1.5B-Instruct` or `qwen2.5:1.5b`. Ollama uses
    # the `:` form natively.
    ollama_id = model_id
    if "/" in ollama_id and ":" not in ollama_id:
        ollama_id = ollama_id.split("/")[-1].lower()

    # A provider node's whole job is serving: Ollama's default 5-minute
    # keep_alive evicts the model between requests, so every burst of
    # traffic after a lull pays the full cold load again (which then
    # blows the caller's latency ceiling). Keep the model resident.
    keep_alive = os.environ.get("PLUGINFER_OLLAMA_KEEP_ALIVE", "60m")

    def _run(prompt: str, payload: Dict[str, Any]) -> bytes:
        body = {
            "model": ollama_id,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                "num_predict": int(opt_num(payload, "max_tokens", 256)),
                "temperature": float(opt_num(payload, "temperature", 0.7)),
            },
        }
        req = urllib.request.Request(
            f"{host}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                resp = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"ollama HTTP {e.code}: model `{ollama_id}` not pulled? "
                f"Run `ollama pull {ollama_id}` on the provider node."
            ) from e
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"ollama unreachable: {e}") from e
        text = str(resp.get("response", ""))
        return text.encode("utf-8")

    _run.served_model_id = ollama_id
    return _run
