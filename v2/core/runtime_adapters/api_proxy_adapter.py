"""API-proxy runtime adapter — bring ANY existing LLM onto the mesh.

The other three adapters run a model that lives ON the provider node
(Ollama process, GGUF file, transformers weights). This one is
different: it makes a provider node a *bridge* to an OpenAI- or
Anthropic-compatible endpoint the operator already has — a local
runtime (LM Studio, vLLM, TGI, Ollama's OpenAI shim, llama.cpp
server), a self-hosted cluster, or a hosted API the operator holds a
key to.

Why this exists (operator's question, 2026-07-17): "let users connect
to any existing LLM and use OUR mesh/compute instead of a cloud." The
honest split this adapter encodes:

  * **Where the model runs = wherever the WEIGHTS already are.** If the
    operator points this adapter at `http://localhost:1234/v1` (LM
    Studio on THEIR box), the compute is theirs and the mesh routes
    work to it — that is a provider contributing real compute. If they
    point it at `https://api.openai.com/v1`, the mesh is only a router
    and the tokens are billed by that upstream — useful for capacity
    overflow, NOT for the "replace the cloud" mission. `is_local`
    reflects which case this is, so receipts never claim mesh compute
    for a hosted passthrough.

  * The mesh NEVER auto-installs a model on someone's machine. A node
    contributes compute by running a model IT chose to host (via the
    Ollama/llama-cpp/transformers adapters) or by bridging an endpoint
    IT already operates (this adapter). Consent is structural.

Config (env, all optional except base_url for the probe to pass):
  PLUGINFER_PROXY_BASE_URL   e.g. http://localhost:1234/v1
  PLUGINFER_PROXY_API_KEY    bearer token if the upstream needs one
  PLUGINFER_PROXY_MODEL      upstream model id to request
  PLUGINFER_PROXY_DIALECT    "openai" (default) | "anthropic"

Probe = a cheap GET/HEAD on the base URL's model list; never downloads
or bills. Refuses (RuntimeAdapterUnavailable) when unconfigured so the
autodetect ladder falls through cleanly on nodes that don't want to
bridge.
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

DEFAULT_TIMEOUT_S = float(os.environ.get("PLUGINFER_PROXY_TIMEOUT_S", "120"))

# Hosts we treat as the operator's OWN compute (mesh genuinely serves
# the work) vs a third-party passthrough (mesh only routes). Substring
# match on the host is deliberately conservative — anything not clearly
# local is flagged non-local so receipts under-claim rather than lie.
_LOCAL_HOST_HINTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1",
                     "host.docker.internal", ".local", "192.168.",
                     "10.", "172.16.", "172.17.", "172.18.")


def _is_local_endpoint(base_url: str) -> bool:
    low = base_url.lower()
    return any(h in low for h in _LOCAL_HOST_HINTS)


def _cfg(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _probe_reachable(base_url: str, api_key: str, timeout_s: float) -> None:
    """Cheap readiness check: GET {base_url}/models. Raises
    RuntimeAdapterUnavailable if unreachable. Never generates tokens."""
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=min(timeout_s, 5.0)) as r:
            r.read(1)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        # A 401/403 still proves the endpoint EXISTS — treat auth
        # errors as "reachable but needs a key" rather than unavailable,
        # so a mis-keyed hosted API surfaces at job time with a clear
        # message instead of silently downgrading to echo.
        if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403):
            return
        raise RuntimeAdapterUnavailable(
            f"api-proxy endpoint {url} unreachable: {e}"
        ) from e


@register_adapter("api-proxy")
def make_api_proxy_runner(
    *,
    model_id: str = "",
    base_url: str = "",
    api_key: str = "",
    dialect: str = "",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    _probe: bool = False,
) -> RunnerFn:
    """Return a RunnerFn that forwards prompts to an OpenAI/Anthropic-
    compatible upstream. Refuses if no base URL is configured."""
    base_url = base_url or _cfg("PLUGINFER_PROXY_BASE_URL")
    if not base_url:
        raise RuntimeAdapterUnavailable(
            "api-proxy not configured (set PLUGINFER_PROXY_BASE_URL to "
            "an OpenAI/Anthropic-compatible endpoint to bridge it onto "
            "the mesh)."
        )
    api_key = api_key or _cfg("PLUGINFER_PROXY_API_KEY")
    dialect = (dialect or _cfg("PLUGINFER_PROXY_DIALECT", "openai")).lower()
    upstream_model = (
        model_id or _cfg("PLUGINFER_PROXY_MODEL")
        or "gpt-3.5-turbo"
    )
    _probe_reachable(base_url, api_key, timeout_s)
    if _probe:
        # Probe path still returns a runner (autodetect calls the
        # factory in both phases); it just did the cheap reachability
        # check above and won't be invoked for generation.
        pass

    is_local = _is_local_endpoint(base_url)
    base = base_url.rstrip("/")

    def _run_openai(prompt: str, payload: Dict[str, Any]) -> bytes:
        body = {
            "model": upstream_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(opt_num(payload, "max_tokens", 256)),
            "temperature": float(opt_num(payload, "temperature", 0.7)),
        }
        req = urllib.request.Request(
            base + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            resp = json.loads(r.read().decode("utf-8"))
        return str(
            resp["choices"][0]["message"]["content"]
        ).encode("utf-8")

    def _run_anthropic(prompt: str, payload: Dict[str, Any]) -> bytes:
        body = {
            "model": upstream_model,
            "max_tokens": int(opt_num(payload, "max_tokens", 256)),
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            base + "/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
        )
        if api_key:
            req.add_header("x-api-key", api_key)
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            resp = json.loads(r.read().decode("utf-8"))
        # Anthropic returns content as a list of blocks.
        parts = resp.get("content") or []
        text = "".join(
            b.get("text", "") for b in parts if isinstance(b, dict))
        return text.encode("utf-8")

    runner = _run_anthropic if dialect == "anthropic" else _run_openai

    def _run(prompt: str, payload: Dict[str, Any]) -> bytes:
        try:
            return runner(prompt, payload)
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"api-proxy upstream HTTP {e.code} ({base}): check "
                f"PLUGINFER_PROXY_API_KEY / model `{upstream_model}`."
            ) from e
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"api-proxy upstream unreachable: {e}") from e

    # Receipts must record the TRUE served model and whether this was
    # mesh compute or a hosted passthrough.
    _run.served_model_id = upstream_model
    _run.is_local_compute = is_local
    _run.proxy_dialect = dialect
    return _run
