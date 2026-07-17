"""Gateway response cache — the cheapest token is the one never generated.

Token-economics innovation for the going-public build: agents and apps
resend byte-identical prompts constantly (retries after client
timeouts, fan-outs that share a system prompt + question, cron agents
asking the same thing hourly). Production LLM gateways report double-
digit exact-repeat rates. Serving a repeat from cache costs ZERO
provider tokens, zero settlement, ~0ms — and drops the buyer's bill by
exactly the repeat rate.

Honesty policy (non-negotiable):
  * Only DETERMINISTIC requests are cached by default: temperature == 0.
    A sampled completion (temperature > 0) is expected to vary per call;
    silently replaying one changes semantics. Operators who accept
    replay-on-repeat for sampled traffic opt in: PLUGINFER_CACHE_ALL=1.
  * Hits are labelled, never disguised: the gateway sends
    `X-Pluginfer-Cache: hit` + `X-Pluginfer-Cache-Age` alongside the
    ORIGINAL receipt id, so the audit trail still points at the real
    execution that produced the bytes.
  * TTL-bounded (default 300s, env PLUGINFER_CACHE_TTL_S) and
    size-bounded LRU (default 1024 entries) — stale answers age out.

Pure stdlib, thread-safe, no external deps.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


class ResponseCache:
    """TTL + LRU cache keyed on the canonical request."""

    def __init__(self, *, max_entries: int = 1024,
                 ttl_s: Optional[float] = None) -> None:
        self.max_entries = int(max_entries)
        self.ttl_s = ttl_s if ttl_s is not None else _env_float(
            "PLUGINFER_CACHE_TTL_S", 300.0)
        self._lock = threading.Lock()
        self._store: "OrderedDict[str, Tuple[float, Dict[str, Any]]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    # -- policy --------------------------------------------------------

    @staticmethod
    def cacheable(payload: Dict[str, Any]) -> bool:
        """Deterministic requests only, unless the operator opted in."""
        if os.environ.get("PLUGINFER_CACHE_DISABLE") == "1":
            return False
        if os.environ.get("PLUGINFER_CACHE_ALL") == "1":
            return True
        temp = payload.get("temperature")
        return temp == 0 or temp == 0.0

    @staticmethod
    def key_for(payload: Dict[str, Any]) -> str:
        """Canonical key over the fields that determine the answer."""
        body = {
            "model": payload.get("model"),
            "messages": payload.get("openai", {}).get("messages")
            or payload.get("messages")
            or payload.get("prompt"),
            "max_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature"),
            "top_p": payload.get("top_p"),
        }
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    # -- storage ---------------------------------------------------------

    def get(self, key: str) -> Optional[Tuple[Dict[str, Any], float]]:
        """Returns (entry, age_seconds) or None."""
        now = time.time()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                self.misses += 1
                return None
            stored_at, entry = item
            if now - stored_at > self.ttl_s:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return dict(entry), now - stored_at

    def put(self, key: str, entry: Dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = (time.time(), dict(entry))
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._store),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
                "ttl_s": self.ttl_s,
            }
