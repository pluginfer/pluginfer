"""Real embedding backends for the semantic cache.

The default SemanticCache backend is lexical 3-gram hashing — honest,
dependency-free, but it matches surface text, not meaning. This module
provides TRUE semantic matching via a local embedding model served by
Ollama (the org's hardware; no text ever leaves the machine).

Honesty rails:
  * Availability is probed at construction — if Ollama is unreachable
    or the model can't embed, construction RAISES. There is no silent
    fallback to lexical: the operator asked for semantic and must know
    if they didn't get it. (The factory `make_embedder` is the place
    that makes an explicit, logged choice.)
  * The cache's `backend_name` carries the real model name, so every
    receipt says exactly which backend produced the match.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable, List, Optional, Tuple

__all__ = ["EmbedderUnavailable", "OllamaEmbedder", "make_embedder"]

# (url, body_bytes, timeout_s) -> (status, body_bytes) — injectable for
# hermetic tests.
HttpPost = Callable[[str, bytes, float], Tuple[int, bytes]]


def _default_post(url: str, body: bytes, timeout_s: float
                  ) -> Tuple[int, bytes]:
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return int(r.status), r.read()
    except urllib.error.HTTPError as e:
        return int(e.code), e.read()


class EmbedderUnavailable(RuntimeError):
    pass


class OllamaEmbedder:
    """Embeds via a local Ollama server. Tries the current `/api/embed`
    shape first, falls back to legacy `/api/embeddings`. Raises
    EmbedderUnavailable at construction if a probe embed fails."""

    def __init__(self, model: str = "nomic-embed-text",
                 base_url: str = "http://127.0.0.1:11434",
                 http_post: Optional[HttpPost] = None,
                 timeout_s: float = 15.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._post = http_post or _default_post
        self.timeout_s = timeout_s
        self._legacy = False
        vec = self._embed("pluginfer availability probe")
        if not vec:
            raise EmbedderUnavailable(
                f"Ollama at {base_url} returned no embedding for model "
                f"{model!r} — is the server running and the model "
                f"pulled? (`ollama pull {model}`)")

    @property
    def backend_name(self) -> str:
        return f"ollama:{self.model}"

    def _embed(self, text: str) -> List[float]:
        # Current API shape.
        if not self._legacy:
            status, body = self._post(
                self.base_url + "/api/embed",
                json.dumps({"model": self.model, "input": text}).encode(),
                self.timeout_s)
            if status == 200:
                try:
                    emb = json.loads(body).get("embeddings") or []
                    if emb and emb[0]:
                        return [float(x) for x in emb[0]]
                except (ValueError, TypeError):
                    pass
            self._legacy = True
        # Legacy shape.
        status, body = self._post(
            self.base_url + "/api/embeddings",
            json.dumps({"model": self.model, "prompt": text}).encode(),
            self.timeout_s)
        if status == 200:
            try:
                emb = json.loads(body).get("embedding") or []
                return [float(x) for x in emb]
            except (ValueError, TypeError):
                pass
        return []

    def __call__(self, text: str) -> List[float]:
        vec = self._embed(text)
        if not vec:
            raise EmbedderUnavailable(
                f"embed failed mid-flight for model {self.model!r}")
        return vec


def make_embedder(backend: str, *, model: str = "nomic-embed-text",
                  base_url: str = "http://127.0.0.1:11434"):
    """Explicit backend choice: 'lexical' → (None, 'lexical-3gram');
    'ollama' → (OllamaEmbedder, its backend name). Raises rather than
    silently downgrading — the operator's config must mean what it
    says."""
    if backend == "lexical":
        return None, "lexical-3gram"
    if backend == "ollama":
        emb = OllamaEmbedder(model=model, base_url=base_url)
        return emb, emb.backend_name
    raise ValueError(f"unknown semantic backend {backend!r} "
                     f"(expected 'lexical' or 'ollama')")
