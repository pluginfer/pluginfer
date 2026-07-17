"""Token-thrift primitives: semantic cache (HG13f) + prompt
compression (HG13h). Pure stdlib — the governance suite must run fully
ON-PREMISES with no mesh, no torch, no network beyond the org's own
LLM upstream. Everything here is deterministic and inspectable.

Honesty design (the rails these features live behind):

  * The default similarity backend is a **lexical** char-3-gram
    hashing embedder — real cosine similarity, zero dependencies, and
    honestly labelled ``backend="lexical-3gram"`` on every receipt. It
    catches near-duplicates (whitespace drift, reordered params,
    timestamps injected into otherwise-identical prompts — the classic
    agent-retry-loop pathology). It is NOT a neural paraphrase
    detector; a true-semantic backend (sentence-transformers etc.) can
    be plugged in via ``embed_fn`` when the org wants to run one on
    their hardware. We never call lexical matching "semantic
    understanding".
  * The high default threshold (0.97) is deliberate: a borderline hit
    served wrongly is a quality lie. Below-threshold candidates are
    misses, full stop.
  * Compression NEVER runs by default — every transform changes the
    prompt the org's model sees, so each one is an explicit operator
    opt-in, and every applied transform is itemised on the receipt.
    Savings from compression are labelled ESTIMATED (chars/4 of what
    was removed) and are kept in a separate bucket from the measured
    cache/cascade counterfactuals — the dashboard never mixes them.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from typing import (
    Any, Callable, Dict, List, Optional, Tuple,
)

_EMBED_DIM = 512


def _ngram_embed(text: str, dim: int = _EMBED_DIM) -> List[float]:
    """Char-3-gram hashing vector, L2-normalised. Deterministic,
    dependency-free, ~O(len(text))."""
    vec = [0.0] * dim
    t = " ".join(text.lower().split())
    for i in range(len(t) - 2):
        g = t[i:i + 3]
        h = int(hashlib.blake2b(g.encode("utf-8"),
                                digest_size=4).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _messages_text(body: Dict[str, Any]) -> str:
    parts: List[str] = []
    sysprompt = body.get("system")
    if isinstance(sysprompt, str):
        parts.append("system:" + sysprompt)
    for m in body.get("messages", []) or []:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(str(m.get("role", "")) + ":" + c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict):
                    parts.append(str(p.get("text", "")))
    return "\n".join(parts)


def _meta_key(body: Dict[str, Any]) -> str:
    """Everything EXCEPT the message text must match exactly for a
    semantic hit — model, max_tokens, tools, response_format... A
    fuzzy match is only ever fuzzy over the words."""
    meta = {k: v for k, v in body.items()
            if k not in ("messages", "system")}
    return hashlib.sha256(
        json.dumps(meta, sort_keys=True, default=str).encode()
    ).hexdigest()


class SemanticCache:
    """Similarity-keyed response reuse. Sits BEHIND the exact-match
    cache: exact handles byte-identical repeats at zero risk; this
    tier catches near-duplicates at an operator-set threshold.

    `embed_fn` is pluggable: default is the lexical 3-gram embedder;
    an org wanting true semantic matching wires a local neural
    embedder here (their hardware, their call)."""

    def __init__(self, *, threshold: float = 0.97,
                 ttl_s: float = 300.0, max_entries: int = 1000,
                 cache_all: bool = False,
                 embed_fn: Optional[Callable[[str], List[float]]] = None,
                 backend_name: Optional[str] = None,
                 clock: Callable[[], float] = time.time):
        if not (0.5 <= threshold <= 1.0):
            raise ValueError("threshold must be in [0.5, 1.0]")
        self.threshold = float(threshold)
        self.ttl_s = float(ttl_s)
        self.max_entries = int(max_entries)
        self.cache_all = cache_all
        self.embed_fn = embed_fn or _ngram_embed
        self.backend_name = backend_name or (
            "lexical-3gram" if embed_fn is None else "custom")
        self._clock = clock
        self._lock = threading.Lock()
        # meta_key -> list of (expires, vec, resp, billed_usd)
        self._buckets: Dict[str, List[Tuple[float, List[float],
                                            Dict[str, Any], float]]] = {}
        self._count = 0
        self.hits = 0
        self.misses = 0

    def _cacheable(self, body: Dict[str, Any]) -> bool:
        if self.ttl_s <= 0 or body.get("stream"):
            return False
        if self.cache_all:
            return True
        temp = body.get("temperature")
        return temp == 0 or temp == 0.0

    def get(self, body: Dict[str, Any]
            ) -> Optional[Tuple[Dict[str, Any], float, float]]:
        """(response, billed_usd_counterfactual, similarity) or None."""
        if not self._cacheable(body):
            return None
        vec = self.embed_fn(_messages_text(body))
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(_meta_key(body), [])
            best: Optional[Tuple[float, Dict[str, Any], float]] = None
            for expires, evec, resp, billed in bucket:
                if now > expires:
                    continue
                sim = _cosine(vec, evec)
                if sim >= self.threshold and (
                        best is None or sim > best[0]):
                    best = (sim, resp, billed)
            if best is None:
                self.misses += 1
                return None
            self.hits += 1
            return best[1], best[2], best[0]

    def put(self, body: Dict[str, Any], resp: Dict[str, Any],
            billed_usd: float) -> None:
        if not self._cacheable(body):
            return
        vec = self.embed_fn(_messages_text(body))
        now = self._clock()
        with self._lock:
            bucket = self._buckets.setdefault(_meta_key(body), [])
            bucket.append((now + self.ttl_s, vec, resp,
                           float(billed_usd)))
            self._count += 1
            if self._count > self.max_entries:
                # Cheap pressure valve: drop expired everywhere, then
                # oldest-first in the biggest bucket.
                for k in list(self._buckets):
                    kept = [e for e in self._buckets[k]
                            if e[0] > now]
                    self._count -= len(self._buckets[k]) - len(kept)
                    if kept:
                        self._buckets[k] = kept
                    else:
                        del self._buckets[k]
                while self._count > self.max_entries and self._buckets:
                    biggest = max(self._buckets,
                                  key=lambda k: len(self._buckets[k]))
                    self._buckets[biggest].pop(0)
                    self._count -= 1
                    if not self._buckets[biggest]:
                        del self._buckets[biggest]


# ---------------------------------------------------------------------------
# HG13h — prompt compression (every transform is operator opt-in)
# ---------------------------------------------------------------------------

class PromptCompressor:
    """Deterministic, itemised input-token reduction.

    Transforms (ALL default-off; each changes the prompt the model
    sees, which is the operator's call, not ours):

      * ``dedup_exact``: drop messages whose (role, content) is a
        byte-exact duplicate of an earlier message — the agent-retry
        pathology where the same tool output or instruction gets
        re-appended every loop iteration.
      * ``collapse_whitespace``: runs of spaces/newlines → single
        space inside message text.
      * ``max_input_tokens``: history pruning to a budget — keeps
        system message(s) and the most recent ``keep_last`` messages,
        dropping oldest middle turns until the chars/4 estimate fits.

    A pluggable ``compress_fn(text) -> text`` slot exists for a local
    LLMLingua-class model when the org runs one — same contract, their
    hardware, itemised like every other transform.

    ``compress(body)`` returns (new_body, report). The report lists
    what was applied and the ESTIMATED tokens removed — estimated is
    the honest word because the counterfactual upstream bill was never
    incurred."""

    def __init__(self, *, dedup_exact: bool = False,
                 collapse_whitespace: bool = False,
                 max_input_tokens: int = 0, keep_last: int = 4,
                 compress_fn: Optional[Callable[[str], str]] = None):
        self.dedup_exact = dedup_exact
        self.collapse_whitespace = collapse_whitespace
        self.max_input_tokens = int(max_input_tokens)
        self.keep_last = max(1, int(keep_last))
        self.compress_fn = compress_fn

    @property
    def enabled(self) -> bool:
        return bool(self.dedup_exact or self.collapse_whitespace
                    or self.max_input_tokens > 0
                    or self.compress_fn is not None)

    @staticmethod
    def _est_tokens(body: Dict[str, Any]) -> int:
        return max(1, len(_messages_text(body)) // 4)

    def compress(self, body: Dict[str, Any]
                 ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not self.enabled or not isinstance(
                body.get("messages"), list):
            return body, {"applied": [], "tokens_removed_est": 0}
        before = self._est_tokens(body)
        out = dict(body)
        msgs = [dict(m) for m in out["messages"]]
        applied: List[str] = []

        if self.dedup_exact:
            seen: set = set()
            kept = []
            for m in msgs:
                sig = (str(m.get("role")),
                       json.dumps(m.get("content"), sort_keys=True,
                                  default=str))
                if sig in seen:
                    continue
                seen.add(sig)
                kept.append(m)
            if len(kept) < len(msgs):
                applied.append(f"dedup_exact(-{len(msgs) - len(kept)})")
            msgs = kept

        if self.collapse_whitespace or self.compress_fn is not None:
            changed = False
            for m in msgs:
                c = m.get("content")
                if not isinstance(c, str):
                    continue
                new = c
                if self.collapse_whitespace:
                    new = " ".join(new.split())
                if self.compress_fn is not None:
                    new = self.compress_fn(new)
                if new != c:
                    changed = True
                    m["content"] = new
            if changed:
                if self.collapse_whitespace:
                    applied.append("collapse_whitespace")
                if self.compress_fn is not None:
                    applied.append("compress_fn")

        if self.max_input_tokens > 0:
            probe = dict(out, messages=msgs)
            if self._est_tokens(probe) > self.max_input_tokens:
                system = [m for m in msgs
                          if m.get("role") == "system"]
                rest = [m for m in msgs if m.get("role") != "system"]
                dropped = 0
                while (len(rest) > self.keep_last
                       and self._est_tokens(
                           dict(out, messages=system + rest))
                       > self.max_input_tokens):
                    rest.pop(0)
                    dropped += 1
                msgs = system + rest
                if dropped:
                    applied.append(f"prune_history(-{dropped})")

        out["messages"] = msgs
        after = self._est_tokens(out)
        return out, {"applied": applied,
                     "tokens_removed_est": max(0, before - after)}
