"""Bitcoin-anchored randomness for self-bootstrapping the mesh.

Pluginfer's launch checklist used to require renting + maintaining a
public-IP seed VPS. That single dependency was the difference between
"the network exists" and "the network is offline because the VPS got
DDOS'd". This module removes that dependency by anchoring bootstrap
randomness to a third-party blockchain (Bitcoin) that neither
Pluginfer nor any one operator controls.

How
---
1. Every node, on first launch, queries the latest Bitcoin block hash
   from N redundant public APIs.
2. The majority value across at least M of N sources is the canonical
   anchor for this epoch.
3. That hash is the deterministic seed used by `anchored_bootstrap.py`
   to permute the signed seed registry; nodes try the first K seeds
   in the permuted order.

Why Bitcoin specifically
------------------------
* Bitcoin's hash is the hardest-to-forge public randomness humanity
  has produced (~700 EH/s of computation backs each block).
* It updates every ~10 minutes, plenty fast for bootstrap freshness.
* It is publicly observable from every consumer device with no
  account, no API key, no rate-limit gate.
* It is NOT something Pluginfer controls -- so we cannot game which
  seeds get picked. That property is what makes the bootstrap
  trust-minimised.

This module deliberately depends ONLY on `urllib.request` so it
does NOT add a runtime dependency. All network access goes through
allowlisted hosts the user can override.

INVENTIONS §A10.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


# Public Bitcoin block-hash sources. Each must return EITHER plain
# 64-hex on a single line OR a JSON document with a known field.
DEFAULT_BTC_HASH_SOURCES: List[dict] = [
    {"name": "blockstream",
     "url": "https://blockstream.info/api/blocks/tip/hash",
     "kind": "text"},
    {"name": "mempool",
     "url": "https://mempool.space/api/blocks/tip/hash",
     "kind": "text"},
    {"name": "blockchain.info",
     "url": "https://blockchain.info/q/latesthash",
     "kind": "text"},
]


HEX64 = set("0123456789abcdefABCDEF")


def _looks_like_btc_hash(s: str) -> bool:
    s = s.strip()
    return len(s) == 64 and all(c in HEX64 for c in s)


def _fetch_one(source: dict, *, timeout: float) -> Optional[str]:
    """Return a normalised lowercase 64-hex hash, or None on failure."""
    try:
        req = urllib.request.Request(
            source["url"],
            headers={"User-Agent": "pluginfer-bitcoin-anchor/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace").strip()
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        logger.warning("btc_anchor: %s failed: %s", source["name"], e)
        return None
    except Exception as e:                                       # pragma: no cover
        logger.warning("btc_anchor: %s unexpected: %s", source["name"], e)
        return None
    kind = source.get("kind", "text")
    if kind == "text":
        if _looks_like_btc_hash(body):
            return body.lower()
        return None
    if kind == "json":
        try:
            j = json.loads(body)
        except Exception:
            return None
        # Best-effort field discovery for a hash.
        for key in ("hash", "blockhash", "best_block_hash", "tip_hash"):
            v = j.get(key)
            if isinstance(v, str) and _looks_like_btc_hash(v):
                return v.lower()
    return None


@dataclass
class BitcoinAnchor:
    """Latest Bitcoin block hash + sources that agreed."""
    block_hash: str                          # 64-hex lowercase
    agreement: int                           # how many sources agreed
    sources_queried: int                     # how many sources we asked
    sources_agreed: List[str] = field(default_factory=list)
    fetched_at: float = field(default_factory=time.time)

    def as_seed_bytes(self) -> bytes:
        """Convert hex to raw 32 bytes for use as a deterministic
        randomness seed. Callers do not need to know about hex."""
        return bytes.fromhex(self.block_hash)


def get_bitcoin_anchor(
    *,
    sources: Optional[Sequence[dict]] = None,
    min_agreement: int = 2,
    timeout: float = 5.0,
    fetcher=None,                            # injection point for tests
) -> BitcoinAnchor:
    """Fetch the latest Bitcoin block hash from N redundant public APIs.

    Returns the hash that AT LEAST `min_agreement` sources agree on. If
    no value reaches that floor, raises `BitcoinAnchorError`.

    `fetcher` lets tests inject a callable `(source, timeout) -> Optional[str]`
    so we don't hit the public network in CI.
    """
    sources = list(sources or DEFAULT_BTC_HASH_SOURCES)
    fetch = fetcher or (lambda s, t: _fetch_one(s, timeout=t))

    seen: Counter = Counter()
    sources_per_hash: dict[str, list[str]] = {}
    queried = 0
    for s in sources:
        queried += 1
        h = fetch(s, timeout)
        if not h:
            continue
        seen[h] += 1
        sources_per_hash.setdefault(h, []).append(s["name"])

    if not seen:
        raise BitcoinAnchorError(
            f"no Bitcoin hash source responded (queried {queried})"
        )
    most_common, n = seen.most_common(1)[0]
    if n < min_agreement:
        raise BitcoinAnchorError(
            f"no hash agreed by >= {min_agreement} sources "
            f"(best: {n} for {most_common})"
        )
    return BitcoinAnchor(
        block_hash=most_common,
        agreement=n,
        sources_queried=queried,
        sources_agreed=sources_per_hash[most_common],
    )


# ---------------------------------------------------------------------------
# On-disk anchor cache (saves the daily-ish public network round-trip)
# ---------------------------------------------------------------------------


_ANCHOR_CACHE_PATH = os.path.join(
    os.path.expanduser("~"), ".pluginfer", "bitcoin_anchor.json"
)


def cache_anchor(anchor: BitcoinAnchor, *,
                 path: str = _ANCHOR_CACHE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "block_hash": anchor.block_hash,
            "agreement": anchor.agreement,
            "sources_queried": anchor.sources_queried,
            "sources_agreed": anchor.sources_agreed,
            "fetched_at": anchor.fetched_at,
        }, f)


def load_cached_anchor(*, path: str = _ANCHOR_CACHE_PATH,
                       max_age_seconds: float = 3600.0
                       ) -> Optional[BitcoinAnchor]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    if (time.time() - float(d.get("fetched_at", 0))) > max_age_seconds:
        return None
    if not _looks_like_btc_hash(d.get("block_hash", "")):
        return None
    return BitcoinAnchor(
        block_hash=d["block_hash"].lower(),
        agreement=int(d.get("agreement", 1)),
        sources_queried=int(d.get("sources_queried", 1)),
        sources_agreed=list(d.get("sources_agreed", [])),
        fetched_at=float(d["fetched_at"]),
    )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BitcoinAnchorError(RuntimeError):
    """Could not establish a trustworthy Bitcoin block-hash anchor."""


__all__ = [
    "BitcoinAnchor",
    "BitcoinAnchorError",
    "DEFAULT_BTC_HASH_SOURCES",
    "get_bitcoin_anchor",
    "cache_anchor",
    "load_cached_anchor",
]
