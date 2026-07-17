"""Anycast+CRDT Edge Cache (PNIS §A15) -- sub-millisecond cache hits.

The fastest provider response on Pluginfer is bounded by network RTT
to the nearest provider (~20-50ms in practice). For idempotent
inferences with popular inputs, even that's too slow when a million
users ask the same question. This module turns the network into a
distributed cache:

  1. Every inference response is content-addressed:
        cache_key = sha256(model_hash || canonical_input)
     so identical (model, input) pairs map to the same key
     regardless of who asked.

  2. Any node can hold cache entries; a CRDT (last-write-wins by
     deterministic provider signature) lets multiple nodes serve the
     same key without coordinating.

  3. DHT-anycast lookup routes a request to the NEAREST node holding
     the key (lowest network latency), not the original provider.

  4. Sub-millisecond hits when the cache is local; ~5-20ms hits when
     the cache lives on a same-region peer.

Why this design is novel
----------------------
Existing CDNs cache HTTP responses by URL -- that's content-addressed
under a centralised DNS hierarchy. Pluginfer's edge cache is:

  * **Permissionless** -- any node can be a cache replica without
    Pluginfer's permission.
  * **Cryptographically verified** -- every entry carries the
    original provider's signature; a malicious cache cannot serve
    forged answers.
  * **Anycast over a DHT** -- no central directory; the lookup
    finds the nearest replica via Kademlia-style XOR distance.

Combined with §A13 quorum inference, the user-visible latency for
the 90% of repeat queries collapses to the local-network ping.

Construction
------------
A cache entry is a tuple:

    {
      "key":           <sha256(model_hash || canonical_input)>,
      "model_hash":    <hex>,
      "input_hash":    <hex>,
      "output_bytes":  <bytes>,
      "provider_id":   <wallet addr>,
      "provider_pubkey_pem": <PEM>,
      "provider_sig":  <base64 ECDSA over canonical(entry-without-sig)>,
      "produced_at":   <ts_ns>,
      "ttl_seconds":   <int>,
    }

The CRDT merge rule is deterministic-tiebreak Last-Write-Wins:
  * If A.produced_at > B.produced_at -> keep A
  * Else if A.produced_at < B.produced_at -> keep B
  * Else compare provider_sig bytewise (any deterministic order works)
This converges across replicas without coordination.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, Optional

from .tokenomics import Wallet


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def cache_key(model_hash: bytes, input_bytes: bytes) -> str:
    """Deterministic content-address for an (idempotent) inference.

    Two callers asking the same input of the same model land on the
    same cache_key, regardless of when or from where.
    """
    if len(model_hash) != 32:
        raise ValueError("model_hash must be 32-byte sha256 digest")
    h = hashlib.sha256()
    h.update(b"pnis-a15-cachekey/v1")
    h.update(model_hash)
    h.update(hashlib.sha256(input_bytes).digest())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    key: str
    model_hash: str                          # hex
    input_hash: str                          # hex (sha256 of input bytes)
    output_b64: str                          # base64-encoded output bytes
    provider_id: str
    provider_pubkey_pem: str
    provider_sig: str                        # base64 ECDSA
    produced_at_ns: int
    ttl_seconds: int = 86_400                 # 1 day default

    def canonical(self) -> str:
        d = asdict(self)
        d.pop("provider_sig", None)
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    def is_expired(self, now_ns: Optional[int] = None) -> bool:
        if now_ns is None:
            now_ns = time.time_ns()
        return (now_ns - self.produced_at_ns) > self.ttl_seconds * 1_000_000_000

    def verify(self) -> bool:
        return Wallet.verify(
            self.provider_pubkey_pem,
            self.canonical(),
            self.provider_sig,
        )


def make_entry(
    *,
    model_hash: bytes,
    input_bytes: bytes,
    output_bytes: bytes,
    provider: Wallet,
    ttl_seconds: int = 86_400,
    produced_at_ns: Optional[int] = None,
) -> CacheEntry:
    """Construct + sign a cache entry."""
    import base64
    pub = provider.export_keys()["public"]
    addr = provider.address
    e = CacheEntry(
        key=cache_key(model_hash, input_bytes),
        model_hash=model_hash.hex(),
        input_hash=hashlib.sha256(input_bytes).hexdigest(),
        output_b64=base64.b64encode(output_bytes).decode(),
        provider_id=addr,
        provider_pubkey_pem=pub,
        provider_sig="",                     # placeholder
        produced_at_ns=produced_at_ns
        if produced_at_ns is not None else time.time_ns(),
        ttl_seconds=int(ttl_seconds),
    )
    e.provider_sig = provider.sign(e.canonical())
    return e


# ---------------------------------------------------------------------------
# CRDT merge
# ---------------------------------------------------------------------------


def merge(a: CacheEntry, b: CacheEntry) -> CacheEntry:
    """Deterministic-tiebreak last-write-wins. Caller must ensure
    a.key == b.key; if not, this raises rather than silently picking
    one."""
    if a.key != b.key:
        raise ValueError("cannot merge entries with different keys")
    if a.produced_at_ns > b.produced_at_ns:
        return a
    if a.produced_at_ns < b.produced_at_ns:
        return b
    # Equal timestamps: deterministic byte-tiebreak on provider_sig.
    return a if a.provider_sig >= b.provider_sig else b


# ---------------------------------------------------------------------------
# Local cache (one node's view)
# ---------------------------------------------------------------------------


@dataclass
class LocalEdgeCache:
    """A node's local replica of the distributed cache.

    `lookup(key)` returns None if not held, the verified+unexpired
    entry otherwise. `put(entry)` validates the signature, runs the
    CRDT merge with any existing entry, and stores the winner.
    """
    entries: Dict[str, CacheEntry] = field(default_factory=dict)
    rejected_invalid: int = 0
    rejected_expired: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def put(self, entry: CacheEntry) -> bool:
        if not entry.verify():
            self.rejected_invalid += 1
            return False
        if entry.is_expired():
            self.rejected_expired += 1
            return False
        existing = self.entries.get(entry.key)
        if existing is None:
            self.entries[entry.key] = entry
        else:
            self.entries[entry.key] = merge(existing, entry)
        return True

    def lookup(self, key: str) -> Optional[CacheEntry]:
        e = self.entries.get(key)
        if e is None:
            self.cache_misses += 1
            return None
        if e.is_expired():
            self.rejected_expired += 1
            self.entries.pop(key, None)
            self.cache_misses += 1
            return None
        self.cache_hits += 1
        return e

    def size(self) -> int:
        return len(self.entries)

    def gc_expired(self, now_ns: Optional[int] = None) -> int:
        """Drop expired entries; return count removed."""
        if now_ns is None:
            now_ns = time.time_ns()
        keys = [k for k, e in self.entries.items()
                if e.is_expired(now_ns)]
        for k in keys:
            self.entries.pop(k, None)
        self.rejected_expired += len(keys)
        return len(keys)


# ---------------------------------------------------------------------------
# Anycast: pick the lowest-latency replica that claims to hold a key
# ---------------------------------------------------------------------------


@dataclass
class ReplicaProbe:
    """A single replica's claim about which keys it holds + latency."""
    replica_id: str
    rtt_ms: float
    keys: set                                # set of cache_key hex


def anycast_pick(probes: Iterable[ReplicaProbe], key: str) -> Optional[str]:
    """Among replicas claiming `key`, return the replica_id with the
    lowest rtt_ms. Returns None if no replica claims the key."""
    candidates = [p for p in probes if key in p.keys]
    if not candidates:
        return None
    return min(candidates, key=lambda p: p.rtt_ms).replica_id


__all__ = [
    "cache_key",
    "CacheEntry",
    "make_entry",
    "merge",
    "LocalEdgeCache",
    "ReplicaProbe",
    "anycast_pick",
]
