"""Tests for A15: Anycast+CRDT Edge Cache."""

import base64
import hashlib
import time

import pytest

from core.edge_cache import (
    CacheEntry,
    LocalEdgeCache,
    ReplicaProbe,
    anycast_pick,
    cache_key,
    make_entry,
    merge,
)
from core.tokenomics import Wallet


def _model() -> bytes:
    return hashlib.sha256(b"filum-127m").digest()


def _entry(*, signer: Wallet, model=None, inp=b"in", out=b"out",
           ttl=3600, ts_ns=None) -> CacheEntry:
    return make_entry(
        model_hash=model or _model(),
        input_bytes=inp,
        output_bytes=out,
        provider=signer,
        ttl_seconds=ttl,
        produced_at_ns=ts_ns,
    )


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def test_cache_key_is_deterministic_for_same_input():
    m = _model()
    assert cache_key(m, b"hello") == cache_key(m, b"hello")


def test_cache_key_differs_on_different_input():
    m = _model()
    assert cache_key(m, b"hello") != cache_key(m, b"hi")


def test_cache_key_differs_on_different_model():
    a = hashlib.sha256(b"ma").digest()
    b = hashlib.sha256(b"mb").digest()
    assert cache_key(a, b"x") != cache_key(b, b"x")


def test_cache_key_rejects_wrong_model_hash_size():
    with pytest.raises(ValueError):
        cache_key(b"too short", b"x")


# ---------------------------------------------------------------------------
# Entry signing
# ---------------------------------------------------------------------------


def test_entry_signed_and_verifies():
    w = Wallet()
    e = _entry(signer=w)
    assert e.verify() is True


def test_entry_with_tampered_output_fails_verify():
    w = Wallet()
    e = _entry(signer=w)
    e.output_b64 = base64.b64encode(b"DIFFERENT").decode()
    assert e.verify() is False


# ---------------------------------------------------------------------------
# CRDT merge
# ---------------------------------------------------------------------------


def test_merge_keeps_newer_timestamp():
    w = Wallet()
    now = time.time_ns()
    e_old = _entry(signer=w, ts_ns=now - 1000)
    e_new = _entry(signer=w, ts_ns=now)
    out = merge(e_old, e_new)
    assert out.produced_at_ns == now
    out2 = merge(e_new, e_old)        # symmetric
    assert out2.produced_at_ns == now


def test_merge_tiebreaks_on_signature_when_timestamps_equal():
    w = Wallet()
    ts = time.time_ns()
    a = _entry(signer=w, ts_ns=ts)
    # Force same key but different sig by re-signing a different ts
    # then forcing ts back.
    b = _entry(signer=w, ts_ns=ts + 1)
    b.produced_at_ns = ts
    out = merge(a, b)
    # Determinism: order independence.
    out2 = merge(b, a)
    assert out.provider_sig == out2.provider_sig


def test_merge_rejects_different_keys():
    w = Wallet()
    a = _entry(signer=w, inp=b"AAA")
    b = _entry(signer=w, inp=b"BBB")
    with pytest.raises(ValueError):
        merge(a, b)


# ---------------------------------------------------------------------------
# LocalEdgeCache
# ---------------------------------------------------------------------------


def test_put_and_lookup_round_trip():
    w = Wallet()
    cache = LocalEdgeCache()
    e = _entry(signer=w)
    assert cache.put(e) is True
    got = cache.lookup(e.key)
    assert got is not None
    assert got.provider_id == w.address
    assert cache.cache_hits == 1


def test_lookup_miss_increments_counter():
    cache = LocalEdgeCache()
    assert cache.lookup("nonexistent") is None
    assert cache.cache_misses == 1


def test_invalid_signature_rejected_on_put():
    w = Wallet()
    cache = LocalEdgeCache()
    e = _entry(signer=w)
    e.provider_sig = "AAAA"   # forged
    assert cache.put(e) is False
    assert cache.rejected_invalid == 1


def test_expired_entry_rejected_on_put():
    w = Wallet()
    cache = LocalEdgeCache()
    long_ago = time.time_ns() - 10 * 86_400 * 1_000_000_000
    e = _entry(signer=w, ts_ns=long_ago, ttl=1)
    assert cache.put(e) is False
    assert cache.rejected_expired == 1


def test_gc_drops_expired_entries():
    w = Wallet()
    cache = LocalEdgeCache()
    # Insert a fresh entry first, then mutate its timestamp far back.
    e = _entry(signer=w, ttl=86400)
    cache.put(e)
    e.produced_at_ns = time.time_ns() - 30 * 86_400 * 1_000_000_000
    n = cache.gc_expired()
    assert n == 1
    assert cache.size() == 0


# ---------------------------------------------------------------------------
# Anycast
# ---------------------------------------------------------------------------


def test_anycast_picks_lowest_rtt_replica_holding_key():
    probes = [
        ReplicaProbe("far", rtt_ms=200.0, keys={"k1"}),
        ReplicaProbe("near", rtt_ms=5.0, keys={"k1"}),
        ReplicaProbe("medium", rtt_ms=50.0, keys={"k1"}),
    ]
    assert anycast_pick(probes, "k1") == "near"


def test_anycast_returns_none_when_no_replica_has_key():
    probes = [
        ReplicaProbe("a", rtt_ms=10.0, keys={"otherkey"}),
        ReplicaProbe("b", rtt_ms=20.0, keys={"otherkey"}),
    ]
    assert anycast_pick(probes, "missing") is None


def test_two_caches_converge_via_crdt():
    """Two nodes hold the same key but learned it in different orders;
    after merging both inbound entries, both nodes show the same
    winning entry (CRDT convergence)."""
    w = Wallet()
    now = time.time_ns()
    e_old = _entry(signer=w, ts_ns=now - 1000)
    e_new = _entry(signer=w, ts_ns=now)
    cache_a, cache_b = LocalEdgeCache(), LocalEdgeCache()
    cache_a.put(e_old); cache_a.put(e_new)
    cache_b.put(e_new); cache_b.put(e_old)
    a = cache_a.lookup(e_old.key)
    b = cache_b.lookup(e_old.key)
    assert a is not None and b is not None
    assert a.produced_at_ns == b.produced_at_ns == now
