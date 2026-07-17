"""Idempotency-store tests for the payments gateway.

The bug guarded against here: a Stripe charge call retried after a
network blip (or after a node crash between Stripe accepting and our
ledger committing) MUST NOT produce a second charge. Stripe's
Idempotency-Key header gives upstream protection; this store gives
local fast-path protection so we never even retry the network call
when we already have a recorded result.
"""

from __future__ import annotations

import time

import pytest

from core.payments import IdempotencyStore, PaymentResult


def test_get_returns_none_when_unset():
    store = IdempotencyStore()
    assert store.get("cust_a", "k1") is None


def test_put_then_get_returns_replay():
    store = IdempotencyStore()
    original = PaymentResult(success=True, transaction_id="pi_1")
    store.put("cust_a", "k1", original)

    cached = store.get("cust_a", "k1")
    assert cached is not None
    assert cached.success is True
    assert cached.transaction_id == "pi_1"
    # The replay flag must be set so receipt logs can distinguish a
    # cached replay from an original — important for double-billing
    # forensics if a customer ever disputes.
    assert cached.idempotent_replay is True


def test_keys_are_scoped_per_customer():
    store = IdempotencyStore()
    store.put("cust_a", "shared_key", PaymentResult(True, "pi_a"))
    store.put("cust_b", "shared_key", PaymentResult(True, "pi_b"))

    assert store.get("cust_a", "shared_key").transaction_id == "pi_a"
    assert store.get("cust_b", "shared_key").transaction_id == "pi_b"


def test_ttl_evicts_expired_entries():
    store = IdempotencyStore(ttl_s=0)  # immediate expiry
    store.put("cust_a", "k1", PaymentResult(True, "pi_1"))
    # Sleep one tick so monotonic clock has moved past expiry.
    time.sleep(0.01)
    assert store.get("cust_a", "k1") is None
    assert len(store) == 0


def test_failed_results_are_also_cached():
    """A retry after a known failure should not silently retry. The
    caller is supposed to use a NEW idempotency key for a deliberate
    retry of a failed charge — caching the failure protects against
    accidental loops."""
    store = IdempotencyStore()
    store.put("cust_a", "k1", PaymentResult(False, None, "card_declined"))
    cached = store.get("cust_a", "k1")
    assert cached is not None
    assert cached.success is False
    assert cached.error == "card_declined"


def test_put_overwrites_existing_key():
    """Same key with same customer overwrites — so a retried call that
    succeeds AFTER we cached its initial failure can update the record.
    Final-write-wins matches Stripe's semantics: your idempotency key
    represents one logical operation, the latest authoritative result
    is what should be returned on subsequent reads in the TTL window."""
    store = IdempotencyStore()
    store.put("cust_a", "k1", PaymentResult(False, None, "transient"))
    store.put("cust_a", "k1", PaymentResult(True, "pi_final"))
    assert store.get("cust_a", "k1").transaction_id == "pi_final"
