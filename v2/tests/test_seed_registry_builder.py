"""G1 — TOFU seed registry + quorum promotion."""

from __future__ import annotations

import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.seed_registry_builder import (  # noqa: E402
    QUORUM_MIN_DISTINCT_SIGNERS,
    TOFU_BOOTSTRAP_FLAG,
    build_initial_registry,
    build_tofu_seed_record,
    co_sign_seed_record,
    distinct_signers,
    is_tofu_only,
    promote_to_quorum,
)
from core.tokenomics import Wallet  # noqa: E402


def test_tofu_record_carries_one_self_signature():
    w = Wallet()
    r = build_tofu_seed_record(host="1.2.3.4", port=8100, wallet=w)
    assert len(r.quorum_signatures) == 1
    assert r.quorum_signatures[0]["label"] == TOFU_BOOTSTRAP_FLAG
    assert r.pubkey_pem == w.export_keys()["public"]


def test_tofu_record_is_flagged_tofu_only():
    w = Wallet()
    r = build_tofu_seed_record(host="1.2.3.4", port=8100, wallet=w)
    assert is_tofu_only(r) is True
    assert distinct_signers(r) == [w.export_keys()["public"]]


def test_co_sign_is_idempotent():
    w = Wallet()
    r = build_tofu_seed_record(host="1.2.3.4", port=8100, wallet=w)
    co_sign_seed_record(record=r, wallet=w)
    co_sign_seed_record(record=r, wallet=w)
    # Still only one distinct signer — the same wallet doesn't double
    # count.
    assert len(distinct_signers(r)) == 1


def test_quorum_promotion_clears_tofu_flag():
    w0 = Wallet()
    w1, w2 = Wallet(), Wallet()
    reg = build_initial_registry(host="1.2.3.4", port=8100, wallet=w0)
    assert is_tofu_only(reg.records[0]) is True
    promote_to_quorum(reg, [w1, w2])
    # Three distinct signers >= QUORUM_MIN_DISTINCT_SIGNERS.
    assert len(distinct_signers(reg.records[0])) >= QUORUM_MIN_DISTINCT_SIGNERS
    assert is_tofu_only(reg.records[0]) is False


def test_tampered_signature_does_not_count_toward_quorum():
    w = Wallet()
    r = build_tofu_seed_record(host="1.2.3.4", port=8100, wallet=w)
    # Inject a fake sig under a different pubkey but with random
    # bytes. distinct_signers() must reject it.
    fake = Wallet()
    r.quorum_signatures.append({
        "pubkey": fake.export_keys()["public"],
        "value": "AAAA",      # garbage
        "label": "fake",
    })
    distinct = distinct_signers(r)
    assert len(distinct) == 1
    assert fake.export_keys()["public"] not in distinct


def test_registry_canonical_round_trip_via_dict():
    """Registry written to JSON + reloaded preserves signatures."""
    w0, w1 = Wallet(), Wallet()
    reg = build_initial_registry(host="1.2.3.4", port=8100, wallet=w0)
    promote_to_quorum(reg, [w1])
    d = reg.to_dict()
    from core.anchored_bootstrap import SeedRegistry
    reg2 = SeedRegistry.from_dict(d)
    assert len(reg2.records) == 1
    # Both signers still verify after the round trip.
    assert len(distinct_signers(reg2.records[0])) == 2
