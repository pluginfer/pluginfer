"""Tests for A10: Bitcoin-anchored bootstrap (no central VPS)."""

import json
from pathlib import Path

import pytest

from core.anchored_bootstrap import (
    SeedRecord,
    SeedRegistry,
    filter_quorum_signed,
    make_bootstrap_plan,
    permute_seeds,
)
from core.bitcoin_anchor import (
    BitcoinAnchor,
    BitcoinAnchorError,
    get_bitcoin_anchor,
)
from core.tokenomics import Wallet


# ---------------------------------------------------------------------------
# Bitcoin anchor (network-free via fetcher injection)
# ---------------------------------------------------------------------------


def _stub_sources():
    return [{"name": f"src{i}", "url": "ignored", "kind": "text"}
            for i in range(3)]


def test_anchor_majority_agreement():
    H = "a" * 64

    def fetcher(s, t):
        return {"src0": H, "src1": H, "src2": "b" * 64}[s["name"]]

    a = get_bitcoin_anchor(sources=_stub_sources(), fetcher=fetcher,
                           min_agreement=2)
    assert a.block_hash == H
    assert a.agreement == 2
    assert "src0" in a.sources_agreed and "src1" in a.sources_agreed


def test_anchor_below_quorum_raises():
    def fetcher(s, t):
        return {"src0": "a" * 64, "src1": "b" * 64, "src2": "c" * 64}[s["name"]]
    with pytest.raises(BitcoinAnchorError, match="agreed"):
        get_bitcoin_anchor(sources=_stub_sources(), fetcher=fetcher,
                           min_agreement=2)


def test_anchor_all_sources_fail_raises():
    def fetcher(s, t):
        return None
    with pytest.raises(BitcoinAnchorError, match="responded"):
        get_bitcoin_anchor(sources=_stub_sources(), fetcher=fetcher,
                           min_agreement=1)


def test_anchor_seed_bytes_is_32():
    a = BitcoinAnchor(block_hash="d" * 64, agreement=3,
                      sources_queried=3, sources_agreed=[])
    assert len(a.as_seed_bytes()) == 32


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------


def _sign_record(rec: SeedRecord, signer: Wallet) -> None:
    msg = rec.canonical()
    pub = signer.export_keys()["public"]
    rec.quorum_signatures.append({
        "pubkey": pub,
        "value": signer.sign(msg),
    })


def _make_records(n: int = 5):
    return [
        SeedRecord(host=f"seed{i}", port=9000 + i,
                   pubkey_pem=f"<pk-{i}>", region=f"r{i}")
        for i in range(n)
    ]


def test_permutation_is_deterministic_under_same_anchor():
    rs = _make_records(5)
    a = BitcoinAnchor(block_hash="1" * 64, agreement=3,
                      sources_queried=3, sources_agreed=[])
    p1 = permute_seeds(rs, a)
    p2 = permute_seeds(list(reversed(rs)), a)   # input order shouldn't matter
    assert [r.host for r in p1] == [r.host for r in p2]


def test_different_anchors_produce_different_permutations():
    rs = _make_records(8)
    a1 = BitcoinAnchor(block_hash="1" * 64, agreement=3,
                       sources_queried=3, sources_agreed=[])
    a2 = BitcoinAnchor(block_hash="2" * 64, agreement=3,
                       sources_queried=3, sources_agreed=[])
    p1 = [r.host for r in permute_seeds(rs, a1)]
    p2 = [r.host for r in permute_seeds(rs, a2)]
    assert p1 != p2          # vanishingly unlikely to collide on 8 items


def test_filter_keeps_only_signed_records():
    signer = Wallet()
    signed = SeedRecord(host="ok", port=9000, pubkey_pem="<pk>")
    _sign_record(signed, signer)
    unsigned = SeedRecord(host="forged", port=9000, pubkey_pem="<pk>")
    out = filter_quorum_signed([signed, unsigned], min_signatures=1)
    assert [r.host for r in out] == ["ok"]


def test_filter_rejects_signature_over_tampered_record():
    signer = Wallet()
    rec = SeedRecord(host="ok", port=9000, pubkey_pem="<pk>")
    _sign_record(rec, signer)
    # Tamper AFTER signing.
    rec.host = "evil"
    out = filter_quorum_signed([rec], min_signatures=1)
    assert out == []


# ---------------------------------------------------------------------------
# End-to-end plan
# ---------------------------------------------------------------------------


def test_bootstrap_plan_returns_at_most_max_seeds():
    signer = Wallet()
    recs = _make_records(20)
    for r in recs:
        _sign_record(r, signer)
    reg = SeedRegistry(records=recs, epoch_btc_height=900_000)
    a = BitcoinAnchor(block_hash="3" * 64, agreement=3,
                      sources_queried=3, sources_agreed=[])
    plan = make_bootstrap_plan(reg, anchor=a, max_seeds=5)
    assert len(plan.seeds) == 5
    assert plan.anchor.block_hash == "3" * 64


def test_bootstrap_plan_drops_unsigned_records():
    signer = Wallet()
    signed_rec = SeedRecord(host="legit", port=9000, pubkey_pem="<pk>")
    _sign_record(signed_rec, signer)
    unsigned_rec = SeedRecord(host="forged", port=9001, pubkey_pem="<pk>")
    reg = SeedRegistry(records=[signed_rec, unsigned_rec])
    a = BitcoinAnchor(block_hash="4" * 64, agreement=3,
                      sources_queried=3, sources_agreed=[])
    plan = make_bootstrap_plan(reg, anchor=a, min_quorum_sigs=1)
    assert [r.host for r in plan.seeds] == ["legit"]


def test_bootstrap_plan_no_network_when_anchor_injected(monkeypatch):
    """If the caller passes an explicit anchor, get_bitcoin_anchor
    must NOT be invoked -- the test asserts no network access."""
    from core import anchored_bootstrap as ab
    called = {"hit": False}

    def trap(*a, **kw):
        called["hit"] = True
        raise AssertionError("network call should not happen here")

    monkeypatch.setattr(ab, "get_bitcoin_anchor", trap)
    a = BitcoinAnchor(block_hash="5" * 64, agreement=3,
                      sources_queried=3, sources_agreed=[])
    reg = SeedRegistry(records=[])
    plan = ab.make_bootstrap_plan(reg, anchor=a)
    assert plan.anchor.block_hash == "5" * 64
    assert called["hit"] is False


def test_seed_registry_round_trip(tmp_path: Path):
    rec = SeedRecord(host="h", port=1, pubkey_pem="p", region="x",
                     registered_at_btc_height=42)
    reg = SeedRegistry(records=[rec], epoch_btc_height=900_000)
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(reg.to_dict()), encoding="utf-8")
    loaded = SeedRegistry.from_dict(json.loads(p.read_text(encoding="utf-8")))
    assert loaded.epoch_btc_height == 900_000
    assert loaded.records[0].host == "h"
