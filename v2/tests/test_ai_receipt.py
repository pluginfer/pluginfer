"""Tests for the Universal AI Receipt Standard (PNIS-Receipt v1)."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from core.ai_receipt import (
    SCHEMA_VERSION,
    AIReceipt,
    ProviderRef,
    make_receipt,
)
from core.tokenomics import Wallet


def _signer() -> tuple[Wallet, ProviderRef]:
    w = Wallet()
    pub = w.export_keys()["public"]
    return w, ProviderRef(id=w.address, pubkey=pub, kind="mesh")


def test_receipt_roundtrip_signs_and_verifies():
    w, prov = _signer()
    r = make_receipt(
        job_id="job-1",
        model_id="filum-127m",
        model_hash="a" * 64,
        provider=prov,
        input_bytes=b"summarise this",
        output_bytes=b"a short summary",
        cost_plg=Decimal("0.001234"),
        cost_usd_estimate=Decimal("0.00008"),
        energy_mj=Decimal("0.42"),
        latency_ms=813,
        signer=w,
    )
    assert r.verify() is True
    assert r.body.schema == SCHEMA_VERSION


def test_receipt_tampered_output_hash_fails_verify():
    w, prov = _signer()
    r = make_receipt(
        job_id="job-2", model_id="m", model_hash="b" * 64, provider=prov,
        input_bytes=b"x", output_bytes=b"y",
        cost_plg=Decimal("0"), signer=w,
    )
    r.body.output_hash = "0" * 64
    assert r.verify() is False


def test_receipt_signature_under_wrong_key_rejected():
    w, prov = _signer()
    r = make_receipt(
        job_id="job-3", model_id="m", model_hash="c" * 64, provider=prov,
        input_bytes=b"x", output_bytes=b"y",
        cost_plg=Decimal("0"), signer=w,
    )
    other = Wallet()
    r.signature_pubkey = other.export_keys()["public"]
    assert r.verify() is False


def test_receipt_provider_pubkey_must_equal_signer():
    w, _prov = _signer()
    other = Wallet()
    bad_provider = ProviderRef(
        id=other.address,
        pubkey=other.export_keys()["public"],
        kind="mesh",
    )
    with pytest.raises(ValueError, match="pubkey"):
        make_receipt(
            job_id="job-4", model_id="m", model_hash="d" * 64,
            provider=bad_provider,
            input_bytes=b"x", output_bytes=b"y",
            cost_plg=Decimal("0"),
            signer=w,
        )


def test_receipt_to_from_json_preserves_signature():
    w, prov = _signer()
    r = make_receipt(
        job_id="job-5", model_id="m", model_hash="e" * 64, provider=prov,
        input_bytes=b"in", output_bytes=b"out",
        cost_plg=Decimal("0.5"), signer=w,
    )
    blob = r.to_json()
    parsed = json.loads(blob)
    assert parsed["schema"] == SCHEMA_VERSION
    assert parsed["signature"]["alg"] == "ecdsa-secp256k1-sha256"
    r2 = AIReceipt.from_json(blob)
    assert r2.verify() is True
    assert r2.body.job_id == "job-5"


def test_receipt_to_file_from_file(tmp_path: Path):
    w, prov = _signer()
    r = make_receipt(
        job_id="job-6", model_id="m", model_hash="f" * 64, provider=prov,
        input_bytes=b"abc", output_bytes=b"xyz",
        cost_plg=Decimal("1.25"), signer=w,
    )
    p = tmp_path / "receipt.json"
    r.to_file(p)
    r2 = AIReceipt.from_file(p)
    assert r2.verify() is True
    assert r2.body.cost.plg == "1.25"


def test_receipt_canonical_json_is_deterministic():
    w, prov = _signer()
    r = make_receipt(
        job_id="job-7", model_id="m", model_hash="g" * 64, provider=prov,
        input_bytes=b"hello", output_bytes=b"world",
        cost_plg=Decimal("2"), signer=w, timestamp_ns=1700000000000000000,
    )
    a = r.body.canonical_json()
    b = r.body.canonical_json()
    assert a == b
    # Re-signing the same body must succeed against same key.
    sig = w.sign(a)
    assert Wallet.verify(prov.pubkey, a, sig)


def test_receipt_input_hash_isolated_from_output():
    w, prov = _signer()
    r1 = make_receipt(
        job_id="j", model_id="m", model_hash="h" * 64, provider=prov,
        input_bytes=b"AAA", output_bytes=b"ZZZ",
        cost_plg=Decimal("0"), signer=w,
    )
    r2 = make_receipt(
        job_id="j", model_id="m", model_hash="h" * 64, provider=prov,
        input_bytes=b"BBB", output_bytes=b"ZZZ",
        cost_plg=Decimal("0"), signer=w,
    )
    assert r1.body.input_hash != r2.body.input_hash
    assert r1.body.output_hash == r2.body.output_hash
