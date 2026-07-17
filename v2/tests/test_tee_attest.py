"""TEE attestation pipeline — vendor-agnostic verifier registry,
freshness window, enclave-measurement match.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from core.tee_attest import (
    AttestationQuote,
    DEFAULT_FRESHNESS_S,
    EnclaveMeasurement,
    LocalProofVerifier,
    TEEAttestationCache,
    can_bid_on_confidential_job,
)


def _local_quote_bytes(measurement_hex: str, pubkey_pem: str,
                       *, issued_at_unix: float = None) -> bytes:
    return json.dumps({
        "measurement_hex": measurement_hex,
        "enclave_pubkey_pem": pubkey_pem,
        "issued_at_unix": issued_at_unix or time.time(),
    }).encode("utf-8")


def test_local_proof_verifier_parses_well_formed_quote():
    v = LocalProofVerifier()
    bs = _local_quote_bytes("abc123", "pub-pem")
    q = v.verify(bs)
    assert q is not None
    assert q.enclave_measurement_hex == "abc123"
    assert q.enclave_pubkey_pem == "pub-pem"


def test_local_proof_verifier_rejects_malformed():
    v = LocalProofVerifier()
    assert v.verify(b"not-json") is None
    assert v.verify(b'{"missing": "fields"}') is None


def test_cache_returns_fresh_quote_within_window():
    cache = TEEAttestationCache()
    cache.register_verifier(LocalProofVerifier())
    cache.submit_quote(
        provider_pubkey="prov-1", vendor="local-proof",
        raw_quote_bytes=_local_quote_bytes("m1", "prov-1-enc"),
    )
    assert cache.get_fresh_quote(provider_pubkey="prov-1") is not None


def test_cache_drops_stale_quote():
    cache = TEEAttestationCache()
    cache.register_verifier(LocalProofVerifier())
    # Issued 10000 seconds ago — way past local-proof's 300s TTL.
    bs = _local_quote_bytes("m1", "enc", issued_at_unix=time.time() - 10_000)
    cache.submit_quote(
        provider_pubkey="prov-1", vendor="local-proof", raw_quote_bytes=bs,
    )
    assert cache.get_fresh_quote(provider_pubkey="prov-1") is None


def test_measurement_match_required_for_attestation():
    cache = TEEAttestationCache()
    cache.register_verifier(LocalProofVerifier())
    cache.submit_quote(
        provider_pubkey="prov-1", vendor="local-proof",
        raw_quote_bytes=_local_quote_bytes("REAL-MEASUREMENT", "enc"),
    )
    # Buyer wants a DIFFERENT measurement → not attested for this job.
    req = EnclaveMeasurement(
        vendor="local-proof", measurement_hex="WRONG-MEASUREMENT",
    )
    assert can_bid_on_confidential_job(
        provider_pubkey="prov-1", cache=cache, required_measurement=req,
    ) is False
    # Buyer's measurement matches → attested.
    req_ok = EnclaveMeasurement(
        vendor="local-proof", measurement_hex="REAL-MEASUREMENT",
    )
    assert can_bid_on_confidential_job(
        provider_pubkey="prov-1", cache=cache, required_measurement=req_ok,
    ) is True


def test_unknown_vendor_rejected():
    cache = TEEAttestationCache()
    # No verifiers registered.
    result = cache.submit_quote(
        provider_pubkey="prov-1", vendor="local-proof",
        raw_quote_bytes=_local_quote_bytes("m", "enc"),
    )
    assert result is None


def test_no_attestation_means_not_attested():
    cache = TEEAttestationCache()
    cache.register_verifier(LocalProofVerifier())
    # No quote submitted for prov-X.
    assert can_bid_on_confidential_job(
        provider_pubkey="prov-X", cache=cache,
    ) is False


def test_enclave_measurement_from_bytes_is_deterministic():
    m1 = EnclaveMeasurement.from_bytes("sgx", b"the same binary")
    m2 = EnclaveMeasurement.from_bytes("sgx", b"the same binary")
    assert m1.measurement_hex == m2.measurement_hex
    m3 = EnclaveMeasurement.from_bytes("sgx", b"a different binary")
    assert m3.measurement_hex != m1.measurement_hex
