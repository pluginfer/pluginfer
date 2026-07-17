"""Hardware-attestation challenge — bound the lying.

Honest peer signs and the timing fits → pass. Liar's hash mismatches
OR timing is impossibly fast/slow → fail. Replay (stale nonce) →
fail. TTL expiry → fail.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from core.hw_attest import (
    AttestChallenge,
    AttestResponse,
    TIMING_BANDS_MS_BY_CLASS,
    issue_challenge,
    respond_to_challenge,
    verify_attestation,
)


class _FakeWallet:
    public_key_pem = "-----BEGIN PUBLIC KEY-----\nFAKE\n-----END PUBLIC KEY-----\n"

    def sign(self, msg):
        return "AAAA"


def test_honest_peer_passes_attestation():
    c = issue_challenge(workload_n=128)
    r = respond_to_challenge(c, signer=_FakeWallet())
    # The test machine is "consumer-cpu" tier — its real timing
    # should fit the band [800, 12000] ms × 2.0 = [800, 24000] ms.
    # If we run on a beefier dev box and finish faster we may need
    # to claim a higher class. The reference compute on n=128 is
    # tiny so we test with "remote-mesh" which accepts wide bands.
    rec = verify_attestation(
        challenge=c, response=r, claimed_hardware_class="remote-mesh",
    )
    assert rec is not None
    assert rec.provider_pubkey == _FakeWallet.public_key_pem
    assert rec.elapsed_ms == r.elapsed_ms


def test_nonce_mismatch_rejected():
    c1 = issue_challenge(workload_n=128)
    c2 = issue_challenge(workload_n=128)
    r1 = respond_to_challenge(c1, signer=_FakeWallet())
    # Send r1 against c2 — nonce mismatch.
    rec = verify_attestation(
        challenge=c2, response=r1, claimed_hardware_class="remote-mesh",
    )
    assert rec is None


def test_hash_tampered_rejected():
    c = issue_challenge(workload_n=128)
    r = respond_to_challenge(c, signer=_FakeWallet())
    r.result_hash_hex = "f" * 64       # tamper
    rec = verify_attestation(
        challenge=c, response=r, claimed_hardware_class="remote-mesh",
    )
    assert rec is None


def test_impossibly_fast_timing_rejected():
    """Claiming consumer-cpu tier but reporting 0.1ms is impossible
    for the reference matmul; the band check rejects."""
    c = issue_challenge(workload_n=128)
    r = respond_to_challenge(c, signer=_FakeWallet())
    r.elapsed_ms = 0.001      # impossibly fast for CPU
    rec = verify_attestation(
        challenge=c, response=r, claimed_hardware_class="consumer-cpu",
    )
    assert rec is None


def test_impossibly_slow_timing_rejected():
    c = issue_challenge(workload_n=128)
    r = respond_to_challenge(c, signer=_FakeWallet())
    r.elapsed_ms = 60_000     # 60s for n=128 is way out of band
    rec = verify_attestation(
        challenge=c, response=r, claimed_hardware_class="consumer-gpu-high",
    )
    assert rec is None


def test_unknown_hardware_class_rejected():
    c = issue_challenge(workload_n=128)
    r = respond_to_challenge(c, signer=_FakeWallet())
    rec = verify_attestation(
        challenge=c, response=r, claimed_hardware_class="quantum-superchip",
    )
    assert rec is None


def test_signature_callback_invoked_when_provided():
    """When a verify_signature callback is provided, it must be
    invoked. Returning False rejects the attestation."""
    c = issue_challenge(workload_n=128)
    r = respond_to_challenge(c, signer=_FakeWallet())
    rec = verify_attestation(
        challenge=c, response=r, claimed_hardware_class="remote-mesh",
        verify_signature=lambda msg, sig, pub: False,
    )
    assert rec is None
    # And when the callback returns True, the attestation passes.
    rec = verify_attestation(
        challenge=c, response=r, claimed_hardware_class="remote-mesh",
        verify_signature=lambda msg, sig, pub: True,
    )
    assert rec is not None


def test_attest_record_freshness_window():
    c = issue_challenge(workload_n=128)
    r = respond_to_challenge(c, signer=_FakeWallet())
    rec = verify_attestation(
        challenge=c, response=r, claimed_hardware_class="remote-mesh",
    )
    assert rec.fresh() is True
    # Force expired record by setting expires in the past.
    rec.expires_unix = time.time() - 1
    assert rec.fresh() is False
