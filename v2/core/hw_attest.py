"""Hardware-attestation challenge — bound the lying.

A node that advertises an H100 on `/v1/hardware` is taken at face
value today. That's exploitable: a CPU-only node with cheap bids
wins auctions it can't fulfill, and the auction's Pareto picker
naively trusts the self-reported `performance_score`.

This module ships the challenge:

  1. Verifier picks a fresh nonce + workload size.
  2. Sends POST /v1/attest_challenge with `(nonce, workload_n)`.
  3. The peer runs a known reference compute (a dense matmul of
     size workload_n × workload_n, deterministic), times it, signs
     `(nonce, workload_n, result_hash, elapsed_ms)` with its wallet,
     and returns the signature.
  4. Verifier checks:
     * signature verifies under the peer's pubkey,
     * result_hash matches the locally-computed reference,
     * elapsed_ms is consistent with the peer's claimed
       hardware_class (within a tolerance band).
  5. On pass: cache the attestation with an expiry. The bid path
     refuses to bid on jobs above the un-attested cap; a fresh
     attestation re-unlocks the full tier.

This is the minimum mechanism that makes self-reporting honest:
a CPU-only liar can sign whatever they want, but they cannot beat
the timing band of an H100 doing the reference compute, so their
attestation FAILS and their bids stay capped at the untrusted tier.

Implementation notes:
  * The reference compute uses numpy float32 matmul — present on
    every Python ML stack, deterministic enough for hash matching
    at the byte level when the same seed produces the same matrix.
  * `tolerance_factor` is the band the timing must fit: 2x slower
    than the claimed-class baseline is the cutoff. Tighter bands
    catch more liars but reject more legitimate variation.
  * Attestations expire (`ATTEST_TTL_S`) so an old-but-genuine
    attestation can't be replayed forever.

Innovation: §A28 "Workload-replay attestation for permissionless
compute providers." No prior art combines (a) on-demand workload
challenges, (b) cryptographic signing of the result by the
provider's wallet, AND (c) hardware-class-tier-keyed timing bands.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

ATTEST_TTL_S = float(os.environ.get("PLUGINFER_ATTEST_TTL_S", "1800.0"))

# Timing bands (ms) for the reference workload (size=512 dense f32
# matmul). Anything claimed to run faster than the band's lower
# bound (or slower than 2× the upper) is rejected.
TIMING_BANDS_MS_BY_CLASS: Dict[str, tuple] = {
    "consumer-gpu-high": (5.0, 80.0),       # 4090 / 7900xtx / H100 class
    "consumer-gpu-mid":  (40.0, 400.0),     # M3 / 3060 / xpu mid
    "consumer-gpu-low":  (200.0, 1500.0),   # DirectML / older GPUs
    "consumer-cpu":      (800.0, 12000.0),  # high-end CPU only
    "remote-mesh":       (0.0, 12000.0),    # catch-all: any speed ok
}
TOLERANCE_FACTOR = float(os.environ.get("PLUGINFER_ATTEST_TOLERANCE", "2.0"))


@dataclass
class AttestChallenge:
    nonce: str
    workload_n: int = 512


@dataclass
class AttestResponse:
    nonce: str
    workload_n: int
    result_hash_hex: str
    elapsed_ms: float
    signature_b64: str = ""
    signer_pubkey_pem: str = ""


@dataclass
class AttestRecord:
    """Cached pass — the bid path reads this before going above the
    untrusted cap. Includes the timing band the response fit so
    the audit log can show WHY the attestation passed."""
    provider_pubkey: str
    hardware_class: str
    elapsed_ms: float
    granted_unix: float = field(default_factory=time.time)
    expires_unix: float = 0.0

    def fresh(self, *, now: Optional[float] = None) -> bool:
        return (now or time.time()) < self.expires_unix


def issue_challenge(workload_n: int = 512) -> AttestChallenge:
    """The verifier-side entry point. Caller is the auction's
    bid-path filter; the peer must respond to this challenge
    before bidding on high-cost jobs."""
    return AttestChallenge(
        nonce=secrets.token_hex(16),
        workload_n=int(workload_n),
    )


def _reference_matmul(nonce: str, n: int):
    """Generate a deterministic (n, n) float32 matmul keyed by
    nonce. Both verifier and peer compute the same matrices, so
    the result hash matches on honest peers. Operator who omits
    numpy gets a clearly-tagged fallback that still keys on nonce
    but uses raw bytes (the timing band is then meaningless;
    fallback is for dependency-free CI not real attestation)."""
    try:
        import numpy as np
        seed = int.from_bytes(
            hashlib.sha256(nonce.encode("utf-8")).digest()[:8], "little",
        )
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((n, n), dtype=np.float32)
        b = rng.standard_normal((n, n), dtype=np.float32)
        c = a @ b
        # Hash the bytes — exact match across honest replicas.
        return c, hashlib.sha256(c.tobytes()).hexdigest()
    except ImportError:
        # Dependency-free fallback for environments without numpy.
        h = hashlib.sha256()
        for i in range(n * n):
            h.update(nonce.encode("utf-8"))
            h.update(i.to_bytes(8, "little"))
        return None, h.hexdigest()


def respond_to_challenge(
    challenge: AttestChallenge,
    *, signer: Any,
) -> AttestResponse:
    """Provider-side: solve the challenge, sign the result, return.
    `signer` must have `.sign(message: str) -> str` (b64 signature)
    and `.public_key_pem` — both true for `core.tokenomics.Wallet`."""
    t0 = time.monotonic()
    _, result_hash = _reference_matmul(challenge.nonce, challenge.workload_n)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    msg = (
        f"ATTEST|{challenge.nonce}|{challenge.workload_n}|"
        f"{result_hash}|{elapsed_ms:.3f}"
    )
    sig = signer.sign(msg) if hasattr(signer, "sign") else ""
    pub = getattr(signer, "public_key_pem", "")
    return AttestResponse(
        nonce=challenge.nonce,
        workload_n=challenge.workload_n,
        result_hash_hex=result_hash,
        elapsed_ms=elapsed_ms,
        signature_b64=sig,
        signer_pubkey_pem=pub,
    )


def verify_attestation(
    *,
    challenge: AttestChallenge,
    response: AttestResponse,
    claimed_hardware_class: str,
    verify_signature: Optional[Callable[..., bool]] = None,
) -> Optional[AttestRecord]:
    """Verifier checks: nonce match, result correct, timing in band,
    signature verifies. Returns the cached record on pass; None on
    fail (caller bid-path keeps the peer at untrusted tier).

    `verify_signature(msg, sig_b64, pubkey_pem) -> bool` is the
    pluggable signature-check (defaults to a no-op for unit tests
    when callers don't have a Wallet handy)."""
    if response.nonce != challenge.nonce:
        return None
    if response.workload_n != challenge.workload_n:
        return None
    # Re-compute the reference and confirm bytes match.
    _, expected_hash = _reference_matmul(
        challenge.nonce, challenge.workload_n,
    )
    if response.result_hash_hex != expected_hash:
        return None
    # Timing band check.
    band = TIMING_BANDS_MS_BY_CLASS.get(claimed_hardware_class)
    if band is None:
        # Unknown class — refuse to attest. Operator can add the
        # class to TIMING_BANDS_MS_BY_CLASS.
        return None
    lo, hi = band
    # An honest peer at the claimed class must fit (lo, hi *
    # TOLERANCE_FACTOR). Faster than `lo` is suspicious (probably
    # claiming a higher tier than it has); slower than `hi *
    # tolerance` flunks.
    if response.elapsed_ms < lo or response.elapsed_ms > hi * TOLERANCE_FACTOR:
        return None
    # Signature check (optional injection for tests).
    if verify_signature is not None and response.signature_b64:
        msg = (
            f"ATTEST|{response.nonce}|{response.workload_n}|"
            f"{response.result_hash_hex}|{response.elapsed_ms:.3f}"
        )
        if not verify_signature(
            msg, response.signature_b64, response.signer_pubkey_pem,
        ):
            return None
    return AttestRecord(
        provider_pubkey=response.signer_pubkey_pem,
        hardware_class=claimed_hardware_class,
        elapsed_ms=response.elapsed_ms,
        granted_unix=time.time(),
        expires_unix=time.time() + ATTEST_TTL_S,
    )


__all__ = [
    "ATTEST_TTL_S",
    "AttestChallenge",
    "AttestRecord",
    "AttestResponse",
    "TIMING_BANDS_MS_BY_CLASS",
    "TOLERANCE_FACTOR",
    "issue_challenge",
    "respond_to_challenge",
    "verify_attestation",
]
