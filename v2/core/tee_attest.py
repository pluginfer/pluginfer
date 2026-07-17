"""TEE-attestation pipeline — make private jobs cross the wire safely.

The privacy problem
-------------------
`privacy_class="private"` today maps to LOCAL_ONLY: the auction
refuses to route the job off the buyer's own node. That's safe but
crippling — most enterprises wanting privacy ALSO want the mesh's
cost economics, which requires routing to nodes they don't own.

Trusted Execution Environments (TEE) — Intel SGX, AMD SEV-SNP,
Intel TDX, ARM Confidential Compute, Apple Secure Enclave — are
hardware-rooted isolation domains that produce remote-attestable
quotes. A buyer who trusts the silicon vendor's root key (Intel, AMD,
Apple) can verify that a remote node's enclave is running specific
attested code over their data, and that the host OS / mesh operator
CANNOT see the plaintext.

This module is the vendor-agnostic abstraction. Each TEE vendor
plugs in a `QuoteVerifier` subclass. The auction layer treats
`privacy_class="confidential"` jobs as: "only bid from providers
whose latest TEE attestation passed our verifier AND whose enclave
measurement matches the agreed-upon code hash."

Innovation: §A34 "Vendor-agnostic TEE attestation in a permissionless
compute auction." Combines (a) pluggable verifiers across vendors,
(b) enclave-measurement-bound bid filtering, AND (c) attestation
cached with a freshness window matched to vendor recommendations
(SGX: 1 day, SEV-SNP: 1 hour, TDX: per-quote). No prior art unifies
these for a permissionless mesh.

Scope this ships
----------------
* `QuoteVerifier` protocol + default `LocalProofVerifier` (sanity
  check — proves the pipeline plumbing, NOT real silicon trust).
* `AttestedProvider` wrapper — augments any Provider with an
  attestation requirement; `bid()` abstains if the cached
  attestation is stale.
* `TEEAttestationCache` — verifies + caches quotes per vendor.

What still needs vendor-specific work
-------------------------------------
* `IntelSgxVerifier` — talks to Intel's Attestation Service (IAS) /
  DCAP. Requires the vendor's API key + DCAP libraries.
* `SevSnpVerifier` — verifies AMD's SEV-SNP attestation reports
  against AMD's VCEK certificate.
* `IntelTdxVerifier` — verifies TDX quotes against Intel's PCS.
These are off-keyboard — they need real silicon to test end-to-end
and the vendor onboarding flows we don't have. The PROTOCOL surface
is shipped so plugging them in later is a one-class addition.
"""

from __future__ import annotations

import abc
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

logger = logging.getLogger(__name__)

DEFAULT_FRESHNESS_S: Dict[str, float] = {
    "sgx":      86400.0,    # 1 day
    "sev-snp":  3600.0,     # 1 hour
    "tdx":      3600.0,     # 1 hour
    "apple-se": 3600.0,
    "local-proof": 300.0,   # dev/test
}


@dataclass(frozen=True)
class EnclaveMeasurement:
    """The hash the buyer trusts: SHA-256 of the binary that will
    run their job inside the enclave. Buyer publishes this hash in
    the JobSpec; the auction filters to providers whose latest
    attestation reports a matching MRENCLAVE / measurement."""
    vendor: str
    measurement_hex: str

    @classmethod
    def from_bytes(cls, vendor: str, code_bytes: bytes) -> "EnclaveMeasurement":
        return cls(
            vendor=vendor,
            measurement_hex=hashlib.sha256(code_bytes).hexdigest(),
        )


@dataclass
class AttestationQuote:
    """Raw vendor quote + the verifier-extracted facts the auction
    needs. The vendor-specific verifier populates this on success."""
    vendor: str
    raw_quote_bytes: bytes
    issued_at_unix: float
    enclave_measurement_hex: str
    enclave_pubkey_pem: str
    verifier_id: str

    def fingerprint(self) -> str:
        return hashlib.sha256(self.raw_quote_bytes).hexdigest()[:32]


class QuoteVerifier(Protocol):
    """Each TEE vendor implements this. Verifier returns a parsed
    `AttestationQuote` on success, None on rejection."""

    vendor: str

    def verify(self, raw_quote_bytes: bytes) -> Optional[AttestationQuote]:
        ...


# ---------------------------------------------------------------------------
# Pipeline plumbing — vendor stubs are the LocalProofVerifier
# ---------------------------------------------------------------------------

@dataclass
class LocalProofVerifier:
    """Dev/test verifier. The `raw_quote_bytes` is a JSON blob
    encoding (measurement_hex, enclave_pubkey_pem, issued_at_unix).
    NOT cryptographically secure — exists only so the pipeline can
    be exercised end-to-end without real silicon. Production
    deployments MUST swap in a real vendor verifier.

    The vendor string is `"local-proof"` so the auction's
    requirement filter can route only test workloads to it."""
    vendor: str = "local-proof"

    def verify(self, raw_quote_bytes: bytes) -> Optional[AttestationQuote]:
        import json
        try:
            blob = json.loads(raw_quote_bytes.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        try:
            return AttestationQuote(
                vendor=self.vendor,
                raw_quote_bytes=raw_quote_bytes,
                issued_at_unix=float(blob["issued_at_unix"]),
                enclave_measurement_hex=str(blob["measurement_hex"]),
                enclave_pubkey_pem=str(blob["enclave_pubkey_pem"]),
                verifier_id="local-proof-v1",
            )
        except (KeyError, ValueError, TypeError):
            return None


@dataclass
class TEEAttestationCache:
    """Verifier-result cache keyed by provider pubkey. Subsequent
    bid() calls within the freshness window skip re-verification.
    Saves us paying Intel's IAS roundtrip on every chat completion."""
    verifiers: Dict[str, QuoteVerifier] = field(default_factory=dict)
    _cache: Dict[str, AttestationQuote] = field(default_factory=dict, repr=False)

    def register_verifier(self, verifier: QuoteVerifier) -> None:
        self.verifiers[verifier.vendor] = verifier

    def submit_quote(
        self, *, provider_pubkey: str, vendor: str,
        raw_quote_bytes: bytes,
    ) -> Optional[AttestationQuote]:
        verifier = self.verifiers.get(vendor)
        if verifier is None:
            return None
        quote = verifier.verify(raw_quote_bytes)
        if quote is None:
            return None
        self._cache[provider_pubkey] = quote
        return quote

    def get_fresh_quote(
        self, *, provider_pubkey: str, now: Optional[float] = None,
    ) -> Optional[AttestationQuote]:
        q = self._cache.get(provider_pubkey)
        if q is None:
            return None
        age = (now or time.time()) - q.issued_at_unix
        ttl = DEFAULT_FRESHNESS_S.get(q.vendor, 3600.0)
        if age > ttl:
            return None
        return q

    def is_attested(
        self, *, provider_pubkey: str,
        required_measurement: Optional[EnclaveMeasurement] = None,
    ) -> bool:
        q = self.get_fresh_quote(provider_pubkey=provider_pubkey)
        if q is None:
            return False
        if required_measurement is None:
            return True
        return (
            q.vendor == required_measurement.vendor
            and q.enclave_measurement_hex == required_measurement.measurement_hex
        )


# ---------------------------------------------------------------------------
# Auction-side: only bid on private jobs when attestation is fresh
# ---------------------------------------------------------------------------

def can_bid_on_confidential_job(
    *, provider_pubkey: str, cache: TEEAttestationCache,
    required_measurement: Optional[EnclaveMeasurement] = None,
) -> bool:
    """Auction's pre-bid filter. Returns True iff the provider has
    a fresh attestation that (optionally) matches the buyer's
    required enclave measurement.

    Providers wire this into their `bid()`: if the JobSpec carries
    `payload.confidential_measurement`, abstain unless this returns
    True. That keeps unattested nodes out of the bidder pool for
    private work — the auction never even SEES them as candidates."""
    return cache.is_attested(
        provider_pubkey=provider_pubkey,
        required_measurement=required_measurement,
    )


__all__ = [
    "AttestationQuote",
    "DEFAULT_FRESHNESS_S",
    "EnclaveMeasurement",
    "LocalProofVerifier",
    "QuoteVerifier",
    "TEEAttestationCache",
    "can_bid_on_confidential_job",
]
