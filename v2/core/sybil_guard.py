"""G6 — Sybil resistance for browser-tab providers.

The browser-tab supply side is a strategic asset (every Chromium tab on
the planet) AND a strategic liability (zero friction to register
millions of fake tabs). This module gates registration with three
stacking defences:

  1. **Per-/24 IP rate limit** — token-bucket scoped to the source
     subnet, much tighter than the global RateLimitMiddleware. An
     attacker can't pivot through 10k IPs in a single subnet to
     register 10k tabs.

  2. **WebGPU adapter fingerprint** — the browser tab reports its
     GPU adapter vendor + architecture + driver via WebGPU's
     `requestAdapterInfo()`. We hash that tuple server-side. The
     same fingerprint registering from many different /24s within
     a short window is treated as Sybil and rate-limited harder.

  3. **Stake-to-register tier promotion** — three tiers:
       - `untrusted` (default) — small jobs only, can be rate-limited
         out at any moment, no leaderboard exposure.
       - `staked` — provider has posted a chain-side deposit ≥
         `MIN_PROVIDER_STAKE_PLG`; full job eligibility; slash-on-
         non-delivery makes refusal expensive.
       - `verified` — staked + Cloudflare Turnstile / hCaptcha
         token at registration; counts as a "human-operated" node
         and earns leaderboard priority.

Tier is reflected in the provider's `evidence` field on the bid, so
the auction's quality / privacy scoring already incorporates it
without further plumbing.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables (env-overridable for ops)
# ---------------------------------------------------------------------------

import os

# How many register / heartbeat / open_jobs ops a single /24 can fire
# per minute. Default 60 = 1/sec — already very generous for a
# single-tab provider. 6000/hr means a /24 with 50 legitimate tabs
# behind a CGNAT can each hit the gateway ~2/sec.
PER_SUBNET_OPS_PER_MIN = int(os.environ.get("PLUGINFER_PER_SUBNET_OPS_PER_MIN", "60"))

# Within how many seconds is a repeat-fingerprint considered Sybil?
FINGERPRINT_SYBIL_WINDOW_S = int(os.environ.get(
    "PLUGINFER_FP_SYBIL_WINDOW_S", "300"
))
# Above this many distinct subnets in the window = block.
FINGERPRINT_SYBIL_MAX_SUBNETS = int(os.environ.get(
    "PLUGINFER_FP_SYBIL_MAX_SUBNETS", "5"
))

MIN_PROVIDER_STAKE_PLG = float(os.environ.get(
    "PLUGINFER_MIN_PROVIDER_STAKE_PLG", "1.0"
))

# Per-tier hard cap on the cost_ceiling of jobs the provider may bid
# on. Untrusted-tier providers refuse to bid (`.bid()` returns None)
# when the job's cost_ceiling_usd exceeds the cap — bounds the blast
# radius of a malicious browser tab to the cap × throughput.
#
# Production-recommended defaults applied below. Operators that want
# looser caps for dev / test override via env:
#     PLUGINFER_UNTRUSTED_MAX_USD=1.0     (loose dev)
#     PLUGINFER_STAKED_MAX_USD=50.0       (loose dev)
#     PLUGINFER_VERIFIED_MAX_USD=10000.0
MAX_JOB_COST_BY_TIER_USD = {
    "untrusted": float(os.environ.get("PLUGINFER_UNTRUSTED_MAX_USD", "0.10")),
    "staked":    float(os.environ.get("PLUGINFER_STAKED_MAX_USD", "10.0")),
    "verified":  float(os.environ.get("PLUGINFER_VERIFIED_MAX_USD", "1000000.0")),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def subnet_of(ip: str) -> str:
    """Reduce an IPv4 to its /24 (`A.B.C.0`). IPv6 collapses to /48 —
    the network operator is who we're rate-limiting against, not the
    individual interface."""
    if not ip:
        return ""
    if ":" in ip:
        # IPv6 — take the first 3 groups (= /48 for routing prefixes).
        parts = ip.split(":")
        return ":".join(parts[:3]) + "::/48"
    parts = ip.split(".")
    if len(parts) != 4:
        return ip
    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"


def fingerprint_hash(*parts: Optional[str]) -> str:
    """Stable hash of (vendor, architecture, device, driver) — call
    with whatever fields the browser reports."""
    canonical = "|".join((p or "") for p in parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Per-subnet rate limiter (token bucket)
# ---------------------------------------------------------------------------

@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class PerSubnetRateLimiter:
    """Token-bucket keyed by /24. One bucket per subnet, refill rate
    derived from the env-tunable. Tests can inject their own clock
    via `now=` for determinism."""

    def __init__(
        self,
        *,
        capacity: int = PER_SUBNET_OPS_PER_MIN,
        refill_per_sec: float = PER_SUBNET_OPS_PER_MIN / 60.0,
    ):
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def allow(self, ip: str, *, now: Optional[float] = None) -> bool:
        """Return True if this subnet can take one more op right now."""
        sub = subnet_of(ip)
        if not sub:
            return True
        t = now if now is not None else time.monotonic()
        with self._lock:
            b = self._buckets.get(sub)
            if b is None:
                b = _Bucket(tokens=self.capacity, last_refill=t)
                self._buckets[sub] = b
            # Refill since last hit.
            elapsed = max(0.0, t - b.last_refill)
            b.tokens = min(self.capacity, b.tokens + elapsed * self.refill_per_sec)
            b.last_refill = t
            if b.tokens < 1.0:
                return False
            b.tokens -= 1.0
            return True


# ---------------------------------------------------------------------------
# Fingerprint Sybil detector
# ---------------------------------------------------------------------------

@dataclass
class FingerprintSybilDetector:
    """Tracks which subnets each fingerprint has been seen from in the
    rolling window. If a single fingerprint shows up from more than
    `max_subnets` distinct subnets within `window_s`, that's Sybil —
    block further registrations under that fingerprint."""

    window_s: int = FINGERPRINT_SYBIL_WINDOW_S
    max_subnets: int = FINGERPRINT_SYBIL_MAX_SUBNETS
    _seen: Dict[str, Dict[str, float]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_and_check(
        self,
        fingerprint: str,
        ip: str,
        *,
        now: Optional[float] = None,
    ) -> bool:
        """Record this (fingerprint, subnet) sighting. Return True if
        the fingerprint is currently *clean*, False if it has tripped
        the Sybil threshold."""
        if not fingerprint or not ip:
            return True
        sub = subnet_of(ip)
        t = now if now is not None else time.monotonic()
        cutoff = t - self.window_s
        with self._lock:
            row = self._seen.setdefault(fingerprint, {})
            row[sub] = t
            stale = [k for k, v in row.items() if v < cutoff]
            for k in stale:
                row.pop(k, None)
            return len(row) <= self.max_subnets


# ---------------------------------------------------------------------------
# Stake tier resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TierResult:
    tier: str           # "untrusted" | "staked" | "verified"
    stake_plg: float = 0.0
    turnstile_ok: bool = False
    # The auction's hard cap on cost_ceiling of jobs this tier may
    # bid on. Sourced from `MAX_JOB_COST_BY_TIER_USD` at resolve
    # time so a runtime change to the env-tunables affects
    # subsequent registrations without redeploying.
    max_job_cost_usd: float = 0.0


def resolve_tier(
    *,
    stake_plg: float,
    turnstile_ok: bool,
    min_stake_plg: float = MIN_PROVIDER_STAKE_PLG,
) -> TierResult:
    """Pure function — given (stake amount, turnstile bool) compute the
    tier + its cost cap."""
    if stake_plg >= min_stake_plg and turnstile_ok:
        return TierResult(
            tier="verified", stake_plg=stake_plg,
            turnstile_ok=turnstile_ok,
            max_job_cost_usd=MAX_JOB_COST_BY_TIER_USD["verified"],
        )
    if stake_plg >= min_stake_plg:
        return TierResult(
            tier="staked", stake_plg=stake_plg,
            turnstile_ok=turnstile_ok,
            max_job_cost_usd=MAX_JOB_COST_BY_TIER_USD["staked"],
        )
    return TierResult(
        tier="untrusted", stake_plg=stake_plg,
        turnstile_ok=turnstile_ok,
        max_job_cost_usd=MAX_JOB_COST_BY_TIER_USD["untrusted"],
    )


__all__ = [
    "FINGERPRINT_SYBIL_MAX_SUBNETS",
    "FINGERPRINT_SYBIL_WINDOW_S",
    "FingerprintSybilDetector",
    "MAX_JOB_COST_BY_TIER_USD",
    "MIN_PROVIDER_STAKE_PLG",
    "PER_SUBNET_OPS_PER_MIN",
    "PerSubnetRateLimiter",
    "TierResult",
    "fingerprint_hash",
    "resolve_tier",
    "subnet_of",
]
