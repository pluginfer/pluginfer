"""Bandwidth-aware bidding — egress costs in the bid template.

The hidden loss
---------------
Provider on a metered uplink (cable, mobile-hotspot, residential
fiber with overages) gets a CPU credit for the compute, but loses
real cash on egress. A 200-token completion in JSON form costs ~0.5
KB to ship; trivial. But:

  * 1M-prompt batch inference returning 500 tokens each → 250 MB
    egress. At Verizon Fios's $10/100GB overage tier that's $0.025
    of pure burn.
  * Image generation returning a 2 MB PNG × 1000 jobs → 2 GB →
    $0.20 in lost margin.
  * Long-form completion (code generation, full essays) at 4 KB
    per call × 10k/day → 40 MB → small but stacks.

Providers should bid net-of-egress so their floor is RECOVERY, not
just compute. This module:

  1. `BandwidthProfile` carries (egress_usd_per_gb, monthly_quota_gb).
     When the operator hasn't supplied real numbers, defaults to
     $0 (no penalty — backward compatible).
  2. `estimate_egress_bytes(job, est_tokens)` heuristics: kind +
     payload shape → bytes/job estimate.
  3. `bandwidth_adjusted_price(base_price, profile, est_bytes)`
     adds the egress cost to the base price so the bid is
     break-even AT WORST.

The cross-node provider's bid template + the browser provider
both consume this. A provider on unmetered fiber leaves the
defaults at 0 and bids unchanged. A provider on a $50/100GB
plan sets the env var and their bids self-adjust.

Innovation: §A31 "Egress-aware bid auction for permissionless
compute mesh." AWS bills egress separately at $0.09/GB and pockets
the difference; we surface it into bids transparently so buyers see
the real cost of distribution and providers don't go broke.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict


# Rough byte-per-token estimates. JSON wire format is ~4 chars per
# token; UTF-8 single-byte for ASCII tokens; embeddings dominated by
# fp32 (1536 dims × 4 bytes = 6KB per embedding).
DEFAULT_BYTES_PER_TOKEN = 5
EMBEDDING_BYTES_PER_VECTOR = 6 * 1024
IMAGE_BYTES_PER_RESPONSE = 2 * 1024 * 1024
DEFAULT_EGRESS_USD_PER_GB = float(
    os.environ.get("PLUGINFER_DEFAULT_EGRESS_USD_PER_GB", "0.0")
)


@dataclass
class BandwidthProfile:
    """Provider's outbound cost. Defaults to 0 — provider has
    unmetered uplink. Operators on metered plans set
    egress_usd_per_gb to their real overage rate."""
    egress_usd_per_gb: float = DEFAULT_EGRESS_USD_PER_GB
    monthly_quota_gb: float = 0.0       # 0 = unlimited
    used_this_month_gb: float = 0.0     # caller tracks


def estimate_egress_bytes(job_payload: Dict[str, Any], *, job_kind: str = "") -> int:
    """Predict outbound bytes for the response. Used by the bid
    template to bake egress into price BEFORE the auction clears."""
    if "embed" in job_kind:
        # Embedding endpoints return fp32 vectors. Default 1536-dim.
        dim = int(job_payload.get("dimensions", 1536))
        return dim * 4
    if "image" in job_kind:
        return IMAGE_BYTES_PER_RESPONSE
    # Default LLM completion path.
    max_tokens = int(job_payload.get("max_tokens", 200))
    return max_tokens * DEFAULT_BYTES_PER_TOKEN


def bandwidth_adjusted_price(
    base_price_usd: float,
    profile: BandwidthProfile,
    est_egress_bytes: int,
) -> float:
    """Add the expected egress cost to base_price. A provider on
    a $0.10/GB plan serving a 500-token completion (2.5KB out) eats
    $0.00000025 — invisible. A 2MB image at $0.10/GB eats $0.0002
    — non-trivial at scale."""
    if profile.egress_usd_per_gb <= 0:
        return base_price_usd
    gb = est_egress_bytes / (1024.0 ** 3)
    return base_price_usd + (gb * profile.egress_usd_per_gb)


__all__ = [
    "BandwidthProfile",
    "DEFAULT_BYTES_PER_TOKEN",
    "DEFAULT_EGRESS_USD_PER_GB",
    "EMBEDDING_BYTES_PER_VECTOR",
    "IMAGE_BYTES_PER_RESPONSE",
    "bandwidth_adjusted_price",
    "estimate_egress_bytes",
]
