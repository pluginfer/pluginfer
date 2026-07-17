"""Market-price observer — close the price-discovery loop.

The static-template problem
---------------------------
Today every cross-node provider bids using a hardcoded template:
`per_1k_usd = 0.0010 / max(1.0, peer_score / 10.0)`. That's a
defensible STARTING bid but it never adapts. If clearing prices in
the real auction settle 30% lower for the same hardware class, the
fresh provider keeps over-bidding and losing every auction.

This module gives providers vision into the recent clearing prices:

  1. The gateway records every winning bid as
     `MarketObserver.record_winning_bid(bid)`.
  2. `MarketObserver.clearing_price(hardware_class, kind)` returns
     the rolling median of the last N clears for that bucket.
  3. The provider's bid template reads the clearing price and
     adjusts its template floor toward it. Fresh providers learn
     the market within a few rounds without operator intervention.

Innovation: §A32 "Self-tuning bid templates for permissionless
compute auctions." The auction itself becomes a price-discovery
engine; providers don't need a separate oracle. Closed-loop
economics.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Deque, Dict, List, Optional

WINDOW_SIZE = 200       # last N clears per bucket
MIN_OBSERVATIONS = 5    # below this we don't influence template


@dataclass
class ClearedBid:
    hardware_class: str
    job_kind: str
    price_usd: float
    eta_ms: int
    timestamp_unix: float = field(default_factory=time.time)


class MarketObserver:
    """Thread-safe rolling window of cleared bids. Bid templates
    read from it without contention; the gateway writes from the
    auction-success path."""

    def __init__(self, *, window_size: int = WINDOW_SIZE) -> None:
        self.window_size = window_size
        self._by_bucket: DefaultDict[tuple, Deque[ClearedBid]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._lock = threading.Lock()

    def record_winning_bid(
        self, *, hardware_class: str, job_kind: str,
        price_usd: float, eta_ms: int,
    ) -> None:
        c = ClearedBid(
            hardware_class=hardware_class, job_kind=job_kind,
            price_usd=float(price_usd), eta_ms=int(eta_ms),
        )
        with self._lock:
            self._by_bucket[(hardware_class, job_kind)].append(c)

    def clearing_price(
        self, hardware_class: str, job_kind: str,
        *, min_observations: int = MIN_OBSERVATIONS,
    ) -> Optional[float]:
        """Return the rolling median price for the bucket, or None
        if we don't have enough samples to be statistically meaningful.
        The bid template uses None to mean 'use the static fallback'."""
        with self._lock:
            samples = list(self._by_bucket.get((hardware_class, job_kind), []))
        if len(samples) < min_observations:
            return None
        return statistics.median(s.price_usd for s in samples)

    def bucket_size(self, hardware_class: str, job_kind: str) -> int:
        with self._lock:
            return len(self._by_bucket.get((hardware_class, job_kind), []))


# ---------------------------------------------------------------------------
# Bid-template blender
# ---------------------------------------------------------------------------

def blended_bid_price(
    *,
    static_template_price: float,
    market_price: Optional[float],
    blend_weight: float = 0.7,
) -> float:
    """Blend the static template with the observed market median.
    Heavy market weight (0.7) means a fresh provider follows the
    market quickly while still leaving room for under-cutters to
    discover lower clears. `market_price=None` falls back to static."""
    if market_price is None or market_price <= 0:
        return static_template_price
    return blend_weight * market_price + (1 - blend_weight) * static_template_price


__all__ = [
    "ClearedBid",
    "MIN_OBSERVATIONS",
    "MarketObserver",
    "WINDOW_SIZE",
    "blended_bid_price",
]
