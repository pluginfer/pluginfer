"""Bandwidth-aware bidding + market price discovery feedback loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from core.bandwidth_pricing import (
    BandwidthProfile,
    bandwidth_adjusted_price,
    estimate_egress_bytes,
)
from core.market_observer import (
    MIN_OBSERVATIONS,
    MarketObserver,
    blended_bid_price,
)


# ---------------------------------------------------------------------------
# Bandwidth pricing
# ---------------------------------------------------------------------------

def test_zero_egress_rate_is_no_op():
    p = BandwidthProfile(egress_usd_per_gb=0.0)
    out = bandwidth_adjusted_price(0.001, p, est_egress_bytes=10 * 1024 * 1024)
    assert out == 0.001


def test_egress_adds_to_base_price_linearly():
    p = BandwidthProfile(egress_usd_per_gb=0.10)
    # 1 GB at $0.10 → +$0.10 on the bid.
    out = bandwidth_adjusted_price(
        base_price_usd=0.50, profile=p, est_egress_bytes=1024 ** 3,
    )
    assert out == pytest.approx(0.60, rel=1e-6)


def test_egress_estimate_for_llm_completion():
    payload = {"max_tokens": 1000}
    b = estimate_egress_bytes(payload, job_kind="llm.completion")
    assert b > 0
    # Default is 5 bytes/token × 1000 = 5000.
    assert b == 5000


def test_egress_estimate_for_embedding():
    b = estimate_egress_bytes({"dimensions": 1536}, job_kind="embed")
    # 1536 × 4 = 6144.
    assert b == 6144


def test_egress_estimate_for_image_is_megabytes():
    b = estimate_egress_bytes({}, job_kind="image.generate")
    assert b == 2 * 1024 * 1024


# ---------------------------------------------------------------------------
# Market observer
# ---------------------------------------------------------------------------

def test_clearing_price_requires_min_observations():
    m = MarketObserver()
    for _ in range(MIN_OBSERVATIONS - 1):
        m.record_winning_bid(
            hardware_class="consumer-gpu-high", job_kind="chat",
            price_usd=0.001, eta_ms=200,
        )
    assert m.clearing_price("consumer-gpu-high", "chat") is None
    # One more reaches threshold.
    m.record_winning_bid(
        hardware_class="consumer-gpu-high", job_kind="chat",
        price_usd=0.001, eta_ms=200,
    )
    assert m.clearing_price("consumer-gpu-high", "chat") == 0.001


def test_clearing_price_is_rolling_median():
    m = MarketObserver()
    for p in [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003]:
        m.record_winning_bid(
            hardware_class="cpu", job_kind="x",
            price_usd=p, eta_ms=100,
        )
    median = m.clearing_price("cpu", "x")
    # Median of 6 values 0.0005..0.003 is (0.0015+0.002)/2 = 0.00175.
    assert median == pytest.approx(0.00175, rel=1e-6)


def test_clearing_price_separates_by_bucket():
    """A clear in (gpu-high, chat) doesn't affect (gpu-mid, chat)."""
    m = MarketObserver()
    for _ in range(MIN_OBSERVATIONS):
        m.record_winning_bid(
            hardware_class="consumer-gpu-high", job_kind="chat",
            price_usd=0.001, eta_ms=100,
        )
    assert m.clearing_price("consumer-gpu-high", "chat") == 0.001
    assert m.clearing_price("consumer-gpu-mid", "chat") is None


def test_window_size_caps_history():
    m = MarketObserver(window_size=3)
    for p in [0.001, 0.002, 0.003, 0.004, 0.005]:
        m.record_winning_bid(
            hardware_class="x", job_kind="y",
            price_usd=p, eta_ms=100,
        )
    # Only the last 3 (0.003, 0.004, 0.005) are in the window.
    assert m.bucket_size("x", "y") == 3
    # Median of those three is 0.004 — but bucket_size==3 < MIN_OBS=5
    # so clearing_price returns None.
    assert m.clearing_price("x", "y") is None
    # Lower min_observations to 1 to see the median.
    assert m.clearing_price("x", "y", min_observations=1) == pytest.approx(0.004)


def test_blended_bid_follows_market_when_present():
    out = blended_bid_price(
        static_template_price=0.0010, market_price=0.0005, blend_weight=0.7,
    )
    # 0.7 × 0.0005 + 0.3 × 0.0010 = 0.00035 + 0.00030 = 0.00065.
    assert out == pytest.approx(0.00065, rel=1e-6)


def test_blended_bid_falls_back_to_static_when_no_market():
    out = blended_bid_price(
        static_template_price=0.0010, market_price=None,
    )
    assert out == 0.0010
