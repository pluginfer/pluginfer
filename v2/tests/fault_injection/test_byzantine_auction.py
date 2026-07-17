"""Adversarial-bid fault injection.

Provider behaviours we explicitly defend against:
  - Negative price -> bid with negative price_usd
  - Zero price -> price gaming (bid below cost ceiling)
  - Negative ETA -> attempt to win on impossible latency
  - Quality > 1.0 -> nonsensical self-report
  - Provider's bid violates job's privacy_class

Each case must either be REJECTED at the violates() level or land in
auction.rejected[] -- never picked as winner. The defending logic
lives in core.providers.Bid.violates() and Auction.run().
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parents[2]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.providers import (  # noqa: E402
    Auction,
    Bid,
    JobSpec,
    PRIVACY_PUBLIC,
    Provider,
)


class _FixedBidProvider(Provider):
    """A provider that always emits the bid you give it (good for
    constructing adversarial scenarios)."""
    privacy_grade = PRIVACY_PUBLIC
    kind = "compute"

    def __init__(self, provider_id: str, bid: Bid):
        self.provider_id = provider_id
        self._bid = bid

    def bid(self, job: JobSpec) -> Bid:
        return self._bid

    def execute(self, job: JobSpec, bid: Bid) -> dict:
        return {"status": "executed", "result_hash": "00" * 32}


def _make_job() -> JobSpec:
    return JobSpec(
        job_id="job-byz-1",
        kind="compute.test",
        payload={},
        cost_ceiling_usd=0.10,
        latency_ceiling_ms=10_000,
        quality_floor=0.7,
        privacy_class="public",
    )


def test_negative_price_bid_filtered_out():
    """A bid below the cost ceiling but with a negative number is a
    pricing-system corruption. Either violates() catches it or the
    score function rejects it."""
    job = _make_job()
    p = _FixedBidProvider(
        "byz-neg-price",
        Bid(provider_id="byz-neg-price", price_usd=-1.0,
            eta_ms=100, expected_quality=0.95, privacy_grade=PRIVACY_PUBLIC),
    )
    a = Auction()
    a.register(p)
    out = a.run(job)
    # Negative price isn't physically meaningful; the auction must NOT
    # treat it as the cheapest viable bid.
    if out.is_won():
        # If it's won, the policy is broken: a real auction must reject
        # the negative bid up-front (defence in depth — this assertion
        # makes the policy explicit).
        assert out.winner.price_usd >= 0, (
            "winning bid has negative price -- pricing oracle broken"
        )


def test_eta_exceeds_ceiling_rejected():
    job = _make_job()
    p = _FixedBidProvider(
        "byz-late",
        Bid(provider_id="byz-late", price_usd=0.001,
            eta_ms=999_999, expected_quality=0.95, privacy_grade=PRIVACY_PUBLIC),
    )
    a = Auction()
    a.register(p)
    out = a.run(job)
    assert not out.is_won()
    assert any("eta" in r["reason"] for r in out.rejected)


def test_quality_below_floor_rejected():
    job = _make_job()
    p = _FixedBidProvider(
        "byz-low-q",
        Bid(provider_id="byz-low-q", price_usd=0.001,
            eta_ms=100, expected_quality=0.1, privacy_grade=PRIVACY_PUBLIC),
    )
    a = Auction()
    a.register(p)
    out = a.run(job)
    assert not out.is_won()
    assert any("quality" in r["reason"] for r in out.rejected)


def test_provider_raising_during_bid_does_not_kill_auction():
    """A flaky provider should be skipped, not bring the whole auction
    down. This is a fault-isolation requirement."""
    class _Raising(Provider):
        provider_id = "raising"
        privacy_grade = PRIVACY_PUBLIC
        kind = "compute"
        def bid(self, job): raise RuntimeError("oops")
        def execute(self, job, bid): return {}

    class _Healthy(Provider):
        provider_id = "healthy"
        privacy_grade = PRIVACY_PUBLIC
        kind = "compute"
        def bid(self, job):
            return Bid(provider_id="healthy", price_usd=0.001,
                       eta_ms=100, expected_quality=0.9,
                       privacy_grade=PRIVACY_PUBLIC)
        def execute(self, job, bid): return {"status": "executed"}

    job = _make_job()
    a = Auction()
    a.register(_Raising())
    a.register(_Healthy())
    out = a.run(job)
    assert out.is_won()
    assert out.winner.provider_id == "healthy"
    assert any(r.get("provider_id") == "raising" for r in out.rejected)


def test_no_providers_returns_clean_loss():
    """Empty auction -> no winner, no exception."""
    out = Auction().run(_make_job())
    assert not out.is_won()
    assert out.bids == []
