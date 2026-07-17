"""Tests for A2: reasoning-time-budget auction axis."""

import pytest

from core.cost_optimizer import CostOptimalRouter
from core.providers import Auction, Bid, JobSpec, Provider


class _StubProvider(Provider):
    def __init__(self, bid: Bid):
        self._bid = bid
        self.provider_id = bid.provider_id

    def bid(self, job: JobSpec) -> Bid:
        return self._bid

    def execute(self, job: JobSpec, bid: Bid) -> dict:
        return {"status": "ok"}

    def attest(self, result: dict) -> dict:
        return {"sig": "stub"}


def _job(*, reasoning_max=0) -> JobSpec:
    return JobSpec(
        job_id="j", kind="inference",
        payload={"prompt": "x"},
        cost_ceiling_usd=0.10,
        latency_ceiling_ms=60_000,
        privacy_class="public",
        quality_floor=0.5,
        reasoning_seconds_max=reasoning_max,
    )


def _bid(pid, *, reasoning=0, price=0.01, eta=2000) -> Bid:
    return Bid(provider_id=pid, price_usd=price, eta_ms=eta,
               expected_quality=0.85, privacy_grade="public",
               reasoning_seconds_committed=reasoning)


def test_default_zero_reasoning_preserves_legacy_behavior():
    """Old bids and old jobs that don't set reasoning_seconds match
    fine -- both default to 0 -> bid passes."""
    a = Auction()
    a.register(_StubProvider(_bid("legacy")))
    res = a.run(_job())
    assert res.is_won()
    assert res.winner.provider_id == "legacy"


def test_bid_exceeding_caller_reasoning_max_is_rejected():
    a = Auction()
    a.register(_StubProvider(_bid("greedy", reasoning=30)))
    a.register(_StubProvider(_bid("ok", reasoning=5)))
    res = a.run(_job(reasoning_max=10))
    assert res.is_won()
    assert res.winner.provider_id == "ok"
    assert any(r.get("bid")
               and r["bid"].provider_id == "greedy"
               and "reasoning_seconds" in r["reason"]
               for r in res.rejected)


def test_negative_reasoning_committed_is_rejected():
    a = Auction()
    a.register(_StubProvider(_bid("buggy", reasoning=-5)))
    a.register(_StubProvider(_bid("ok", reasoning=0)))
    res = a.run(_job())
    assert res.is_won()
    assert res.winner.provider_id == "ok"


def test_cost_optimizer_enforces_reasoning_budget():
    r = CostOptimalRouter()
    # A cheaper bid that wants too much reasoning -> rejected.
    r.register(_StubProvider(_bid("cheap_but_slow_to_think",
                                  price=0.001, reasoning=120)))
    r.register(_StubProvider(_bid("ok",
                                  price=0.005, reasoning=10)))
    sel = r.select(_job(reasoning_max=30))
    assert sel.is_won()
    assert sel.winner.provider_id == "ok"


def test_caller_with_high_budget_admits_deeper_thinker():
    """A high reasoning budget unlocks the 'deep thinking' provider
    -- the o1-style reasoning axis on the mesh."""
    r = CostOptimalRouter()
    r.register(_StubProvider(_bid("deep_thinker",
                                  price=0.005, reasoning=60)))
    r.register(_StubProvider(_bid("shallow",
                                  price=0.010, reasoning=2)))
    sel = r.select(_job(reasoning_max=120))
    assert sel.is_won()
    # deep_thinker is cheaper AND now within budget.
    assert sel.winner.provider_id == "deep_thinker"


def test_zero_budget_filters_any_committed_reasoning():
    r = CostOptimalRouter()
    r.register(_StubProvider(_bid("any_thinking",
                                  price=0.001, reasoning=1)))
    sel = r.select(_job(reasoning_max=0))
    assert not sel.is_won()
    assert any("reasoning_seconds" in rr["reason"]
               for rr in sel.rejected)
