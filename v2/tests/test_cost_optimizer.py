"""Tests for the Cost-Optimal Multi-Constraint Router (PNIS §A11)."""

import pytest

from core.cost_optimizer import (
    CostOptimalRouter,
    cost_savings_vs_baseline,
    pareto_frontier,
)
from core.providers import Bid, JobSpec, Provider


class _StubProvider(Provider):
    """Provider that returns a fixed bid (or None to abstain)."""

    def __init__(self, bid: Bid | None, raises: bool = False):
        self._bid = bid
        self._raises = raises
        self.provider_id = bid.provider_id if bid else "stub-abstain"

    def bid(self, job: JobSpec) -> Bid | None:
        if self._raises:
            raise RuntimeError("provider error")
        return self._bid

    def execute(self, job: JobSpec, bid: Bid) -> dict:
        return {"status": "ok"}

    def attest(self, result: dict) -> dict:
        return {"sig": "stub"}


def _job(*, cost_ceiling=0.10, latency_ceiling=10_000,
         quality_floor=0.7, privacy="public") -> JobSpec:
    return JobSpec(
        job_id="j", kind="inference",
        payload={"prompt": "x"},
        cost_ceiling_usd=cost_ceiling,
        latency_ceiling_ms=latency_ceiling,
        privacy_class=privacy,
        quality_floor=quality_floor,
    )


def _bid(pid, *, price, eta, q=0.85, priv="public") -> Bid:
    return Bid(provider_id=pid, price_usd=price, eta_ms=eta,
               expected_quality=q, privacy_grade=priv)


def test_picks_cheapest_passing_bid():
    r = CostOptimalRouter()
    r.register(_StubProvider(_bid("a", price=0.05, eta=2000)))
    r.register(_StubProvider(_bid("b", price=0.01, eta=2000)))   # cheapest
    r.register(_StubProvider(_bid("c", price=0.03, eta=2000)))
    sel = r.select(_job())
    assert sel.is_won()
    assert sel.winner.provider_id == "b"


def test_filters_out_over_ceiling_bid():
    r = CostOptimalRouter()
    r.register(_StubProvider(_bid("expensive", price=0.50, eta=2000)))
    r.register(_StubProvider(_bid("ok", price=0.05, eta=2000)))
    sel = r.select(_job(cost_ceiling=0.10))
    assert sel.is_won()
    assert sel.winner.provider_id == "ok"
    assert any(r["provider_id"] == "expensive"
               and "ceiling" in r["reason"] for r in sel.rejected)


def test_no_winner_returns_pareto_frontier():
    r = CostOptimalRouter()
    # Both bids fail the latency ceiling.
    r.register(_StubProvider(_bid("slow1", price=0.01, eta=20_000)))
    r.register(_StubProvider(_bid("slow2", price=0.02, eta=15_000)))
    sel = r.select(_job(latency_ceiling=5_000))
    assert not sel.is_won()
    # Frontier should still surface candidates so caller can relax.
    assert len(sel.frontier) >= 1


def test_tiebreak_prefers_lower_eta():
    r = CostOptimalRouter()
    r.register(_StubProvider(_bid("slow", price=0.02, eta=4000, q=0.9)))
    r.register(_StubProvider(_bid("fast", price=0.02, eta=1000, q=0.9)))
    sel = r.select(_job())
    assert sel.winner.provider_id == "fast"


def test_tiebreak_prefers_higher_quality():
    r = CostOptimalRouter()
    r.register(_StubProvider(_bid("low_q", price=0.02, eta=2000, q=0.75)))
    r.register(_StubProvider(_bid("high_q", price=0.02, eta=2000, q=0.95)))
    sel = r.select(_job(quality_floor=0.7))
    assert sel.winner.provider_id == "high_q"


def test_provider_raise_recorded_in_rejected():
    r = CostOptimalRouter()
    r.register(_StubProvider(_bid("ok", price=0.02, eta=2000)))
    r.register(_StubProvider(None, raises=True))
    sel = r.select(_job())
    assert sel.is_won()
    assert any("raised" in r["reason"] for r in sel.rejected)


def test_abstain_recorded_in_rejected():
    r = CostOptimalRouter()
    r.register(_StubProvider(None))
    r.register(_StubProvider(_bid("ok", price=0.02, eta=2000)))
    sel = r.select(_job())
    assert any(r["reason"] == "abstained" for r in sel.rejected)


def test_privacy_constraint_filters_lower_grade_bid():
    r = CostOptimalRouter()
    r.register(_StubProvider(
        _bid("public_only", price=0.001, eta=1000, priv="public")))
    r.register(_StubProvider(
        _bid("private_ok", price=0.05, eta=1000, priv="private")))
    sel = r.select(_job(privacy="private"))
    assert sel.is_won()
    assert sel.winner.provider_id == "private_ok"


def test_selection_proof_is_deterministic():
    r1 = CostOptimalRouter()
    r1.register(_StubProvider(_bid("a", price=0.01, eta=1000)))
    r1.register(_StubProvider(_bid("b", price=0.02, eta=2000)))
    r2 = CostOptimalRouter()
    # Same providers in REVERSE order; proof must still match because
    # the proof sorts the bid set canonically.
    r2.register(_StubProvider(_bid("b", price=0.02, eta=2000)))
    r2.register(_StubProvider(_bid("a", price=0.01, eta=1000)))
    j = _job()
    s1 = r1.select(j)
    s2 = r2.select(j)
    assert s1.selection_proof == s2.selection_proof


def test_cost_savings_vs_baseline_reports_ratio():
    r = CostOptimalRouter()
    r.register(_StubProvider(_bid("cheap_mesh", price=0.001, eta=2000)))
    sel = r.select(_job(cost_ceiling=0.10))
    rep = cost_savings_vs_baseline(sel, centralised_baseline_usd=0.030)
    assert rep["won"] is True
    assert rep["savings_pct"] > 95.0
    assert rep["ratio_cheaper"] >= 30.0


def test_pareto_frontier_drops_dominated():
    bids = [
        _bid("a", price=0.10, eta=5000, q=0.7),     # expensive + slow + low Q
        _bid("b", price=0.05, eta=2000, q=0.8),     # strictly better than a
        _bid("c", price=0.02, eta=4000, q=0.9),     # cheaper but slower
    ]
    front = pareto_frontier(bids)
    ids = sorted(b.provider_id for b in front)
    assert "a" not in ids
    assert "b" in ids and "c" in ids
