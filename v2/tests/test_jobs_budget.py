"""§RFC-3 — Budget-as-Contract wired into JobsService.

Pins the money-critical placement rules:
  * an exhausted envelope refuses the job BEFORE the auction — the
    provider is never even asked to bid (zero economic side effects),
  * a completed job settles the budget at the ACTUAL auction-cleared
    price, not the ceiling that was held,
  * a job that finds no provider releases its hold,
  * quote() prices via the real auction without executing anything.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from api.jobs_service import JobsService
from governance.budget_ledger import BudgetLedger
from core.providers import Auction, Bid, PRIVACY_PUBLIC, Provider


class _W(Provider):
    def __init__(self, *, pid: str = "p1", price: float = 0.05):
        self.provider_id = pid
        self.privacy_grade = PRIVACY_PUBLIC
        self._price = price
        self.bid_calls = 0
        self.execute_calls = 0

    def bid(self, job):
        self.bid_calls += 1
        return Bid(
            provider_id=self.provider_id, price_usd=self._price,
            eta_ms=100, expected_quality=0.9,
            privacy_grade=PRIVACY_PUBLIC, evidence={},
        )

    def execute(self, job, bid):
        self.execute_calls += 1
        out = b"governed-output"
        return {
            "status": "executed", "job_id": job.job_id,
            "result_bytes": base64.b64encode(out).decode("ascii"),
            "result_hash": hashlib.sha256(out).hexdigest(),
            "execution_ms": 5.0, "provider_sig": "AAAA",
            "provider_pubkey_pem": "fake",
        }


def _stack(cap_usd: float, price: float = 0.05):
    budget = BudgetLedger(None)
    budget.set_envelope("acme", cap_usd, "month")
    prov = _W(price=price)
    auction = Auction()
    auction.register(prov)
    svc = JobsService(auction=auction)
    svc.budget = budget
    return svc, budget, prov


def _submit(svc, **overrides):
    """Submit and wait for the terminal state (execution is a
    background task after submit() returns)."""
    kw = dict(
        kind="inference",
        payload={"prompt": "hi", "model": "m", "max_tokens": 8},
        cost_ceiling_usd=0.10,
        latency_ceiling_ms=30_000,
        privacy_class=PRIVACY_PUBLIC,
        quality_floor=0.5,
        requester_identity="tester",
        budget_envelope="acme/support/bot",
    )
    kw.update(overrides)

    async def _run():
        rec = await svc.submit(**kw)
        end = asyncio.get_event_loop().time() + 5.0
        while asyncio.get_event_loop().time() < end:
            if rec.state in ("completed", "completed_partial",
                             "failed", "interrupted", "timeout"):
                return rec
            await asyncio.sleep(0.02)
        return rec

    return asyncio.run(_run())


def test_exhausted_envelope_refuses_before_auction():
    svc, budget, prov = _stack(cap_usd=0.05)   # ceiling 0.10 > cap
    rec = _submit(svc)
    assert rec.state == "failed"
    assert rec.detail and rec.detail.startswith("budget_ledger:")
    assert rec.budget_envelope == "acme/support/bot"
    # Zero economic side effects: the provider was never asked to bid,
    # nothing executed, and no hold is left dangling.
    assert prov.bid_calls == 0
    assert prov.execute_calls == 0
    assert budget.reserve("probe", "acme", 0.05) is None


def test_completed_job_settles_at_cleared_price_not_ceiling():
    svc, budget, prov = _stack(cap_usd=1.00, price=0.05)
    rec = _submit(svc)                          # holds 0.10, clears 0.05
    assert rec.state == "completed"
    assert prov.execute_calls == 1
    rep = budget.report()
    assert rep["total_spend_usd"] == pytest.approx(0.05)
    assert rep["by_envelope"]["acme/support/bot"]["jobs"] == 1
    # The whole remaining cap is free again — no stuck reservation.
    assert budget.reserve("probe", "acme", 0.95) is None
    assert rec.to_info()["budget_envelope"] == "acme/support/bot"


def test_no_provider_matched_releases_the_hold():
    budget = BudgetLedger(None)
    budget.set_envelope("acme", 1.00, "month")
    svc = JobsService(auction=Auction())        # empty auction
    svc.budget = budget
    rec = _submit(svc)
    assert rec.state == "failed"
    assert rec.detail == "no_provider_matched"
    assert budget.reserve("probe", "acme", 1.00) is None


def test_envelope_defaults_to_requester_identity():
    svc, budget, _ = _stack(cap_usd=1.00)
    rec = _submit(svc, budget_envelope=None,
                  requester_identity="team-x")
    assert rec.state == "completed"
    assert rec.budget_envelope == "team-x"
    assert budget.report()["by_envelope"]["team-x"]["jobs"] == 1


def test_no_budget_attached_behaves_exactly_as_before():
    prov = _W()
    auction = Auction()
    auction.register(prov)
    svc = JobsService(auction=auction)          # budget stays None
    rec = _submit(svc)
    assert rec.state == "completed"
    assert rec.budget_envelope == "acme/support/bot"


# ---------------------------------------------------------------------------
# quote-before-run
# ---------------------------------------------------------------------------

def test_quote_prices_without_executing():
    svc, budget, prov = _stack(cap_usd=1.00, price=0.05)
    q = svc.quote(kind="inference",
                  payload={"prompt": "hi"},
                  cost_ceiling_usd=0.10)
    assert q["would_clear"] is True
    assert q["price_usd"] == pytest.approx(0.05)
    assert q["provider_id"] == "p1"
    assert prov.bid_calls == 1
    assert prov.execute_calls == 0              # priced, never run
    # And no budget/job side effects at all.
    assert svc.jobs == {}
    assert budget.report()["total_spend_usd"] == 0.0


def test_quote_reports_why_it_would_not_clear():
    svc, _, _ = _stack(cap_usd=1.00, price=0.50)
    q = svc.quote(kind="inference", payload={"prompt": "hi"},
                  cost_ceiling_usd=0.10)        # ceiling < any bid
    assert q["would_clear"] is False
    assert q["price_usd"] is None
    assert any("ceiling" in r["reason"] for r in q["reasons"])
