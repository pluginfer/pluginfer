"""JobsService + BuyerLedger end-to-end economic flow.

Pins the full money path through a real job:
  1. Buyer wallet has X. Submit a job priced at Y → balance becomes
     X-Y (locked).
  2. Provider executes successfully → buyer's locked drains to zero,
     provider gets Y*(1-c), treasury gets Y*c.
  3. Provider raises → buyer's locked refunds to available. Treasury
     unchanged.
  4. Insufficient-funds buyer → job fails at submit, provider never
     touched.
  5. Consortium with partial failure → successful members paid their
     share; failed members' share refunded to buyer; treasury gets
     commission only on the released portion.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import sys
from decimal import Decimal
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from api.jobs_service import JobsService
from core.buyer_ledger import (
    BuyerLedger,
    COMMISSION_RATE,
    TREASURY_WALLET_ID,
)
from core.providers import (
    Auction,
    Bid,
    JobSpec,
    PRIVACY_PUBLIC,
    Provider,
)


class _PricedWorker(Provider):
    def __init__(self, *, pid: str, price: float, output: bytes,
                 raises: bool = False):
        self.provider_id = pid
        self.privacy_grade = PRIVACY_PUBLIC
        self._price = price
        self._output = output
        self._raises = raises

    def bid(self, job):
        return Bid(
            provider_id=self.provider_id, price_usd=self._price, eta_ms=100,
            expected_quality=0.9, privacy_grade=PRIVACY_PUBLIC, evidence={},
        )

    def execute(self, job, bid):
        if self._raises:
            raise RuntimeError("worker died")
        return {
            "status": "executed", "job_id": job.job_id,
            "result_bytes": base64.b64encode(self._output).decode("ascii"),
            "result_hash": hashlib.sha256(self._output).hexdigest(),
            "execution_ms": 100.0, "provider_sig": "AAAA",
            "provider_pubkey_pem": "fake",
        }


def _submit(svc, *, buyer="alice", price_floor=0.0, payload=None,
            cost_ceiling=1.0):
    return svc.submit(
        kind="compute.test", payload=payload or {},
        cost_ceiling_usd=cost_ceiling, latency_ceiling_ms=10_000,
        privacy_class="public", quality_floor=price_floor,
        requester_identity="tester", buyer_wallet_id=buyer,
    )


async def _wait_terminal(svc, job_id, deadline_s=5.0):
    end = asyncio.get_event_loop().time() + deadline_s
    while asyncio.get_event_loop().time() < end:
        rec = svc.get(job_id)
        if rec and rec.state in (
            "completed", "completed_partial", "failed", "interrupted",
        ):
            return rec
        await asyncio.sleep(0.05)
    return svc.get(job_id)


# ---------------------------------------------------------------------------
# Happy path: provider gets paid net of commission, buyer's balance debited
# ---------------------------------------------------------------------------

def test_successful_job_credits_provider_minus_commission():
    ledger = BuyerLedger()
    ledger.credit("alice", Decimal("100.0"))
    auction = Auction()
    auction.register(_PricedWorker(pid="bob-gpu", price=0.50, output=b"x"))
    svc = JobsService(auction=auction, ledger=ledger)

    async def _run():
        rec = await _submit(svc)
        return await _wait_terminal(svc, rec.job_id)

    rec = asyncio.run(_run())
    assert rec.state == "completed"
    alice = ledger.get_wallet("alice")
    bob = ledger.get_wallet("bob-gpu")
    treas = ledger.get_wallet(TREASURY_WALLET_ID)
    # Alice paid 0.50; bob got 0.45; treasury got 0.05 (at 10% rate).
    expected_commission = Decimal("0.50") * COMMISSION_RATE
    expected_to_bob = Decimal("0.50") - expected_commission
    assert alice.available_usd == Decimal("100.0") - Decimal("0.50")
    assert alice.locked_usd == Decimal("0")
    assert bob.available_usd == expected_to_bob
    assert treas.available_usd == expected_commission


# ---------------------------------------------------------------------------
# Failure path: 100% refund, treasury untouched
# ---------------------------------------------------------------------------

def test_failed_job_refunds_buyer_and_treasury_unchanged():
    ledger = BuyerLedger()
    ledger.credit("alice", Decimal("100.0"))
    auction = Auction()
    auction.register(_PricedWorker(
        pid="bob-broken", price=0.50, output=b"x", raises=True,
    ))
    svc = JobsService(auction=auction, ledger=ledger)

    async def _run():
        rec = await _submit(svc)
        return await _wait_terminal(svc, rec.job_id)

    rec = asyncio.run(_run())
    assert rec.state == "failed"
    alice = ledger.get_wallet("alice")
    treas = ledger.get_wallet(TREASURY_WALLET_ID)
    assert alice.available_usd == Decimal("100.0")
    assert alice.locked_usd == Decimal("0")
    assert treas.available_usd == Decimal("0")


# ---------------------------------------------------------------------------
# Insufficient funds: rejected at submit, provider never touched
# ---------------------------------------------------------------------------

def test_underfunded_buyer_pauses_instead_of_dropping():
    """Wallet-drained == paused, not lost. Buyer sees a clear
    "top up to continue" status and can resume the job after
    crediting the wallet. Same pattern as Claude Code pausing on
    usage limits and continuing on resume."""
    ledger = BuyerLedger()
    ledger.credit("alice", Decimal("0.10"))
    auction = Auction()
    auction.register(_PricedWorker(pid="bob", price=0.50, output=b"x"))
    svc = JobsService(auction=auction, ledger=ledger)

    async def _run():
        rec = await _submit(svc)
        # The submit() path sets paused_funding synchronously; no
        # need to wait for terminal state — the job is alive,
        # waiting for resume.
        return svc.get(rec.job_id)

    rec = asyncio.run(_run())
    assert rec.state == "paused_funding"
    assert "top up" in (rec.detail or "").lower()
    # Treasury and bob untouched until resume succeeds.
    assert ledger.get_wallet(TREASURY_WALLET_ID).available_usd == Decimal("0")
    bob = ledger.get_wallet("bob")
    assert bob is None or bob.available_usd == Decimal("0")
    # Alice's balance is unchanged (no lock placed).
    assert ledger.get_wallet("alice").available_usd == Decimal("0.10")


# ---------------------------------------------------------------------------
# Consortium partial-failure economics
# ---------------------------------------------------------------------------

def test_consortium_partial_pays_survivors_refunds_failures():
    ledger = BuyerLedger()
    ledger.credit("alice", Decimal("100.0"))
    auction = Auction()
    auction.register(_PricedWorker(pid="p1", price=1.0, output=b"a"))
    auction.register(_PricedWorker(pid="p2", price=2.0, output=b"b"))
    auction.register(_PricedWorker(pid="p3", price=3.0, output=b"c", raises=True))
    svc = JobsService(auction=auction, ledger=ledger)

    async def _run():
        rec = await svc.submit(
            kind="compute.test",
            payload={"consortium": {"size": 3}},
            cost_ceiling_usd=10.0, latency_ceiling_ms=10_000,
            privacy_class="public", quality_floor=0.5,
            requester_identity="tester", buyer_wallet_id="alice",
        )
        return await _wait_terminal(svc, rec.job_id)

    rec = asyncio.run(_run())
    assert rec.state == "completed_partial", (rec.state, rec.detail)
    # Total locked was 6 (sum of bids). p1 (1.0) + p2 (2.0) succeeded.
    # p3 (3.0) failed → refunded.
    alice = ledger.get_wallet("alice")
    c = COMMISSION_RATE
    p1 = ledger.get_wallet("p1")
    p2 = ledger.get_wallet("p2")
    p3 = ledger.get_wallet("p3")
    assert p1.available_usd == Decimal("1.0") - Decimal("1.0") * c
    assert p2.available_usd == Decimal("2.0") - Decimal("2.0") * c
    # p3 didn't exist as a wallet OR has zero — never paid.
    assert p3 is None or p3.available_usd == Decimal("0")
    # Alice: started 100, locked 6, released 3 to consortium, 3 refunded.
    assert alice.available_usd == Decimal("100.0") - Decimal("3.0")
    # Treasury: 10% of $3 successfully released = 0.30.
    treas = ledger.get_wallet(TREASURY_WALLET_ID)
    assert treas.available_usd == Decimal("3.0") * c
