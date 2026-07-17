"""Wallet-pause/resume/abandon — buyers never lose work or money for
a recoverable accounting condition.

Invariants pinned:
  * Underfunded submit ⇒ `paused_funding` (not `failed`); JobRecord
    keeps the pending winner/spec metadata so resume can dispatch
    without re-running the auction.
  * `resume_funding(job_id)` re-locks + dispatches; if still
    underfunded it stays paused with an updated detail.
  * `abandon(job_id)` while paused → terminal `abandoned` state;
    any partial output is preserved on the record.
  * Wallet credit + resume → job executes + completes + provider
    paid + treasury commissioned.
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


class _W(Provider):
    def __init__(self, *, pid: str, price: float, output: bytes):
        self.provider_id = pid
        self.privacy_grade = PRIVACY_PUBLIC
        self._price = price
        self._output = output

    def bid(self, job):
        return Bid(
            provider_id=self.provider_id, price_usd=self._price, eta_ms=100,
            expected_quality=0.9, privacy_grade=PRIVACY_PUBLIC, evidence={},
        )

    def execute(self, job, bid):
        return {
            "status": "executed", "job_id": job.job_id,
            "result_bytes": base64.b64encode(self._output).decode("ascii"),
            "result_hash": hashlib.sha256(self._output).hexdigest(),
            "execution_ms": 100.0, "provider_sig": "AAAA",
            "provider_pubkey_pem": "fake",
        }


async def _wait_state(svc, job_id, target_states, deadline_s=5.0):
    if isinstance(target_states, str):
        target_states = (target_states,)
    end = asyncio.get_event_loop().time() + deadline_s
    while asyncio.get_event_loop().time() < end:
        rec = svc.get(job_id)
        if rec and rec.state in target_states:
            return rec
        await asyncio.sleep(0.05)
    return svc.get(job_id)


def _svc_with_one_priced_worker(price=0.50):
    ledger = BuyerLedger()
    auction = Auction()
    auction.register(_W(pid="provider-1", price=price, output=b"result"))
    svc = JobsService(auction=auction, ledger=ledger)
    return svc, ledger


# ---------------------------------------------------------------------------
# Underfunded → paused_funding (NOT failed)
# ---------------------------------------------------------------------------

def test_underfunded_submit_pauses_does_not_drop():
    svc, ledger = _svc_with_one_priced_worker(price=0.50)
    ledger.credit("alice", Decimal("0.10"))    # not enough for $0.50 job

    async def _run():
        rec = await svc.submit(
            kind="compute.test", payload={"prompt": "x"},
            cost_ceiling_usd=1.0, latency_ceiling_ms=10_000,
            privacy_class="public", quality_floor=0.5,
            requester_identity="tester",
            buyer_wallet_id="alice",
        )
        return await _wait_state(svc, rec.job_id, "paused_funding")

    rec = asyncio.run(_run())
    assert rec.state == "paused_funding"
    # The job carries clear remediation text the buyer can act on.
    assert "top up" in (rec.detail or "")
    # The pending winner/spec are stashed so resume doesn't re-auction.
    assert getattr(rec, "_pending_winner", None) is not None
    assert getattr(rec, "_pending_spec", None) is not None
    # Money path unchanged: alice's $0.10 still hers, treasury still empty.
    assert ledger.get_wallet("alice").available_usd == Decimal("0.10")
    assert ledger.get_wallet(TREASURY_WALLET_ID).available_usd == Decimal("0")


# ---------------------------------------------------------------------------
# Resume after top-up: dispatches + completes
# ---------------------------------------------------------------------------

def test_resume_after_topup_dispatches_and_completes():
    svc, ledger = _svc_with_one_priced_worker(price=0.50)
    ledger.credit("alice", Decimal("0.10"))

    async def _run():
        rec = await svc.submit(
            kind="compute.test", payload={"prompt": "x"},
            cost_ceiling_usd=1.0, latency_ceiling_ms=10_000,
            privacy_class="public", quality_floor=0.5,
            requester_identity="tester", buyer_wallet_id="alice",
        )
        await _wait_state(svc, rec.job_id, "paused_funding")
        # Top up the wallet.
        ledger.credit("alice", Decimal("1.0"))
        await svc.resume_funding(rec.job_id)
        return await _wait_state(svc, rec.job_id, ("completed", "failed"))

    rec = asyncio.run(_run())
    assert rec.state == "completed", (rec.state, rec.detail)
    # Provider got paid (1-c) × 0.50; treasury got c × 0.50.
    c = COMMISSION_RATE
    assert ledger.get_wallet("provider-1").available_usd == Decimal("0.50") - Decimal("0.50") * c
    assert ledger.get_wallet(TREASURY_WALLET_ID).available_usd == Decimal("0.50") * c


def test_resume_when_still_underfunded_stays_paused():
    svc, ledger = _svc_with_one_priced_worker(price=0.50)
    ledger.credit("alice", Decimal("0.10"))

    async def _run():
        rec = await svc.submit(
            kind="compute.test", payload={"prompt": "x"},
            cost_ceiling_usd=1.0, latency_ceiling_ms=10_000,
            privacy_class="public", quality_floor=0.5,
            requester_identity="tester", buyer_wallet_id="alice",
        )
        await _wait_state(svc, rec.job_id, "paused_funding")
        # Top-up that's STILL not enough.
        ledger.credit("alice", Decimal("0.20"))
        return await svc.resume_funding(rec.job_id)

    rec = asyncio.run(_run())
    assert rec.state == "paused_funding"
    assert "still_underfunded" in (rec.detail or "") or "underfunded" in (rec.detail or "")


# ---------------------------------------------------------------------------
# Abandon: terminal state, partial output preserved if any
# ---------------------------------------------------------------------------

def test_abandon_paused_job_marks_terminal_state():
    svc, ledger = _svc_with_one_priced_worker(price=0.50)
    ledger.credit("alice", Decimal("0.10"))

    async def _run():
        rec = await svc.submit(
            kind="compute.test", payload={"prompt": "x"},
            cost_ceiling_usd=1.0, latency_ceiling_ms=10_000,
            privacy_class="public", quality_floor=0.5,
            requester_identity="tester", buyer_wallet_id="alice",
        )
        await _wait_state(svc, rec.job_id, "paused_funding")
        return await svc.abandon(rec.job_id)

    rec = asyncio.run(_run())
    assert rec.state in ("abandoned", "abandoned_partial")
    # No partial deliverable on a pre-execution pause.
    assert rec.state == "abandoned"
    # Alice's money is untouched — she never paid for work that
    # didn't run.
    assert ledger.get_wallet("alice").available_usd == Decimal("0.10")
    assert ledger.get_wallet(TREASURY_WALLET_ID).available_usd == Decimal("0")


def test_abandon_after_topup_with_partial_result_delivers():
    """Synthetic: simulate a paused job that already has partial
    result bytes attached (from a prior aborted run). Abandon
    transitions to `abandoned_partial` and preserves the bytes."""
    svc, ledger = _svc_with_one_priced_worker(price=0.50)
    ledger.credit("alice", Decimal("0.10"))

    async def _run():
        rec = await svc.submit(
            kind="compute.test", payload={"prompt": "x"},
            cost_ceiling_usd=1.0, latency_ceiling_ms=10_000,
            privacy_class="public", quality_floor=0.5,
            requester_identity="tester", buyer_wallet_id="alice",
        )
        await _wait_state(svc, rec.job_id, "paused_funding")
        # Pretend a previous run delivered partial bytes already.
        rec.result_b64 = "cGFydGlhbC1ieXRlcw=="
        rec.result_hash_hex = "partialhash"
        return await svc.abandon(rec.job_id, deliver_partial=True)

    rec = asyncio.run(_run())
    assert rec.state == "abandoned_partial"
    assert rec.result_b64 == "cGFydGlhbC1ieXRlcw=="
