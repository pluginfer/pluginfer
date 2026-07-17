"""Buyer-side dispute window — claim a refund within N seconds of
completion. Buyer wallet credit restored, provider clawed back,
shortfall (if any) parked for slashing.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import sys
import time
from decimal import Decimal
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from api.jobs_service import JobsService
from core.buyer_ledger import BuyerLedger, COMMISSION_RATE, TREASURY_WALLET_ID
from core.providers import (
    Auction,
    Bid,
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


async def _wait(svc, job_id, deadline_s=5.0):
    end = asyncio.get_event_loop().time() + deadline_s
    while asyncio.get_event_loop().time() < end:
        rec = svc.get(job_id)
        if rec and rec.state in (
            "completed", "completed_partial", "failed",
        ):
            return rec
        await asyncio.sleep(0.05)
    return svc.get(job_id)


def _build():
    ledger = BuyerLedger()
    ledger.credit("alice", Decimal("10.0"))
    auction = Auction()
    auction.register(_W(pid="bob", price=1.0, output=b"wrong-answer"))
    svc = JobsService(auction=auction, ledger=ledger)
    return svc, ledger


def test_dispute_within_window_refunds_buyer_and_claws_back_provider():
    svc, ledger = _build()

    async def _run():
        rec = await svc.submit(
            kind="compute.test", payload={"prompt": "x"},
            cost_ceiling_usd=10.0, latency_ceiling_ms=10_000,
            privacy_class="public", quality_floor=0.5,
            requester_identity="t", buyer_wallet_id="alice",
        )
        rec = await _wait(svc, rec.job_id)
        assert rec.state == "completed"
        return await svc.dispute(rec.job_id, reason="output_wrong")

    rec = asyncio.run(_run())
    assert rec.state == "disputed_refunded"
    # Bob was clawed back; alice got the full $1 refunded (provider
    # share + treasury commission).
    alice = ledger.get_wallet("alice")
    bob = ledger.get_wallet("bob")
    treas = ledger.get_wallet(TREASURY_WALLET_ID)
    assert alice.available_usd == Decimal("10.0")
    assert bob.available_usd == Decimal("0")
    assert treas.available_usd == Decimal("0")


def test_dispute_after_window_rejected(monkeypatch):
    monkeypatch.setenv("PLUGINFER_DISPUTE_WINDOW_S", "0.1")
    svc, ledger = _build()

    async def _run():
        rec = await svc.submit(
            kind="compute.test", payload={"prompt": "x"},
            cost_ceiling_usd=10.0, latency_ceiling_ms=10_000,
            privacy_class="public", quality_floor=0.5,
            requester_identity="t", buyer_wallet_id="alice",
        )
        rec = await _wait(svc, rec.job_id)
        # Sleep past the dispute window.
        await asyncio.sleep(0.5)
        return await svc.dispute(rec.job_id)

    rec = asyncio.run(_run())
    assert rec.state == "completed"  # state unchanged
    assert "window_closed" in (rec.detail or "")
    # Bob keeps his earnings.
    c = COMMISSION_RATE
    bob = ledger.get_wallet("bob")
    assert bob.available_usd == Decimal("1.0") - Decimal("1.0") * c


def test_dispute_on_non_completed_job_rejected():
    svc, ledger = _build()

    async def _run():
        # Submit an underfunded job → paused.
        ledger.get_wallet("alice").available_usd = Decimal("0.001")
        rec = await svc.submit(
            kind="compute.test", payload={"prompt": "x"},
            cost_ceiling_usd=10.0, latency_ceiling_ms=10_000,
            privacy_class="public", quality_floor=0.5,
            requester_identity="t", buyer_wallet_id="alice",
        )
        return await svc.dispute(rec.job_id)

    rec = asyncio.run(_run())
    # Dispute rejected because the job never completed.
    assert "dispute_rejected" in (rec.detail or "")


def test_dispute_idempotent_does_not_double_refund():
    """Calling dispute twice doesn't compound the refund."""
    svc, ledger = _build()

    async def _run():
        rec = await svc.submit(
            kind="compute.test", payload={"prompt": "x"},
            cost_ceiling_usd=10.0, latency_ceiling_ms=10_000,
            privacy_class="public", quality_floor=0.5,
            requester_identity="t", buyer_wallet_id="alice",
        )
        rec = await _wait(svc, rec.job_id)
        await svc.dispute(rec.job_id)
        await svc.dispute(rec.job_id)     # second call
        return svc.get(rec.job_id)

    rec = asyncio.run(_run())
    # Alice still has exactly $10 — not $11.
    assert ledger.get_wallet("alice").available_usd == Decimal("10.0")
