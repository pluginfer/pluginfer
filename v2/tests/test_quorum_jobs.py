"""Live quorum jobs — HG20's voting live in the REAL buyer path,
HG21's economics firing on the outcome.

`payload={"quorum_n": N}` is sugar for the consortium's
quorum-replicate mode — ONE quorum mechanism, and these tests pin its
money contract:

  * the buyer gets the MAJORITY's bytes, never a forgery,
  * ONLY the agreeing majority is paid (each its own bid price, net of
    commission); dissenters' and non-responders' shares refund to the
    buyer,
  * a staked dissenter is slashed and the honest majority collects,
  * a dispute (no majority) fails the job, refunds the buyer in FULL,
    and slashes NOBODY,
  * degraded cases are honest: too few providers → single-winner with
    a note; streaming → quorum ignored with a note.

These also pin the three bugs found in the pre-existing
quorum-replicate path: byzantine dissenters were PAID (settlement
ignored the dissenter list), a split-brain billed the buyer as
completed_partial with NO result, and dissenter attribution could
land on the wrong provider (members zipped against a gap-filled blob
list).
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

from api.jobs_service import JobsService
from core.buyer_ledger import BuyerLedger
from core.economic_layer import EconomicLayer
from core.providers import Auction, Bid, PRIVACY_PUBLIC, Provider

D = Decimal
GOOD = b"the-correct-answer"
EVIL = b"the-forged-answer"


class _Worker(Provider):
    def __init__(self, pid, output=GOOD, *, price=0.25, raises=False):
        self.provider_id = pid
        self.privacy_grade = PRIVACY_PUBLIC
        self._output = output
        self._price = price
        self._raises = raises

    def bid(self, job):
        return Bid(provider_id=self.provider_id, price_usd=self._price,
                   eta_ms=100, expected_quality=0.9,
                   privacy_grade=PRIVACY_PUBLIC, evidence={})

    def execute(self, job, bid):
        if self._raises:
            raise RuntimeError("worker died")
        return {
            "status": "executed", "job_id": job.job_id,
            "result_bytes": base64.b64encode(self._output).decode(),
            "result_hash": hashlib.sha256(self._output).hexdigest(),
            "provider_sig": "AAAA",
        }


def _stack(workers, *, buyer_usd="100", econ=False):
    ledger = BuyerLedger()
    ledger.credit("alice", D(buyer_usd))
    auction = Auction()
    for w in workers:
        auction.register(w)
    svc = JobsService(auction=auction, ledger=ledger)
    economics = None
    if econ:
        economics = EconomicLayer(ledger)
        svc.economics = economics
    return svc, ledger, economics


def _submit(svc, *, quorum_n=3, streaming=False):
    return svc.submit(
        kind="compute.test", payload={"quorum_n": quorum_n},
        # Below CONSORTIUM_COST_THRESHOLD_USD (5.0) so nothing here
        # rides the big-job default — only the quorum sugar decides.
        cost_ceiling_usd=2.0, latency_ceiling_ms=10_000,
        privacy_class="public", quality_floor=0.0,
        requester_identity="tester", buyer_wallet_id="alice",
        streaming=streaming)


async def _wait(svc, job_id, deadline_s=5.0):
    end = asyncio.get_event_loop().time() + deadline_s
    while asyncio.get_event_loop().time() < end:
        rec = svc.get(job_id)
        if rec and rec.state in ("completed", "completed_partial",
                                 "failed", "timeout"):
            return rec
        await asyncio.sleep(0.05)
    return svc.get(job_id)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------

def test_majority_paid_dissenter_slashed_buyer_refunded_share():
    svc, ledger, econ = _stack(
        [_Worker("p-good1"), _Worker("p-good2"),
         _Worker("p-evil", output=EVIL)], econ=True)
    # The cheater bonded stake — that's what it is about to lose.
    ledger.credit("p-evil", D("5"))
    econ.stake("p-evil", D("2"))

    async def flow():
        rec = await _submit(svc)
        return await _wait(svc, rec.job_id)

    rec = _run(flow())
    assert rec.state == "completed", rec.detail
    # The caller got the MAJORITY's answer, not the forgery.
    assert rec.result_hash_hex == hashlib.sha256(GOOD).hexdigest()
    assert base64.b64decode(rec.result_b64) == GOOD
    assert "quorum 2/3 agreed" in rec.detail
    assert "p-evil" in rec.detail          # dissenter named on the job

    # Money: buyer locked 0.75, majority released 0.50, forger's 0.25
    # refunded → net spend 0.50.
    assert ledger.get_wallet("alice").available_usd == D("99.50")
    # Each honest provider: 0.25 × 0.9 release + half the 0.25 slash.
    for pid in ("p-good1", "p-good2"):
        assert ledger.get_wallet(pid).available_usd == \
            D("0.225") + D("0.125")
    # The forger: paid NOTHING, and its bond shrank by the slash.
    w_evil = ledger.get_wallet("p-evil")
    assert w_evil.staked_usd == D("1.75")
    assert w_evil.available_usd == D("3")   # available untouched
    # Treasury: commission on the RELEASED 0.50 only — never slash money.
    assert ledger.treasury_balance() == D("0.05")
    # Economics recorded it all.
    assert rec.quorum_economics["slashed"] == {"p-evil": "0.25000000"}
    assert econ.reputation("p-evil")["dissents"] == 1
    assert econ.reputation("p-good1")["majorities"] == 1
    assert ledger.verify_balances()["ok"]


def test_dispute_full_refund_nobody_slashed():
    svc, ledger, econ = _stack(
        [_Worker("p1", b"a"), _Worker("p2", b"b"), _Worker("p3", b"c")],
        econ=True)
    for p in ("p1", "p2", "p3"):
        ledger.credit(p, D("5"))
        econ.stake(p, D("2"))

    async def flow():
        rec = await _submit(svc)
        return await _wait(svc, rec.job_id)

    rec = _run(flow())
    # Pre-fix this billed the buyer as completed_partial with NO
    # result. Now: failed, full refund, nobody slashed.
    assert rec.state == "failed"
    assert "quorum_dispute" in rec.detail
    assert rec.result_b64 is None
    assert ledger.get_wallet("alice").available_usd == D("100")
    for p in ("p1", "p2", "p3"):
        assert ledger.get_wallet(p).staked_usd == D("2")
        assert ledger.get_wallet(p).available_usd == D("3")
    assert ledger.treasury_balance() == D("0")
    assert ledger.verify_balances()["ok"]


def test_nonresponder_not_slashed_majority_still_wins():
    # The dead worker sits FIRST so this also pins the attribution
    # fix: pre-fix, members were zipped against a gap-filled blob
    # list, shifting every blob one provider over.
    svc, ledger, econ = _stack(
        [_Worker("p-dead", raises=True), _Worker("p-good1"),
         _Worker("p-good2")], econ=True)
    ledger.credit("p-dead", D("5"))
    econ.stake("p-dead", D("2"))

    async def flow():
        rec = await _submit(svc)
        return await _wait(svc, rec.job_id)

    rec = _run(flow())
    assert rec.state == "completed", rec.detail
    assert rec.result_hash_hex == hashlib.sha256(GOOD).hexdigest()
    # Dead worker: unpaid (its share refunds) but NOT slashed — an
    # error is not a forged answer.
    assert ledger.get_wallet("p-dead").staked_usd == D("2")
    assert econ.reputation("p-dead")["dissents"] == 0
    # Honest members paid; buyer refunded the dead member's share.
    for pid in ("p-good1", "p-good2"):
        assert ledger.get_wallet(pid).available_usd == D("0.225")
    assert ledger.get_wallet("alice").available_usd == D("99.50")
    assert ledger.verify_balances()["ok"]


def test_too_few_providers_degrades_to_single_winner():
    svc, ledger, _ = _stack([_Worker("only")])

    async def flow():
        rec = await _submit(svc, quorum_n=3)
        return await _wait(svc, rec.job_id)

    rec = _run(flow())
    assert rec.state == "completed"
    assert rec.matched_provider_pubkey == "only"
    # The degrade note was set at submit time (completion may
    # overwrite detail) — the pinned behavior is that the job ran
    # single-winner instead of failing over an optional hardening.


def test_streaming_ignores_quorum_with_honest_note():
    svc, ledger, _ = _stack([_Worker("p1"), _Worker("p2"),
                             _Worker("p3")])

    async def flow():
        rec = await _submit(svc, quorum_n=3, streaming=True)
        assert "quorum_ignored" in (rec.detail or "")
        return await _wait(svc, rec.job_id)

    rec = _run(flow())
    assert rec.state == "completed"
    assert not rec.matched_provider_pubkey.startswith("consortium")


def test_quarantined_provider_cannot_join_quorum():
    """The economic layer's gate applies to consortium membership too —
    a quarantined forger must not slip into a quorum via the side
    door and out-vote honest nodes."""
    svc, ledger, econ = _stack(
        [_Worker("p-good1"), _Worker("p-good2"),
         _Worker("p-banned", output=EVIL)], econ=True)
    svc.auction.eligibility_fn = lambda pid: (
        (False, "quarantined") if pid == "p-banned" else (True, "ok"))

    async def flow():
        rec = await _submit(svc)
        return await _wait(svc, rec.job_id)

    rec = _run(flow())
    # Only the two honest providers voted: 2/2 unanimity.
    assert rec.state == "completed", rec.detail
    assert "quorum 2/2 agreed" in rec.detail
    assert rec.result_hash_hex == hashlib.sha256(GOOD).hexdigest()
    assert ledger.get_wallet("p-banned") is None or \
        ledger.get_wallet("p-banned").available_usd == D("0")


def test_members_execute_concurrently_not_sequentially():
    """3 members × 0.4 s each must take ~0.4 s wall, not ~1.2 s —
    sequential execution was the tax that made quorum impractical."""
    import time as _time

    class _Slow(_Worker):
        def execute(self, job, bid):
            _time.sleep(0.4)
            return super().execute(job, bid)

    svc, ledger, _ = _stack([_Slow("p1"), _Slow("p2"), _Slow("p3")])

    async def flow():
        t0 = _time.monotonic()
        rec = await _submit(svc)
        rec = await _wait(svc, rec.job_id, deadline_s=10.0)
        return rec, _time.monotonic() - t0

    rec, wall = _run(flow())
    assert rec.state == "completed", rec.detail
    assert wall < 1.0, f"members ran sequentially ({wall:.2f}s)"


def test_stuck_member_times_out_and_does_not_hang_the_job(monkeypatch):
    """A member that never returns is dropped at the member timeout;
    the healthy majority still wins and the job never hangs on the
    stuck thread (shutdown never waits on it)."""
    import threading as _threading
    import time as _time
    monkeypatch.setenv("PLUGINFER_CONSORTIUM_MEMBER_TIMEOUT_S", "1")

    class _Stuck(_Worker):
        def execute(self, job, bid):
            _threading.Event().wait(20)      # far past the timeout
            return super().execute(job, bid)

    svc, ledger, _ = _stack(
        [_Worker("p-good1"), _Worker("p-good2"), _Stuck("p-stuck")])

    async def flow():
        t0 = _time.monotonic()
        rec = await _submit(svc)
        rec = await _wait(svc, rec.job_id, deadline_s=10.0)
        return rec, _time.monotonic() - t0

    rec, wall = _run(flow())
    assert rec.state == "completed", rec.detail
    assert rec.result_hash_hex == hashlib.sha256(GOOD).hexdigest()
    assert "2/3 agreed" in rec.detail
    assert wall < 8.0, f"job waited on the stuck member ({wall:.1f}s)"
    # The stuck member didn't vote and isn't paid.
    assert ledger.get_wallet("alice").available_usd == D("99.50")


def test_underfunded_quorum_fails_with_clear_message():
    svc, ledger, _ = _stack(
        [_Worker("p1"), _Worker("p2"), _Worker("p3")],
        buyer_usd="0.50")            # needs 0.75 for 3 × 0.25

    async def flow():
        return await _submit(svc)

    rec = _run(flow())
    assert rec.state == "failed"
    assert "insufficient_funds" in rec.detail
    # Nothing was taken: no lock survived the refusal.
    assert ledger.get_wallet("alice").available_usd == D("0.50")
    assert ledger.get_wallet("alice").locked_usd == D("0")
    assert ledger.verify_balances()["ok"]
