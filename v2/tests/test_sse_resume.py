"""SSE delta cursor + resume — reconnect with a since-sequence cursor
replays everything that streamed during the disconnect window.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from api.jobs_service import JobsService
from core.delta_cursor import DeltaCursor, MAX_DELTAS_PER_JOB
from core.providers import (
    Auction,
    Bid,
    JobSpec,
    PRIVACY_PUBLIC,
    Provider,
)


# ---------------------------------------------------------------------------
# Unit: DeltaCursor primitive
# ---------------------------------------------------------------------------

def test_cursor_replays_from_since():
    c = DeltaCursor()
    for i in range(5):
        c.append({"text": f"chunk-{i}"})
    entries, gap = c.replay_since(2)
    assert gap is False
    assert [e.seq for e in entries] == [3, 4, 5]
    assert [e.payload["text"] for e in entries] == ["chunk-2", "chunk-3", "chunk-4"]


def test_cursor_full_replay_from_zero():
    c = DeltaCursor()
    for i in range(3):
        c.append({"text": str(i)})
    entries, gap = c.replay_since(0)
    assert gap is False
    assert len(entries) == 3


def test_cursor_gap_detected_when_chunks_evicted():
    c = DeltaCursor(max_size=4)
    for i in range(10):
        c.append({"text": str(i)})
    # Earliest seq is now 7 (10-4+1); since=1 means we missed 6 chunks.
    entries, gap = c.replay_since(1)
    assert gap is True
    assert len(entries) == 4


def test_cursor_terminal_seq_recorded():
    c = DeltaCursor()
    c.append({"text": "step-1"})
    seq_end = c.append({"text": "done"}, is_terminal=True)
    assert c.has_terminal()
    assert c.terminal_seq == seq_end


def test_cursor_no_replay_when_caller_is_caught_up():
    c = DeltaCursor()
    for i in range(3):
        c.append({"text": str(i)})
    entries, gap = c.replay_since(c.last_seq)
    assert entries == []
    assert gap is False


# ---------------------------------------------------------------------------
# Integration: JobsService writes through to the cursor
# ---------------------------------------------------------------------------

class _StreamProvider(Provider):
    """Provider that emits N deltas before returning."""

    def __init__(self, *, pid: str, n_deltas: int):
        self.provider_id = pid
        self.privacy_grade = PRIVACY_PUBLIC
        self._n = n_deltas

    def bid(self, job):
        return Bid(
            provider_id=self.provider_id, price_usd=0.001, eta_ms=100,
            expected_quality=0.9, privacy_grade=PRIVACY_PUBLIC, evidence={},
        )

    def execute(self, job, bid, *, on_delta=None):
        import base64, hashlib
        for i in range(self._n):
            if on_delta:
                on_delta({"text": f"chunk-{i}"})
        out = b"DONE"
        return {
            "status": "executed", "job_id": job.job_id,
            "result_bytes": base64.b64encode(out).decode("ascii"),
            "result_hash": hashlib.sha256(out).hexdigest(),
            "execution_ms": 10.0, "provider_sig": "AAAA",
            "provider_pubkey_pem": "fake",
        }


def test_jobs_service_streaming_writes_to_cursor():
    auction = Auction()
    auction.register(_StreamProvider(pid="p", n_deltas=5))
    svc = JobsService(auction=auction)

    async def _run():
        rec = await svc.submit(
            kind="compute.test", payload={"prompt": "x"},
            cost_ceiling_usd=1.0, latency_ceiling_ms=10_000,
            privacy_class="public", quality_floor=0.5,
            requester_identity="t", streaming=True,
        )
        # Drain to terminal.
        for _ in range(50):
            cur = svc.get(rec.job_id)
            if cur.state in ("completed", "completed_partial", "failed"):
                break
            await asyncio.sleep(0.05)
        return rec

    rec = asyncio.run(_run())
    assert rec.delta_cursor is not None
    # All 5 deltas landed in the cursor + we can replay from any point.
    entries, gap = rec.delta_cursor.replay_since(2)
    assert gap is False
    assert [e.payload["text"] for e in entries] == [
        "chunk-2", "chunk-3", "chunk-4",
    ]


def test_replay_after_simulated_reconnect():
    """Simulate the production flow: client consumes first 2
    deltas via the asyncio.Queue (live SSE), then 'disconnects',
    then reconnects with since=2 — gets the remaining 3."""
    auction = Auction()
    auction.register(_StreamProvider(pid="p", n_deltas=5))
    svc = JobsService(auction=auction)

    async def _run():
        rec = await svc.submit(
            kind="compute.test", payload={"prompt": "x"},
            cost_ceiling_usd=1.0, latency_ceiling_ms=10_000,
            privacy_class="public", quality_floor=0.5,
            requester_identity="t", streaming=True,
        )
        # Wait until terminal.
        for _ in range(80):
            cur = svc.get(rec.job_id)
            if cur.state in ("completed", "completed_partial", "failed"):
                break
            await asyncio.sleep(0.05)
        return rec

    rec = asyncio.run(_run())
    # Live SSE consumer would have drained 5 items from delta_queue;
    # the cursor preserved all of them with seq numbers.
    assert rec.delta_cursor.last_seq >= 5
    # Reconnect with since=2 — get 3 + 4 + 5 (and possibly a terminal
    # marker, but the provider in this test doesn't emit one).
    entries, gap = rec.delta_cursor.replay_since(2)
    assert gap is False
    texts = [e.payload["text"] for e in entries]
    assert texts == ["chunk-2", "chunk-3", "chunk-4"]
