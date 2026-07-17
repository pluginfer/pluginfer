"""JobStore — durable persistence across gateway restarts.

Pins:
  * InMemoryJobStore round-trips JobRecord state.
  * SQLiteJobStore survives process restart (file-on-disk).
  * Restore-on-boot demotes still-running jobs to `interrupted`.
  * JobsService writes through to the store on every transition.
  * In-flight jobs become `interrupted` after restart, NOT silently lost.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from api.job_store import (
    InMemoryJobStore,
    OPEN_STATES,
    SQLiteJobStore,
    TERMINAL_STATES,
)
from api.jobs_service import JobRecord, JobsService
from core.providers import (
    Auction,
    Bid,
    JobSpec,
    PRIVACY_PUBLIC,
    Provider,
)


def _mk_rec(job_id="j-1", state="queued"):
    return JobRecord(
        job_id=job_id,
        kind="compute.test",
        payload={"prompt": "hello"},
        cost_ceiling_usd=1.0,
        latency_ceiling_ms=10_000,
        privacy_class="public",
        quality_floor=0.5,
        submitted_at_unix=1.0,
        requester_identity="tester",
        state=state,
    )


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

def test_inmemory_put_get_roundtrip():
    s = InMemoryJobStore()
    rec = _mk_rec()
    s.put(rec)
    got = s.get("j-1")
    assert got is rec        # same object — in-memory keeps identity


def test_inmemory_all_open_filters_terminal():
    s = InMemoryJobStore()
    s.put(_mk_rec(job_id="alive", state="running"))
    s.put(_mk_rec(job_id="done", state="completed"))
    open_jobs = s.all_open()
    assert {r.job_id for r in open_jobs} == {"alive"}


def test_inmemory_restore_open_marks_them_interrupted():
    s = InMemoryJobStore()
    s.put(_mk_rec(job_id="stuck", state="running"))
    s.put(_mk_rec(job_id="done", state="completed"))
    survivors = s.restore_open_jobs()
    assert len(survivors) == 1
    assert survivors[0].state == "interrupted"
    assert "gateway_restart" in (survivors[0].detail or "")
    # Idempotent: a second restore returns nothing (already terminal).
    assert s.restore_open_jobs() == []


# ---------------------------------------------------------------------------
# SQLite store — durability across "restart"
# ---------------------------------------------------------------------------

def test_sqlite_put_get_survives_close_and_reopen():
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "jobs.db")
        s1 = SQLiteJobStore(db_path=path)
        rec = _mk_rec(job_id="durable", state="running")
        rec.result_b64 = "QUJDRA=="
        rec.result_hash_hex = "deadbeef"
        rec.matched_provider_pubkey = "pub-1"
        rec.price_locked_usd = 0.0042
        rec.execution_ms = 120.0
        s1.put(rec)
        s1.close()

        # Reopen — emulating a process restart.
        s2 = SQLiteJobStore(db_path=path)
        got = s2.get("durable")
        assert got is not None
        assert got.job_id == "durable"
        assert got.state == "running"
        assert got.result_hash_hex == "deadbeef"
        assert got.matched_provider_pubkey == "pub-1"
        assert got.price_locked_usd == pytest.approx(0.0042)
        s2.close()


def test_sqlite_restore_open_jobs_demotes_running_to_interrupted():
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "jobs.db")
        s1 = SQLiteJobStore(db_path=path)
        s1.put(_mk_rec(job_id="r1", state="running"))
        s1.put(_mk_rec(job_id="q1", state="queued"))
        s1.put(_mk_rec(job_id="m1", state="matched"))
        s1.put(_mk_rec(job_id="d1", state="completed"))
        s1.close()

        s2 = SQLiteJobStore(db_path=path)
        survivors = s2.restore_open_jobs()
        assert {r.job_id for r in survivors} == {"r1", "q1", "m1"}
        for r in survivors:
            assert r.state == "interrupted"
        # Completed job stays completed.
        assert s2.get("d1").state == "completed"
        s2.close()


def test_sqlite_put_idempotent_via_upsert():
    """A second `put` for the same job_id updates, doesn't error."""
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "jobs.db")
        s = SQLiteJobStore(db_path=path)
        rec = _mk_rec(job_id="upd", state="queued")
        s.put(rec)
        rec.state = "running"
        rec.detail = "now-running"
        s.put(rec)
        rec.state = "completed"
        rec.detail = None
        s.put(rec)
        got = s.get("upd")
        assert got.state == "completed"
        assert got.detail is None
        s.close()


# ---------------------------------------------------------------------------
# JobsService write-through
# ---------------------------------------------------------------------------

class _DummyProvider(Provider):
    def __init__(self):
        self.provider_id = "dummy"
        self.privacy_grade = PRIVACY_PUBLIC

    def bid(self, job):
        import base64, hashlib
        return Bid(
            provider_id=self.provider_id, price_usd=0.001, eta_ms=100,
            expected_quality=0.9, privacy_grade=PRIVACY_PUBLIC, evidence={},
        )

    def execute(self, job, bid):
        import base64, hashlib
        out = b"dummy-out"
        return {
            "status": "executed",
            "job_id": job.job_id,
            "result_bytes": base64.b64encode(out).decode("ascii"),
            "result_hash": hashlib.sha256(out).hexdigest(),
            "execution_ms": 100.0,
            "provider_sig": "AAAA",
            "provider_pubkey_pem": "fake",
        }


def test_jobs_service_writes_through_to_store():
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "jobs.db")
        store = SQLiteJobStore(db_path=path)
        auction = Auction()
        auction.register(_DummyProvider())
        svc = JobsService(auction=auction, store=store)

        async def _run():
            rec = await svc.submit(
                kind="compute.test", payload={"prompt": "x"},
                cost_ceiling_usd=1.0, latency_ceiling_ms=10_000,
                privacy_class="public", quality_floor=0.5,
                requester_identity="tester",
            )
            # Drain to terminal.
            for _ in range(50):
                if svc.get(rec.job_id) is not None and svc.get(rec.job_id).state in (
                    "completed", "failed", "completed_partial"
                ):
                    break
                await asyncio.sleep(0.05)
            return rec.job_id

        job_id = asyncio.run(_run())
        # The store has the terminal-state row, NOT just the
        # in-memory dict.
        got = store.get(job_id)
        assert got is not None
        assert got.state == "completed"
        assert got.result_b64 is not None
        store.close()


def test_jobs_service_resumes_state_across_restart():
    """The product test: an in-flight job survives a gateway restart
    as an `interrupted` record instead of vanishing silently."""
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "jobs.db")
        # Run 1: submit a job and write its 'running' state.
        store1 = SQLiteJobStore(db_path=path)
        rec = _mk_rec(job_id="midflight", state="running")
        rec.matched_provider_pubkey = "p-mid"
        rec.price_locked_usd = 0.5
        store1.put(rec)
        store1.close()

        # Run 2: a fresh JobsService construction calls
        # restore_open_jobs which marks it interrupted.
        store2 = SQLiteJobStore(db_path=path)
        svc = JobsService(auction=Auction(), store=store2)
        got = svc.get("midflight")
        assert got is not None
        assert got.state == "interrupted"
        assert got.matched_provider_pubkey == "p-mid"
        assert got.price_locked_usd == pytest.approx(0.5)
        # Still-persistent row also reflects the interrupted state.
        on_disk = store2.get("midflight")
        assert on_disk.state == "interrupted"
        store2.close()
