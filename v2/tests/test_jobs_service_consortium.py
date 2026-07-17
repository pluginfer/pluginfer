"""JobsService routes consortium-eligible jobs through the
sharded execution path. Pins:
  * explicit `payload.consortium = {"size": N}` triggers the N-way
    consortium even when the cost is small.
  * cost_ceiling above the threshold auto-triggers a 4-way default.
  * the aggregate result_b64 contains every member's bytes.
  * partial-failure tolerance — if one member raises, the job
    becomes `completed_partial`, not `failed`.
  * single-winner path still works for normal-sized jobs.
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
from core.consortium_auction import CONSORTIUM_COST_THRESHOLD_USD
from core.providers import (
    Auction,
    Bid,
    JobSpec,
    PRIVACY_PUBLIC,
    Provider,
)


class _Worker(Provider):
    def __init__(self, *, pid: str, output: bytes, raise_exec: bool = False):
        self.provider_id = pid
        self.privacy_grade = PRIVACY_PUBLIC
        self._output = output
        self._raise = raise_exec

    def bid(self, job):
        return Bid(
            provider_id=self.provider_id,
            price_usd=0.001, eta_ms=100,
            expected_quality=0.9,
            privacy_grade=PRIVACY_PUBLIC,
            evidence={},
        )

    def execute(self, job, bid):
        if self._raise:
            raise RuntimeError("simulated worker death")
        return {
            "status": "executed",
            "job_id": job.job_id,
            "result_bytes": base64.b64encode(self._output).decode("ascii"),
            "result_hash": hashlib.sha256(self._output).hexdigest(),
            "execution_ms": 100.0,
            "provider_sig": "AAAA",
            "provider_pubkey_pem": "-----BEGIN PUBLIC KEY-----\nfake\n-----END PUBLIC KEY-----\n",
        }


def _svc(workers):
    a = Auction()
    for w in workers:
        a.register(w)
    return JobsService(auction=a)


def _submit_args(*, cost=1.0, payload=None):
    return dict(
        kind="compute.test",
        payload=payload or {},
        cost_ceiling_usd=cost,
        latency_ceiling_ms=60_000,
        privacy_class="public",
        quality_floor=0.5,
        requester_identity="tester",
    )


async def _wait_terminal(svc, job_id, deadline_s=5.0):
    end = asyncio.get_event_loop().time() + deadline_s
    while asyncio.get_event_loop().time() < end:
        rec = svc.get(job_id)
        if rec and rec.state in ("completed", "completed_partial", "failed"):
            return rec
        await asyncio.sleep(0.05)
    return svc.get(job_id)


def test_explicit_payload_consortium_triggers_sharded_run():
    workers = [
        _Worker(pid="w1", output=b"alpha"),
        _Worker(pid="w2", output=b"beta"),
        _Worker(pid="w3", output=b"gamma"),
    ]
    svc = _svc(workers)

    async def _run():
        rec = await svc.submit(**_submit_args(
            cost=0.01,
            payload={"consortium": {"size": 3}},
        ))
        return await _wait_terminal(svc, rec.job_id)

    rec = asyncio.run(_run())
    assert rec.state == "completed", (rec.state, rec.detail)
    assert rec.matched_provider_pubkey.startswith("consortium:3:")
    combined = base64.b64decode(rec.result_b64)
    assert b"alpha" in combined
    assert b"beta" in combined
    assert b"gamma" in combined
    # Price-locked is the SUM of member prices.
    assert rec.price_locked_usd == pytest.approx(0.003)


def test_big_job_auto_triggers_consortium():
    workers = [_Worker(pid=f"w{i}", output=f"shard-{i}".encode())
               for i in range(4)]
    svc = _svc(workers)

    async def _run():
        rec = await svc.submit(**_submit_args(
            cost=CONSORTIUM_COST_THRESHOLD_USD + 1.0,
        ))
        return await _wait_terminal(svc, rec.job_id)

    rec = asyncio.run(_run())
    assert rec.state == "completed"
    assert "consortium:4" in rec.matched_provider_pubkey
    combined = base64.b64decode(rec.result_b64)
    for i in range(4):
        assert f"shard-{i}".encode() in combined


def test_consortium_partial_failure_tolerated():
    workers = [
        _Worker(pid="ok-1", output=b"ok1"),
        _Worker(pid="ok-2", output=b"ok2"),
        _Worker(pid="dies", output=b"x", raise_exec=True),
    ]
    svc = _svc(workers)

    async def _run():
        rec = await svc.submit(**_submit_args(
            cost=0.01,
            payload={"consortium": {"size": 3}},
        ))
        return await _wait_terminal(svc, rec.job_id)

    rec = asyncio.run(_run())
    assert rec.state == "completed_partial", (rec.state, rec.detail)
    combined = base64.b64decode(rec.result_b64)
    assert b"ok1" in combined
    assert b"ok2" in combined


def test_consortium_all_members_fail_marks_job_failed():
    workers = [
        _Worker(pid=f"dies-{i}", output=b"x", raise_exec=True)
        for i in range(2)
    ]
    svc = _svc(workers)

    async def _run():
        rec = await svc.submit(**_submit_args(
            cost=0.01,
            payload={"consortium": {"size": 2}},
        ))
        return await _wait_terminal(svc, rec.job_id)

    rec = asyncio.run(_run())
    assert rec.state == "failed"
    assert "all_consortium_members_failed" in (rec.detail or "")


def test_small_job_does_not_trigger_consortium():
    workers = [
        _Worker(pid="solo", output=b"single-winner-result"),
        _Worker(pid="other", output=b"alt"),
    ]
    svc = _svc(workers)

    async def _run():
        rec = await svc.submit(**_submit_args(cost=0.01))
        return await _wait_terminal(svc, rec.job_id)

    rec = asyncio.run(_run())
    assert rec.state == "completed"
    # Single-winner path means matched_provider_pubkey is the
    # provider_id of the auction winner, not the "consortium:N" prefix.
    assert not rec.matched_provider_pubkey.startswith("consortium:")


def test_consortium_tensor_parallel_raises_honestly():
    """The design notes scopes data-parallel + diloco today;
    tensor-parallel needs core.diloco_serialize integration. The
    JobsService must surface the NotImplementedError as a clean
    failure, not crash the runtime."""
    workers = [_Worker(pid="w1", output=b"x"), _Worker(pid="w2", output=b"y")]
    svc = _svc(workers)

    async def _run():
        rec = await svc.submit(**_submit_args(
            cost=0.01,
            payload={"consortium": {"size": 2, "mode": "tensor-parallel"}},
        ))
        return await _wait_terminal(svc, rec.job_id)

    rec = asyncio.run(_run())
    assert rec.state == "failed"
    assert "consortium_mode_unsupported" in (rec.detail or "")
