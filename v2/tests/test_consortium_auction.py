"""Consortium auction — N best providers, sharded execution.

The unit tests pin: ranking is consistent with the single-winner
auction; quota math is correct; failed-member tolerance is real;
tensor-parallel mode raises (not silently lies).
"""

from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from core.consortium_auction import (  # noqa: E402
    CONSORTIUM_COST_THRESHOLD_USD,
    Consortium,
    ConsortiumExecution,
    execute_consortium,
    job_needs_consortium,
    select_consortium,
)
from core.providers import (  # noqa: E402
    Auction,
    Bid,
    JobSpec,
    PRIVACY_PUBLIC,
    Provider,
)


class _Worker(Provider):
    """A provider whose bid and result are configurable per fixture."""

    def __init__(self, *, pid: str, price: float, eta: int,
                 quality: float, output: bytes,
                 raise_on_execute: bool = False):
        self.provider_id = pid
        self.privacy_grade = PRIVACY_PUBLIC
        self._price = price
        self._eta = eta
        self._quality = quality
        self._output = output
        self._raise = raise_on_execute

    def bid(self, job: JobSpec):
        return Bid(
            provider_id=self.provider_id,
            price_usd=self._price, eta_ms=self._eta,
            expected_quality=self._quality,
            privacy_grade=PRIVACY_PUBLIC,
            evidence={},
        )

    def execute(self, job, bid):
        if self._raise:
            raise RuntimeError("boom")
        return {
            "status": "executed",
            "job_id": job.job_id,
            "result_bytes": base64.b64encode(self._output).decode("ascii"),
            "result_hash": hashlib.sha256(self._output).hexdigest(),
            "execution_ms": float(self._eta),
            "provider_sig": "AAAA",
            "provider_pubkey_pem": "-----BEGIN PUBLIC KEY-----\nfake\n-----END PUBLIC KEY-----\n",
        }


def _spec(*, cost=1.0, latency=10_000, quality=0.5, payload=None):
    return JobSpec(
        job_id="t",
        kind="compute.test",
        payload=payload or {},
        cost_ceiling_usd=cost, latency_ceiling_ms=latency,
        privacy_class="public", quality_floor=quality,
    )


# ---------------------------------------------------------------------------
# job_needs_consortium heuristic
# ---------------------------------------------------------------------------

def test_consortium_not_needed_for_small_job():
    assert job_needs_consortium(_spec(cost=0.01)) is None


def test_consortium_explicit_payload_request_wins():
    spec = _spec(cost=0.01, payload={"consortium": {"size": 5}})
    assert job_needs_consortium(spec) == 5


def test_consortium_auto_for_big_jobs():
    spec = _spec(cost=CONSORTIUM_COST_THRESHOLD_USD + 1.0)
    assert job_needs_consortium(spec) == 4   # default 4-way split


# ---------------------------------------------------------------------------
# select_consortium ranking
# ---------------------------------------------------------------------------

def test_select_returns_top_n_ranked_by_pareto():
    a = Auction()
    a.register(_Worker(pid="w1", price=0.01, eta=200, quality=0.9, output=b"x1"))
    a.register(_Worker(pid="w2", price=0.05, eta=400, quality=0.85, output=b"x2"))
    a.register(_Worker(pid="w3", price=0.10, eta=800, quality=0.80, output=b"x3"))
    a.register(_Worker(pid="w4", price=0.15, eta=1000, quality=0.70, output=b"x4"))
    c = select_consortium(a, _spec(cost=1.0, latency=2000, quality=0.5),
                          target_size=3)
    assert c.size == 3
    # Top-3 by score: w1, w2, w3 in that order (w4 has worse price + eta).
    ranks = [m.provider_id for m in c.members]
    assert ranks == ["w1", "w2", "w3"]
    assert c.members[0].rank == 0
    assert c.members[0].shard_fraction == pytest.approx(1 / 3)


def test_select_drops_bids_above_cost_ceiling():
    a = Auction()
    a.register(_Worker(pid="ok", price=0.01, eta=200, quality=0.9, output=b"x"))
    a.register(_Worker(pid="too-expensive", price=10.0, eta=200, quality=0.9, output=b"x"))
    c = select_consortium(a, _spec(cost=1.0), target_size=4)
    # Only 1 bid survives, but consortium is partial — returns 1 member.
    assert c.size == 1
    assert c.members[0].provider_id == "ok"
    assert any(r["reason"] for r in c.rejected)


def test_select_zero_when_nothing_qualifies():
    a = Auction()
    a.register(_Worker(pid="bad", price=10.0, eta=200, quality=0.9, output=b"x"))
    c = select_consortium(a, _spec(cost=1.0), target_size=4)
    assert c.size == 0
    assert c.is_filled(minimum=4) is False


def test_select_handles_zero_providers_gracefully():
    a = Auction()
    c = select_consortium(a, _spec(), target_size=4)
    assert c.size == 0


# ---------------------------------------------------------------------------
# execute_consortium aggregation
# ---------------------------------------------------------------------------

def test_execute_runs_every_member_and_concats_results():
    a = Auction()
    a.register(_Worker(pid="w1", price=0.01, eta=200, quality=0.9, output=b"alpha"))
    a.register(_Worker(pid="w2", price=0.01, eta=200, quality=0.9, output=b"beta"))
    a.register(_Worker(pid="w3", price=0.01, eta=200, quality=0.9, output=b"gamma"))
    c = select_consortium(a, _spec(), target_size=3)
    spec = _spec()
    out = execute_consortium(a, c, spec)
    assert out.consortium.size == 3
    assert out.combined_result_bytes is not None
    assert b"alpha" in out.combined_result_bytes
    assert b"beta" in out.combined_result_bytes
    assert b"gamma" in out.combined_result_bytes
    assert out.total_cost_usd == pytest.approx(0.03)
    assert not out.failed_members


def test_execute_records_per_member_status():
    a = Auction()
    a.register(_Worker(pid="ok", price=0.01, eta=200, quality=0.9, output=b"x"))
    a.register(_Worker(pid="raises", price=0.02, eta=200, quality=0.9,
                       output=b"y", raise_on_execute=True))
    c = select_consortium(a, _spec(), target_size=2)
    out = execute_consortium(a, c, _spec())
    # Both members got a per-member entry; one is success, one
    # failed (and is recorded in failed_members).
    statuses = [e["status"] for e in out.per_member]
    assert "executed" in statuses
    assert "failed" in statuses
    assert "raises" in out.failed_members


def test_execute_tensor_parallel_raises_honestly():
    """The §A23 design notes will eventually cover tensor-parallel
    sharding; today the auction would lie if it pretended to
    serve it. The honest discipline: raise, with a pointer."""
    a = Auction()
    a.register(_Worker(pid="w1", price=0.01, eta=200, quality=0.9, output=b"x"))
    c = select_consortium(a, _spec(), target_size=1)
    c.sharding_mode = "tensor-parallel"
    with pytest.raises(NotImplementedError):
        execute_consortium(a, c, _spec())


def test_execute_diloco_dispatch_is_data_parallel_under_the_hood():
    """diloco sharding shares its dispatch with data-parallel; the
    aggregator does the gradient-averaging post-hoc, not at exec
    time."""
    a = Auction()
    a.register(_Worker(pid="w1", price=0.01, eta=200, quality=0.9, output=b"g1"))
    a.register(_Worker(pid="w2", price=0.01, eta=200, quality=0.9, output=b"g2"))
    c = select_consortium(a, _spec(), target_size=2)
    c.sharding_mode = "diloco"
    out = execute_consortium(a, c, _spec())
    assert out.combined_result_bytes is not None
    assert len(out.per_member) == 2
