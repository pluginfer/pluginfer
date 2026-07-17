"""Quorum-replicate consortium mode — every member runs the same
job; majority-hash wins; byzantine dissenters surface for slashing.

This is the "critical job, no quality loss, no byzantine corruption"
mode. Pins:
  * Unanimous agreement → single result returned, no failed members.
  * 2-of-3 majority with one byzantine → majority result wins;
    byzantine surfaced in failed_members.
  * 1-of-3 split-brain → no result, all marked failed.
  * payload.criticality=high auto-triggers quorum-replicate mode.
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

from core.consortium_auction import (
    execute_consortium,
    job_default_sharding_mode,
    job_needs_consortium,
    select_consortium,
)
from core.providers import (
    Auction,
    Bid,
    JobSpec,
    PRIVACY_PUBLIC,
    Provider,
)


class _Replicator(Provider):
    def __init__(self, *, pid: str, output: bytes):
        self.provider_id = pid
        self.privacy_grade = PRIVACY_PUBLIC
        self._output = output

    def bid(self, job):
        return Bid(
            provider_id=self.provider_id, price_usd=0.001, eta_ms=100,
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


def _spec(*, payload=None, cost=1.0):
    return JobSpec(
        job_id="qt", kind="compute.test", payload=payload or {},
        cost_ceiling_usd=cost, latency_ceiling_ms=60_000,
        privacy_class="public", quality_floor=0.5,
    )


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

def test_criticality_high_auto_picks_quorum_replicate():
    spec = _spec(payload={"criticality": "high"})
    assert job_needs_consortium(spec) == 3      # 3-way default
    assert job_default_sharding_mode(spec) == "quorum-replicate"


def test_explicit_mode_overrides_criticality():
    spec = _spec(payload={
        "criticality": "high",
        "consortium": {"size": 5, "mode": "data-parallel"},
    })
    assert job_needs_consortium(spec) == 5
    assert job_default_sharding_mode(spec) == "data-parallel"


def test_normal_jobs_default_to_data_parallel():
    spec = _spec(cost=10.0)
    assert job_default_sharding_mode(spec) == "data-parallel"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def test_unanimous_quorum_returns_single_result():
    auction = Auction()
    auction.register(_Replicator(pid="p1", output=b"42"))
    auction.register(_Replicator(pid="p2", output=b"42"))
    auction.register(_Replicator(pid="p3", output=b"42"))
    c = select_consortium(
        auction, _spec(), target_size=3, sharding_mode="quorum-replicate",
    )
    out = execute_consortium(auction, c, _spec())
    assert out.combined_result_bytes == b"42"
    assert out.combined_result_hash == hashlib.sha256(b"42").hexdigest()
    assert out.failed_members == []


def test_byzantine_minority_loses_and_is_flagged():
    """One byzantine member returns garbage; the 2/3 majority wins,
    the byzantine is added to failed_members for slashing."""
    auction = Auction()
    auction.register(_Replicator(pid="honest-1", output=b"correct"))
    auction.register(_Replicator(pid="honest-2", output=b"correct"))
    auction.register(_Replicator(pid="byzantine", output=b"garbage"))
    c = select_consortium(
        auction, _spec(), target_size=3, sharding_mode="quorum-replicate",
    )
    out = execute_consortium(auction, c, _spec())
    assert out.combined_result_bytes == b"correct"
    assert "byzantine" in out.failed_members
    assert "honest-1" not in out.failed_members
    assert "honest-2" not in out.failed_members


def test_split_brain_returns_no_result():
    """When every member returns different bytes, no majority exists.
    The exec result has no combined bytes; the buyer can retry with
    a larger quorum."""
    auction = Auction()
    auction.register(_Replicator(pid="p1", output=b"a"))
    auction.register(_Replicator(pid="p2", output=b"b"))
    auction.register(_Replicator(pid="p3", output=b"c"))
    c = select_consortium(
        auction, _spec(), target_size=3, sharding_mode="quorum-replicate",
    )
    out = execute_consortium(auction, c, _spec())
    assert out.combined_result_bytes is None
    assert out.combined_result_hash is None
