"""Elastic-scale-to-compute consortium — the strategic moat.

A startup with one H100 (~score 200) can pay AWS $4-5/hour or pay
Pluginfer the auction-cleared price for an equivalent compute mass
assembled from the mesh:

  * 4 × RTX 4090 @ score 50 = score 200 — small consortium
  * 50 × GTX 1650 @ score 4 = score 200 — large consortium
  * mixed: 1 × 4090 + 30 × 1650 = score 170 — partial; aggregate
    or the auction returns 'not enough'

This test pins those exact scenarios. The mesh's selling point:
no single firm can match the aggregate compute mass of a global
consumer-GPU pool — Pluginfer's auction surfaces it permissionlessly
and assembles it dynamically per job.
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
    job_needs_consortium,
    select_scale_to_compute,
)
from core.providers import (
    Auction,
    Bid,
    JobSpec,
    PRIVACY_PUBLIC,
    Provider,
)


class _ScoredProvider(Provider):
    def __init__(self, *, pid: str, score: float, price_per_score: float = 0.001):
        self.provider_id = pid
        self.privacy_grade = PRIVACY_PUBLIC
        self._score = score
        self._price_per_score = price_per_score

    def bid(self, job):
        return Bid(
            provider_id=self.provider_id,
            price_usd=self._price_per_score * self._score,
            eta_ms=100, expected_quality=0.9,
            privacy_grade=PRIVACY_PUBLIC,
            evidence={"peer_score": self._score, "hardware_class": "remote-mesh"},
        )

    def execute(self, job, bid):
        out = f"shard-from-{self.provider_id}".encode("utf-8")
        return {
            "status": "executed", "job_id": job.job_id,
            "result_bytes": base64.b64encode(out).decode("ascii"),
            "result_hash": hashlib.sha256(out).hexdigest(),
            "execution_ms": 100.0,
            "provider_sig": "AAAA",
            "provider_pubkey_pem": "fake",
        }


def _spec(*, required_score=200.0, cost=1000.0):
    return JobSpec(
        job_id="big", kind="compute.test",
        payload={"required_compute_score": required_score},
        cost_ceiling_usd=cost, latency_ceiling_ms=600_000,
        privacy_class="public", quality_floor=0.5,
    )


def test_job_needs_consortium_sentinel_for_scale_to_compute():
    """payload.required_compute_score triggers the
    auction-decides-size path. job_needs_consortium returns 0
    (sentinel)."""
    assert job_needs_consortium(_spec()) == 0


def test_four_rtx_4090s_equal_one_h100():
    """The strategic claim: 4 × RTX 4090 (score 50 each) combine to
    cover an H100-sized job (score ~200)."""
    auction = Auction()
    for i in range(4):
        auction.register(_ScoredProvider(pid=f"rtx4090-{i}", score=50.0))
    c = select_scale_to_compute(
        auction, _spec(required_score=200.0), required_compute_score=200.0,
    )
    assert c.size == 4
    summed_score = sum(
        m.bid.evidence["peer_score"] for m in c.members
    )
    assert summed_score == 200.0


def test_fifty_gtx_1650s_can_cover_an_h100_job():
    """The 'long tail' moat: 50 × GTX 1650 (score 4 each) covers
    score 200. No single firm has a 50-GTX-1650 datacenter; the
    mesh has 50 living rooms."""
    auction = Auction()
    for i in range(50):
        auction.register(_ScoredProvider(pid=f"gtx1650-{i}", score=4.0))
    c = select_scale_to_compute(
        auction, _spec(required_score=200.0), required_compute_score=200.0,
        max_members=100,
    )
    assert c.size == 50
    summed_score = sum(
        m.bid.evidence["peer_score"] for m in c.members
    )
    assert summed_score == 200.0


def test_mixed_fleet_picks_cheapest_per_score():
    """When 4090s and 1650s are both available, the auction picks
    cheapest USD/score first — the same compute should cost less
    via 4090s than via 1650s if their per-score prices match."""
    auction = Auction()
    # 1650s are configured 10% cheaper per score than 4090s, so they
    # win the picking order despite being more numerous.
    for i in range(2):
        auction.register(_ScoredProvider(
            pid=f"rtx4090-{i}", score=50.0, price_per_score=0.0010,
        ))
    for i in range(40):
        auction.register(_ScoredProvider(
            pid=f"gtx1650-{i}", score=4.0, price_per_score=0.00091,
        ))
    c = select_scale_to_compute(
        auction, _spec(required_score=200.0, cost=1000.0),
        required_compute_score=200.0, max_members=100,
    )
    # 1650s are cheapest per score; auction grabs them first until
    # the requirement is met.
    pids = [m.provider_id for m in c.members]
    n_1650 = sum(1 for p in pids if "1650" in p)
    n_4090 = sum(1 for p in pids if "4090" in p)
    # 40 × 1650 = 160 score. Need 200. Either 50 1650s OR 40 1650s
    # plus one 4090. Greedy fills 40 1650s (160) then a 4090 (50),
    # total 210. That's the cheapest set that meets requirement.
    assert n_1650 == 40
    assert n_4090 == 1


def test_returns_empty_consortium_when_mesh_too_small():
    """If the total available compute is below the requirement,
    the consortium is empty. Caller (JobsService) can downgrade or
    pause the job rather than running it half-compute."""
    auction = Auction()
    # Only 4 nodes × score 4 = 16, way under 200.
    for i in range(4):
        auction.register(_ScoredProvider(pid=f"weak-{i}", score=4.0))
    c = select_scale_to_compute(
        auction, _spec(required_score=200.0),
        required_compute_score=200.0,
    )
    assert c.size == 0


def test_max_members_cap_prevents_runaway_consortium():
    auction = Auction()
    for i in range(200):
        auction.register(_ScoredProvider(pid=f"tiny-{i}", score=1.0))
    c = select_scale_to_compute(
        auction, _spec(required_score=200.0, cost=1000.0),
        required_compute_score=200.0,
        max_members=64,
    )
    # 64 × 1 = 64 < 200 → empty. The cap kicks in before the score
    # requirement is met; mesh refuses rather than over-consume.
    assert c.size == 0
