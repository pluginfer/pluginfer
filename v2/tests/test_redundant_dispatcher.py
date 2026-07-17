"""Tests for K-redundant dispatch with majority-vote consensus."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sys
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.providers import Bid, JobSpec, PRIVACY_PUBLIC, Provider  # noqa: E402
from core.redundant_dispatcher import RedundantDispatcher  # noqa: E402


def _spec() -> JobSpec:
    return JobSpec(
        job_id="job-K", kind="compute.echo",
        payload={"x": 1},
        cost_ceiling_usd=0.10, latency_ceiling_ms=5_000,
    )


def _bid(pid: str) -> Bid:
    return Bid(provider_id=pid, price_usd=0.001, eta_ms=10,
               expected_quality=0.99, privacy_grade=PRIVACY_PUBLIC)


class _HonestProvider(Provider):
    privacy_grade = PRIVACY_PUBLIC
    kind = "compute"

    def __init__(self, pid: str, output: bytes = b"good") -> None:
        self.provider_id = pid
        self.output = output

    def bid(self, job: JobSpec) -> Bid:
        return _bid(self.provider_id)

    def execute(self, job: JobSpec, bid: Bid) -> dict:
        return {
            "status": "executed",
            "result_bytes_b64": base64.b64encode(self.output).decode(),
            "result_hash": hashlib.sha256(self.output).hexdigest(),
            "provider_sig": base64.b64encode(b"sig-" + self.provider_id.encode()).decode(),
        }


class _LyingProvider(Provider):
    """Returns a different output (different hash) than the honest set."""
    privacy_grade = PRIVACY_PUBLIC
    kind = "compute"

    def __init__(self, pid: str) -> None:
        self.provider_id = pid

    def bid(self, job: JobSpec) -> Bid:
        return _bid(self.provider_id)

    def execute(self, job: JobSpec, bid: Bid) -> dict:
        out = b"forged"
        return {
            "status": "executed",
            "result_bytes_b64": base64.b64encode(out).decode(),
            "result_hash": hashlib.sha256(out).hexdigest(),
            "provider_sig": base64.b64encode(b"forge").decode(),
        }


class _CrashingProvider(Provider):
    privacy_grade = PRIVACY_PUBLIC
    kind = "compute"

    def __init__(self, pid: str) -> None:
        self.provider_id = pid

    def bid(self, job: JobSpec) -> Bid:
        return _bid(self.provider_id)

    def execute(self, job: JobSpec, bid: Bid) -> dict:
        raise RuntimeError("simulated crash")


# ---------------------------------------------------------------------------


def test_three_honest_providers_unanimous():
    async def _run() -> None:
        ps = [_HonestProvider(f"p{i}", b"good") for i in range(3)]
        d = RedundantDispatcher(
            providers_and_bids=[(p, _bid(p.provider_id)) for p in ps],
            quorum_k=2,
        )
        res = await d.dispatch(_spec())
        assert res.won is True
        assert res.consensus_hash_hex == hashlib.sha256(b"good").hexdigest()
        assert res.dissenters == []
        assert res.majority_size() >= 2
    asyncio.run(_run())


def test_two_honest_one_liar_majority_wins_and_flags_liar():
    async def _run() -> None:
        ps = [
            _HonestProvider("p0", b"good"),
            _HonestProvider("p1", b"good"),
            _LyingProvider("p2-liar"),
        ]
        d = RedundantDispatcher(
            providers_and_bids=[(p, _bid(p.provider_id)) for p in ps],
            quorum_k=2,
        )
        res = await d.dispatch(_spec())
        assert res.won is True
        assert res.consensus_hash_hex == hashlib.sha256(b"good").hexdigest()
        assert "p2-liar" in res.dissenters
        assert res.majority_size() == 2
    asyncio.run(_run())


def test_one_honest_two_liars_dissents_to_majority_of_liars():
    """If two of three providers collude and lie, the majority IS
    wrong. The dispatcher returns won=True with the wrong consensus
    -- THIS IS BY DESIGN. Higher-level systems must layer additional
    defences (reputation, ZK gradient provenance, on-chain dispute)
    when adversarial collusion is in scope."""
    async def _run() -> None:
        liars_out = b"forged"
        forged_hash = hashlib.sha256(liars_out).hexdigest()
        # Two liars colluding return the SAME forged hash.
        class _ColludingLiar(Provider):
            privacy_grade = PRIVACY_PUBLIC; kind = "compute"
            def __init__(self, pid): self.provider_id = pid
            def bid(self, j): return _bid(self.provider_id)
            def execute(self, j, b):
                return {
                    "status": "executed",
                    "result_bytes_b64": base64.b64encode(liars_out).decode(),
                    "result_hash": forged_hash,
                    "provider_sig": base64.b64encode(b"forge").decode(),
                }
        ps = [
            _HonestProvider("honest", b"good"),
            _ColludingLiar("liar-A"),
            _ColludingLiar("liar-B"),
        ]
        d = RedundantDispatcher(
            providers_and_bids=[(p, _bid(p.provider_id)) for p in ps],
            quorum_k=2,
        )
        res = await d.dispatch(_spec())
        assert res.won is True
        assert res.consensus_hash_hex == forged_hash
        assert "honest" in res.dissenters       # honest now flagged as dissenter
    asyncio.run(_run())


def test_crashing_provider_does_not_block_consensus():
    async def _run() -> None:
        ps = [
            _HonestProvider("p0", b"good"),
            _HonestProvider("p1", b"good"),
            _CrashingProvider("crash"),
        ]
        d = RedundantDispatcher(
            providers_and_bids=[(p, _bid(p.provider_id)) for p in ps],
            quorum_k=2,
        )
        res = await d.dispatch(_spec())
        assert res.won is True
        assert "crash" in res.dissenters
        # The consensus is still the honest result
        assert res.consensus_hash_hex == hashlib.sha256(b"good").hexdigest()
    asyncio.run(_run())


def test_no_quorum_when_all_disagree():
    """Three different outputs -> no quorum-of-2 -> won=False."""
    async def _run() -> None:
        ps = [
            _HonestProvider("p0", b"a"),
            _HonestProvider("p1", b"b"),
            _HonestProvider("p2", b"c"),
        ]
        d = RedundantDispatcher(
            providers_and_bids=[(p, _bid(p.provider_id)) for p in ps],
            quorum_k=2,
        )
        res = await d.dispatch(_spec())
        assert res.won is False
        assert res.detail == "no_quorum"
        # All three are dissenters from the (single-vote "consensus").
        assert len(res.dissenters) == 2     # one happens to be the 'majority' of 1
    asyncio.run(_run())


def test_quorum_k_bounds_validated():
    with pytest.raises(ValueError):
        RedundantDispatcher(
            providers_and_bids=[(_HonestProvider("p"), _bid("p"))],
            quorum_k=2,         # > k
        )
    with pytest.raises(ValueError):
        RedundantDispatcher(
            providers_and_bids=[(_HonestProvider("p"), _bid("p"))],
            quorum_k=0,
        )
    with pytest.raises(ValueError):
        RedundantDispatcher(providers_and_bids=[])
