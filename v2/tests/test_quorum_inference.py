"""Tests for A13: Stateless Quorum Inference."""

import asyncio
import hashlib

import pytest

from core.quorum_inference import (
    QuorumResult,
    expected_failure_rate,
    quorum_dispatch,
)


# ---------------------------------------------------------------------------
# expected_failure_rate sanity
# ---------------------------------------------------------------------------


def test_expected_failure_rate_k1_equals_p():
    assert expected_failure_rate(0.05, 1) == pytest.approx(0.05)


def test_expected_failure_rate_k3_one_in_a_million_at_p1pct():
    assert expected_failure_rate(0.01, 3) == pytest.approx(1e-6)


def test_expected_failure_rate_k_must_be_positive():
    with pytest.raises(ValueError):
        expected_failure_rate(0.01, 0)


# ---------------------------------------------------------------------------
# quorum_dispatch (async)
# ---------------------------------------------------------------------------


def _provider(pid, *, latency=0.05, fail=False, output=b"hi"):
    return {"provider_id": pid, "latency": latency, "fail": fail,
            "output": output}


async def _fake_execute(rec):
    await asyncio.sleep(rec["latency"])
    if rec["fail"]:
        raise RuntimeError(f"{rec['provider_id']} simulated failure")
    return rec["output"]


def test_first_valid_wins(event_loop=None):
    async def go():
        providers = [
            _provider("slow", latency=0.20, output=b"slow"),
            _provider("fast", latency=0.02, output=b"fast"),
            _provider("medium", latency=0.10, output=b"medium"),
        ]
        res = await quorum_dispatch(
            providers=providers,
            execute=_fake_execute,
            quorum_k=3,
            overall_timeout_s=2.0,
        )
        assert res.is_won
        assert res.winner_provider_id == "fast"
        assert res.output_bytes == b"fast"
        assert res.output_sha256 == hashlib.sha256(b"fast").hexdigest()
        assert "slow" in res.losers_skipped or "medium" in res.losers_skipped

    asyncio.run(go())


def test_survives_k_minus_1_failures():
    async def go():
        providers = [
            _provider("p1", latency=0.05, fail=True),
            _provider("p2", latency=0.10, fail=True),
            _provider("p3", latency=0.15, output=b"third-tries"),
        ]
        res = await quorum_dispatch(
            providers=providers, execute=_fake_execute,
            quorum_k=3, overall_timeout_s=2.0,
        )
        assert res.is_won
        assert res.winner_provider_id == "p3"
        assert len(res.losers_failed) == 2

    asyncio.run(go())


def test_total_failure_returns_unwon_result():
    async def go():
        providers = [
            _provider(f"p{i}", latency=0.01, fail=True) for i in range(3)
        ]
        res = await quorum_dispatch(
            providers=providers, execute=_fake_execute,
            quorum_k=3, overall_timeout_s=2.0,
        )
        assert not res.is_won
        assert len(res.losers_failed) == 3

    asyncio.run(go())


def test_overall_timeout_aborts_slow_dispatch():
    async def go():
        providers = [
            _provider(f"p{i}", latency=5.0, output=b"x") for i in range(3)
        ]
        res = await quorum_dispatch(
            providers=providers, execute=_fake_execute,
            quorum_k=3, overall_timeout_s=0.1,
        )
        assert not res.is_won

    asyncio.run(go())


def test_expected_output_sha256_filters_wrong_answer():
    async def go():
        providers = [
            _provider("liar", latency=0.01, output=b"WRONG"),
            _provider("honest", latency=0.05, output=b"truth"),
        ]
        expected = hashlib.sha256(b"truth").hexdigest()
        res = await quorum_dispatch(
            providers=providers, execute=_fake_execute,
            quorum_k=2, overall_timeout_s=2.0,
            expected_output_sha256=expected,
        )
        assert res.is_won
        assert res.winner_provider_id == "honest"
        assert any(f["provider_id"] == "liar" and "mismatch" in f["error"]
                   for f in res.losers_failed)

    asyncio.run(go())


def test_empty_providers_returns_unwon():
    async def go():
        res = await quorum_dispatch(
            providers=[], execute=_fake_execute,
            quorum_k=3, overall_timeout_s=1.0,
        )
        assert not res.is_won

    asyncio.run(go())


def test_k_caps_dispatch_count():
    async def go():
        seen = []

        async def tracking_execute(rec):
            seen.append(rec["provider_id"])
            return rec["output"]

        providers = [
            _provider(f"p{i}", latency=0.001, output=b"out")
            for i in range(10)
        ]
        await quorum_dispatch(
            providers=providers, execute=tracking_execute,
            quorum_k=3, overall_timeout_s=2.0,
        )
        # Only first 3 by ranking should be invoked.
        assert seen == ["p0", "p1", "p2"]

    asyncio.run(go())
