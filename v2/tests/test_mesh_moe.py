"""Tests for A8: MoE-on-the-Mesh."""

import asyncio

import pytest

from core.mesh_moe import (
    ExpertOutput,
    ExpertRecord,
    MeshMoERouter,
    weighted_softmax_combine,
)


def _make_registry(n: int = 4) -> dict:
    return {
        f"exp-{i}": ExpertRecord(
            expert_id=f"exp-{i}",
            domain=f"d{i}",
            model_hash=f"{i:064x}",
            author_address=f"addr-{i}",
        )
        for i in range(n)
    }


# ---------------------------------------------------------------------------
# Selection (router-side, no async)
# ---------------------------------------------------------------------------


def test_select_top_k_keeps_largest_weights_normalised():
    reg = _make_registry(4)
    router = MeshMoERouter(
        router=lambda inp: {"exp-0": 0.1, "exp-1": 0.6,
                             "exp-2": 0.2, "exp-3": 0.1},
        registry=reg,
        dispatch_one=lambda r, i: None,
        combine=lambda w, o: None,
        top_k=2,
    )
    sel = router.select(b"prompt")
    assert set(sel.keys()) == {"exp-1", "exp-2"}
    # Normalised to sum 1.0 within the kept set.
    assert sum(sel.values()) == pytest.approx(1.0)


def test_select_drops_unknown_experts():
    reg = _make_registry(2)
    router = MeshMoERouter(
        router=lambda inp: {"exp-0": 0.4, "BOGUS": 0.6},
        registry=reg,
        dispatch_one=lambda r, i: None,
        combine=lambda w, o: None,
        top_k=2,
    )
    sel = router.select(b"x")
    assert set(sel.keys()) == {"exp-0"}


def test_select_returns_empty_when_router_returns_empty():
    reg = _make_registry(2)
    router = MeshMoERouter(
        router=lambda inp: {},
        registry=reg,
        dispatch_one=lambda r, i: None,
        combine=lambda w, o: None,
        top_k=2,
    )
    assert router.select(b"x") == {}


# ---------------------------------------------------------------------------
# End-to-end dispatch (async)
# ---------------------------------------------------------------------------


def test_full_path_dispatches_and_collects():
    reg = _make_registry(3)

    async def stub_dispatch(rec, inp):
        await asyncio.sleep(0.01)
        return ExpertOutput(expert_id=rec.expert_id, output=[1.0, 2.0],
                            latency_ms=10.0)

    router = MeshMoERouter(
        router=lambda inp: {"exp-0": 0.6, "exp-1": 0.4},
        registry=reg,
        dispatch_one=stub_dispatch,
        combine=lambda w, o: None,
        top_k=2,
    )

    async def go():
        return await router(b"input")

    result = asyncio.run(go())
    assert sorted(result.chosen_experts) == ["exp-0", "exp-1"]
    assert len(result.outputs) == 2
    assert result.failed_experts == []


def test_provider_failure_reflected_in_failed_experts_list():
    reg = _make_registry(2)

    async def half_fail(rec, inp):
        if rec.expert_id == "exp-0":
            raise RuntimeError("boom")
        return ExpertOutput(expert_id=rec.expert_id, output=[1.0],
                            latency_ms=5.0)

    router = MeshMoERouter(
        router=lambda inp: {"exp-0": 0.5, "exp-1": 0.5},
        registry=reg,
        dispatch_one=half_fail,
        combine=lambda w, o: None,
        top_k=2,
    )

    async def go():
        return await router(b"x")

    result = asyncio.run(go())
    assert "exp-0" in result.failed_experts
    assert "exp-1" not in result.failed_experts


def test_overall_timeout_marks_all_pending_as_failed():
    reg = _make_registry(2)

    async def slow(rec, inp):
        await asyncio.sleep(5.0)
        return ExpertOutput(expert_id=rec.expert_id, output=[1.0],
                            latency_ms=5000.0)

    router = MeshMoERouter(
        router=lambda inp: {"exp-0": 0.5, "exp-1": 0.5},
        registry=reg,
        dispatch_one=slow,
        combine=lambda w, o: None,
        top_k=2,
    )

    async def go():
        return await router(b"x", overall_timeout_s=0.05)

    result = asyncio.run(go())
    assert set(result.failed_experts) == {"exp-0", "exp-1"}


# ---------------------------------------------------------------------------
# Combiner
# ---------------------------------------------------------------------------


def test_weighted_softmax_combine_blends_two_experts():
    weights = {"a": 0.7, "b": 0.3}
    outputs = [
        ExpertOutput(expert_id="a", output=[1.0, 0.0, 0.0], latency_ms=5),
        ExpertOutput(expert_id="b", output=[0.0, 1.0, 0.0], latency_ms=5),
    ]
    out = weighted_softmax_combine(weights, outputs)
    assert out == pytest.approx([0.7, 0.3, 0.0])


def test_weighted_softmax_handles_failed_expert():
    weights = {"a": 0.5, "b": 0.5}
    outputs = [
        ExpertOutput(expert_id="a", output=[1.0, 0.0], latency_ms=5),
        ExpertOutput(expert_id="b", output=None, latency_ms=0,
                     error="timeout"),
    ]
    out = weighted_softmax_combine(weights, outputs)
    # Only 'a' contributed; sum is 1.0 from 'a' alone.
    assert out == pytest.approx([1.0, 0.0])


def test_weighted_softmax_returns_none_when_all_fail():
    weights = {"a": 0.5, "b": 0.5}
    outputs = [
        ExpertOutput(expert_id="a", output=None, latency_ms=0, error="x"),
        ExpertOutput(expert_id="b", output=None, latency_ms=0, error="y"),
    ]
    assert weighted_softmax_combine(weights, outputs) is None


def test_weighted_softmax_pads_unequal_lengths():
    weights = {"a": 0.5, "b": 0.5}
    outputs = [
        ExpertOutput(expert_id="a", output=[1.0, 1.0], latency_ms=5),
        ExpertOutput(expert_id="b", output=[1.0], latency_ms=5),
    ]
    out = weighted_softmax_combine(weights, outputs)
    # b gets padded to [1.0, 0.0]; mean = [1.0, 0.5]
    assert out == pytest.approx([1.0, 0.5])
