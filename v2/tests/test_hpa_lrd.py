"""HPA-LRD CPU smoke tests.

These tests verify the four pure-Python pieces of the
hardware-pressure-adaptive trainer without ever touching CUDA.
They are intentionally cheap so they can run on the user's CPU dev
box without re-creating the GPU hang we just diagnosed.

Coverage:

* PressureSampler             - background thread starts, stops, returns a sample
* pressure_scalar             - max-of-known-signals, ignores -1 fields
* choose_rank / RankPolicy    - monotonic non-increasing in P
* AdaptiveLowRankProjector    - SVD round-trip on a fake gradient (numpy/torch)
* DiskTeacherCache            - put / take / consumed eviction, persistence
* CooperativeYield            - yields when pressure_fn > threshold
* cuda_oom_guard              - catches simulated OOM and retries

These run as part of the full v2/tests/ suite.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest


# --- telemetry ---------------------------------------------------------------

def test_pressure_scalar_max_of_known_signals():
    from ai.filum.hpa.telemetry import PressureSample, pressure_scalar

    s = PressureSample(
        ts=0.0,
        vram_used_frac=0.40,
        gpu_util_frac=0.10,
        gpu_temp_frac=0.50,
        ram_used_frac=0.20,
        cpu_used_frac=0.10,
    )
    # Max signal is vram_used_frac (cpu is downweighted by 0.5 in the
    # implementation; gpu_temp clamped to 1.0 max).
    assert pressure_scalar(s) == pytest.approx(0.50)


def test_pressure_scalar_ignores_unknown():
    from ai.filum.hpa.telemetry import PressureSample, pressure_scalar
    s = PressureSample(ts=0.0)  # all -1
    assert pressure_scalar(s) == 0.0


def test_pressure_sampler_lifecycle():
    from ai.filum.hpa.telemetry import PressureSampler
    sampler = PressureSampler(period_s=0.05).start()
    try:
        time.sleep(0.15)
        s = sampler.last()
        assert s.ts > 0
        p = sampler.pressure()
        assert 0.0 <= p <= 1.0
    finally:
        sampler.stop()


# --- adaptive rank -----------------------------------------------------------

def test_choose_rank_monotonic():
    from ai.filum.hpa.galore_adaptive import RankPolicy, choose_rank

    pol = RankPolicy(r_min=8, r_max=256, p_lo=0.30, p_hi=0.85)
    rs = [choose_rank(p, pol) for p in [0.0, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]]
    # Non-increasing
    for a, b in zip(rs, rs[1:]):
        assert a >= b, f"rank not monotonic: {rs}"
    assert rs[0] == 256
    assert rs[-1] == 8


def test_choose_rank_clamped_in_band():
    from ai.filum.hpa.galore_adaptive import RankPolicy, choose_rank
    pol = RankPolicy(r_min=4, r_max=64, p_lo=0.0, p_hi=1.0)
    for p in [-1.0, 0.0, 0.5, 1.0, 2.0]:
        r = choose_rank(p, pol)
        assert 4 <= r <= 64


@pytest.mark.skipif(
    os.environ.get("FILUM_SKIP_TORCH_TESTS") == "1",
    reason="torch not available in this CI tier",
)
def test_projector_roundtrip_preserves_rank():
    pytest.importorskip("torch")
    import torch
    from ai.filum.hpa.galore_adaptive import AdaptiveLowRankProjector, RankPolicy

    torch.manual_seed(7)
    proj = AdaptiveLowRankProjector(
        policy=RankPolicy(r_min=4, r_max=16, p_lo=0.2, p_hi=0.8),
        refresh_steps=1000,
    )
    grad = torch.randn(64, 32)
    low = proj.project("w", grad, pressure=0.0)  # P low -> r_max
    assert low.shape == (16, 32)
    # Re-projection in low space stays consistent on the next call
    # (no refresh because step counter unchanged + same target r).
    low2 = proj.project("w", grad, pressure=0.0)
    assert torch.allclose(low, low2)
    # Unproject yields a 64x32 tensor back.
    full = proj.unproject("w", low)
    assert full.shape == (64, 32)


@pytest.mark.skipif(
    os.environ.get("FILUM_SKIP_TORCH_TESTS") == "1",
    reason="torch not available in this CI tier",
)
def test_projector_rank_changes_with_pressure():
    pytest.importorskip("torch")
    import torch
    from ai.filum.hpa.galore_adaptive import AdaptiveLowRankProjector, RankPolicy

    torch.manual_seed(11)
    proj = AdaptiveLowRankProjector(
        policy=RankPolicy(r_min=4, r_max=32, p_lo=0.2, p_hi=0.8),
        refresh_steps=1,  # always refresh
    )
    grad = torch.randn(64, 64)
    proj.project("a", grad, pressure=0.0)
    r_low = proj.current_rank_for("a")
    proj.step()
    proj.project("a", grad, pressure=1.0)
    r_high = proj.current_rank_for("a")
    assert r_low > r_high, f"rank should drop under pressure: {r_low} vs {r_high}"


# --- teacher cache -----------------------------------------------------------

def test_teacher_cache_put_get(tmp_path: Path):
    from ai.filum.hpa.teacher_cache import DiskTeacherCache, TeacherSample

    cache = DiskTeacherCache(tmp_path)
    s = TeacherSample(prompt="hi", response_text="hello",
                      teacher_id="mock", ts=time.time())
    assert cache.put(s) is True
    assert cache.put(s) is False  # dedup
    assert len(cache) == 1

    out = cache.take(2)
    assert len(out) == 1
    assert out[0].prompt == "hi"


def test_teacher_cache_evicts_after_max_consumed(tmp_path: Path):
    from ai.filum.hpa.teacher_cache import DiskTeacherCache, TeacherSample

    cache = DiskTeacherCache(tmp_path, max_consumed=2)
    s = TeacherSample(prompt="p", response_text="r",
                      teacher_id="m", ts=time.time())
    cache.put(s)
    cache.take(1)
    cache.take(1)
    # After 2 consumes the sample is evicted.
    assert len(cache) == 0
    assert cache.take(1) == []


def test_teacher_cache_persists_across_instances(tmp_path: Path):
    from ai.filum.hpa.teacher_cache import DiskTeacherCache, TeacherSample

    c1 = DiskTeacherCache(tmp_path)
    c1.put(TeacherSample(prompt="x", response_text="y",
                         teacher_id="m", ts=time.time()))
    c2 = DiskTeacherCache(tmp_path)
    assert len(c2) == 1


def test_teacher_cache_async_fill(tmp_path: Path):
    from ai.filum.hpa.teacher_cache import DiskTeacherCache

    cache = DiskTeacherCache(tmp_path)

    async def gen(prompt: str) -> tuple[str, str]:
        await asyncio.sleep(0)  # yield control
        return f"reply to {prompt}", "mock"

    prompts = [f"q{i}" for i in range(8)]
    added = asyncio.run(cache.fill(prompts, gen, target_size=5,
                                   max_concurrent=3))
    assert added >= 5
    assert len(cache) >= 5


# --- cooperative yield -------------------------------------------------------

def test_cooperative_yield_only_when_above_threshold():
    from ai.filum.hpa.cooperative import CooperativeYield

    p = [0.10]

    coop = CooperativeYield(pressure_fn=lambda: p[0],
                            threshold=0.85,
                            base_sleep_s=0.001,
                            max_sleep_s=0.002)
    assert coop.maybe_yield() is False
    p[0] = 0.95
    assert coop.maybe_yield() is True
    assert coop.yield_count == 1


def test_cuda_oom_guard_catches_string_match():
    from ai.filum.hpa.cooperative import cuda_oom_guard

    called = {"n": 0}

    def on_oom(_e):
        called["n"] += 1
        return True  # recover

    with cuda_oom_guard(on_oom):
        raise RuntimeError("CUDA out of memory: tried to allocate 5GB")
    assert called["n"] == 1


def test_cuda_oom_guard_reraises_unrelated():
    from ai.filum.hpa.cooperative import cuda_oom_guard

    with pytest.raises(ValueError):
        with cuda_oom_guard(lambda _e: True):
            raise ValueError("not an OOM")
