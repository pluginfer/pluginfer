"""Host-guard regression tests — the 2026-07-17 host-freeze class.

The laptop froze because three gates were missing at once:
  (a) torch tests saturated every core at NORMAL priority,
  (b) Ollama negotiation's `pulled[0]` fallback picked qwen2.5:14b
      (~10 GB, keep_alive=60m) on a 16 GB machine with zero headroom
      check,
  (c) nothing capped the process tree's memory, so the OS swapped.
Each gate is pinned here so the class can never silently regress.
The suite itself runs UNDER the guard (tests/conftest.py installs it
before torch loads), so the install assertions exercise the real
production path, not a fixture.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

V2 = str(Path(__file__).resolve().parents[1])
if V2 not in sys.path:
    sys.path.insert(0, V2)

import host_guard  # noqa: E402

_GB = 1024 * 1024 * 1024


def test_install_idempotent_and_layers_applied():
    st1 = host_guard.install("pytest")
    st2 = host_guard.install("something-else")
    assert st1["installed"] and st2["installed"]
    # Second call is a no-op: role from the first install sticks.
    assert st2["role"] == st1["role"]
    # Thread caps were exported before conftest's torch import.
    assert int(os.environ["OMP_NUM_THREADS"]) >= 1
    assert st1["thread_cap"] >= 1
    if sys.platform == "win32" and st1["enabled"]:
        assert st1["job_assigned"], "job object must wrap the tree"
        assert host_guard.in_job()
        import psutil
        assert (psutil.Process().nice()
                == psutil.BELOW_NORMAL_PRIORITY_CLASS)
        assert st1["job_mem_limit_bytes"] < host_guard.total_bytes()


def test_headroom_sampling_sane():
    total = host_guard.total_bytes()
    avail = host_guard.headroom_bytes()
    assert 0 < avail <= total


def test_classify_thresholds_and_hysteresis(monkeypatch):
    monkeypatch.setenv("PLUGINFER_MEM_SOFT_MB", "2048")
    monkeypatch.setenv("PLUGINFER_MEM_CRIT_MB", "1024")
    host_guard._STATE["pressure"] = "ok"
    assert host_guard._classify(4 * _GB) == "ok"
    assert host_guard._classify(int(1.5 * _GB)) == "soft"
    assert host_guard._classify(int(0.5 * _GB)) == "critical"
    # Hysteresis: a soft state does NOT clear at 2.2 GB free (needs
    # soft * 1.25 = 2.5 GB), so the gate can't flap on the boundary.
    host_guard._STATE["pressure"] = "soft"
    assert host_guard._classify(int(2.2 * _GB)) == "soft"
    assert host_guard._classify(int(2.6 * _GB)) == "ok"
    host_guard._STATE["pressure"] = "ok"


def test_pressure_gates_work_acceptance():
    host_guard._apply_state("soft")
    try:
        assert not host_guard.should_accept_work()
    finally:
        host_guard._apply_state("ok")
    assert host_guard.should_accept_work()


def test_shed_callback_fires_once_per_transition():
    seen = []
    host_guard.register_shed_callback(seen.append)
    try:
        host_guard._apply_state("critical")
        host_guard._apply_state("critical")  # same state -> no refire
        assert seen == ["critical"]
    finally:
        host_guard._apply_state("ok")
        host_guard._SHED_CALLBACKS.clear()


def test_fits_model_math(monkeypatch):
    monkeypatch.setattr(host_guard, "headroom_bytes", lambda: 6 * _GB)
    monkeypatch.setenv("PLUGINFER_MEM_SOFT_MB", "2048")
    # Budget = 6 GB free - 2 GB soft floor = 4 GB.
    # 2 GB on disk -> est 2.3 + 1.5 = 3.8 GB: fits.
    assert host_guard.fits_model(2 * _GB)
    # 9 GB on disk (the qwen2.5:14b case) -> est ~11.9 GB: refused.
    assert not host_guard.fits_model(9 * _GB)
    # Unknown size passes — we can't judge what we can't measure.
    assert host_guard.fits_model(0)


def test_ollama_negotiation_refuses_oversized_models(monkeypatch):
    from core.runtime_adapters import ollama_adapter as oa

    monkeypatch.setattr(host_guard, "headroom_bytes", lambda: 6 * _GB)
    monkeypatch.setenv("PLUGINFER_MEM_SOFT_MB", "2048")
    pulled = ["qwen2.5:14b", "gemma3:4b"]
    sizes = {"qwen2.5:14b": 9 * _GB, "gemma3:4b": 2 * _GB}
    assert oa._filter_by_headroom(pulled, sizes) == ["gemma3:4b"]
    # Nothing fits -> empty list -> _resolve_endpoint records the
    # honest "none fit host RAM headroom" error and refuses, rather
    # than loading a model that would freeze the host.
    assert oa._filter_by_headroom(
        ["qwen2.5:14b"], {"qwen2.5:14b": 9 * _GB},
    ) == []


def test_jobs_service_rejects_under_pressure():
    from api.jobs_service import JobsService

    # auction=None is safe: the host-guard gate returns terminally
    # BEFORE the auction or ledger are touched — that ordering (no
    # economic side effects on rejection) is part of the contract.
    svc = JobsService(auction=None)
    host_guard._apply_state("soft")
    try:
        rec = asyncio.run(svc.submit(
            kind="prompt",
            payload={"prompt": "hi"},
            cost_ceiling_usd=1.0,
            latency_ceiling_ms=1000,
            privacy_class="open",
            quality_floor=0.0,
            requester_identity="test-host-guard",
        ))
    finally:
        host_guard._apply_state("ok")
    assert rec.state == "failed"
    assert rec.detail and "host_guard" in rec.detail
    assert svc.jobs[rec.job_id] is rec
    assert rec.price_locked_usd is None  # no money moved
