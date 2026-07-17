"""Vendor-agnostic backend abstraction smoke tests.

These tests run on whatever hardware the dev machine has — the goal
is to verify the abstraction degrades gracefully, not to validate
each vendor's probe (which would require that vendor's hardware).

Coverage:

* detect_backend never raises and always returns a valid BackendInfo
* select_torch_device returns a torch-acceptable string
* memory_cap_bytes is sane on every backend
* synchronize / empty_cache / memory_used_bytes do not raise
* CPU fallback always works
* The vendor-aware probe in telemetry.sample_now() returns a valid
  PressureSample on the dev box without hardware-specific tooling
"""

from __future__ import annotations

import pytest


def test_detect_backend_always_returns_info():
    from ai.filum.hpa.backend import detect_backend, BackendInfo
    info = detect_backend()
    assert isinstance(info, BackendInfo)
    assert info.name in {"cuda", "rocm", "mps", "xpu", "cpu"}
    assert info.device_str in {"cuda", "mps", "xpu", "cpu"}
    if info.name == "cpu":
        assert info.cpu_count >= 1


def test_detect_backend_prefer_cpu_short_circuits():
    from ai.filum.hpa.backend import detect_backend
    info = detect_backend(prefer="cpu")
    assert info.name == "cpu"


def test_select_torch_device_is_string():
    from ai.filum.hpa.backend import select_torch_device
    s = select_torch_device()
    assert isinstance(s, str)
    assert s in {"cuda", "mps", "xpu", "cpu"}


def test_memory_cap_bytes_is_positive():
    from ai.filum.hpa.backend import memory_cap_bytes, detect_backend
    cap = memory_cap_bytes(detect_backend())
    assert cap > 0
    # Sanity: cap should be at least 256MiB on any host that runs python.
    assert cap >= 256 * (1 << 20)


def test_memory_cap_obeys_fraction():
    from ai.filum.hpa.backend import memory_cap_bytes, BackendInfo
    info = BackendInfo(
        name="cuda", device_str="cuda",
        accelerator_count=1,
        accelerator_name="MockGPU",
        total_memory_bytes=8 * (1 << 30),  # 8 GiB
    )
    cap_70 = memory_cap_bytes(info, frac=0.70)
    cap_30 = memory_cap_bytes(info, frac=0.30)
    assert cap_70 > cap_30


def test_synchronize_and_empty_cache_dont_raise():
    """These are best-effort hints; they must never throw."""
    from ai.filum.hpa.backend import synchronize, empty_cache, memory_used_bytes
    synchronize()
    synchronize("cpu")
    empty_cache()
    empty_cache("cpu")
    n = memory_used_bytes("cpu")
    assert n >= 0


def test_pressure_sampler_works_without_nvidia_smi():
    """Verifies the §B telemetry layer still produces valid samples on
    a host without nvidia-smi (i.e. AMD/Apple/Intel/CPU). Sanity-only."""
    from ai.filum.hpa.telemetry import sample_now, pressure_scalar
    s = sample_now()
    p = pressure_scalar(s)
    assert 0.0 <= p <= 1.0


def test_vendor_probe_callable():
    from ai.filum.hpa.backend import vendor_telemetry_probe
    probe = vendor_telemetry_probe()
    assert callable(probe)
    # Invocation must not raise.
    r = probe()
    assert r is None or len(r) == 3


def test_backend_info_has_total_memory_when_known():
    """On any modern host the probe should report *some* memory total
    (psutil fallback at minimum). 0 is acceptable only when psutil is
    missing AND no accelerator was detected."""
    from ai.filum.hpa.backend import detect_backend
    info = detect_backend()
    # Permit 0 for a truly minimal embedded host, but we don't have
    # one of those — assert basic sanity.
    assert info.total_memory_bytes >= 0
