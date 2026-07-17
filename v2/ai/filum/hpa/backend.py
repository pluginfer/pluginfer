"""Vendor-agnostic hardware backend abstraction.

The §B/§C bundle is *protocol-agnostic about hardware*. The actual
training kernels run on whatever the host has: NVIDIA CUDA, AMD ROCm,
Apple Metal Performance Shaders (MPS), Intel XPU (oneAPI), or
plain CPU. The §C grain protocol does not care: a gradient computed
on a Mac Mini and a gradient computed on a 3090 are interchangeable
once they're in low-rank form.

This module gives the rest of the codebase one place to ask:

* "What hardware do I have?"          -> ``detect_backend()``
* "Which torch device should I use?"  -> ``select_torch_device()``
* "How do I sample pressure here?"    -> ``vendor_telemetry_probe()``
* "What's a safe memory cap?"         -> ``memory_cap_bytes()``

Backends supported in priority order (first available wins):

    cuda   - NVIDIA via CUDA + nvidia-smi
    rocm   - AMD via ROCm + rocm-smi
    mps    - Apple Silicon via Metal Performance Shaders
    xpu    - Intel via oneAPI XPU
    cpu    - any host, always available

The backend detection is *defensive*: if a tool exists but doesn't
work, we degrade gracefully to the next backend. The user's mesh
node never refuses to start because of a missing driver — it just
participates with whatever it has.

design rationale impact: §B1 and §C1-§C8 are written generally to
"hardware pressure" not "GPU memory pressure"; this module is the
embodiment that makes the claims hold across vendors.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class BackendInfo:
    """A description of what the host has available."""
    name: str               # "cuda" | "rocm" | "mps" | "xpu" | "cpu"
    device_str: str         # what to pass to torch.device(...)
    accelerator_count: int = 0
    accelerator_name: str = ""
    total_memory_bytes: int = 0
    cpu_count: int = 0
    notes: str = ""


# ----------------------------------------------------------------------------
# Probes (one per vendor). Each returns a BackendInfo or None.
# ----------------------------------------------------------------------------

def _probe_cuda() -> Optional[BackendInfo]:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        n = torch.cuda.device_count()
        if n <= 0:
            return None
        name = torch.cuda.get_device_name(0)
        free, total = (0, 0)
        try:
            free, total = torch.cuda.mem_get_info()
        except Exception:
            try:
                total = torch.cuda.get_device_properties(0).total_memory
            except Exception:
                pass
        return BackendInfo(
            name="cuda", device_str="cuda",
            accelerator_count=n, accelerator_name=name,
            total_memory_bytes=int(total),
        )
    except Exception:
        return None


def _probe_rocm() -> Optional[BackendInfo]:
    """AMD ROCm reuses torch.cuda.* with the ROCm-built wheel.

    We distinguish it from real CUDA by checking ``torch.version.hip``.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    if getattr(torch.version, "hip", None) is None:
        return None
    try:
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        try:
            total = torch.cuda.get_device_properties(0).total_memory
        except Exception:
            total = 0
        return BackendInfo(
            name="rocm", device_str="cuda",   # torch ROCm uses cuda string
            accelerator_count=n, accelerator_name=name,
            total_memory_bytes=int(total),
            notes="ROCm via torch.cuda interface",
        )
    except Exception:
        return None


def _probe_mps() -> Optional[BackendInfo]:
    """Apple Silicon MPS. torch.mps is available on macOS / Apple Silicon."""
    try:
        import torch
    except ImportError:
        return None
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return None
    try:
        if not backend.is_available():
            return None
    except Exception:
        return None
    # Apple unified memory: there isn't a separate VRAM number.
    # We approximate with system RAM as the bound (MPS can use it all).
    total = _system_ram_bytes()
    return BackendInfo(
        name="mps", device_str="mps",
        accelerator_count=1, accelerator_name="Apple MPS",
        total_memory_bytes=total,
        notes="Unified memory; cap measured against system RAM",
    )


def _probe_xpu() -> Optional[BackendInfo]:
    """Intel XPU via Intel Extension for PyTorch (ipex)."""
    try:
        import torch
    except ImportError:
        return None
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return None
    try:
        if not xpu.is_available():
            return None
        n = xpu.device_count()
        name = xpu.get_device_name(0)
        try:
            total = xpu.get_device_properties(0).total_memory
        except Exception:
            total = 0
        return BackendInfo(
            name="xpu", device_str="xpu",
            accelerator_count=n, accelerator_name=name,
            total_memory_bytes=int(total),
        )
    except Exception:
        return None


def _probe_cpu() -> BackendInfo:
    """Always-available fallback. Real for CPU-only nodes."""
    cpu_count = os.cpu_count() or 1
    return BackendInfo(
        name="cpu", device_str="cpu",
        accelerator_count=0, accelerator_name="cpu",
        total_memory_bytes=_system_ram_bytes(),
        cpu_count=cpu_count,
    )


def _system_ram_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        return 0


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

_PROBES = [
    _probe_cuda,
    _probe_rocm,
    _probe_mps,
    _probe_xpu,
]


def detect_backend(
    *,
    prefer: Optional[str] = None,
    allow_cpu: bool = True,
) -> BackendInfo:
    """Return the best available backend.

    ``prefer``: if set ("cuda", "rocm", "mps", "xpu", "cpu"), tries that
    first and falls through if not available.

    ``allow_cpu``: if False, raises RuntimeError when no accelerator is
    found. Default True — CPU is a perfectly valid mesh contributor.
    """
    if prefer == "cpu":
        return _probe_cpu()

    if prefer:
        named = {
            "cuda": _probe_cuda, "rocm": _probe_rocm,
            "mps": _probe_mps,   "xpu":  _probe_xpu,
        }.get(prefer)
        if named is not None:
            r = named()
            if r is not None:
                return r
        # Preferred backend not available; fall through to scan.

    for probe in _PROBES:
        r = probe()
        if r is not None:
            return r
    if not allow_cpu:
        raise RuntimeError("No accelerator available and CPU disallowed.")
    return _probe_cpu()


def select_torch_device(prefer: Optional[str] = None) -> str:
    """Return the torch device string for the best backend."""
    return detect_backend(prefer=prefer).device_str


def memory_cap_bytes(
    backend: Optional[BackendInfo] = None,
    frac: float = 0.70,
    headroom_mib: int = 600,
) -> int:
    """Return a soft memory budget in bytes.

    For accelerators with their own VRAM (CUDA/ROCm/XPU): cap by
    fraction of dedicated VRAM, leaving ``headroom_mib`` for the OS.
    For unified-memory backends (MPS): cap by fraction of system RAM,
    leaving more headroom (default 25% of system RAM) for the OS.
    For CPU-only: cap by fraction of system RAM with ample headroom.
    """
    info = backend or detect_backend()
    total = info.total_memory_bytes
    if total <= 0:
        return 1 << 30
    if info.name in ("cuda", "rocm", "xpu"):
        by_frac = int(total * max(0.1, min(0.95, frac)))
        by_headroom = int(total - headroom_mib * (1 << 20))
        return max(1 << 28, min(by_frac, by_headroom))
    if info.name == "mps":
        # Unified memory: be more conservative (compositor + apps).
        by_frac = int(total * min(0.5, frac))
        return max(1 << 28, by_frac)
    # cpu: leave 30% to OS at minimum.
    return max(1 << 28, int(total * min(0.7, frac)))


# ----------------------------------------------------------------------------
# Vendor-aware pressure probes (used by telemetry)
# ----------------------------------------------------------------------------

def vendor_telemetry_probe() -> Callable[[], Optional[tuple[float, float, float]]]:
    """Return a probe function that yields (vram_used_frac, gpu_util_frac,
    gpu_temp_frac) on demand, choosing the right vendor tool.

    Returns ``None`` from the inner function if the probe fails. Callers
    fall back to torch-level probes (already in telemetry.py).
    """
    info = detect_backend()
    name = info.name
    if name == "cuda":
        return _probe_cuda_telemetry
    if name == "rocm":
        return _probe_rocm_telemetry
    if name == "mps":
        return _probe_mps_telemetry
    if name == "xpu":
        return _probe_xpu_telemetry
    return _probe_cpu_telemetry


def _probe_cuda_telemetry() -> Optional[tuple[float, float, float]]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi,
             "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0, check=True,
        ).stdout.strip().splitlines()[0]
        parts = [p.strip() for p in out.split(",")]
        mem_used, mem_total, util, temp = (float(parts[0]), float(parts[1]),
                                            float(parts[2]), float(parts[3]))
        if mem_total <= 0:
            return None
        return (mem_used / mem_total, util / 100.0, temp / 83.0)
    except Exception:
        return None


def _probe_rocm_telemetry() -> Optional[tuple[float, float, float]]:
    """AMD rocm-smi has a JSON mode that's stable across versions."""
    smi = shutil.which("rocm-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "-a", "--json"],
            capture_output=True, text=True, timeout=2.5, check=True,
        ).stdout
        import json
        data = json.loads(out)
        # Pick the first card.
        card = next(iter(data.values()))
        # Field names vary across versions; probe for likely keys.
        used = _first_float(card, ["VRAM Total Used Memory (B)",
                                    "GPU memory use (%)", "GPU Memory Used"])
        total = _first_float(card, ["VRAM Total Memory (B)",
                                     "GPU memory total"])
        util = _first_float(card, ["GPU use (%)", "GPU Utilization (%)"])
        temp = _first_float(card, ["Temperature (Sensor edge) (C)",
                                    "Temperature (Sensor junction) (C)",
                                    "Temperature (C)"])
        # Throttle T for AMD consumer ~ 100C.
        if total and total > 0:
            mem_frac = used / total if used <= total else used / 100.0
        else:
            mem_frac = (used or 0.0) / 100.0
        return (mem_frac, (util or 0.0) / 100.0, (temp or 0.0) / 100.0)
    except Exception:
        return None


def _probe_mps_telemetry() -> Optional[tuple[float, float, float]]:
    """Apple unified memory: reuse psutil for system RAM as a proxy.

    Apple does not expose per-process VRAM the way NVIDIA does because
    GPU and CPU share the same memory pool. The cleanest proxy is the
    system memory pressure metric.
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        # On macOS, 'percent' tracks total memory pressure.
        mem_frac = mem.percent / 100.0
    except Exception:
        return None
    # GPU temp / util on Mac requires powermetrics with sudo; skip.
    return (mem_frac, -1.0, -1.0)


def _probe_xpu_telemetry() -> Optional[tuple[float, float, float]]:
    """Intel XPU: try xpu-smi if present."""
    smi = shutil.which("xpu-smi")
    if not smi:
        return None
    try:
        # xpu-smi 'stats' produces JSON-like output in modern versions.
        out = subprocess.run(
            [smi, "stats", "-d", "0"],
            capture_output=True, text=True, timeout=2.5, check=True,
        ).stdout
        # Best-effort parse: look for GPU Utilization / GPU Memory Util keys.
        util = _scan_for_percent(out, ["GPU Utilization", "GPU Util"])
        mem  = _scan_for_percent(out, ["GPU Memory Util", "Memory Util"])
        temp = _scan_for_percent(out, ["GPU Temperature", "Temperature"])
        return (mem / 100.0, util / 100.0, temp / 100.0)
    except Exception:
        return None


def _probe_cpu_telemetry() -> Optional[tuple[float, float, float]]:
    """CPU-only mesh node: 'pressure' is RAM + CPU, no GPU signals."""
    try:
        import psutil
        mem = psutil.virtual_memory().percent / 100.0
        cpu = psutil.cpu_percent(interval=None) / 100.0
    except Exception:
        return None
    # Treat CPU-only's 'gpu_util' as CPU util. Useful for the pressure
    # scalar consumer — it sees one number that means "host is busy".
    return (mem, cpu, -1.0)


def _first_float(d: dict, keys: list[str]) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        try:
            return float(str(v).split()[0])
        except Exception:
            continue
    return None


def _scan_for_percent(text: str, keys: list[str]) -> float:
    for line in text.splitlines():
        for k in keys:
            if k.lower() in line.lower():
                try:
                    digits = "".join(
                        c for c in line if c.isdigit() or c == "."
                    )
                    return float(digits) if digits else 0.0
                except ValueError:
                    continue
    return 0.0


# ----------------------------------------------------------------------------
# Backend-aware torch helpers — the rest of the trainer calls these so it
# never has to branch on backend itself.
# ----------------------------------------------------------------------------

def synchronize(device_str: str = "") -> None:
    """Drain pending kernels on whatever backend we're on."""
    try:
        import torch
    except ImportError:
        return
    if not device_str:
        device_str = select_torch_device()
    try:
        if device_str == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device_str == "mps":
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "synchronize"):
                mps.synchronize()
        elif device_str == "xpu":
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and hasattr(xpu, "synchronize"):
                xpu.synchronize()
        # cpu: nothing to do
    except Exception:
        pass


def empty_cache(device_str: str = "") -> None:
    """Hint the allocator to release fragments. Safe no-op when unsupported."""
    try:
        import torch
    except ImportError:
        return
    if not device_str:
        device_str = select_torch_device()
    try:
        if device_str == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device_str == "mps":
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()
        elif device_str == "xpu":
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and hasattr(xpu, "empty_cache"):
                xpu.empty_cache()
    except Exception:
        pass


def memory_used_bytes(device_str: str = "") -> int:
    """Return current memory in use on the active backend, or 0 if unknown."""
    try:
        import torch
    except ImportError:
        return 0
    if not device_str:
        device_str = select_torch_device()
    try:
        if device_str == "cuda" and torch.cuda.is_available():
            return int(torch.cuda.memory_allocated())
        if device_str == "mps":
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "current_allocated_memory"):
                return int(mps.current_allocated_memory())
        if device_str == "xpu":
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and hasattr(xpu, "memory_allocated"):
                return int(xpu.memory_allocated())
    except Exception:
        return 0
    return 0
