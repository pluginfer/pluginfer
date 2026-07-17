"""Hardware pressure telemetry.

Samples the host's hardware state at a configurable cadence and
reduces it to a single scalar ``P in [0, 1]`` that downstream
components use to throttle themselves. ``P = 0`` means "machine is
idle, train as hard as you want"; ``P = 1`` means "back off
immediately, the OS is about to TDR the GPU".

The pressure scalar is the *max* of normalised contributions from:

* VRAM used / VRAM total          (gpu memory pressure)
* GPU utilisation                  (gpu compute pressure)
* GPU temperature vs. throttle T   (thermal pressure)
* System RAM used / total          (host memory pressure)

Max-of (not sum-of) is deliberate: we want to back off as soon as
*any single resource* hits its ceiling, not wait for the average to
get bad.

novel claim B1 (see the design notes): a method of training a
neural network in which optimizer-state memory footprint is adjusted
in real time as a function of a hardware pressure scalar derived
from concurrent telemetry across at least three of {GPU memory, GPU
utilisation, GPU temperature, host memory, host CPU temperature}.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

try:
    import torch
    _HAS_TORCH = True
except Exception:                                                  # pragma: no cover
    torch = None
    _HAS_TORCH = False

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:                                                  # pragma: no cover
    psutil = None
    _HAS_PSUTIL = False


@dataclass(frozen=True)
class PressureSample:
    """One snapshot of hardware state. All fractions in [0, 1]; -1 means unknown."""
    ts: float
    vram_used_frac: float = -1.0
    gpu_util_frac: float = -1.0
    gpu_temp_frac: float = -1.0      # current_temp / throttle_temp
    ram_used_frac: float = -1.0
    cpu_used_frac: float = -1.0


def _query_nvidia_smi() -> Optional[tuple[float, float, float]]:
    """Return (vram_used_frac, gpu_util_frac, gpu_temp_frac) or None.

    Uses ``nvidia-smi --query-gpu=...``. Robust to missing tooling --
    returns None and the caller falls back. Throttle temp is taken
    as 83 C (typical for consumer Turing/Ampere); see datasheet.
    """
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
    except (subprocess.SubprocessError, OSError, IndexError):
        return None
    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 4:
        return None
    try:
        mem_used = float(parts[0])
        mem_total = float(parts[1])
        util = float(parts[2])
        temp = float(parts[3])
    except ValueError:
        return None
    if mem_total <= 0:
        return None
    THROTTLE_C = 83.0
    return (mem_used / mem_total, util / 100.0, temp / THROTTLE_C)


def _query_torch_vram() -> Optional[float]:
    """Returns vram_used / vram_total for the active CUDA device, or None."""
    if not _HAS_TORCH or not torch.cuda.is_available():
        return None
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception:
        return None
    if total <= 0:
        return None
    return 1.0 - (free / total)


def _query_psutil() -> tuple[float, float]:
    """Returns (ram_used_frac, cpu_used_frac). 0.0 if psutil unavailable."""
    if not _HAS_PSUTIL:
        return (0.0, 0.0)
    try:
        ram = psutil.virtual_memory().percent / 100.0
        cpu = psutil.cpu_percent(interval=None) / 100.0
    except Exception:
        return (0.0, 0.0)
    return (ram, cpu)


def sample_now() -> PressureSample:
    """Take one pressure sample synchronously. Cheap (~5-10ms with smi).

    Probe order:
    1. Vendor-aware probe (cuda / rocm / mps / xpu / cpu) via backend.py.
       Picks the right tool for whatever hardware is present.
    2. Fall back to nvidia-smi if backend probe returned nothing.
    3. Fall back to torch.cuda.mem_get_info if even smi failed.
    4. Always layer on psutil for host RAM + CPU.
    """
    vram_frac = -1.0
    gpu_util = -1.0
    gpu_temp = -1.0
    try:
        from .backend import vendor_telemetry_probe
        probe = vendor_telemetry_probe()
        r = probe()
        if r is not None:
            vram_frac, gpu_util, gpu_temp = r
    except Exception:
        pass
    if vram_frac < 0:
        smi = _query_nvidia_smi()
        if smi is not None:
            vram_frac, gpu_util, gpu_temp = smi
        else:
            torch_vram = _query_torch_vram()
            if torch_vram is not None:
                vram_frac = torch_vram
    ram_frac, cpu_frac = _query_psutil()
    return PressureSample(
        ts=time.monotonic(),
        vram_used_frac=vram_frac,
        gpu_util_frac=gpu_util,
        gpu_temp_frac=gpu_temp,
        ram_used_frac=ram_frac,
        cpu_used_frac=cpu_frac,
    )


def pressure_scalar(s: PressureSample) -> float:
    """Reduce a sample to scalar P in [0, 1]. Max over known signals."""
    signals = []
    if s.vram_used_frac >= 0:
        signals.append(s.vram_used_frac)
    if s.gpu_util_frac >= 0:
        signals.append(s.gpu_util_frac)
    if s.gpu_temp_frac >= 0:
        signals.append(min(s.gpu_temp_frac, 1.0))
    if s.ram_used_frac >= 0:
        signals.append(s.ram_used_frac)
    if s.cpu_used_frac >= 0:
        signals.append(s.cpu_used_frac * 0.5)  # CPU pressure weighted lower
    if not signals:
        return 0.0
    return max(signals)


class PressureSampler:
    """Background sampler. Latest sample is always available via ``.last()``.

    Cheap: thread sleeps between samples; default cadence 250ms.
    """

    def __init__(self, period_s: float = 0.25):
        self._period = max(0.05, period_s)
        self._last = sample_now()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def start(self) -> "PressureSampler":
        if self._t is not None:
            return self
        self._stop.clear()
        self._t = threading.Thread(
            target=self._run, name="hpa-telemetry", daemon=True,
        )
        self._t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)
            self._t = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                s = sample_now()
            except Exception:
                s = PressureSample(ts=time.monotonic())
            with self._lock:
                self._last = s
            self._stop.wait(self._period)

    def last(self) -> PressureSample:
        with self._lock:
            return self._last

    def pressure(self) -> float:
        return pressure_scalar(self.last())

    def __enter__(self) -> "PressureSampler":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
