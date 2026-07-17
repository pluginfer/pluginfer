"""Cooperative GPU yield + soft VRAM cap.

The user's GTX 1650 hung the entire laptop because:

1. A training kernel ran past the Windows WDDM 2-second TDR
   (Timeout Detection and Recovery) deadline. WDDM kills the GPU
   driver, which knocks the desktop compositor offline -- the
   "frozen mouse cursor" symptom.
2. Optimizer state crossed the 4 GB physical VRAM limit and CUDA
   raised ``cudaErrorIllegalAddress``.

We address both with three primitives in this module:

* ``soft_vram_cap_bytes(frac)`` - returns a target VRAM budget so
  the trainer can pick a micro-batch that fits inside it. Defaults
  to 70% of total VRAM; leaves enough headroom for the OS
  compositor + browser.
* ``CooperativeYield`` - inserts ``torch.cuda.synchronize()`` plus a
  brief sleep when pressure spikes. This gives the OS scheduler a
  window to paint a frame so the system doesn't *appear* hung even
  if training is hot.
* ``cuda_oom_guard()`` - context manager that catches CUDA OOM /
  illegal-address and resets the optimizer's microbatch by half.

novel claim B3 (see the design notes): a method of training a
neural network on a graphics processor shared with a display
compositor, comprising periodically inserting compute-yield points
in response to a hardware pressure scalar exceeding a threshold,
such that the operating system retains the ability to refresh the
display and respond to user input during training.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional

try:
    import torch
    _HAS_TORCH = True
except Exception:                                                  # pragma: no cover
    torch = None
    _HAS_TORCH = False


def soft_vram_cap_bytes(frac: float = 0.70, headroom_mib: int = 600) -> int:
    """Return target VRAM budget in bytes for the active CUDA device.

    ``frac`` is the fraction of *total* VRAM we'll consume; ``headroom_mib``
    is an absolute floor we always leave to the OS. Whichever is more
    conservative wins.

    Falls back to a 1 GiB budget when CUDA is not available -- callers
    should treat that as "we're on CPU, don't bother capping".
    """
    if not _HAS_TORCH or not torch.cuda.is_available():
        return 1 << 30
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception:
        total = torch.cuda.get_device_properties(0).total_memory
    by_frac = int(total * max(0.1, min(0.95, frac)))
    by_headroom = int(total - headroom_mib * (1 << 20))
    return max(1 << 28, min(by_frac, by_headroom))


def vram_used_bytes() -> int:
    if not _HAS_TORCH or not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.memory_allocated())
    except Exception:
        return 0


class CooperativeYield:
    """Inserts cooperative yield points based on a pressure callable.

    Usage::

        coop = CooperativeYield(pressure_fn=sampler.pressure)
        for step in range(N):
            ...
            coop.maybe_yield()

    On yield: synchronize the active CUDA stream (blocks until pending
    work drains, so the GPU is briefly idle) and ``time.sleep`` for a
    pressure-scaled duration. The sleep is short (10-50ms typical) but
    enough for the OS compositor to paint a frame and the WDDM
    watchdog to reset.
    """

    def __init__(
        self,
        pressure_fn,
        threshold: float = 0.85,
        base_sleep_s: float = 0.010,
        max_sleep_s: float = 0.080,
    ):
        self._pressure_fn = pressure_fn
        self._threshold = float(threshold)
        self._base = float(base_sleep_s)
        self._max = float(max_sleep_s)
        self._yields = 0

    @property
    def yield_count(self) -> int:
        return self._yields

    def maybe_yield(self) -> bool:
        """Yield if pressure is over threshold. Returns True iff yielded."""
        try:
            p = float(self._pressure_fn())
        except Exception:
            return False
        if p < self._threshold:
            return False
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                # Free anything we can give back before the sleep so the
                # OS sees a real reduction in VRAM pressure during the
                # window.
                torch.cuda.empty_cache()
            except Exception:
                pass
        # Sleep duration scales linearly above threshold.
        slack = max(0.0, p - self._threshold) / max(1e-6, 1.0 - self._threshold)
        s = self._base + slack * (self._max - self._base)
        time.sleep(min(self._max, s))
        self._yields += 1
        return True


@contextmanager
def cuda_oom_guard(on_oom):
    """Context manager that catches CUDA OOM / illegal-access and calls
    ``on_oom(exc)``; the caller decides whether to retry.

    Re-raises if ``on_oom`` returns False (or anything falsy).
    """
    try:
        yield
    except RuntimeError as e:                                      # pragma: no cover
        msg = str(e).lower()
        is_oom = (
            "out of memory" in msg
            or "cuda out of memory" in msg
            or "cudaerrorillegaladdress" in msg
            or "illegal memory access" in msg
        )
        if not is_oom:
            raise
        if not on_oom(e):
            raise
        # Recover GPU state before returning.
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
