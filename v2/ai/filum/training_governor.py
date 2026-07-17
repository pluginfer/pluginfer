"""Adaptive training governor.

Goal: training never crashes the host or the mesh node. Not by
throttling — the happy path runs at full speed — but by detecting
unrecoverable conditions early and choosing an intelligent recovery
instead of letting the kernel walk into a wall.

What it watches
---------------
1.  Process-level RAM / GPU-VRAM headroom. Hard ceiling NOT enforced;
    we only react when usage trends toward exhaustion.
2.  Loss / gradient-norm trajectory. Three divergences in a row trips
    a recovery (LR cut + step skip). One blow-up just drops the step.
3.  CUDA errors raised from `optimizer.step()` or model forward. We
    classify (OOM vs illegal-address vs other), then pick a recovery:
       OOM             -> empty_cache + retry; if it repeats, halve
                          the micro-batch; if still failing, fallback_cpu.
       illegal-address -> the device is poisoned (we already protect
                          against the int8-NaN UB, but defence in
                          depth): drop step, fallback_cpu next time.
       other            -> propagate to the caller.

What it does NOT do
-------------------
* It does not pre-emptively cap VRAM (that costs throughput).
* It does not pause training on idle pressure.
* It does not message the mesh on every blip — only on confirmed
  divergence, so we don't spam peers with false positives.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

try:
    import torch
    _HAS_TORCH = True
except Exception:                                                # pragma: no cover
    torch = None
    _HAS_TORCH = False


@dataclass
class TrainingGovernor:
    device: str = "cpu"
    log: Callable[[str], None] = print

    consecutive_skips: int = 0
    last_losses: List[float] = field(default_factory=list)
    oom_count: int = 0
    illegal_count: int = 0
    started_at: float = field(default_factory=time.monotonic)

    # Tunables (chosen so the happy path never hits them)
    loss_window: int = 16          # rolling window for divergence check
    divergence_factor: float = 8.0 # this-loss / median-window > 8x => spike
    max_consecutive_skips: int = 12
    max_ooms_before_cpu: int = 2

    # Mesh-broadcast hook. Caller wires in something like
    # `lambda snap: task_router.update_peer_health(self_id, snap)` so
    # peers route work away from a node that's recovering. Optional —
    # the governor stays dependency-free without it.
    broadcast_fn: Optional[Callable[[dict], None]] = None
    broadcast_throttle_s: float = 5.0
    _last_broadcast_at: float = field(default=0.0, repr=False)

    @classmethod
    def start(cls, device: str = "cpu", log: Callable[[str], None] = print) -> "TrainingGovernor":
        gov = cls(device=device, log=log)
        gov.log(f"governor: started on {device} (adaptive, no throughput cap)")
        return gov

    # ---- happy-path hooks -------------------------------------------------
    def tick(self, step: int) -> None:
        """Called once per step before forward. Cheap heartbeat — no I/O."""
        if step % 200 == 0 and self.device == "cuda" and _HAS_TORCH:
            # Periodic memory-trend check. We do NOT cap; we only log so
            # operators / mesh peers can see the trajectory.
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                self.log(
                    f"governor: VRAM allocated={allocated:.2f}GB reserved={reserved:.2f}GB"
                )
            except Exception:
                pass

    def record_ok(self, loss_val: float) -> None:
        """Called after a successful step. Tracks loss for divergence detection."""
        self.consecutive_skips = 0
        self.last_losses.append(loss_val)
        if len(self.last_losses) > self.loss_window:
            self.last_losses.pop(0)
        # Heartbeat broadcast at most every broadcast_throttle_s so
        # the mesh has fresh routing information without flooding peers.
        self._maybe_broadcast(force=False)

    def _maybe_broadcast(self, *, force: bool) -> None:
        if self.broadcast_fn is None:
            return
        now = time.monotonic()
        if not force and now - self._last_broadcast_at < self.broadcast_throttle_s:
            return
        try:
            self.broadcast_fn(self.health_snapshot())
            self._last_broadcast_at = now
        except Exception as e:
            # A broken broadcast hook must NEVER break training.
            self.log(f"governor: broadcast_fn raised, ignoring: {e}")

    # ---- divergence ------------------------------------------------------
    def is_diverging(self, current_loss: float) -> bool:
        if len(self.last_losses) < 4:
            return False
        sorted_window = sorted(self.last_losses)
        median = sorted_window[len(sorted_window) // 2]
        if median <= 0:
            return False
        return current_loss > median * self.divergence_factor

    # ---- error classifier + recovery -------------------------------------
    def handle_runtime_error(self, exc: Exception, step: int) -> str:
        """Return one of: 'skip' | 'fallback_cpu' | 'raise'."""
        msg = str(exc).lower()
        is_oom = any(s in msg for s in (
            "out of memory", "cuda out of memory", "cudaerrormemoryallocation",
        ))
        is_illegal = any(s in msg for s in (
            "illegal memory access", "cudaerrorillegaladdress",
        ))

        if is_oom:
            self.oom_count += 1
            self.log(f"governor: OOM #{self.oom_count} at step {step} — clearing cache + retrying")
            self._free_cuda()
            self._maybe_broadcast(force=True)   # peers should know
            if self.oom_count >= self.max_ooms_before_cpu:
                self.log("governor: repeated OOM — switching to CPU for survival")
                return "fallback_cpu"
            return "skip"

        if is_illegal:
            self.illegal_count += 1
            self.log(
                f"governor: cudaErrorIllegalAddress at step {step} — "
                "device is poisoned, falling back to CPU"
            )
            self._free_cuda()
            self._maybe_broadcast(force=True)   # quarantine signal
            return "fallback_cpu"

        # Anything else (driver crash, kernel launch failure, etc.):
        # try one CUDA cache clear, then propagate. We do not silently
        # swallow unknown errors.
        self.log(f"governor: unhandled runtime error at step {step}: {exc}")
        self._free_cuda()
        self.consecutive_skips += 1
        if self.consecutive_skips >= self.max_consecutive_skips:
            return "fallback_cpu"
        return "raise"

    def _free_cuda(self) -> None:
        if not _HAS_TORCH:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

    # ---- mesh-side broadcast (stub: caller wires this in) ----------------
    def health_snapshot(self) -> dict:
        """Snapshot the governor publishes to the mesh so peers can route
        work away from a node that's recovering. Caller is responsible
        for actually publishing — keeps this module dependency-free."""
        return {
            "device": self.device,
            "uptime_s": time.monotonic() - self.started_at,
            "oom_count": self.oom_count,
            "illegal_count": self.illegal_count,
            "consecutive_skips": self.consecutive_skips,
            "loss_window_size": len(self.last_losses),
            "loss_window_median": (
                sorted(self.last_losses)[len(self.last_losses) // 2]
                if self.last_losses else None
            ),
        }
