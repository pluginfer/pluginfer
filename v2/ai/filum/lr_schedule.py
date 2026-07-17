"""Learning-rate schedules + numerical-stability guards.

The original demo training loop saw loss explode from 13 to 1179
within 50 steps. Three fixes, all packaged here so every trainer
(demo, real, hpa) shares the same wiring:

* **Warmup** — linear ramp from 0 to ``target_lr`` over
  ``warmup_steps`` (default 100). Tiny models with byte-level vocab
  are extremely sensitive to early high learning rates.
* **Cosine decay** — after warmup, lr follows a cosine curve to a
  ``min_lr`` floor over ``total_steps - warmup_steps``. Standard
  practice; included here so demo and production share it.
* **NaN/Inf guard** — ``is_finite_loss(loss)`` returns False on
  non-finite or absurdly-large loss. Caller can skip the step.

The schedule is *stateless except for step count* — pass ``step``
in and get the right LR back. No global state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LRSchedule:
    target_lr: float
    warmup_steps: int = 100
    total_steps: int = 5000
    min_lr_frac: float = 0.10        # min_lr = min_lr_frac * target_lr

    def lr_at(self, step: int) -> float:
        if step < 0:
            return 0.0
        # Phase 1: linear warmup.
        if step < self.warmup_steps:
            return self.target_lr * (step + 1) / max(1, self.warmup_steps)
        # Phase 2: cosine decay to min_lr.
        denom = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / denom)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_lr = self.target_lr * self.min_lr_frac
        return min_lr + (self.target_lr - min_lr) * cos


def is_finite_loss(loss_value, *, max_abs: float = 1e6) -> bool:
    """Returns False if loss is NaN, Inf, or absurdly large.

    The callers use this to skip a step without touching parameters,
    so a single bad batch doesn't poison the run. Detaches first so
    the loss tensor's autograd graph is not pulled into a Python
    float conversion (avoids the requires_grad UserWarning).
    """
    detach = getattr(loss_value, "detach", None)
    if callable(detach):
        loss_value = detach()
    try:
        v = float(loss_value)
    except (TypeError, ValueError):
        return False
    if v != v:                                    # NaN
        return False
    if v == float("inf") or v == -float("inf"):
        return False
    return abs(v) <= max_abs


def apply_lr(optimizer, lr: float) -> None:
    """Set optimizer's lr in-place. Compatible with torch.optim.Optimizer
    and the AdamW8bit class in this repo."""
    for group in optimizer.param_groups:
        group["lr"] = lr


class DivergenceGuard:
    """Skip parameter updates when the current loss spikes far above
    the best seen so far — catches the canonical "loss was decreasing,
    then exploded" failure that grad clipping alone does not stop on
    small models with byte vocabularies.

    The guard is conservative: it only fires *after* a warm-in window
    so initial-noise losses do not lock in a low best. After warm-in,
    if ``loss > best * spike_ratio`` it returns True (caller skips
    the step). Best is only updated when the step is accepted, so a
    string of skipped steps cannot drift the bar upward.
    """

    def __init__(
        self,
        *,
        warm_in_steps: int = 30,
        spike_ratio: float = 4.0,
        max_consecutive_skips: int = 10,
    ):
        self.warm_in_steps = max(0, int(warm_in_steps))
        self.spike_ratio = float(spike_ratio)
        self.max_consecutive_skips = max(1, int(max_consecutive_skips))
        self.best: float = float("inf")
        self.consecutive_skips: int = 0
        self.total_skips: int = 0

    def should_skip(self, step: int, loss_scalar: float) -> bool:
        if step < self.warm_in_steps:
            return False
        if loss_scalar <= self.best * self.spike_ratio:
            return False
        if self.consecutive_skips >= self.max_consecutive_skips:
            # Never wedge — give up skipping after N in a row, let the
            # optimizer either recover or reveal a deeper bug.
            return False
        return True

    def accept(self, loss_scalar: float) -> None:
        if loss_scalar < self.best:
            self.best = loss_scalar
        self.consecutive_skips = 0

    def skip(self) -> None:
        self.consecutive_skips += 1
        self.total_skips += 1
