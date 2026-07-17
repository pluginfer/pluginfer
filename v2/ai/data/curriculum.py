"""Simple curriculum scheduler for multi-task training.

Returns a per-step weighting vector across tasks. Three phases:

  Phase 0 [0, warmup): heavy weight on easy tasks (job_router) so the
                       model gets fluent in domain text first.
  Phase 1 [warmup, 2*warmup): mix in provider_quality.
  Phase 2 [2*warmup, ...): full uniform mix.

Used by Phase 4's MultiTaskLoss to scale per-task losses.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumScheduler:
    warmup_steps: int = 200
    tasks: tuple[str, ...] = ("job_router", "provider_quality", "price", "anomaly")

    def weights_at(self, step: int) -> dict[str, float]:
        if step < self.warmup_steps:
            # Easy task only
            base = {t: 0.0 for t in self.tasks}
            base["job_router"] = 1.0
            return base
        if step < 2 * self.warmup_steps:
            # Two-task mix
            return {
                "job_router": 0.6,
                "provider_quality": 0.4,
                "price": 0.0,
                "anomaly": 0.0,
            }
        # Full uniform
        n = len(self.tasks)
        return {t: 1.0 / n for t in self.tasks}
