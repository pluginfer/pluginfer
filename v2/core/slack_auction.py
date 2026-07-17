"""
core.slack_auction — Slack-Aware Compute Pricing (TODO §4.2, W14b)
==================================================================

**Design claim:**
  "Slack-aware compute auction with time-of-day x workload-class
   opportunity-cost curves, where a peer node publishes a
   piecewise-linear opportunity-cost function and a marketplace broker
   matches submitted jobs against the union of peer slack curves to
   find a Pareto-optimal price/latency/quality assignment."

Why it matters
--------------
Standard distributed-compute markets quote $/FLOP. But:
  * A 4090 at 03:00 has near-zero opportunity cost (no human is using
    the machine; idle GPU costs the operator $0.02/h in idle power).
  * The same 4090 at 14:00 has high opportunity cost (the operator is
    using it for gaming or work; renting it out costs them
    productivity).
  * Existing markets can't price this — they quote one number per
    GPU regardless of when the job runs.

A node publishes a TimeOfDaySlackCurve over the 24-hour day,
encoding their workload-class-specific opportunity cost. The broker
matches inbound jobs against the union of all curves.

Result: training jobs that are time-elastic ("I need this gradient
done sometime in the next 6 h") cost 5–10x less than centralised
cloud during off-peak hours. Time-sensitive jobs ("RAG response in
<2 s") still match peak-hour bids when nothing else is available.

The opportunity-cost factor
---------------------------
The curve returns a multiplier in [0.0, ∞):
  * < 1.0 : node is in slack window — cheaper than baseline.
  * = 1.0 : node is at baseline cost.
  * > 1.0 : node is in busy window — more expensive than baseline,
            because operator is foregoing real activity.

A Pareto-optimal multi-provider auction over slack curves is the
the design notes. The implementation is a piecewise-linear function for now,
but the design rationale covers any time-varying opportunity-cost
function.

This module is a primitive — it's used by `MeshGPUProvider` in
`core.providers` to scale its bid price.

API
---
    >>> import datetime
    >>> curve = TimeOfDaySlackCurve(
    ...     points=[(0, 0.2), (6, 0.3), (9, 1.0), (18, 1.3),
    ...             (22, 0.6), (24, 0.2)],
    ...     workload_class="general",
    ... )
    >>> curve.opportunity_cost_factor(at=datetime.time(3, 0))
    # ~0.25
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TimeOfDaySlackCurve:
    """A piecewise-linear curve mapping wall-clock hour-of-day to
    opportunity-cost factor.

    Args:
        points: List of (hour, factor) breakpoints. Hour is in
            [0, 24], factor in [0, ∞). Linear interpolation between
            adjacent breakpoints.
        workload_class: Tag for which job kinds this curve applies
            to (e.g. 'general', 'gaming-on', 'video-call'). The
            design rationale includes this dimension because a single
            node may publish multiple curves (one per workload class)
            so the broker can pick the *active* curve based on the
            node's current state.
        timezone_offset_hours: For nodes that want to publish in a
            specific local timezone; default 0 = UTC.

    Conformance to the design rationale:
      * Time-varying: yes (24-hour cycle).
      * Workload-class-aware: yes (the workload_class tag).
      * Opportunity-cost-coded: yes (factor < 1 = idle, > 1 = busy).
    """
    points: List[Tuple[float, float]]      # [(hour, factor)]
    workload_class: str = "general"
    timezone_offset_hours: float = 0.0

    def __post_init__(self):
        # Normalise: sorted by hour, hour clamped to [0, 24], factor >= 0.
        self.points = sorted(
            (max(0.0, min(24.0, float(h))),
             max(0.0, float(f)))
            for h, f in self.points
        )
        if not self.points:
            raise ValueError("TimeOfDaySlackCurve needs at least one point")

    def opportunity_cost_factor(
        self,
        at: Optional[datetime.time] = None,
    ) -> float:
        """Return the opportunity-cost factor at the given clock time.

        If `at` is None, uses the current local time (after applying
        timezone_offset_hours)."""
        if at is None:
            now = datetime.datetime.utcnow() + datetime.timedelta(
                hours=self.timezone_offset_hours)
            at = now.time()
        # Convert to fractional hour [0, 24).
        hour = at.hour + at.minute / 60.0 + at.second / 3600.0

        pts = self.points
        # Clamp before first / after last.
        if hour <= pts[0][0]:
            return pts[0][1]
        if hour >= pts[-1][0]:
            return pts[-1][1]
        # Find bracketing pair and interpolate.
        for (h0, f0), (h1, f1) in zip(pts[:-1], pts[1:]):
            if h0 <= hour <= h1:
                if h1 == h0:
                    return f0
                t = (hour - h0) / (h1 - h0)
                return f0 + t * (f1 - f0)
        # Should be unreachable.
        return pts[-1][1]


def default_consumer_curve() -> TimeOfDaySlackCurve:
    """A reasonable default for consumer GPUs in a typical work-from-home
    schedule (used by tests and as a starting point for new nodes that
    haven't tuned their own curve yet)."""
    return TimeOfDaySlackCurve(
        points=[
            (0.0, 0.20),    # midnight — deeply idle
            (6.0, 0.30),    # early morning — still cheap
            (9.0, 1.00),    # work begins
            (12.0, 1.10),   # lunch peak
            (14.0, 1.20),   # afternoon meetings
            (18.0, 1.30),   # gaming starts
            (22.0, 0.60),   # winding down
            (24.0, 0.20),   # back to deep idle (wraps to 0.0)
        ],
        workload_class="general",
    )
