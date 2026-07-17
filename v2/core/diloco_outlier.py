"""Gradient outlier detector for Diloco aggregation.

When N workers submit gradients for the same training round, an
attacker (or a worker with a corrupted dataset / numerical bug) can
poison the aggregate by submitting an arbitrary tensor. The
gradient-provenance ZK (`core/gradient_provenance.py`, §1) proves the
submission came from a committed (data, model) tuple, but it can't
detect a worker who trained on a *poisoned* shard or whose dataset
truly diverges.

This module provides an outlier detector grounded in robust
statistics rather than the mean+stddev that an attacker can shift:

  * Per-coordinate **median** -- the population's robust centre.
  * Per-coordinate **MAD** (median absolute deviation) -- robust
    spread. MAD is robust up to 50% contamination; the mean+stddev
    is robust to ZERO outliers.
  * Per-worker **L2 distance from cohort median** as the score.
  * Reject anyone whose score > `mad_factor * sum(MAD)` (default 6).

This is BFT for aggregation: as long as < 50% of workers in a round
are malicious (and not collusively producing a coherent fake median),
honest workers are accepted and outliers are rejected.

Pure-stdlib implementation (no torch / numpy hard dep) so the
diloco_aggregator can use it on a coordinator that doesn't have a
GPU stack installed. If torch is available it'd be ~10x faster, but
we don't gate on it.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class OutlierVerdict:
    """Per-worker outlier-detection result for a single round."""
    worker_id: str
    accepted: bool
    distance: float
    threshold: float
    detail: Optional[str] = None


@dataclass
class CohortReport:
    """Result of running the detector over an entire round."""
    accepted: List[str]
    rejected: List[str]
    median: List[float]
    mad: List[float]
    threshold: float
    per_worker: Dict[str, OutlierVerdict] = field(default_factory=dict)

    def cohort_size(self) -> int:
        return len(self.accepted) + len(self.rejected)


def _median(xs: Sequence[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def _mad(xs: Sequence[float], med: float) -> float:
    """Median absolute deviation. Scaled by 1.4826 to be a consistent
    estimator of stddev under normal data (so a `mad_factor` of ~3
    corresponds to the familiar 3-sigma rule)."""
    if not xs:
        return 0.0
    return 1.4826 * statistics.median(abs(x - med) for x in xs)


def _l2(xs: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in xs))


def detect_outliers(
    *,
    gradients: Dict[str, List[float]],
    mad_factor: float = 6.0,
    min_cohort: int = 3,
) -> CohortReport:
    """Score every worker; return who's in vs out.

    `gradients` -- {worker_id: flattened gradient vector}. All vectors
    must be the same length (the trainer guarantees this; we double
    check and reject mismatched workers as `dimension_mismatch`).

    `mad_factor` -- distance threshold in MAD-units. Default 6 is
    conservative (allows ~6-sigma natural variation between honest
    workers); tighten for adversarial training, loosen for
    early-training when gradient norms differ a lot.

    `min_cohort` -- with fewer than this many workers, the detector
    refuses to drop anyone (the statistic is unreliable below 3-4
    samples). All workers are accepted.
    """
    if not gradients:
        return CohortReport(accepted=[], rejected=[], median=[], mad=[],
                            threshold=0.0)

    # Reject dimension mismatches up front.
    expected_dim: Optional[int] = None
    bad_dim: Dict[str, int] = {}
    for wid, g in gradients.items():
        if expected_dim is None:
            expected_dim = len(g)
        elif len(g) != expected_dim:
            bad_dim[wid] = len(g)
    valid = {wid: g for wid, g in gradients.items() if wid not in bad_dim}

    n = len(valid)
    per_worker: Dict[str, OutlierVerdict] = {}
    for wid, dim in bad_dim.items():
        per_worker[wid] = OutlierVerdict(
            worker_id=wid, accepted=False, distance=float("inf"),
            threshold=0.0,
            detail=f"dimension_mismatch ({dim} vs expected {expected_dim})",
        )

    if expected_dim is None or n == 0:
        return CohortReport(
            accepted=[], rejected=list(bad_dim.keys()),
            median=[], mad=[], threshold=0.0, per_worker=per_worker,
        )

    # Per-coordinate median + MAD. With small dim the cost is fine;
    # for a 1B-param gradient this would be 1B medians on each round
    # -- in production swap for a sketch (e.g. count-min) or sample.
    medians: List[float] = []
    mads: List[float] = []
    for j in range(expected_dim):
        col = [g[j] for g in valid.values()]
        med = _median(col)
        medians.append(med)
        mads.append(_mad(col, med))

    threshold = mad_factor * _l2(mads)

    accepted: List[str] = []
    rejected: List[str] = []

    if n < min_cohort:
        # Can't reliably distinguish honest variance from poisoning
        # with too few samples. Accept everyone, log the situation
        # via the `detail` field.
        for wid in valid:
            per_worker[wid] = OutlierVerdict(
                worker_id=wid, accepted=True, distance=0.0,
                threshold=threshold,
                detail=f"cohort {n} < min_cohort {min_cohort}; "
                       f"detector did not run",
            )
            accepted.append(wid)
        return CohortReport(
            accepted=accepted, rejected=rejected,
            median=medians, mad=mads, threshold=threshold,
            per_worker=per_worker,
        )

    # Score every worker.
    for wid, g in valid.items():
        diff = [g[j] - medians[j] for j in range(expected_dim)]
        d = _l2(diff)
        verdict = OutlierVerdict(
            worker_id=wid, accepted=(d <= threshold),
            distance=d, threshold=threshold,
        )
        per_worker[wid] = verdict
        if verdict.accepted:
            accepted.append(wid)
        else:
            rejected.append(wid)
            verdict.detail = (
                f"L2 {d:.4f} > {mad_factor}*||MAD||={threshold:.4f}"
            )

    rejected.extend(bad_dim.keys())
    return CohortReport(
        accepted=accepted, rejected=rejected,
        median=medians, mad=mads, threshold=threshold,
        per_worker=per_worker,
    )


def filter_gradients(
    gradients: Dict[str, List[float]],
    *,
    mad_factor: float = 6.0,
    min_cohort: int = 3,
) -> tuple[Dict[str, List[float]], CohortReport]:
    """Convenience: run the detector and return (kept_gradients, report)."""
    report = detect_outliers(
        gradients=gradients, mad_factor=mad_factor, min_cohort=min_cohort,
    )
    kept = {wid: gradients[wid] for wid in report.accepted}
    return kept, report
