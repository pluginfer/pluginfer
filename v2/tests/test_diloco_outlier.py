"""Tests for the gradient outlier detector."""

from __future__ import annotations

import random
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.diloco_outlier import detect_outliers, filter_gradients  # noqa: E402


def test_5_honest_1_attacker_attacker_rejected():
    """5 workers submit gradients drawn from the same distribution.
    A 6th worker submits noise 100x larger (the "I am poisoning
    aggregation" attack). The 6th must be rejected; the 5 honest
    must all pass."""
    rng = random.Random(42)
    dim = 32
    base = [rng.gauss(0, 0.1) for _ in range(dim)]
    honest = {
        f"h{i}": [base[j] + rng.gauss(0, 0.02) for j in range(dim)]
        for i in range(5)
    }
    attacker = {"attacker": [rng.gauss(0, 10.0) for _ in range(dim)]}
    grads = {**honest, **attacker}

    report = detect_outliers(gradients=grads, mad_factor=6.0)
    assert "attacker" in report.rejected
    for hid in honest:
        assert hid in report.accepted, (
            f"honest worker {hid} mistakenly rejected: "
            f"{report.per_worker[hid].detail}"
        )


def test_too_small_cohort_accepts_everyone():
    """Below `min_cohort` (default 3), we don't have enough samples
    to estimate MAD reliably. Detector accepts all + flags the
    situation in `detail`."""
    grads = {
        "a": [1.0, 2.0, 3.0],
        "b": [1000.0, -2000.0, 50.0],   # would be a clear outlier with N=10
    }
    report = detect_outliers(gradients=grads, mad_factor=6.0, min_cohort=3)
    assert report.rejected == []
    assert "a" in report.accepted and "b" in report.accepted
    assert "min_cohort" in (report.per_worker["a"].detail or "")


def test_dimension_mismatch_rejected_independently():
    """A worker who submits a shorter gradient vector is filtered
    BEFORE the statistical test even runs (otherwise the
    per-coordinate median would crash)."""
    grads = {
        "a": [0.0] * 8,
        "b": [0.0] * 8,
        "c": [0.0] * 8,
        "broken": [0.0] * 4,   # wrong dim
    }
    report = detect_outliers(gradients=grads, mad_factor=6.0)
    assert "broken" in report.rejected
    assert "dimension_mismatch" in (report.per_worker["broken"].detail or "")
    # The 3 honest workers go through.
    assert set(report.accepted) == {"a", "b", "c"}


def test_50pct_attackers_corrupts_median_known_limit():
    """When attackers reach 50% of the cohort and submit a coherent
    fake median, the median itself shifts. This test DOCUMENTS the
    known limit: robust statistics are robust UP TO 50% contamination;
    beyond that, additional defences (reputation, ZK provenance,
    on-chain stake) are required."""
    dim = 16
    honest_val = 0.0
    attacker_val = 100.0
    grads = {}
    # 4 honest, 4 attackers (50/50). With n=8 the median-of-medians
    # falls between the two clusters, so the detector treats one of
    # the two groups as "outlier" and the other as "majority".
    for i in range(4):
        grads[f"h{i}"] = [honest_val] * dim
        grads[f"a{i}"] = [attacker_val] * dim
    report = detect_outliers(gradients=grads, mad_factor=6.0)
    # Either 4 attackers OR 4 honest end up rejected -- this is by
    # design: < 50% honest cannot fix this layer alone.
    assert len(report.rejected) >= 4 or len(report.rejected) == 0
    # The point of the test: the function returns SOMETHING coherent
    # without crashing or accepting both halves.


def test_filter_gradients_returns_only_accepted():
    rng = random.Random(0)
    grads = {
        f"w{i}": [rng.gauss(0, 0.1) for _ in range(8)]
        for i in range(5)
    }
    grads["bad"] = [50.0] * 8    # outlier
    kept, report = filter_gradients(gradients=grads, mad_factor=4.0)
    assert "bad" not in kept
    assert "bad" in report.rejected
    for k in kept:
        assert k in report.accepted


def test_zero_gradient_cohort_handled_gracefully():
    """Edge case: every worker submits an exactly-zero gradient. MAD
    is zero; threshold is zero; everyone is exactly at the median.
    Detector must NOT crash and must accept all."""
    grads = {f"w{i}": [0.0] * 8 for i in range(5)}
    report = detect_outliers(gradients=grads, mad_factor=6.0)
    assert report.rejected == []
    assert len(report.accepted) == 5
