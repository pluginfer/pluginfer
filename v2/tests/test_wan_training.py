"""WAN-tolerant training: DiLoCo + int4 quant + top-K sparsification
with error feedback.

Invariants:
  * int4 round-trip preserves the values within scale (≤ 1 lsb).
  * Top-K keeps EXACTLY the top-k-by-magnitude indices.
  * Error feedback eventually transmits the un-sent portion (no
    gradient signal lost; just deferred).
  * Aggregate of N identical pseudo-gradients = the pseudo-gradient
    itself (mean is idempotent).
  * Bandwidth estimate for 70B + default settings is < 100 MB per
    step-equivalent (proves the WAN-feasibility claim).
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from core.wan_training import (
    ErrorFeedbackState,
    Int4Quantized,
    accumulate_pseudo_gradient,
    aggregate_pseudo_gradients,
    dequantize_int4_dense,
    dequantize_sparse_int4,
    estimate_sync_bandwidth,
    quantize_int4_dense,
    top_k_sparsify_int4,
)


# ---------------------------------------------------------------------------
# Int4 round-trip
# ---------------------------------------------------------------------------

def test_int4_round_trip_within_one_lsb():
    values = [0.0, 0.1, -0.3, 0.7, -0.99, 0.5, -0.5]
    q = quantize_int4_dense(values)
    recovered = dequantize_int4_dense(q)
    assert len(recovered) == len(values)
    for orig, rec in zip(values, recovered):
        # Each step is `scale`; quantization error ≤ 0.5 × scale
        # ≈ 0.5 × (1.0/7) ≈ 0.071.
        assert abs(orig - rec) <= q.scale, (orig, rec, q.scale)


def test_int4_packing_density_two_per_byte():
    values = list(range(-8, 8))     # 16 values, hits every nibble
    q = quantize_int4_dense([float(v) / 7.0 for v in values])
    # 16 values / 2 per byte = 8 bytes.
    assert len(q.data) == 8


def test_int4_handles_empty():
    q = quantize_int4_dense([])
    assert q.length == 0
    assert dequantize_int4_dense(q) == []


# ---------------------------------------------------------------------------
# Top-K sparsification
# ---------------------------------------------------------------------------

def test_top_k_keeps_largest_magnitudes():
    grad = [0.01, 5.0, -0.02, 3.0, 0.03, -10.0, 0.04, 0.5]
    # k_fraction=0.5 → keep 4 indices: 1 (5.0), 3 (3.0), 5 (-10.0), 7 (0.5)
    q = top_k_sparsify_int4(grad, k_fraction=0.5)
    assert q.nnz == 4
    indices = [
        struct.unpack_from("<I", q.indices, off)[0]
        for off in range(0, len(q.indices), 4)
    ]
    assert set(indices) == {1, 3, 5, 7}


def test_top_k_reconstructs_zeros_for_non_kept_indices():
    grad = [0.0, 5.0, 0.0, 3.0, 0.0]
    q = top_k_sparsify_int4(grad, k_fraction=0.4)
    dense = dequantize_sparse_int4(q, full_length=5)
    # Indices 1 and 3 carry real values; others zero.
    assert dense[0] == 0.0
    assert dense[2] == 0.0
    assert dense[4] == 0.0
    assert dense[1] != 0.0
    assert dense[3] != 0.0


# ---------------------------------------------------------------------------
# Error feedback
# ---------------------------------------------------------------------------

def test_error_feedback_eventually_transmits_un_sent_values():
    """Across many sparsification rounds with the SAME gradient,
    the residual accumulates and the un-transmitted indices
    eventually become the largest — they get picked.

    This proves convergence: signal isn't lost, just deferred."""
    grad = [0.05] * 10        # 10 equal-magnitude values
    ef = ErrorFeedbackState()
    transmitted_indices_per_round: list = []
    for _ in range(30):
        q = top_k_sparsify_int4(grad, k_fraction=0.1, error_feedback=ef)
        idx = struct.unpack("<I", q.indices)[0]
        transmitted_indices_per_round.append(idx)
    # Over 30 rounds, every index should have been transmitted at
    # least once (k=1, 10 indices, perfect round-robin would visit
    # each 3x in 30 rounds).
    assert set(transmitted_indices_per_round) == set(range(10))


def test_error_feedback_off_loses_un_sent_signal():
    """Without error feedback, the un-sent indices are dropped
    forever — proves the EF mechanism is doing the right thing."""
    grad = [0.05] * 10
    transmitted = set()
    for _ in range(30):
        q = top_k_sparsify_int4(grad, k_fraction=0.1)   # no EF
        idx = struct.unpack("<I", q.indices)[0]
        transmitted.add(idx)
    # Without EF, the same index gets picked every round because
    # the gradient never changes.
    assert len(transmitted) == 1


# ---------------------------------------------------------------------------
# DiLoCo pseudo-gradient + aggregation
# ---------------------------------------------------------------------------

def test_accumulate_pseudo_gradient_packages_inner_window():
    before = [1.0, 2.0, 3.0, 4.0]
    after = [1.1, 2.1, 2.5, 4.2]   # deltas: 0.1, 0.1, -0.5, 0.2
    g = accumulate_pseudo_gradient(
        before, after, parameter_id="w0", inner_steps=500, k_fraction=0.25,
    )
    assert g.full_length == 4
    assert g.inner_steps == 500
    assert g.sparse_int4.nnz == 1     # 25% of 4 = 1
    # The single transmitted index is the one with the biggest delta,
    # which is index 2 (delta -0.5).
    idx = struct.unpack("<I", g.sparse_int4.indices)[0]
    assert idx == 2


def test_aggregate_pseudo_gradients_means_dense_dequantization():
    before = [1.0, 2.0, 3.0, 4.0]
    g1 = accumulate_pseudo_gradient(
        before, [1.1, 2.1, 2.5, 4.2],
        parameter_id="w0", inner_steps=500, k_fraction=0.5,
    )
    g2 = accumulate_pseudo_gradient(
        before, [1.1, 2.1, 2.5, 4.2],     # identical
        parameter_id="w0", inner_steps=500, k_fraction=0.5,
    )
    avg = aggregate_pseudo_gradients([g1, g2])
    # Two identical inputs → average == any one of them.
    expected = dequantize_sparse_int4(g1.sparse_int4, full_length=4)
    for a, b in zip(avg, expected):
        assert abs(a - b) < 1e-6


def test_aggregate_handles_empty_input():
    assert aggregate_pseudo_gradients([]) == []


# ---------------------------------------------------------------------------
# Bandwidth math — the marketing-slide proof
# ---------------------------------------------------------------------------

def test_70b_model_int4_top1pct_fits_under_100mb_per_step():
    """The headline number: 70B params, int4 + top-1% + DiLoCo-500
    should be << 1 GB per step-equivalent. We want under 100 MB so
    a 100 MB/s consumer uplink syncs in under a second."""
    est = estimate_sync_bandwidth(
        param_count=70_000_000_000, inner_steps=500,
        k_fraction=0.01, int4=True,
    )
    gb = est.gb_per_step_equivalent
    # Generous bound; the actual math should come out under 0.1 GB.
    assert gb < 0.1, f"expected < 0.1 GB, got {gb} GB"


def test_fp32_no_diloco_top1_baseline_is_still_huge():
    """Without any of the multipliers, a 70B model can't sync over
    consumer internet — proves the moat is real."""
    est = estimate_sync_bandwidth(
        param_count=70_000_000_000, inner_steps=1,
        k_fraction=1.0, int4=False,
    )
    gb = est.gb_per_step_equivalent
    # ~260 GB per step. WAN-impossible.
    assert gb > 100


def test_int4_helps_significantly_over_fp32():
    """int4 alone (no top-K) divides bytes/value by 8 for the value
    payload; the index overhead means the net win is smaller, but
    still positive at scale."""
    fp32 = estimate_sync_bandwidth(
        param_count=70_000_000_000, inner_steps=500,
        k_fraction=0.01, int4=False,
    )
    int4 = estimate_sync_bandwidth(
        param_count=70_000_000_000, inner_steps=500,
        k_fraction=0.01, int4=True,
    )
    assert int4.bytes_per_step_equivalent < fp32.bytes_per_step_equivalent
