"""§E Equal-Access layer smoke tests.

Covers compute-as-currency (§E1), delta-sync (§E2), fragment
training (§E3), data-labor royalty math (§E4).

CPU-only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ---------- §E1 compute-as-currency ----------------------------------------

def test_compute_currency_seed_and_submit(tmp_path: Path):
    from ai.filum.hpa.compute_currency import (
        ComputeCurrencyExchange, ComputeCurrencyConfig,
    )

    ex = ComputeCurrencyExchange(ComputeCurrencyConfig(
        state_path=str(tmp_path / "cc.json"),
        max_debt_per_pubkey=100.0,
    ))
    ex.seed(50.0)
    debt = ex.submit_for_compute_debt("alice", 10.0, job_id="j1")
    assert debt.initial_tflop_hr == 10.0
    assert ex.balance().available_tflop_hr == 40.0
    assert ex.outstanding_for("alice") == 10.0


def test_compute_currency_repays_own_debt_first(tmp_path: Path):
    from ai.filum.hpa.compute_currency import (
        ComputeCurrencyExchange, ComputeCurrencyConfig,
    )

    ex = ComputeCurrencyExchange(ComputeCurrencyConfig(
        state_path=str(tmp_path / "cc.json"),
    ))
    ex.seed(100.0)
    ex.submit_for_compute_debt("bob", 20.0)
    assert ex.outstanding_for("bob") == 20.0

    res = ex.contribute_compute("bob", 25.0)
    # 20 should go to bob's debt, 5 should go to the pool.
    assert res["applied_to_debt"] == 20.0
    assert res["added_to_pool"] == 5.0
    assert ex.outstanding_for("bob") == 0.0


def test_compute_currency_caps_debt_per_pubkey(tmp_path: Path):
    from ai.filum.hpa.compute_currency import (
        ComputeCurrencyExchange, ComputeCurrencyConfig,
    )

    ex = ComputeCurrencyExchange(ComputeCurrencyConfig(
        state_path=str(tmp_path / "cc.json"),
        max_debt_per_pubkey=30.0,
    ))
    ex.seed(1000.0)
    ex.submit_for_compute_debt("c", 25.0)
    with pytest.raises(ValueError):
        ex.submit_for_compute_debt("c", 10.0)   # 25+10 > 30


def test_compute_currency_persists_across_restart(tmp_path: Path):
    from ai.filum.hpa.compute_currency import (
        ComputeCurrencyExchange, ComputeCurrencyConfig,
    )

    cfg = ComputeCurrencyConfig(state_path=str(tmp_path / "cc.json"))
    a = ComputeCurrencyExchange(cfg); a.seed(50.0); a.submit_for_compute_debt("d", 5.0)
    b = ComputeCurrencyExchange(cfg)
    assert b.outstanding_for("d") == 5.0
    assert b.balance().available_tflop_hr == 45.0


def test_compute_currency_write_off_capped(tmp_path: Path):
    from ai.filum.hpa.compute_currency import (
        ComputeCurrencyExchange, ComputeCurrencyConfig,
    )

    ex = ComputeCurrencyExchange(ComputeCurrencyConfig(
        state_path=str(tmp_path / "cc.json"),
        default_repay_window_days=0.0,        # immediately expired
        insolvency_haircut_pct=0.10,
    ))
    ex.seed(100.0)
    # Submit 50 tflop-hr — debt expires instantly.
    ex.submit_for_compute_debt("e", 50.0)
    # cumulative_consumed = 50. cap = 50 * 0.10 = 5.
    written = ex.write_off_expired()
    assert written <= 5.0 + 1e-6


# ---------- §E3 fragment training ------------------------------------------

def test_fragment_split_makes_correct_blocks():
    from ai.filum.hpa.fragment_training import split_layer_into_fragments

    frags = split_layer_into_fragments(
        layer_idx=7, matrix_id="q_proj",
        full_rows=512, cols=128,
        target_memory_bytes=200 * 1024,        # 200 KB
        bytes_per_param=4,
    )
    # All rows covered, no overlap.
    covered = []
    for f in frags:
        for r in range(f.row_start, f.row_end):
            covered.append(r)
    assert sorted(covered) == list(range(512))
    # Each fragment fits the budget.
    from ai.filum.hpa.fragment_training import estimate_memory_for_fragment
    for f in frags:
        assert estimate_memory_for_fragment(f) <= 200 * 1024


def test_fits_in_memory_with_headroom():
    from ai.filum.hpa.fragment_training import (
        FragmentSpec, fits_in_memory, estimate_memory_for_fragment,
    )

    f = FragmentSpec(layer_idx=0, matrix_id="x",
                     row_start=0, row_end=10,
                     full_rows=100, cols=64)
    cost = estimate_memory_for_fragment(f)
    # Plenty of headroom -> fits.
    assert fits_in_memory(f, available_bytes=cost * 10) is True
    # Tight budget -> doesn't fit.
    assert fits_in_memory(f, available_bytes=int(cost * 0.5)) is False


def test_micro_grain_accumulator_emits_when_covered():
    from ai.filum.hpa.fragment_training import (
        FragmentSpec, MicroGrainAccumulator,
    )

    acc = MicroGrainAccumulator(coverage_threshold=0.80)
    full_rows = 10
    cols = 4
    payload = None
    # Submit fragments covering rows 0..7 (80% coverage).
    for start in range(0, 8, 2):
        f = FragmentSpec(layer_idx=0, matrix_id="W",
                         row_start=start, row_end=start + 2,
                         full_rows=full_rows, cols=cols)
        grad = np.ones((2, cols), dtype="float32") * 0.1
        result = acc.submit_fragment_grad(f, grad)
        if result is not None:
            payload = result
    assert payload is not None
    assert payload["layer_idx"] == 0
    assert payload["matrix_id"] == "W"
    assert payload["coverage"] >= 0.80


# ---------- §E2 delta-sync ------------------------------------------------

def test_delta_sync_roundtrip():
    from ai.filum.hpa.delta_sync import (
        produce_delta, apply_delta,
        serialize_patch, deserialize_patch,
    )
    rng = np.random.default_rng(7)
    old = {
        "W1": rng.standard_normal((64, 32)).astype("float32"),
        "W2": rng.standard_normal((128, 16)).astype("float32"),
        "b": rng.standard_normal((32,)).astype("float32"),
    }
    new = {
        "W1": old["W1"] + rng.standard_normal((64, 32)).astype("float32") * 0.01,
        "W2": old["W2"] + rng.standard_normal((128, 16)).astype("float32") * 0.01,
        "b": old["b"] + rng.standard_normal((32,)).astype("float32") * 0.001,
    }
    patch = produce_delta(old, new, rank=8)
    blob = serialize_patch(patch)
    patch2 = deserialize_patch(blob)
    reconstructed = apply_delta(old, patch2)

    # Reconstruction is approximate (rank-r) — verify error is small.
    for k in new:
        diff = np.linalg.norm(reconstructed[k] - new[k])
        denom = max(1e-9, float(np.linalg.norm(new[k])))
        assert diff / denom < 0.5, (
            f"reconstruction error too large for {k}: "
            f"{diff:.4f} / {denom:.4f}"
        )


def test_delta_sync_zero_for_unchanged_tensor():
    from ai.filum.hpa.delta_sync import produce_delta

    a = {"x": np.zeros((4, 4), dtype="float32")}
    b = {"x": np.zeros((4, 4), dtype="float32")}
    patch = produce_delta(a, b)
    assert len(patch.tensors) == 1
    assert patch.tensors[0].kind == "zero"


def test_delta_sync_rejects_wrong_base():
    from ai.filum.hpa.delta_sync import produce_delta, apply_delta

    a = {"x": np.zeros((4, 4), dtype="float32")}
    b = {"x": np.ones((4, 4), dtype="float32")}
    patch = produce_delta(a, b)
    wrong_base = {"x": np.full((4, 4), 0.5, dtype="float32")}
    with pytest.raises(ValueError):
        apply_delta(wrong_base, patch, verify_base=True)


def test_delta_sync_size_smaller_than_full_tensor():
    from ai.filum.hpa.delta_sync import (
        produce_delta, serialize_patch, estimate_patch_size_bytes,
    )

    rng = np.random.default_rng(7)
    old = {"big": rng.standard_normal((256, 256)).astype("float32")}
    # Tiny perturbation -> rank-8 should compress hard.
    new = {"big": old["big"] + 0.001 * rng.standard_normal((256, 256)).astype("float32")}
    patch = produce_delta(old, new, rank=8)
    blob = serialize_patch(patch)
    full_bytes = old["big"].nbytes
    assert len(blob) < full_bytes / 4, (
        f"patch {len(blob)} bytes vs full {full_bytes} bytes"
    )


# ---------- §E4 data-labor ledger ------------------------------------------

def test_data_labor_records_contribution(tmp_path: Path):
    from ai.filum.hpa.data_labor import (
        DataLaborLedger, DataLaborConfig, Contribution,
        KIND_VOICE, KIND_PREFERENCE,
    )

    led = DataLaborLedger(DataLaborConfig(
        state_path=str(tmp_path / "dl.json"),
    ))
    led.record(Contribution(
        pubkey="alice", kind=KIND_VOICE,
        content_sha256="a" * 64,
        language="sw", domain="general",
    ))
    led.record(Contribution(
        pubkey="alice", kind=KIND_PREFERENCE,
        content_sha256="b" * 64,
        language="en", domain="medical",
    ))
    s = led.stats()
    assert s["n_contributors"] == 1
    assert s["n_contributions"] == 2


def test_data_labor_low_resource_language_weighted_higher(tmp_path: Path):
    from ai.filum.hpa.data_labor import (
        DataLaborLedger, DataLaborConfig, Contribution, KIND_VOICE,
    )

    led = DataLaborLedger(DataLaborConfig(
        state_path=str(tmp_path / "dl.json"),
    ))
    en = led.record(Contribution(
        pubkey="alice", kind=KIND_VOICE,
        content_sha256="a" * 64, language="en",
    ))
    sw = led.record(Contribution(
        pubkey="bob", kind=KIND_VOICE,
        content_sha256="b" * 64, language="sw",   # Swahili (low-resource)
    ))
    assert sw.weight > en.weight


def test_data_labor_split_royalties_sums_to_pool(tmp_path: Path):
    from ai.filum.hpa.data_labor import (
        DataLaborLedger, DataLaborConfig, Contribution, KIND_TEXT,
    )

    led = DataLaborLedger(DataLaborConfig(
        state_path=str(tmp_path / "dl.json"),
    ))
    for pk in ("a", "b", "c"):
        led.record(Contribution(
            pubkey=pk, kind=KIND_TEXT,
            content_sha256=pk * 32, language="en",
        ))
    splits = led.split_royalties(total_pool=100.0)
    assert len(splits) == 3
    assert abs(sum(splits.values()) - 100.0) < 1e-3
    # Equal weights -> equal splits.
    for v in splits.values():
        assert abs(v - 100.0 / 3) < 1e-3


def test_data_labor_empty_ledger_returns_empty_split(tmp_path: Path):
    from ai.filum.hpa.data_labor import DataLaborLedger, DataLaborConfig

    led = DataLaborLedger(DataLaborConfig(
        state_path=str(tmp_path / "dl.json"),
    ))
    assert led.split_royalties(100.0) == {}


def test_data_labor_credit_payout_updates_account(tmp_path: Path):
    from ai.filum.hpa.data_labor import (
        DataLaborLedger, DataLaborConfig, Contribution, KIND_TEXT,
    )

    led = DataLaborLedger(DataLaborConfig(
        state_path=str(tmp_path / "dl.json"),
    ))
    led.record(Contribution(
        pubkey="alice", kind=KIND_TEXT,
        content_sha256="a" * 64,
    ))
    led.credit_payout({"alice": 12.5})
    s = led.stats()
    assert s["total_paid_out"] == 12.5
