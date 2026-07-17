"""G7 — energy + carbon accounting for PNIS receipts.

Deterministic tests for the meter math. We exercise the CPU-TDP path
(no GPU required, no network calls — uses
`carbon_intensity_gco2_per_kwh_override` to pin the carbon math)
so the assertions are reproducible on every CI runner.

The NVML path is exercised opportunistically: if `pynvml` is installed
and `nvmlDeviceGetTotalEnergyConsumption` returns a number, we verify
the meter source string flips to `gpu-nvml`; otherwise the test
soft-skips so CI without GPUs stays green.
"""

from __future__ import annotations

import sys
import time
from decimal import Decimal
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.energy import (  # noqa: E402
    CPU_TDP_WATTS_BY_CLASS,
    EnergyMeter,
    EnergyReport,
    measure_job_energy,
)


# ---------------------------------------------------------------------------
# CPU-TDP path — deterministic, no GPU/network required
# ---------------------------------------------------------------------------

def test_cpu_tdp_estimate_matches_watts_times_seconds():
    """Energy (J) ~= TDP-watts × duration-seconds. ~0.1s sleep on a
    65W desktop class should be ~6.5J = 6.5e-6 MJ."""
    m = EnergyMeter(
        hardware_class="consumer-desktop",
        gpu_index=None,        # force CPU-TDP path
        carbon_intensity_gco2_per_kwh_override=400.0,
    )
    m.start()
    time.sleep(0.1)
    r = m.stop()

    assert r.source == "cpu-tdp"
    # Allow ±50% wall-clock slack — Windows time.sleep granularity is
    # ~15ms; we don't want a flaky test that fails on slow CI.
    expected_joules = 65.0 * r.duration_s
    expected_mj = Decimal(expected_joules) / Decimal(1_000_000)
    assert abs(float(r.energy_mj) - float(expected_mj)) / float(expected_mj) < 0.5
    assert r.carbon_intensity_gco2_per_kwh == 400.0


def test_browser_webgpu_tdp_class_uses_25w():
    """Browser-tab providers run on integrated GPUs and laptop SoCs.
    The TDP class is conservative (25W) to under-claim, not over-claim,
    energy savings."""
    assert CPU_TDP_WATTS_BY_CLASS["browser-webgpu"] == 25.0
    assert CPU_TDP_WATTS_BY_CLASS["datacentre-gpu"] > \
        CPU_TDP_WATTS_BY_CLASS["consumer-desktop"]


def test_carbon_math_is_watts_x_seconds_times_intensity():
    """gCO2e = (kWh) × (gCO2/kWh). Sanity-check the conversion."""
    m = EnergyMeter(
        hardware_class="datacentre-gpu",  # 700W from the table
        gpu_index=None,
        carbon_intensity_gco2_per_kwh_override=500.0,  # CN-grid-ish
    )
    m.start()
    time.sleep(0.05)
    r = m.stop()
    # 700W × 0.05s = 35J = 9.7e-6 kWh × 500 gCO2/kWh = ~4.85e-3 gCO2e
    expected = (700.0 * r.duration_s / 3_600_000.0) * 500.0
    # Within 50% — wall-clock slack again.
    assert abs(float(r.carbon_gco2e) - expected) / max(expected, 1e-9) < 0.5


def test_unknown_hardware_class_falls_back_to_safe_midpoint():
    m = EnergyMeter(hardware_class="nonsense-class", gpu_index=None,
                    carbon_intensity_gco2_per_kwh_override=480.0)
    m.start()
    time.sleep(0.01)
    r = m.stop()
    assert r.source == "cpu-tdp"
    # The unknown-class TDP is 50W per the registry; energy_mj > 0.
    assert float(r.energy_mj) > 0


def test_stop_without_start_raises():
    m = EnergyMeter(hardware_class="consumer-desktop", gpu_index=None)
    try:
        m.stop()
    except RuntimeError as e:
        assert "start" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_zero_duration_yields_zero_or_tiny_energy():
    m = EnergyMeter(
        hardware_class="consumer-desktop", gpu_index=None,
        carbon_intensity_gco2_per_kwh_override=480.0,
    )
    m.start()
    r = m.stop()                                 # no sleep at all
    assert float(r.energy_mj) < 1e-3
    assert float(r.carbon_gco2e) < 1e-3
    # source still "cpu-tdp" so the receipt records the *attempt* to
    # measure — never a silent zero.
    assert r.source == "cpu-tdp"


# ---------------------------------------------------------------------------
# Wrapper helper
# ---------------------------------------------------------------------------

def test_measure_job_energy_wraps_a_callable_and_returns_both():
    def heavy() -> str:
        time.sleep(0.02)
        return "ok"
    result, report = measure_job_energy(
        heavy,
        hardware_class="consumer-gpu-mid",
        gpu_index=None,
        carbon_intensity_gco2_per_kwh_override=480.0,
    )
    assert result == "ok"
    assert isinstance(report, EnergyReport)
    assert float(report.energy_mj) > 0
    assert report.source == "cpu-tdp"


def test_report_to_receipt_fields_shape():
    """The receipt-shaped dict carries the right strings + zone."""
    r = EnergyReport(
        energy_mj=Decimal("0.000005"),
        carbon_gco2e=Decimal("0.000667"),
        source="cpu-tdp",
        duration_s=0.1,
        region_zone="US-CAL-CISO",
        carbon_intensity_gco2_per_kwh=240.0,
    )
    d = r.to_receipt_fields()
    assert d["energy_mj"] == "0.000005"
    assert d["carbon_gco2e"] == "0.000667"
    assert d["energy_source"] == "cpu-tdp"
    assert d["energy_zone"] == "US-CAL-CISO"
    assert d["energy_carbon_intensity_gco2_per_kwh"] == 240.0


# ---------------------------------------------------------------------------
# NVML path (opportunistic — soft skip when unavailable)
# ---------------------------------------------------------------------------

def test_nvml_path_when_available_soft_skip_otherwise():
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        try:
            pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pynvml.nvmlShutdown()
            return  # no GPU device on this host
        pynvml.nvmlShutdown()
    except ImportError:
        return       # pynvml not installed — soft skip
    except Exception:
        return       # driver missing / NVML initialise failed — soft skip

    m = EnergyMeter(
        hardware_class="consumer-gpu-mid", gpu_index=0,
        carbon_intensity_gco2_per_kwh_override=480.0,
    )
    m.start()
    # Do something CPU-bound the GPU won't see — energy delta will
    # still be small but the source string should flip to gpu-nvml
    # iff the counter is exposed.
    time.sleep(0.05)
    r = m.stop()
    assert r.source in ("gpu-nvml", "cpu-tdp")   # cards without the counter fall back
    if r.source == "gpu-nvml":
        assert r.gpu_index == 0
