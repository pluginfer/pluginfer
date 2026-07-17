"""Tests for A3: estimate-earnings."""

import pytest

from core.earnings_estimator import (
    HARDWARE_TABLE,
    estimate_earnings,
    format_estimate,
    list_known_hardware,
)


def test_known_hardware_lists_at_least_the_consumer_tier():
    names = list_known_hardware()
    for required in ("gtx-1650", "rtx-3090", "rtx-4090"):
        assert required in names


def test_unknown_hardware_raises():
    with pytest.raises(ValueError, match="unknown hardware"):
        estimate_earnings(hardware_id="potato")


def test_estimate_low_lt_expected_lt_high():
    est = estimate_earnings(hardware_id="rtx-3060",
                            idle_hours_per_day=8.0)
    assert est.revenue_usd_low <= est.revenue_usd_expected <= est.revenue_usd_high
    assert est.net_usd_low <= est.net_usd_expected <= est.net_usd_high


def test_more_idle_hours_means_more_revenue():
    a = estimate_earnings(hardware_id="rtx-3090",
                          idle_hours_per_day=4.0)
    b = estimate_earnings(hardware_id="rtx-3090",
                          idle_hours_per_day=10.0)
    assert b.revenue_usd_expected > a.revenue_usd_expected


def test_higher_power_cost_reduces_net():
    cheap = estimate_earnings(hardware_id="rtx-4080",
                              power_cost_usd_per_kwh=0.05)
    expensive = estimate_earnings(hardware_id="rtx-4080",
                                  power_cost_usd_per_kwh=0.40)
    assert cheap.net_usd_expected > expensive.net_usd_expected


def test_protocol_fee_reduces_revenue():
    a = estimate_earnings(hardware_id="rtx-3090",
                          protocol_fee_pct=0.0)
    b = estimate_earnings(hardware_id="rtx-3090",
                          protocol_fee_pct=10.0)
    assert b.revenue_usd_expected < a.revenue_usd_expected


def test_negative_net_reported_for_losing_setup():
    """Tiny GPU + expensive electricity + deep-slack hours -> negative."""
    est = estimate_earnings(
        hardware_id="cpu-only",
        idle_hours_per_day=8.0,
        idle_window=(0, 8),
        power_cost_usd_per_kwh=0.50,
    )
    assert est.net_usd_expected < 0
    assert any("negative" in n.lower() for n in est.notes)


def test_format_includes_assumptions_section():
    est = estimate_earnings(hardware_id="rtx-3060")
    s = format_estimate(est)
    assert "Assumptions" in s
    assert "electricity" in s.lower()
    assert "RTX 3060" in s


def test_higher_tier_hardware_outearns_lower_tier():
    """A 4090 should outearn a GTX 1650 with identical other params."""
    low = estimate_earnings(hardware_id="gtx-1650")
    high = estimate_earnings(hardware_id="rtx-4090")
    assert high.revenue_usd_expected > 5.0 * low.revenue_usd_expected


def test_hardware_profile_has_minimum_fields():
    for hw in HARDWARE_TABLE.values():
        assert hw.vram_gb >= 0
        assert hw.typical_power_w > 0
        assert hw.inferences_per_hour > 0
        assert hw.base_price_per_inference_usd > 0
