"""Tests for A16: On-Chain Revenue Distribution Protocol."""

from decimal import Decimal

import pytest

from core.revenue_distribution import (
    SLA_ESCROW_ADDRESS,
    TREASURY_ADDRESS,
    RevenueProjection,
    RevenueRule,
    split_revenue,
)


def test_default_split_5pct_treasury_95pct_provider():
    s = split_revenue(
        job_id="j1",
        gross_plg=Decimal("100"),
        provider_address="prov-A",
    )
    by = s.by_role()
    assert by["treasury"] == Decimal("5")
    assert by["provider"] == Decimal("95")


def test_split_is_conservative_to_chain_precision():
    s = split_revenue(
        job_id="j2",
        gross_plg=Decimal("10.00000001"),
        provider_address="prov-B",
    )
    assert s.is_conserved()
    assert s.total_distributed() == Decimal("10.00000001")


def test_capability_royalty_routes_to_author():
    rule = RevenueRule(
        protocol_fee_pct=Decimal("5"),
        capability_royalty_pct=Decimal("3"),
    )
    s = split_revenue(
        job_id="j3",
        gross_plg=Decimal("100"),
        provider_address="prov",
        capability_author_address="lora-author",
        rule=rule,
    )
    by = s.by_role()
    assert by["capability_royalty"] == Decimal("3")
    assert by["treasury"] == Decimal("5")
    assert by["provider"] == Decimal("92")


def test_referrer_rebate_only_when_referrer_address_supplied():
    rule = RevenueRule(referrer_rebate_pct=Decimal("1"))
    no_ref = split_revenue(
        job_id="a", gross_plg=Decimal("100"),
        provider_address="prov", rule=rule,
    )
    with_ref = split_revenue(
        job_id="b", gross_plg=Decimal("100"),
        provider_address="prov", referrer_address="ref-1", rule=rule,
    )
    no_by = no_ref.by_role()
    yes_by = with_ref.by_role()
    assert "referrer_rebate" not in no_by
    assert yes_by["referrer_rebate"] == Decimal("1")


def test_sla_escrow_for_premium_tier():
    rule = RevenueRule(
        protocol_fee_pct=Decimal("5"),
        sla_escrow_pct=Decimal("3"),
    )
    s = split_revenue(
        job_id="sla1", gross_plg=Decimal("100"),
        provider_address="prov", rule=rule,
    )
    by = s.by_role()
    assert by["sla_escrow"] == Decimal("3")
    assert any(b.address == SLA_ESCROW_ADDRESS and b.role == "sla_escrow"
               for b in s.transfers)


def test_zero_gross_rejected():
    with pytest.raises(ValueError, match="gross_plg"):
        split_revenue(job_id="x", gross_plg=Decimal("0"),
                      provider_address="p")


def test_fee_sum_at_or_above_100_rejected():
    rule = RevenueRule(
        protocol_fee_pct=Decimal("60"),
        capability_royalty_pct=Decimal("50"),
    )
    with pytest.raises(ValueError, match="sum of fee"):
        split_revenue(job_id="x", gross_plg=Decimal("100"),
                      provider_address="p", rule=rule)


def test_rounding_dust_accrues_to_provider():
    """A 5%-of-1.000000007 fee rounds to 0.05000000 at 8 decimals;
    the missing 1e-8 dust must accrue to provider not vanish."""
    s = split_revenue(
        job_id="dust", gross_plg=Decimal("1.000000007"),
        provider_address="prov",
    )
    assert s.is_conserved()


def test_treasury_address_defaults_to_constant():
    s = split_revenue(
        job_id="t", gross_plg=Decimal("100"),
        provider_address="prov",
    )
    treasury_b = next(b for b in s.transfers if b.role == "treasury")
    assert treasury_b.address == TREASURY_ADDRESS


def test_yearly_projection_matches_per_job_arithmetic():
    proj = RevenueProjection(
        daily_jobs=100_000,
        avg_price_plg=Decimal("0.001"),
        plg_usd=Decimal("0.10"),                 # PLG is worth $0.10
        rule=RevenueRule(),                      # 5% treasury, no others
    )
    out = proj.projected()
    expected_gross = Decimal(100_000) * 365 * Decimal("0.001")
    assert out["yearly_gross_plg"] == expected_gross
    assert out["yearly_treasury_plg"] == expected_gross * Decimal("0.05")
    # Treasury USD revenue = 36500 PLG * 0.05 = 1825 PLG, * $0.10 = $182.50.
    assert out["yearly_treasury_usd"] == Decimal("182.5000")


def test_full_premium_split_routes_correctly():
    rule = RevenueRule(
        protocol_fee_pct=Decimal("5"),
        capability_royalty_pct=Decimal("2"),
        referrer_rebate_pct=Decimal("1"),
        sla_escrow_pct=Decimal("3"),
    )
    s = split_revenue(
        job_id="all-streams",
        gross_plg=Decimal("1000"),
        provider_address="prov-all",
        capability_author_address="lora",
        referrer_address="ref",
        rule=rule,
    )
    by = s.by_role()
    assert by["treasury"] == Decimal("50")
    assert by["capability_royalty"] == Decimal("20")
    assert by["referrer_rebate"] == Decimal("10")
    assert by["sla_escrow"] == Decimal("30")
    assert by["provider"] == Decimal("890")
    assert s.is_conserved()
