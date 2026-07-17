"""Revenue-flow invariant tests.

Pluginfer's economic model: BUYER pays, PLATFORM takes 5% commission,
PROVIDER receives the rest. PLUGINFER NEVER pays providers from its
treasury. Tests assert this with hard-coded splits.

Bonus pools (post-funding) are additive and optional; they do not
break the buyer-funded invariant.
"""

from __future__ import annotations

from decimal import Decimal

import pytest


def test_split_buyer_pays_invariant():
    from ai.filum.hpa.revenue_flow import RevenueFlow, RevenueFlowConfig

    rf = RevenueFlow()
    s = rf.split_payment(Decimal("100"))
    # 5% commission, 95% provider, no royalty (no adapter used).
    assert s.platform_commission == Decimal("5.00000000")
    assert s.provider_take == Decimal("95.00000000")
    assert s.buyer_payment == Decimal("100")
    assert s.bonus_from_pool == Decimal("0")
    assert rf.invariants_hold(s)


def test_split_with_capability_royalty():
    from ai.filum.hpa.revenue_flow import RevenueFlow, RevenueFlowConfig

    rf = RevenueFlow()
    s = rf.split_payment(Decimal("100"), capability_used=True)
    # 5% commission, 5% royalty, 90% provider.
    assert s.platform_commission == Decimal("5.00000000")
    assert s.capability_royalty == Decimal("5.00000000")
    assert s.provider_take == Decimal("90.00000000")
    assert rf.invariants_hold(s)


def test_split_invariant_holds_with_decimals():
    from ai.filum.hpa.revenue_flow import RevenueFlow

    rf = RevenueFlow()
    for amount in ("0.01", "1", "1.5", "33.33", "100", "999.999"):
        s = rf.split_payment(Decimal(amount))
        # Provider + platform == buyer payment (no other beneficiaries).
        total = s.platform_commission + s.provider_take
        assert total == Decimal(amount), (
            f"buyer-pays invariant broken at {amount}: "
            f"sum {total} != buyer {amount}"
        )
        assert rf.invariants_hold(s)


def test_split_rejects_zero_or_negative():
    from ai.filum.hpa.revenue_flow import RevenueFlow

    rf = RevenueFlow()
    with pytest.raises(ValueError):
        rf.split_payment(Decimal("0"))
    with pytest.raises(ValueError):
        rf.split_payment(Decimal("-1"))


def test_split_rejects_overconfigured_rates():
    """If commission + royalty + rebate + sla >= 100%, the split is broken."""
    from ai.filum.hpa.revenue_flow import RevenueFlow, RevenueFlowConfig

    rf = RevenueFlow(RevenueFlowConfig(
        platform_commission_rate=Decimal("0.50"),
        capability_royalty_rate=Decimal("0.40"),
        referrer_rebate_rate=Decimal("0.20"),
        sla_escrow_rate=Decimal("0.0"),
    ))
    with pytest.raises(ValueError):
        rf.split_payment(Decimal("100"),
                          capability_used=True, referrer_present=True)


def test_bonus_pool_off_by_default():
    """Default config: no bonus, no Pluginfer outflow."""
    from ai.filum.hpa.revenue_flow import RevenueFlow

    rf = RevenueFlow()
    s = rf.split_payment(Decimal("100"))
    assert s.bonus_from_pool == Decimal("0")


def test_bonus_pool_when_enabled_funds_provider_extra():
    """Post-funding scenario: bonus pool is set up, cold-start providers
    earn extra. The base buyer-pays split is unchanged."""
    from ai.filum.hpa.revenue_flow import (
        RevenueFlow, RevenueFlowConfig, BonusPool,
    )

    pool = BonusPool(
        balance="50",
        cold_start_bonus_pct=0.10,    # 10% bonus
        cold_start_pubkeys={"alice"},
    )
    cfg = RevenueFlowConfig(bonus_pool_enabled=True)
    rf = RevenueFlow(config=cfg, bonus_pool=pool)

    # Alice (cold-start): gets bonus.
    s_a = rf.split_payment(Decimal("100"), provider_pubkey="alice")
    assert s_a.bonus_from_pool == Decimal("9.50000000")  # 10% of 95
    assert s_a.total_to_provider() == Decimal("104.50000000")
    # Buyer-pays invariant still holds (commission + provider == 100).
    assert s_a.platform_commission + s_a.provider_take == Decimal("100")

    # Bob (no cold-start status): no bonus.
    s_b = rf.split_payment(Decimal("100"), provider_pubkey="bob")
    assert s_b.bonus_from_pool == Decimal("0")


def test_bonus_pool_caps_at_balance():
    """When pool is exhausted, bonuses stop. Mesh keeps working."""
    from ai.filum.hpa.revenue_flow import (
        RevenueFlow, RevenueFlowConfig, BonusPool,
    )

    pool = BonusPool(
        balance="3",                 # tiny pool
        cold_start_bonus_pct=0.50,
        cold_start_pubkeys={"alice"},
    )
    cfg = RevenueFlowConfig(bonus_pool_enabled=True)
    rf = RevenueFlow(config=cfg, bonus_pool=pool)

    s1 = rf.split_payment(Decimal("100"), provider_pubkey="alice")
    # Wanted 50% of 95 = 47.5, but pool only has 3.
    assert s1.bonus_from_pool == Decimal("3")
    assert pool.balance == Decimal("0")

    # Next call: pool empty, no bonus.
    s2 = rf.split_payment(Decimal("100"), provider_pubkey="alice")
    assert s2.bonus_from_pool == Decimal("0")
    # Base flow unchanged.
    assert s2.platform_commission == Decimal("5.00000000")


def test_audit_tallies_track_flow():
    from ai.filum.hpa.revenue_flow import RevenueFlow

    rf = RevenueFlow()
    rf.split_payment(Decimal("100"))
    rf.split_payment(Decimal("200"), capability_used=True)
    s = rf.stats()
    assert s["total_buyer_payments"] == 300.0
    assert s["total_platform_commission"] == 15.0
    # provider take = 95 (no royalty) + 180 (200 - 10% royalty - 5% commission) = 275
    assert s["total_provider_payouts"] == 275.0
    assert s["total_capability_royalty"] == 10.0


def test_pluginfer_zero_outflow_invariant():
    """The most important invariant: Pluginfer's commission is INCOMING
    revenue, not an outflow. Stats should never show Pluginfer paying
    providers."""
    from ai.filum.hpa.revenue_flow import RevenueFlow

    rf = RevenueFlow()
    for _ in range(100):
        rf.split_payment(Decimal("10"))
    s = rf.stats()
    # Total provider payouts == buyer payments - all fees.
    expected_provider = (s["total_buyer_payments"]
                          - s["total_platform_commission"]
                          - s["total_capability_royalty"]
                          - s["total_referrer_rebate"])
    assert abs(s["total_provider_payouts"] - expected_provider) < 1e-6
    # Bonus is separately funded; default zero.
    assert s["total_bonus_paid"] == 0.0


def test_bonus_pool_fund_increases_balance():
    from ai.filum.hpa.revenue_flow import BonusPool

    pool = BonusPool(balance="100")
    pool.fund(Decimal("50"))
    assert pool.balance == Decimal("150")
    with pytest.raises(ValueError):
        pool.fund(Decimal("-10"))
