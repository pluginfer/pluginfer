"""Canonical revenue flow on Pluginfer.

ONE source of truth for who pays what to whom. Every other module
(reverse_auction, revenue_distribution, compute_currency, gamer
earnings calculator) defers to this module's invariants:

  1. The BUYER is the only source of money entering the mesh for a
     given job. Pluginfer's treasury never funds a provider's
     earnings.
  2. Pluginfer takes a commission (default 5%) on every settled
     trade. That commission is the platform's *only* revenue.
  3. The PROVIDER receives the remainder, minus any §A16
     capability-royalty share (default 5% to the LoRA author when
     a capability adapter was used) and minus any §C7 referral
     rebate.
  4. BonusPool grants are *post-funding, additive, and optional*.
     They DO NOT change the base flow above. They are credited to
     the provider on top of the buyer-funded payment, sourced from
     a separately-funded pool. Off by default.

This module exposes:

* ``RevenueFlow.split_payment(buyer_payment, ...)`` — returns a
  RevenueSplit dict with provider, platform, capability_royalty,
  referrer, sla_escrow, and (optional) bonus.
* ``RevenueFlow.invariants_hold(...)`` — asserts that the splits
  sum to the buyer's payment plus any bonus drawn from the bonus
  pool. Catches accounting bugs at the boundary.

novel surface: this is the embodiment of §A16 with the
buyer-pays invariant made explicit. Worth referencing in the §C7
design notes as a defensive "the platform never subsidises
providers" clause — strengthens the ZERO-MARGINAL-COST property
of the platform's economics that's central to the
"AWS-incompatible competitor" pitch.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from decimal import Decimal, getcontext
from typing import Optional

# 8 decimal digits matches the §A16 chain-precision spec.
getcontext().prec = 28

logger = logging.getLogger(__name__)


# ---------- the split -----------------------------------------------------

@dataclass
class RevenueSplit:
    """Per-job split. All values denominated in the same unit (USD or PLG)
    and use Decimal for chain-precision arithmetic at runtime."""
    buyer_payment:        Decimal = Decimal("0")
    platform_commission:  Decimal = Decimal("0")  # Pluginfer treasury
    provider_take:        Decimal = Decimal("0")  # the provider's earnings
    capability_royalty:   Decimal = Decimal("0")  # §A16 LoRA author share
    referrer_rebate:      Decimal = Decimal("0")  # §A16 referrer share
    sla_escrow:           Decimal = Decimal("0")  # §A16 premium-tier escrow
    bonus_from_pool:      Decimal = Decimal("0")  # post-funding only; 0 by default

    def total_to_provider(self) -> Decimal:
        """What the provider actually receives. Includes any bonus."""
        return self.provider_take + self.bonus_from_pool

    def buyer_paid(self) -> Decimal:
        return self.buyer_payment

    def to_dict(self) -> dict:
        return {k: float(v) for k, v in asdict(self).items()}


@dataclass
class RevenueFlowConfig:
    """All percentages are fractions in [0, 1]. Defaults match §A16."""
    platform_commission_rate:    Decimal = Decimal("0.05")   # 5%
    capability_royalty_rate:     Decimal = Decimal("0.05")   # 5% if adapter used
    referrer_rebate_rate:        Decimal = Decimal("0.00")   # off by default
    sla_escrow_rate:             Decimal = Decimal("0.00")   # off by default
    bonus_pool_enabled:          bool    = False             # post-funding only


# ---------- the canonical splitter ----------------------------------------

class RevenueFlow:
    """Canonical money-flow accountant. ZERO subsidy from Pluginfer."""

    def __init__(self, config: RevenueFlowConfig = RevenueFlowConfig(),
                 bonus_pool: Optional["BonusPool"] = None):
        self.cfg = config
        self.bonus_pool = bonus_pool   # None unless post-funding
        # Cumulative tallies (audit log).
        self._total_buyer_payments = Decimal("0")
        self._total_platform_commission = Decimal("0")
        self._total_provider_payouts = Decimal("0")
        self._total_capability_royalty = Decimal("0")
        self._total_referrer_rebate = Decimal("0")
        self._total_bonus_paid = Decimal("0")

    def split_payment(
        self,
        buyer_payment: Decimal | float | str,
        *,
        capability_used: bool = False,
        referrer_present: bool = False,
        is_premium_sla: bool = False,
        provider_pubkey: str = "",
    ) -> RevenueSplit:
        """Split a buyer's payment across all beneficiaries.

        Buyer payment is the *total* the buyer pays. Pluginfer takes
        commission first; remaining is split among capability author
        (if used), referrer (if present), SLA escrow (if premium),
        and finally the provider.

        Bonus from pool is added on top — it is sourced from the
        BonusPool, not from the buyer's payment, and comes from
        Pluginfer's funded incentive budget (post-funding only,
        off by default).
        """
        bp = Decimal(str(buyer_payment))
        if bp <= 0:
            raise ValueError("buyer_payment must be positive")

        commission = (bp * self.cfg.platform_commission_rate).quantize(Decimal("1.00000000"))
        royalty = (
            (bp * self.cfg.capability_royalty_rate).quantize(Decimal("1.00000000"))
            if capability_used else Decimal("0")
        )
        rebate = (
            (bp * self.cfg.referrer_rebate_rate).quantize(Decimal("1.00000000"))
            if referrer_present else Decimal("0")
        )
        sla = (
            (bp * self.cfg.sla_escrow_rate).quantize(Decimal("1.00000000"))
            if is_premium_sla else Decimal("0")
        )
        # The provider gets whatever's left. Rounding-dust always
        # accrues to the provider so the split is exactly conservative.
        provider = bp - commission - royalty - rebate - sla
        if provider < 0:
            raise ValueError(
                f"split misconfigured: rates sum to >= 100% of buyer payment "
                f"(buyer={bp}, commission={commission}, royalty={royalty}, "
                f"rebate={rebate}, sla={sla})"
            )

        bonus = Decimal("0")
        if self.cfg.bonus_pool_enabled and self.bonus_pool is not None:
            bonus = self.bonus_pool.draw_for(provider_pubkey, base=provider)

        split = RevenueSplit(
            buyer_payment=bp,
            platform_commission=commission,
            provider_take=provider,
            capability_royalty=royalty,
            referrer_rebate=rebate,
            sla_escrow=sla,
            bonus_from_pool=bonus,
        )
        # Update audit tallies.
        self._total_buyer_payments += bp
        self._total_platform_commission += commission
        self._total_provider_payouts += provider
        self._total_capability_royalty += royalty
        self._total_referrer_rebate += rebate
        self._total_bonus_paid += bonus
        return split

    def invariants_hold(self, split: RevenueSplit) -> bool:
        """Assert split obeys the buyer-pays + zero-subsidy invariants.

        Returns False on any violation; logs the reason. Caller can
        treat False as a hard accounting bug.
        """
        # 1. Sum of buyer-funded shares == buyer_payment exactly.
        buyer_funded_total = (
            split.platform_commission + split.provider_take
            + split.capability_royalty + split.referrer_rebate
            + split.sla_escrow
        )
        if abs(buyer_funded_total - split.buyer_payment) > Decimal("1e-8"):
            logger.error(
                "buyer-pays invariant broken: %s != %s",
                buyer_funded_total, split.buyer_payment,
            )
            return False
        # 2. Bonus is *additive* — does NOT come from buyer.
        # No assertion needed; the data structure already separates it.
        # 3. No share is negative (no provider paying back, etc.).
        for k in ("platform_commission", "provider_take", "capability_royalty",
                   "referrer_rebate", "sla_escrow", "bonus_from_pool"):
            v = getattr(split, k)
            if v < 0:
                logger.error("negative share %s = %s", k, v)
                return False
        return True

    def stats(self) -> dict:
        return {
            "total_buyer_payments":      float(self._total_buyer_payments),
            "total_platform_commission": float(self._total_platform_commission),
            "total_provider_payouts":    float(self._total_provider_payouts),
            "total_capability_royalty":  float(self._total_capability_royalty),
            "total_referrer_rebate":     float(self._total_referrer_rebate),
            "total_bonus_paid":          float(self._total_bonus_paid),
        }


# ---------- the bonus pool ------------------------------------------------

class BonusPool:
    """Optional, post-funding incentive pool.

    OFF by default. To enable: set ``RevenueFlowConfig.bonus_pool_enabled =
    True`` and pass a funded BonusPool to ``RevenueFlow``.

    The pool's balance is set externally (from Pluginfer's funded
    incentive budget). Drawing from the pool depletes its balance;
    when empty, ``draw_for`` returns 0 and the provider just gets
    their normal buyer-funded share. The base mesh keeps working.

    Common bonus rules (all opt-in):

    * Cold-start: new providers (no attestations yet) earn an extra
      X% of base for their first month.
    * Referrer: matching the §A16 rebate rate.
    * Green-energy: providers using verified-renewable get a
      premium percentage.

    Implementation kept minimal: a single rate function. Production
    can plug in arbitrary policy.
    """

    def __init__(self, *, balance: Decimal | float | str = "0",
                 cold_start_bonus_pct: float = 0.0,
                 green_bonus_pct: float = 0.0,
                 cold_start_pubkeys: Optional[set] = None,
                 green_pubkeys: Optional[set] = None):
        self._balance = Decimal(str(balance))
        self._cold_pct = Decimal(str(cold_start_bonus_pct))
        self._green_pct = Decimal(str(green_bonus_pct))
        self._cold = set(cold_start_pubkeys or set())
        self._green = set(green_pubkeys or set())
        self._total_drawn = Decimal("0")

    @property
    def balance(self) -> Decimal:
        return self._balance

    def fund(self, amount: Decimal | float | str) -> None:
        """Add to the bonus pool. Only callable by Pluginfer treasury."""
        a = Decimal(str(amount))
        if a < 0:
            raise ValueError("fund amount must be non-negative")
        self._balance += a

    def draw_for(self, provider_pubkey: str, *,
                  base: Decimal) -> Decimal:
        """Compute and deduct the bonus for this provider.

        Returns the bonus amount; caller adds it to the provider's
        take. Returns 0 if the pool is empty or no rule matches.
        """
        if self._balance <= 0:
            return Decimal("0")
        rate = Decimal("0")
        if provider_pubkey in self._cold:
            rate += self._cold_pct
        if provider_pubkey in self._green:
            rate += self._green_pct
        if rate <= 0:
            return Decimal("0")
        bonus = (base * rate).quantize(Decimal("1.00000000"))
        if bonus > self._balance:
            bonus = self._balance
        self._balance -= bonus
        self._total_drawn += bonus
        return bonus

    def stats(self) -> dict:
        return {
            "balance":     float(self._balance),
            "total_drawn": float(self._total_drawn),
        }
