"""On-Chain Revenue Distribution Protocol (PNIS §A16).

The monetisation surface for Pluginfer + Filum. Every PLG settled in
the network is split deterministically across a small number of
beneficiaries by an on-chain rule that anyone can verify. Streams:

  * **Provider payout** -- 80-95% of every job's settled price flows
    to the provider that executed it. This is the core "earn money
    from your idle GPU" claim.

  * **Protocol fee (treasury)** -- 5% of every settled job goes to the
    Pluginfer treasury for development, audits, infrastructure. This
    is the project's primary revenue.

  * **Capability royalty** -- when a job uses a registered LoRA /
    capability adapter (§8, §9), 0.5-3% of the price flows to the
    capability's author. This makes "publish a specialty model, get
    paid forever" a first-class action.

  * **Referrer rebate** -- when a paying user is brought in by an
    explicit referrer wallet, a configurable rebate (default 1%) is
    paid to that referrer for the first N days. This is the
    user-acquisition flywheel.

  * **SLA escrow** -- premium-tier jobs that demand the §A13 quorum
    (zero downtime) and §A12 fan-out (apparent zero latency) pay a
    higher protocol fee (default 8%) into an SLA-escrow address that
    is rebated to the user on missed-SLA events.

The split is enforced at settlement time (§core.compute_ledger):
the broker constructs N transfer transactions, all atomic in the
same block. Any deviation from the rule is rejected by validators.

Beyond the fee/royalty stream, four product lines monetise on top
of the open network and use this same primitive:

  1. **Filum API as a hosted service** -- Pluginfer-operated nodes
     that wrap the local Filum model behind a paid endpoint.
     Revenue stream: provider_payout to the Pluginfer wallet.
  2. **Enterprise AI-Receipt Dashboard** -- subscription analytics
     over §A1 receipts. Revenue stream: off-chain SaaS billed
     monthly; the receipts themselves stay open.
  3. **Custom LoRA training-as-a-service** -- one-call training
     run for a customer's dataset. Revenue stream: §4 fine-tune
     SDK billed per-step + capability_royalty if the resulting LoRA
     is published back to the marketplace.
  4. **SLA-backed infra contracts** -- enterprise NDA contracts
     with custom protocol_fee_pct and dedicated quorum. Revenue
     stream: per-quarter retainer + protocol fee.

This module formalises the on-chain split. The off-chain product
lines call these helpers to construct the right RevenueSplit at
settlement time so the ledger automatically routes everyone's share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal, getcontext
from typing import Dict, List, Optional, Tuple

# Decimal precision: 28 sig figs is enough for any plausible PLG amount
# while preventing repeating-fraction creep at split time.
getcontext().prec = 28


PLG = Decimal               # type alias for clarity


# ---------------------------------------------------------------------------
# Address conventions
# ---------------------------------------------------------------------------


TREASURY_ADDRESS = "PLG_TREASURY_v1"
SLA_ESCROW_ADDRESS = "PLG_SLA_ESCROW_v1"


# ---------------------------------------------------------------------------
# Beneficiary record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Beneficiary:
    address: str
    role: str                                # "provider"|"treasury"|...
    amount: PLG


# ---------------------------------------------------------------------------
# Split rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RevenueSplit:
    """The deterministic split applied to a settled job."""
    job_id: str
    gross_plg: PLG
    transfers: Tuple[Beneficiary, ...]

    def to_list(self) -> List[Beneficiary]:
        return list(self.transfers)

    def total_distributed(self) -> PLG:
        return sum((b.amount for b in self.transfers), Decimal("0"))

    def by_role(self) -> Dict[str, PLG]:
        out: Dict[str, PLG] = {}
        for b in self.transfers:
            out[b.role] = out.get(b.role, Decimal("0")) + b.amount
        return out

    def is_conserved(self) -> bool:
        return self.total_distributed() == self.gross_plg


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _round_down_to_chain(amount: PLG, decimals: int = 8) -> PLG:
    """PLG amounts are tracked at 8 decimals on-chain. Round down so
    the sum of split parts can never exceed the gross."""
    q = Decimal(10) ** -decimals
    return amount.quantize(q, rounding=ROUND_DOWN)


@dataclass
class RevenueRule:
    """Configurable on-chain split rule. Defaults are the Pluginfer
    v1 economic policy; governance can tune them via parameters."""
    protocol_fee_pct: Decimal = Decimal("5")        # to TREASURY
    capability_royalty_pct: Decimal = Decimal("0")  # to capability author
    referrer_rebate_pct: Decimal = Decimal("0")     # to referrer
    sla_escrow_pct: Decimal = Decimal("0")          # premium SLA tier


def _validate_rule(rule: RevenueRule) -> None:
    total_fees = (
        rule.protocol_fee_pct
        + rule.capability_royalty_pct
        + rule.referrer_rebate_pct
        + rule.sla_escrow_pct
    )
    if total_fees < 0 or total_fees >= Decimal("100"):
        raise ValueError(
            f"sum of fee percentages ({total_fees}) must be in [0, 100)"
        )


def split_revenue(
    *,
    job_id: str,
    gross_plg: PLG,
    provider_address: str,
    rule: Optional[RevenueRule] = None,
    capability_author_address: Optional[str] = None,
    referrer_address: Optional[str] = None,
    treasury_address: str = TREASURY_ADDRESS,
    sla_escrow_address: str = SLA_ESCROW_ADDRESS,
) -> RevenueSplit:
    """Construct the deterministic split. Rounds-down each fee leg to
    8-decimal precision; the leftover dust (always < 1e-8 PLG)
    accrues to the provider so the split is exactly conservative."""
    rule = rule or RevenueRule()
    _validate_rule(rule)
    if gross_plg <= 0:
        raise ValueError(f"gross_plg must be > 0, got {gross_plg}")

    gross = Decimal(gross_plg)
    transfers: List[Beneficiary] = []

    treasury_amt = _round_down_to_chain(gross * rule.protocol_fee_pct / 100)
    if treasury_amt > 0:
        transfers.append(Beneficiary(treasury_address, "treasury",
                                     treasury_amt))

    if capability_author_address and rule.capability_royalty_pct > 0:
        royalty = _round_down_to_chain(
            gross * rule.capability_royalty_pct / 100,
        )
        if royalty > 0:
            transfers.append(Beneficiary(capability_author_address,
                                         "capability_royalty", royalty))

    if referrer_address and rule.referrer_rebate_pct > 0:
        rebate = _round_down_to_chain(
            gross * rule.referrer_rebate_pct / 100,
        )
        if rebate > 0:
            transfers.append(Beneficiary(referrer_address,
                                         "referrer_rebate", rebate))

    if rule.sla_escrow_pct > 0:
        escrow = _round_down_to_chain(
            gross * rule.sla_escrow_pct / 100,
        )
        if escrow > 0:
            transfers.append(Beneficiary(sla_escrow_address,
                                         "sla_escrow", escrow))

    distributed = sum((b.amount for b in transfers), Decimal("0"))
    provider_amt = gross - distributed
    transfers.append(Beneficiary(provider_address, "provider", provider_amt))

    return RevenueSplit(
        job_id=str(job_id),
        gross_plg=gross,
        transfers=tuple(transfers),
    )


# ---------------------------------------------------------------------------
# Stream-level revenue projections (for finance / pitch decks)
# ---------------------------------------------------------------------------


@dataclass
class RevenueProjection:
    """Annualised revenue projection across the four streams."""
    daily_jobs: int
    avg_price_plg: Decimal
    plg_usd: Decimal
    rule: RevenueRule
    capability_share_of_jobs: Decimal = Decimal("0.20")
    referrer_share_of_jobs: Decimal = Decimal("0.10")

    @property
    def yearly_jobs(self) -> int:
        return self.daily_jobs * 365

    def projected(self) -> Dict[str, Decimal]:
        gross = Decimal(self.yearly_jobs) * self.avg_price_plg
        treasury = gross * self.rule.protocol_fee_pct / 100
        royalty = (
            gross * self.capability_share_of_jobs
            * self.rule.capability_royalty_pct / 100
        )
        referrer = (
            gross * self.referrer_share_of_jobs
            * self.rule.referrer_rebate_pct / 100
        )
        sla = gross * self.rule.sla_escrow_pct / 100
        provider = gross - treasury - royalty - referrer - sla
        return {
            "yearly_gross_plg": gross,
            "yearly_provider_plg": provider,
            "yearly_treasury_plg": treasury,
            "yearly_capability_royalty_plg": royalty,
            "yearly_referrer_rebate_plg": referrer,
            "yearly_sla_escrow_plg": sla,
            "yearly_treasury_usd": treasury * self.plg_usd,
        }


__all__ = [
    "PLG",
    "TREASURY_ADDRESS",
    "SLA_ESCROW_ADDRESS",
    "Beneficiary",
    "RevenueSplit",
    "RevenueRule",
    "split_revenue",
    "RevenueProjection",
]
