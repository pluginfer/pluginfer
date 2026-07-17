"""Gamer earnings calculator.

CRITICAL READING NOTE — the money flow:

  Buyer (the customer wanting a training job)
     |
     |  pays buyer_payment for the job
     v
  Pluginfer (matcher)
     |
     |  takes 5% commission (treasury — Pluginfer's only revenue)
     |  takes optional cuts: capability-author 5% if adapter used,
     |                        referrer rebate, SLA escrow
     v
  Provider (the gamer with the idle GPU)
     receives the rest as their take

PLUGINFER NEVER PAYS PROVIDERS FROM ITS TREASURY. Every dollar a
gamer earns originates from a buyer who paid for a real training
job. The platform is a pure marketplace operator (Uber/Airbnb
model), not a subsidizer. This keeps Pluginfer's margins ~zero-
marginal-cost and lets the mesh scale without burning capital.

Bonus pools (cold-start, referrer, green-energy) are *post-
funding, additive, and optional*. Off by default. They sit on top
of the buyer-funded base; the base economy works without any of
them.

Pricing assumptions (visible, edit them):
* Sunk-cost-floor: $0.10/TFLOP-hr  (provider's stated bid)
* AWS-equivalent ceiling: $1.00/TFLOP-hr  (buyer max-price)
* Daily TOU averages to ~1.0x across 24hr but earnings concentrate
  in the 8 peak hours that surge to ~1.5x
* Cold-start attestation multiplier: 0.5x for first month, 1.0x after.
  This means a new provider earns less initially — the *buyer*
  saves the difference (or the §A16 split allocates it to other
  beneficiaries). NO Pluginfer subsidy.

Hardware specs are mid-2026 retail cards. Power numbers are
GPU-only at typical training utilisation (~85% TDP). Idle power
when not training is ignored — we account only for the *marginal*
draw while running Pluginfer jobs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GpuSpec:
    name: str
    tflops_fp16: float       # raw fp16 TFLOPS (sparse FlOPs ignored)
    tdp_watts: int           # nominal full-load
    typical_load_pct: float = 0.85   # training rarely hits 100%


CARDS = [
    GpuSpec("GTX 1650 (mobile)",     5.0,  50),
    GpuSpec("RTX 3060",             12.0, 170),
    GpuSpec("RTX 3070",             20.0, 220),
    GpuSpec("RTX 4060",             19.0, 115),
    GpuSpec("RTX 3080",             30.0, 320),
    GpuSpec("RTX 4070 Super",       35.0, 220),
    GpuSpec("RTX 4080",             49.0, 320),
    GpuSpec("RTX 4090",             83.0, 450),
]


@dataclass
class EarningsAssumptions:
    base_price_per_tflop_hr: float = 0.10   # sunk-cost-only floor
    avg_daily_multiplier:    float = 1.20   # weighted across 24hr
    peak_surge_multiplier:   float = 1.50   # 8 peak hours
    pluginfer_treasury_cut:  float = 0.05   # §A16
    capability_royalty_cut:  float = 0.05   # §A16 author share
    stability_score:         float = 0.90   # uptime ~90%
    attestation_cold_start:  float = 0.5    # half-rate first month
    attestation_after_30d:   float = 1.0
    idle_hours_per_day:      float = 12.0   # gamer typical
    days_per_month:          int   = 30
    electricity_per_kwh_usd: float = 0.15   # US average


def buyer_payment_per_month(card: GpuSpec, ass: EarningsAssumptions) -> float:
    """What the BUYER pays for a month's worth of this card's compute.

    This is the upstream cash that funds the gamer's earnings. None
    of this comes from Pluginfer's treasury — it is paid by the
    end-customer (a researcher, startup, university, hospital, etc.)
    who submitted the training job that the gamer's GPU executed.
    """
    tflop_hr_month = (
        card.tflops_fp16
        * card.typical_load_pct
        * ass.idle_hours_per_day
        * ass.days_per_month
    )
    avg_price = ass.base_price_per_tflop_hr * ass.avg_daily_multiplier
    return tflop_hr_month * avg_price


def pluginfer_commission_per_month(card: GpuSpec, ass: EarningsAssumptions) -> float:
    """Pluginfer's 5% cut. The platform's only revenue per provider."""
    bp = buyer_payment_per_month(card, ass)
    return bp * ass.pluginfer_treasury_cut


def gross_earnings_per_month(card: GpuSpec, ass: EarningsAssumptions,
                               *, attestation: float) -> float:
    """Provider's gross take BEFORE electricity. This is the buyer's
    payment minus all fees, multiplied by the §C8 bonded earnings
    factor (stability * cold-start attestation).

    Cold-start providers earn LESS — the difference is NOT paid by
    Pluginfer. It's just a smaller buyer-funded share for new
    providers, equilibrating their unverified status."""
    # TFLOP-hr produced per month (only during idle hours).
    tflop_hr_month = (
        card.tflops_fp16
        * card.typical_load_pct
        * ass.idle_hours_per_day
        * ass.days_per_month
    )
    # Provider gets the matched price = base * multiplier.
    avg_price = ass.base_price_per_tflop_hr * ass.avg_daily_multiplier
    # §C8 bonded earnings curve: stability * (0.5 + 0.5*attestation).
    bonded_factor = ass.stability_score * (0.5 + 0.5 * attestation)
    gross = tflop_hr_month * avg_price * bonded_factor
    return gross


def electricity_cost_per_month(card: GpuSpec, ass: EarningsAssumptions) -> float:
    kwh = (card.tdp_watts * card.typical_load_pct
            * ass.idle_hours_per_day * ass.days_per_month) / 1000.0
    return kwh * ass.electricity_per_kwh_usd


def net_take(card: GpuSpec, ass: EarningsAssumptions,
              *, attestation: float) -> dict:
    """Returns the full money-flow breakdown for one card-month.

    Keys returned:
      buyer_pays_usd   — what the BUYER pays Pluginfer (input cash flow)
      platform_fee_usd — Pluginfer's 5% commission (the platform's revenue)
      gross_usd        — what the provider takes BEFORE electricity (after fees)
      electricity_usd  — what the provider's wall socket costs them
      net_usd          — what the provider keeps (their actual income)
      tflop_hr_month   — work performed
    """
    buyer = buyer_payment_per_month(card, ass)
    fees = buyer * (ass.pluginfer_treasury_cut + ass.capability_royalty_cut)
    gross = gross_earnings_per_month(card, ass, attestation=attestation)
    elec = electricity_cost_per_month(card, ass)
    net  = gross - elec
    return {
        "buyer_pays_usd":    round(buyer, 2),
        "platform_fee_usd":  round(buyer * ass.pluginfer_treasury_cut, 2),
        "gross_usd":         round(gross, 2),
        "fees_usd":          round(fees, 2),
        "electricity_usd":   round(elec, 2),
        "net_usd":           round(net, 2),
        "tflop_hr_month":    round(
            card.tflops_fp16 * card.typical_load_pct
            * ass.idle_hours_per_day * ass.days_per_month, 1,
        ),
    }


def run_table(ass: EarningsAssumptions = EarningsAssumptions()) -> None:
    print(f"{'Card':<22}  {'BuyerPays':>10}  {'PluginferFee':>12}  "
          f"{'ProviderGross':>14}  {'Elec':>7}  "
          f"{'NetM1':>7}  {'NetM2+':>8}")
    print("-" * 92)
    for card in CARDS:
        m1 = net_take(card, ass, attestation=ass.attestation_cold_start)
        m2 = net_take(card, ass, attestation=ass.attestation_after_30d)
        print(f"{card.name:<22}  ${m2['buyer_pays_usd']:>8.2f}  "
              f"${m2['platform_fee_usd']:>10.2f}  "
              f"${m2['gross_usd']:>12.2f}  "
              f"${m2['electricity_usd']:>5.2f}  "
              f"${m1['net_usd']:>5.2f}  ${m2['net_usd']:>6.2f}")


def annualised(card: GpuSpec, ass: EarningsAssumptions) -> float:
    cold_first_month = net_take(card, ass,
                                  attestation=ass.attestation_cold_start)["net_usd"]
    veteran_after = net_take(card, ass,
                                attestation=ass.attestation_after_30d)["net_usd"]
    return cold_first_month + veteran_after * 11   # one cold month + 11 veteran


SCENARIOS = {
    "conservative": EarningsAssumptions(
        base_price_per_tflop_hr=0.025,    # competitive with Salad.com today
        avg_daily_multiplier=1.0,
        idle_hours_per_day=8.0,            # gamer + work, less idle
    ),
    "realistic": EarningsAssumptions(
        base_price_per_tflop_hr=0.05,     # 50% above commodity inference market
        avg_daily_multiplier=1.10,
        idle_hours_per_day=12.0,
    ),
    "target": EarningsAssumptions(
        base_price_per_tflop_hr=0.10,     # the §C7 sunk-cost-only floor
        avg_daily_multiplier=1.20,
        idle_hours_per_day=12.0,
    ),
}


if __name__ == "__main__":
    for name, ass in SCENARIOS.items():
        print("=" * 92)
        print(f"  Scenario: {name.upper()}")
        print(f"  base ${ass.base_price_per_tflop_hr}/TFLOP-hr, "
              f"avg mult {ass.avg_daily_multiplier}x, "
              f"{ass.idle_hours_per_day}hr/day idle, "
              f"electricity ${ass.electricity_per_kwh_usd}/kWh")
        print("=" * 92)
        run_table(ass)
        print()
        print(f"  12-month total (1 cold + 11 veteran months):")
        for card in CARDS:
            ann = annualised(card, ass)
            print(f"    {card.name:<22}  ${ann:>9.2f}/yr")
        print()
