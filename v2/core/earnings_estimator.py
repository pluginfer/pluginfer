"""Earnings estimator (PNIS §A3 -- UX surface for §3 slack pricing).

Given a candidate node's hardware + idle hours + electricity cost,
return an honest predicted PLG/day earnings range. This is the
conversion funnel for new providers: the prospective operator sees a
NUMBER before installing -- "you'd earn $0.40/day off-peak on this
GTX 1650, net of electricity".

Design ethics
-------------
This estimate must be HONEST. Past projects in this space (and the
prior Pluginfer auto_onboarding fabricated-earnings code that we
explicitly removed in W19) earned actionable consumer-protection
liability by inflating earnings claims. This module:

  * Reports a RANGE [low, expected, high], not a single number.
  * Reports the assumptions inline (utilization, cost basis, slack
    curve, jobs/hour at the inferred GPU class).
  * Returns NEGATIVE earnings when electricity > expected revenue.
  * Refuses to extrapolate beyond what the hardware can plausibly
    support (e.g. claims about a 4090 are not made based on data
    from a 1650).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional

from .slack_auction import TimeOfDaySlackCurve, default_consumer_curve


# ---------------------------------------------------------------------------
# Hardware tier table -- jobs/hour rough estimates at WORLDWIDE-AVG
# task mix. These numbers are conservative estimates; production tunes
# them from chain telemetry per provider.
# ---------------------------------------------------------------------------


@dataclass
class HardwareProfile:
    name: str
    vram_gb: int
    typical_power_w: int
    inferences_per_hour: int                 # ballpark
    base_price_per_inference_usd: float      # at slack=1.0


HARDWARE_TABLE: Dict[str, HardwareProfile] = {
    # Consumer
    "gtx-1650":  HardwareProfile("GTX 1650",  4,  75,  120, 0.0008),
    "gtx-1660":  HardwareProfile("GTX 1660",  6,  120, 220, 0.0010),
    "rtx-3060":  HardwareProfile("RTX 3060",  12, 170, 600, 0.0014),
    "rtx-3090":  HardwareProfile("RTX 3090",  24, 350, 1500, 0.0024),
    "rtx-4080":  HardwareProfile("RTX 4080",  16, 320, 2200, 0.0030),
    "rtx-4090":  HardwareProfile("RTX 4090",  24, 450, 3500, 0.0042),
    # Enterprise
    "a100-40":   HardwareProfile("A100 40GB", 40, 400, 6500, 0.0090),
    "h100":      HardwareProfile("H100 80GB", 80, 700, 18000, 0.0140),
    # CPU-only fallback
    "cpu-only":  HardwareProfile("CPU only",  0,  65,  4,    0.0001),
}


def list_known_hardware() -> list[str]:
    return list(HARDWARE_TABLE.keys())


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


@dataclass
class EarningsEstimate:
    """Honest earnings forecast for a candidate node."""
    hardware: HardwareProfile
    idle_hours_per_day: float
    avg_slack_factor: float
    inferences_per_day_low: float
    inferences_per_day_expected: float
    inferences_per_day_high: float
    revenue_usd_low: float
    revenue_usd_expected: float
    revenue_usd_high: float
    electricity_cost_usd_per_day: float
    net_usd_low: float
    net_usd_expected: float
    net_usd_high: float
    # Per the §A11 cost optimizer, the protocol fee is taken on the
    # SETTLEMENT side, not the bid side -- documented here so the
    # operator sees the after-fee figure.
    protocol_fee_pct: float = 5.0
    notes: list[str] = field(default_factory=list)
    assumptions: Dict[str, str] = field(default_factory=dict)


def _avg_slack_over_idle_window(
    curve: TimeOfDaySlackCurve,
    idle_window: tuple[int, int],
) -> float:
    """Mean slack factor over the user-declared idle window
    (start_hour, end_hour). Wraps midnight cleanly."""
    start, end = idle_window
    samples = []
    h = start
    while True:
        samples.append(curve.opportunity_cost_factor(
            at=datetime.time(int(h) % 24, 0)))
        h += 0.5
        if (start < end and h >= end) or (start >= end and h % 24 >= end and h - start >= 0.5):
            if start < end:
                break
            if h - start >= ((end + 24) - start):
                break
    if not samples:
        return 1.0
    return sum(samples) / len(samples)


def estimate_earnings(
    *,
    hardware_id: str,
    idle_hours_per_day: float = 8.0,
    idle_window: tuple[int, int] = (0, 8),
    power_cost_usd_per_kwh: float = 0.12,
    utilization: float = 0.6,                # % of idle time spent on real jobs
    curve: Optional[TimeOfDaySlackCurve] = None,
    protocol_fee_pct: float = 5.0,
) -> EarningsEstimate:
    """Compute honest earnings range for a node.

    `idle_hours_per_day` -- how many hours per day the operator is
    willing to lend the GPU.
    `idle_window` -- typical local-time window (start_h, end_h, 24h)
    when the GPU is idle, used to look up the slack curve.
    `power_cost_usd_per_kwh` -- the operator's electricity rate.
    `utilization` -- fraction of idle time that actually serves jobs
    (the network can't always saturate every node).
    `curve` -- the node's published slack curve; defaults to consumer-
    work-from-home preset.
    `protocol_fee_pct` -- the chain's settlement fee.
    """
    hw = HARDWARE_TABLE.get(hardware_id.lower())
    if hw is None:
        raise ValueError(
            f"unknown hardware '{hardware_id}'. "
            f"Known: {', '.join(list_known_hardware())}"
        )
    curve = curve or default_consumer_curve()
    avg_slack = _avg_slack_over_idle_window(curve, idle_window)
    base_revenue_per_idle_hour = (
        hw.inferences_per_hour * hw.base_price_per_inference_usd * avg_slack
    )
    # Range derives from utilisation uncertainty.
    low_util = max(0.05, utilization * 0.5)
    high_util = min(1.0, utilization * 1.3)
    inf_per_day_low = hw.inferences_per_hour * idle_hours_per_day * low_util
    inf_per_day_exp = hw.inferences_per_hour * idle_hours_per_day * utilization
    inf_per_day_high = hw.inferences_per_hour * idle_hours_per_day * high_util
    rev_low = inf_per_day_low * hw.base_price_per_inference_usd * avg_slack
    rev_exp = inf_per_day_exp * hw.base_price_per_inference_usd * avg_slack
    rev_high = inf_per_day_high * hw.base_price_per_inference_usd * avg_slack
    # Apply protocol fee (revenue is what the operator KEEPS).
    keep = 1.0 - protocol_fee_pct / 100.0
    rev_low *= keep
    rev_exp *= keep
    rev_high *= keep
    # Electricity for the idle hours during which the GPU is genuinely
    # serving (utilisation), at typical power draw.
    avg_load_w = hw.typical_power_w * utilization
    kwh_per_day = (avg_load_w / 1000.0) * idle_hours_per_day
    electricity_cost = kwh_per_day * power_cost_usd_per_kwh

    notes = []
    if avg_slack < 0.5:
        notes.append(
            "Idle window is in deep slack; revenue is at the LOWER end "
            "of the published price -- this is by design (cheap during "
            "low-demand windows)."
        )
    if rev_exp - electricity_cost < 0:
        notes.append(
            "Net is negative: electricity exceeds expected revenue at "
            "this slack. Consider running during higher-demand hours."
        )

    return EarningsEstimate(
        hardware=hw,
        idle_hours_per_day=idle_hours_per_day,
        avg_slack_factor=avg_slack,
        inferences_per_day_low=inf_per_day_low,
        inferences_per_day_expected=inf_per_day_exp,
        inferences_per_day_high=inf_per_day_high,
        revenue_usd_low=rev_low,
        revenue_usd_expected=rev_exp,
        revenue_usd_high=rev_high,
        electricity_cost_usd_per_day=electricity_cost,
        net_usd_low=rev_low - electricity_cost,
        net_usd_expected=rev_exp - electricity_cost,
        net_usd_high=rev_high - electricity_cost,
        protocol_fee_pct=protocol_fee_pct,
        notes=notes,
        assumptions={
            "hardware": hw.name,
            "idle_hours_per_day": f"{idle_hours_per_day:.1f}",
            "idle_window": f"{idle_window[0]:02d}:00 -> {idle_window[1]:02d}:00",
            "utilization": f"{utilization:.0%}",
            "power_w_active": f"{avg_load_w:.0f}",
            "power_cost_usd_per_kwh": f"{power_cost_usd_per_kwh:.4f}",
            "avg_slack_factor": f"{avg_slack:.3f}",
            "inferences_per_hour_at_baseline": str(hw.inferences_per_hour),
            "base_price_usd_per_inference": (
                f"{hw.base_price_per_inference_usd:.6f}"
            ),
            "protocol_fee_pct": f"{protocol_fee_pct:.1f}%",
        },
    )


def format_estimate(est: EarningsEstimate) -> str:
    """Human-readable summary suitable for stdout."""
    lines = []
    lines.append(f"Pluginfer earnings estimate -- {est.hardware.name}")
    lines.append("=" * 56)
    lines.append("Predicted earnings (USD/day, after {:.0f}% protocol fee):"
                 .format(est.protocol_fee_pct))
    lines.append(
        f"  expected   : ${est.revenue_usd_expected:7.4f}   "
        f"(low ${est.revenue_usd_low:.4f}, high ${est.revenue_usd_high:.4f})"
    )
    lines.append(f"  electricity: ${est.electricity_cost_usd_per_day:7.4f}")
    lines.append(
        f"  NET        : ${est.net_usd_expected:7.4f}   "
        f"(low ${est.net_usd_low:.4f}, high ${est.net_usd_high:.4f})"
    )
    lines.append("")
    lines.append("Assumptions (override via flags):")
    for k, v in est.assumptions.items():
        lines.append(f"  {k:32s} : {v}")
    if est.notes:
        lines.append("")
        lines.append("Notes:")
        for n in est.notes:
            lines.append(f"  - {n}")
    return "\n".join(lines)


__all__ = [
    "HARDWARE_TABLE",
    "HardwareProfile",
    "EarningsEstimate",
    "estimate_earnings",
    "format_estimate",
    "list_known_hardware",
]


# ---------------------------------------------------------------------------
# CLI: python -m core.earnings_estimator <hardware> [flags]
# ---------------------------------------------------------------------------


def _cli_main(argv: Optional[list] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="pluginfer-estimate-earnings",
        description=(
            "Predict honest PLG/USD earnings for running a Pluginfer "
            "node on the given hardware."
        ),
    )
    p.add_argument("hardware",
                   help=f"hardware id (one of: {', '.join(list_known_hardware())})")
    p.add_argument("--idle-hours", type=float, default=8.0,
                   help="hours per day you are willing to share the GPU")
    p.add_argument("--idle-start", type=int, default=0,
                   help="local hour when the idle window begins (24h)")
    p.add_argument("--idle-end", type=int, default=8,
                   help="local hour when the idle window ends (24h)")
    p.add_argument("--power-cost", type=float, default=0.12,
                   help="electricity cost in USD per kWh")
    p.add_argument("--utilization", type=float, default=0.6,
                   help="fraction of idle time actually serving jobs")
    p.add_argument("--protocol-fee", type=float, default=5.0,
                   help="chain protocol fee percentage")
    args = p.parse_args(argv)
    try:
        est = estimate_earnings(
            hardware_id=args.hardware,
            idle_hours_per_day=args.idle_hours,
            idle_window=(args.idle_start, args.idle_end),
            power_cost_usd_per_kwh=args.power_cost,
            utilization=args.utilization,
            protocol_fee_pct=args.protocol_fee,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        return 2
    print(format_estimate(est))
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    import sys
    sys.exit(_cli_main())
