"""Cost-Optimal Multi-Constraint Router (PNIS §A11).

The standard Pluginfer Auction (`core/providers.Auction`) ranks bids
by a Pareto-scalarised score that blends price, latency, quality, and
privacy. That's the right default. But for cost-sensitive callers --
the dominant case for production AI workloads where 99% of inferences
run for cents and only the long tail needs maximum quality -- the
right policy is *strict cost minimisation subject to hard constraints*.

This module is the explicit "cheapest-that-works" router:

  * Caller specifies HARD constraints: cost_ceiling_usd, latency
    ceiling, quality floor, privacy class -- all enforced as filters.
  * Among the bids that satisfy every constraint, pick the one with
    the LOWEST price_usd. Ties broken by lower latency, then by higher
    quality.
  * If no bid satisfies every constraint, return the Pareto frontier
    so the caller can decide which constraint to relax.

Why this design is novel
----------------------
Existing decentralised compute markets (Akash, io.net, Render) auction
on price-only and let the user manually filter on hardware specs.
Centralised LLM APIs price-fix per-token. Pluginfer's contribution:

  "A multi-constraint sealed-bid auction protocol that returns either
   the cost-minimal bid satisfying every caller-supplied hard
   constraint, or the Pareto frontier when no single bid satisfies
   them, with cryptographic provenance binding the selection to the
   bid set the broker observed."

This is the formal structure that powers the "AI for a fraction of
the cost" promise. Combined with §3 slack-aware pricing, real
workloads route to consumer-GPU off-peak slack at 5-20x below the
centralised-API price floor.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional

from .providers import (
    AuctionResult,
    Bid,
    JobSpec,
    Provider,
    _PRIVACY_RANK,
)


# ---------------------------------------------------------------------------
# Pareto frontier (used when no bid satisfies every hard constraint)
# ---------------------------------------------------------------------------


def _dominates(a: Bid, b: Bid) -> bool:
    """Bid `a` Pareto-dominates `b` iff a is no worse on every
    objective and strictly better on at least one. Objectives:
    minimize price, minimize eta, maximize expected_quality.
    """
    no_worse = (
        a.price_usd <= b.price_usd
        and a.eta_ms <= b.eta_ms
        and a.expected_quality >= b.expected_quality
    )
    strictly_better = (
        a.price_usd < b.price_usd
        or a.eta_ms < b.eta_ms
        or a.expected_quality > b.expected_quality
    )
    return no_worse and strictly_better


def pareto_frontier(bids: List[Bid]) -> List[Bid]:
    """Return the subset of `bids` that no other bid Pareto-dominates."""
    frontier: List[Bid] = []
    for b in bids:
        if any(_dominates(other, b) for other in bids if other is not b):
            continue
        frontier.append(b)
    return frontier


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------


@dataclass
class CostOptimalSelection:
    """Output of `CostOptimalRouter.select`.

    `winner` is None iff no bid satisfied every hard constraint; in
    that case `frontier` is the Pareto frontier of the relaxed bid set
    so the caller can pick which constraint to weaken.
    """
    winner: Optional[Bid]
    rejected: List[dict] = field(default_factory=list)
    frontier: List[Bid] = field(default_factory=list)
    bids_observed: List[Bid] = field(default_factory=list)
    selection_proof: Optional[str] = None        # sha256 over canonical bid set

    def is_won(self) -> bool:
        return self.winner is not None


@dataclass
class CostOptimalRouter:
    """Cost-minimising sealed-bid router.

    Same registration/bid-collection contract as `core.providers.Auction`
    -- a drop-in alternative when the caller wants explicit
    cost-minimisation rather than Pareto-blended scoring.
    """
    providers: List[Provider] = field(default_factory=list)

    def register(self, p: Provider) -> None:
        self.providers.append(p)

    # ----------------------------------------------------------------------

    def collect_bids(self, job: JobSpec) -> tuple[List[Bid], List[dict]]:
        bids: List[Bid] = []
        rejected: List[dict] = []
        for p in self.providers:
            try:
                b = p.bid(job)
            except Exception as e:
                rejected.append({
                    "provider_id": getattr(p, "provider_id", "?"),
                    "reason": f"provider raised: {e}",
                })
                continue
            if b is None:
                rejected.append({
                    "provider_id": getattr(p, "provider_id", "?"),
                    "reason": "abstained",
                })
                continue
            bids.append(b)
        return bids, rejected

    # ----------------------------------------------------------------------

    @staticmethod
    def _passes_hard_constraints(b: Bid, job: JobSpec) -> Optional[str]:
        # Cheaper to evaluate physics first.
        if b.price_usd < 0:
            return f"negative price ({b.price_usd})"
        if b.eta_ms <= 0:
            return f"non-positive eta ({b.eta_ms})"
        if not (0.0 <= b.expected_quality <= 1.0):
            return f"quality {b.expected_quality} out of [0,1]"
        if b.price_usd > job.cost_ceiling_usd:
            return f"price {b.price_usd:.6f} > ceiling {job.cost_ceiling_usd}"
        if b.eta_ms > job.latency_ceiling_ms:
            return f"eta {b.eta_ms} > ceiling {job.latency_ceiling_ms}"
        if b.expected_quality < job.quality_floor:
            return (f"quality {b.expected_quality:.3f} "
                    f"< floor {job.quality_floor}")
        if _PRIVACY_RANK.get(b.privacy_grade, -1) < _PRIVACY_RANK.get(
                job.privacy_class, 0):
            return f"privacy {b.privacy_grade} < required {job.privacy_class}"
        if b.reasoning_seconds_committed < 0:
            return (f"negative reasoning_seconds_committed "
                    f"({b.reasoning_seconds_committed})")
        if b.reasoning_seconds_committed > job.reasoning_seconds_max:
            return (f"reasoning_seconds {b.reasoning_seconds_committed} "
                    f"> caller max {job.reasoning_seconds_max}")
        return None

    # ----------------------------------------------------------------------

    @staticmethod
    def _selection_proof(bids: List[Bid], job: JobSpec) -> str:
        """Hash committing to (job, observed bid set). Lets the
        requester later prove the winner truly was cost-minimal among
        the bids the broker actually saw.
        """
        canonical = {
            "job": {
                "id": job.job_id,
                "kind": job.kind,
                "cost_ceiling_usd": job.cost_ceiling_usd,
                "latency_ceiling_ms": job.latency_ceiling_ms,
                "privacy_class": job.privacy_class,
                "quality_floor": job.quality_floor,
            },
            "bids": [
                {
                    "provider_id": b.provider_id,
                    "price_usd": b.price_usd,
                    "eta_ms": b.eta_ms,
                    "expected_quality": b.expected_quality,
                    "privacy_grade": b.privacy_grade,
                }
                for b in sorted(bids, key=lambda x: x.provider_id)
            ],
        }
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    # ----------------------------------------------------------------------

    def select(self, job: JobSpec) -> CostOptimalSelection:
        """Run the cost-optimal selection.

        Strict cost-minimal among bids that satisfy every hard
        constraint. Tie-break: lower eta, then higher quality. If no
        bid passes, the Pareto frontier of the *unfiltered* set is
        returned so the caller can relax intelligently.
        """
        bids, rejected = self.collect_bids(job)

        # Filter to satisfying set.
        passing: List[Bid] = []
        for b in bids:
            why = self._passes_hard_constraints(b, job)
            if why:
                rejected.append({
                    "provider_id": b.provider_id,
                    "reason": why,
                    "price_usd": b.price_usd,
                    "eta_ms": b.eta_ms,
                    "quality": b.expected_quality,
                })
                continue
            passing.append(b)

        proof = self._selection_proof(bids, job) if bids else None

        if not passing:
            # No bid meets every hard constraint. Surface the Pareto
            # frontier of every bid the broker saw so the caller can
            # decide which axis to relax.
            return CostOptimalSelection(
                winner=None,
                rejected=rejected,
                frontier=pareto_frontier(bids),
                bids_observed=bids,
                selection_proof=proof,
            )

        winner = min(
            passing,
            key=lambda b: (b.price_usd, b.eta_ms, -b.expected_quality),
        )
        return CostOptimalSelection(
            winner=winner,
            rejected=rejected,
            frontier=passing,
            bids_observed=bids,
            selection_proof=proof,
        )


# ---------------------------------------------------------------------------
# Convenience: cost-savings reporter (the headline number for users)
# ---------------------------------------------------------------------------


def cost_savings_vs_baseline(
    selection: CostOptimalSelection,
    centralised_baseline_usd: float,
) -> dict:
    """Compare the chosen mesh bid to a centralised-API baseline price.

    Used for the public "AI for a fraction of the cost" headline metric
    -- e.g. "your job ran 11.7x cheaper than gpt-4o-mini for the same
    quality floor."
    """
    if not selection.is_won():
        return {"won": False}
    chosen = selection.winner.price_usd
    if centralised_baseline_usd <= 0:
        return {"won": True, "chosen_usd": chosen,
                "baseline_usd": centralised_baseline_usd, "ratio": None}
    return {
        "won": True,
        "chosen_usd": chosen,
        "baseline_usd": centralised_baseline_usd,
        "savings_usd": centralised_baseline_usd - chosen,
        "savings_pct": 100.0 * (1.0 - chosen / centralised_baseline_usd),
        "ratio_cheaper": (centralised_baseline_usd / chosen
                          if chosen > 0 else float("inf")),
    }


__all__ = [
    "CostOptimalRouter",
    "CostOptimalSelection",
    "pareto_frontier",
    "cost_savings_vs_baseline",
]
