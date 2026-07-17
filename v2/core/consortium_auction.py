"""Consortium auction — N best providers, not one.

The §A11 single-winner auction picks the cheapest qualifying bid for
small / mid-size inference jobs. For LARGE jobs — distributed
training, batch inference of millions of prompts, model parallel for
70B+ params — one winner is not enough. We need to **shard the work
across many providers** and aggregate their results.

This module is layer 3 of the topology described in
`core.gossip_discovery`:

  Layer 1: bootstrap (seed) -> Layer 2: gossip (find everyone)
  -> Layer 3: consortium auction (share the work).

The auction returns a `Consortium`: an ordered list of N bids
ranked by the same Pareto scoring used in the single-winner case.
The Bids' `evidence` dicts carry per-provider sharding hints
(rank, shard_size, expected_tokens) so the dispatcher knows how to
slice the work.

Three sharding modes are wired through the same primitive:

  * **data-parallel** — split a batch of M prompts into N shards;
    each provider runs `M/N` prompts; results concatenate.
    (For inference, no inter-provider communication needed.)
  * **tensor-parallel** — split the model's weight matrices across
    N providers; each computes a slice of the forward pass;
    activations flow between providers via direct TCP.
    (Needed for 70B+ models that don't fit on one GPU.)
  * **diloco** — each provider trains its own replica on a local
    data shard; periodic sync via `core.diloco_aggregator`
    averages parameters.

The auction layer is sharding-mode-agnostic — `JobSpec.payload`
declares which mode the dispatcher should use. The auction's only
job is to RANK and PICK the consortium.

Filing target: INVENTIONS §A23 "Consortium Auction Over a
Permissionless Compute Mesh" (drafted in the WORKLOG entry for
this commit).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .providers import Auction, AuctionResult, Bid, JobSpec, Provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

@dataclass
class ConsortiumMember:
    """One slot in the consortium."""
    provider_id: str
    bid: Bid
    rank: int                       # 0-based seat number
    shard_fraction: float           # 1/N for uniform sharding


@dataclass
class Consortium:
    """The auction's output for a shardable job."""
    members: List[ConsortiumMember] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    sharding_mode: str = "data-parallel"

    @property
    def size(self) -> int:
        return len(self.members)

    def is_filled(self, minimum: int) -> bool:
        """Is the consortium big enough to serve the job?"""
        return len(self.members) >= minimum

    def to_dict(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "sharding_mode": self.sharding_mode,
            "members": [
                {
                    "provider_id": m.provider_id,
                    "rank": m.rank,
                    "shard_fraction": m.shard_fraction,
                    "price_usd": m.bid.price_usd,
                    "eta_ms": m.bid.eta_ms,
                    "expected_quality": m.bid.expected_quality,
                }
                for m in self.members
            ],
            "rejected_count": len(self.rejected),
        }


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _pareto_score(bid: Bid, job: JobSpec) -> float:
    """Reuses the single-winner auction's scoring shape so a
    consortium's ranking is consistent with the §A11 single-winner
    pick. Lower score = better."""
    cost_term = bid.price_usd / max(1e-9, job.cost_ceiling_usd)
    eta_term = bid.eta_ms / max(1.0, job.latency_ceiling_ms)
    quality_term = max(0.0, job.quality_floor - bid.expected_quality)
    return cost_term + eta_term + 2.0 * quality_term


def _peer_score(b: Bid) -> float:
    """Extract the bidder's advertised compute score from the bid's
    evidence dict. _CrossNodeProvider stamps it as `peer_score`;
    local flagships expose it via `hardware_class` mapping (mid-tier
    = 5, high-tier = 50; matches HardwareDetector tiers). Defaults
    to 1.0 so a bid with no score still participates."""
    ev = b.evidence or {}
    s = ev.get("peer_score")
    if isinstance(s, (int, float)) and s > 0:
        return float(s)
    hwc = ev.get("hardware_class", "")
    return {
        "consumer-gpu-high": 50.0,
        "consumer-gpu-mid":  5.0,
        "consumer-gpu-low":  3.0,
        "consumer-cpu":      1.0,
    }.get(hwc, 1.0)


def select_scale_to_compute(
    auction: Auction,
    job: JobSpec,
    *,
    required_compute_score: float,
    max_members: int = 64,
    sharding_mode: str = "data-parallel",
) -> Consortium:
    """Pick the smallest, cheapest set of bidders whose summed
    `peer_score` ≥ required_compute_score. The thing that makes the
    mesh special: 4 GTX 1650s (score≈4 each) can together cover an
    H100-sized job (score≈200/4 = 50 each, or some combination).

    Selection strategy: sort bidders by USD/score ratio ascending,
    then greedy-pick until the required score is met or the
    max_members cap kicks in. Caps prevent a single huge job from
    consuming the entire mesh.

    Returns an unfilled Consortium when not enough total compute is
    available; the caller can downgrade the job or wait."""
    bids: List[Bid] = []
    rejected: List[Dict[str, Any]] = []
    for p in auction.providers:
        try:
            b = p.bid(job)
        except Exception as e:
            rejected.append({"provider_id": getattr(p, "provider_id", "?"),
                             "reason": f"provider raised: {e}"})
            continue
        if b is None:
            rejected.append({"provider_id": getattr(p, "provider_id", "?"),
                             "reason": "abstained"})
            continue
        why = b.violates(job)
        if why:
            rejected.append({"provider_id": b.provider_id, "reason": why,
                             "bid": b})
            continue
        bids.append(b)
    bids.sort(key=lambda b: b.price_usd / max(0.001, _peer_score(b)))
    picked: List[Bid] = []
    accumulated = 0.0
    for b in bids:
        if accumulated >= required_compute_score:
            break
        if len(picked) >= max_members:
            break
        picked.append(b)
        accumulated += _peer_score(b)
    if not picked or accumulated < required_compute_score:
        return Consortium(
            members=[], rejected=rejected, sharding_mode=sharding_mode,
        )
    shard_fraction = 1.0 / len(picked)
    members = [
        ConsortiumMember(
            provider_id=b.provider_id, bid=b, rank=i,
            shard_fraction=shard_fraction,
        )
        for i, b in enumerate(picked)
    ]
    return Consortium(
        members=members, rejected=rejected, sharding_mode=sharding_mode,
    )


def select_consortium(
    auction: Auction,
    job: JobSpec,
    *,
    target_size: int,
    minimum_size: Optional[int] = None,
    sharding_mode: str = "data-parallel",
) -> Consortium:
    """Run the auction and pick up to `target_size` providers, ranked
    by Pareto score (best first). Drops bids that violate the job's
    constraints, exactly as the single-winner path does.

    `minimum_size` defaults to `target_size`; if fewer bids survive
    filtering, the consortium is returned partial — caller decides
    whether that's enough (e.g. data-parallel can degrade gracefully;
    tensor-parallel cannot)."""
    minimum_size = minimum_size if minimum_size is not None else target_size
    bids: List[Bid] = []
    rejected: List[Dict[str, Any]] = []
    for p in auction.providers:
        try:
            b = p.bid(job)
        except Exception as e:
            rejected.append({"provider_id": getattr(p, "provider_id", "?"),
                             "reason": f"provider raised: {e}"})
            continue
        if b is None:
            rejected.append({"provider_id": getattr(p, "provider_id", "?"),
                             "reason": "abstained"})
            continue
        why = b.violates(job)
        if why:
            rejected.append({"provider_id": b.provider_id, "reason": why,
                             "bid": b})
            continue
        bids.append(b)

    bids.sort(key=lambda b: _pareto_score(b, job))
    picked = bids[:target_size]
    if not picked:
        return Consortium(members=[], rejected=rejected,
                          sharding_mode=sharding_mode)

    shard_fraction = 1.0 / len(picked)
    members = [
        ConsortiumMember(
            provider_id=b.provider_id, bid=b, rank=i,
            shard_fraction=shard_fraction,
        )
        for i, b in enumerate(picked)
    ]
    return Consortium(members=members, rejected=rejected,
                      sharding_mode=sharding_mode)


# ---------------------------------------------------------------------------
# Heuristic — when does a job need a consortium?
# ---------------------------------------------------------------------------

# A consortium is the right answer when the job is large enough that
# splitting beats one-provider serial work. Two simple heuristics:
#   * the buyer asked for it: `payload.consortium = {"size": N}`.
#   * the cost ceiling is so high that the largest provider alone
#     would saturate (latency-bound, e.g. training). The threshold
#     defaults to `$5.00`; operator-tunable.
CONSORTIUM_COST_THRESHOLD_USD = 5.0


def job_needs_consortium(job: JobSpec) -> Optional[int]:
    """Return the requested consortium size if the job is shardable,
    else None. Inspects the job's payload + cost ceiling.

    Two ways the buyer signals a consortium is needed:
      * `payload.consortium = {"size": N, "mode": ...}` — explicit.
      * `payload.required_compute_score = X` — elastic-scale path;
        JobsService routes through `select_scale_to_compute` and
        size is dynamic, decided by the auction at clearing time
        (the smallest combination of providers whose summed peer
        scores meet X). Returns `0` as a sentinel meaning
        "auction-decides".
    """
    payload = job.payload or {}
    if (payload.get("required_compute_score") or 0) > 0:
        return 0       # sentinel: auction decides size based on score
    explicit = payload.get("consortium")
    if isinstance(explicit, dict):
        size = explicit.get("size")
        if isinstance(size, int) and size >= 2:
            return size
    # Critical jobs default to a 3-way quorum-replicate consortium —
    # majority of 3 (2-of-3) is the smallest meaningful byzantine
    # quorum. Buyers can override with payload.consortium.size for
    # stronger fault tolerance (5-of-9, etc.).
    criticality = str(payload.get("criticality") or "").lower()
    if criticality in ("high", "critical"):
        return 3
    if job.cost_ceiling_usd >= CONSORTIUM_COST_THRESHOLD_USD:
        # Default to 4-way split for big jobs unless the buyer
        # specified otherwise.
        return 4
    return None


def job_default_sharding_mode(job: JobSpec) -> str:
    """For consortium-eligible jobs: which sharding mode does the
    JobsService dispatcher use? Critical jobs default to
    `quorum-replicate` (byzantine-tolerant); others default to
    `data-parallel`. Buyer can override with payload.consortium.mode."""
    payload = job.payload or {}
    explicit = payload.get("consortium")
    if isinstance(explicit, dict) and explicit.get("mode"):
        return str(explicit["mode"])
    criticality = str(payload.get("criticality") or "").lower()
    if criticality in ("high", "critical"):
        return "quorum-replicate"
    return "data-parallel"


# ---------------------------------------------------------------------------
# Dispatcher hook — calls each member's execute() with its shard
# ---------------------------------------------------------------------------

@dataclass
class ConsortiumExecution:
    """The aggregated result of running a job across the consortium."""
    consortium: Consortium
    per_member: List[Dict[str, Any]] = field(default_factory=list)
    combined_result_bytes: Optional[bytes] = None
    combined_result_hash: Optional[str] = None
    total_cost_usd: float = 0.0
    failed_members: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        import base64
        return {
            "consortium": self.consortium.to_dict(),
            "per_member": [
                {
                    "provider_id": e.get("provider_id"),
                    "rank": e.get("rank"),
                    "status": e.get("status"),
                    "result_hash": e.get("result_hash"),
                    "execution_ms": e.get("execution_ms"),
                }
                for e in self.per_member
            ],
            "combined_result_hash": self.combined_result_hash,
            "combined_result_b64": (
                base64.b64encode(self.combined_result_bytes).decode("ascii")
                if self.combined_result_bytes else None
            ),
            "total_cost_usd": self.total_cost_usd,
            "failed_members": self.failed_members,
        }


def execute_consortium(
    auction: Auction,
    consortium: Consortium,
    job: JobSpec,
) -> ConsortiumExecution:
    """Dispatch the job across every consortium member, combine the
    results per `consortium.sharding_mode`, return the aggregate.

    Currently supports:
      * `data-parallel` — every member runs the SAME job; results
        are byte-concatenated (downstream caller decides what to
        do with multiple identical-shape outputs — voting, sampling,
        averaging).
      * `quorum-replicate` — every member runs the SAME job, results
        compared by SHA-256(result_bytes); the majority-hash result
        wins. Byzantine dissenters surface in `failed_members` so
        the slashing pipeline can act. Use this mode when the buyer
        cares more about correctness than throughput (critical jobs,
        regulatory workloads).
      * `diloco` — same dispatch shape as data-parallel; the
        post-processing hook in `core.diloco_aggregator` averages
        the gradients. Wired here as data-parallel; the aggregator
        consumes `per_member` directly.
      * `tensor-parallel` — NotImplementedError (requires per-shard
        weight slicing; that lives in `core.diloco_serialize` and
        needs integration the consortium doesn't yet do).

    The single-winner auction path is identical to a 1-member
    consortium — they share `Provider.execute()`."""
    import base64
    import hashlib
    if consortium.sharding_mode == "tensor-parallel":
        raise NotImplementedError(
            "tensor-parallel consortium execution requires "
            "core.diloco_serialize integration — multi-week protocol "
            "work. Use data-parallel or diloco mode for now."
        )

    pid_to_provider: Dict[str, Provider] = {
        getattr(p, "provider_id", "?"): p for p in auction.providers
    }
    exec_result = ConsortiumExecution(consortium=consortium)
    parts: List[bytes] = []
    for member in consortium.members:
        provider = pid_to_provider.get(member.provider_id)
        if provider is None:
            exec_result.failed_members.append(member.provider_id)
            exec_result.per_member.append({
                "provider_id": member.provider_id,
                "rank": member.rank,
                "status": "provider_missing",
            })
            continue
        try:
            out = provider.execute(job, member.bid)
        except Exception as e:
            exec_result.failed_members.append(member.provider_id)
            exec_result.per_member.append({
                "provider_id": member.provider_id,
                "rank": member.rank,
                "status": "failed",
                "reason": f"{type(e).__name__}: {e}",
            })
            continue
        exec_result.per_member.append({
            "provider_id": member.provider_id,
            "rank": member.rank,
            "status": out.get("status"),
            "result_hash": out.get("result_hash"),
            "execution_ms": out.get("execution_ms"),
        })
        exec_result.total_cost_usd += float(member.bid.price_usd)
        b64 = out.get("result_bytes") or out.get("result_bytes_b64")
        if b64:
            try:
                parts.append(base64.b64decode(b64))
            except Exception:
                pass
    if parts:
        if consortium.sharding_mode == "quorum-replicate":
            # Byzantine-tolerant: every member ran the same job;
            # winner is the majority-hash of result_bytes. Dissenters
            # get appended to failed_members so the slashing
            # pipeline can act on them.
            from collections import Counter
            hash_counter: Counter = Counter()
            hash_to_bytes: Dict[str, bytes] = {}
            hash_to_providers: Dict[str, List[str]] = {}
            for member, blob in zip(consortium.members, parts):
                h = hashlib.sha256(blob).hexdigest()
                hash_counter[h] += 1
                hash_to_bytes[h] = blob
                hash_to_providers.setdefault(h, []).append(member.provider_id)
            if hash_counter:
                winning_hash, winning_count = hash_counter.most_common(1)[0]
                threshold = (len(parts) // 2) + 1
                if winning_count >= threshold:
                    exec_result.combined_result_bytes = hash_to_bytes[winning_hash]
                    exec_result.combined_result_hash = winning_hash
                    # Mark every NON-winning member as a byzantine
                    # dissenter — they ran the same job, produced
                    # different bytes. Auditor / slashing pipeline
                    # consumes this list.
                    for h, providers in hash_to_providers.items():
                        if h != winning_hash:
                            for pid in providers:
                                if pid not in exec_result.failed_members:
                                    exec_result.failed_members.append(pid)
                else:
                    # No majority — split-brain. Surface as a
                    # failure; the buyer can retry with a larger
                    # consortium for stronger quorum.
                    exec_result.combined_result_bytes = None
                    exec_result.combined_result_hash = None
        else:
            # Data-parallel / diloco concat. The aggregator decides
            # whether to use the bytes directly or to interpret them
            # as gradient shards.
            combined = b"\n----PLUGINFER-SHARD----\n".join(parts)
            exec_result.combined_result_bytes = combined
            exec_result.combined_result_hash = hashlib.sha256(combined).hexdigest()
    return exec_result


__all__ = [
    "CONSORTIUM_COST_THRESHOLD_USD",
    "Consortium",
    "ConsortiumExecution",
    "ConsortiumMember",
    "execute_consortium",
    "job_default_sharding_mode",
    "job_needs_consortium",
    "select_consortium",
]
