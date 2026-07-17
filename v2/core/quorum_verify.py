"""Quorum verification for untrusted compute (§mesh-trust).

The audit named this the mesh's existential gap: a signed PNIS receipt
proves WHO returned bytes, not that the bytes are a CORRECT inference.
A malicious or lazy node can return garbage, a cheaper model's output,
or a cached lie, and the signature still checks out.

There is no cheap perfect solution (verifying an inference without
re-running it is an open research problem). The honest engineering
mitigation — the one real distributed systems use — is REDUNDANCY:
dispatch the same deterministic job to N INDEPENDENT providers and
accept the result only if at least K of them agree on its hash. This
module is that logic, kept pure so it is exhaustively testable without
a network.

What it defeats and what it does NOT (stated plainly, never oversold):

  * DEFEATS independent faults: a single node returning wrong/garbage/
    wrong-model output is outvoted and paid nothing. Lazy nodes that
    echo or truncate are caught the same way.
  * Raises the bar on cheating: to forge an accepted-but-wrong result a
    node must COLLUDE with >= K-1 others who all return the SAME wrong
    hash — expensive and detectable if providers are chosen
    independently.
  * Does NOT defeat a determined colluding majority. That needs an
    economic layer (stake + slashing + reputation) on top; this module
    exposes the disagreement signal that layer consumes. It also only
    applies to DETERMINISTIC jobs (temperature 0 / fixed seed) — a
    sampling job has no single correct hash, so quorum is skipped and
    the caller falls back to single-provider + reputation.

Cost is the honest tradeoff: K-of-N redundancy multiplies compute by N.
So it is a POLICY the buyer opts into per job (high-stakes work), not a
default tax on every call — surfaced via `QuorumPolicy`.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


@dataclass
class ProviderVerdict:
    provider_id: str
    result_hash: Optional[str]      # None = failed / no result
    in_majority: bool = False
    error: Optional[str] = None


@dataclass
class QuorumOutcome:
    accepted: bool
    agreed_result_hash: Optional[str]
    agreement_count: int
    responded: int
    dispatched: int
    quorum: int
    dispute: bool
    reason: str
    verdicts: List[ProviderVerdict] = field(default_factory=list)

    def paid_providers(self) -> List[str]:
        """Only providers in the agreeing majority get paid. Everyone
        else — wrong hash, no response, error — is withheld. That is
        the incentive: return the correct answer or don't get paid."""
        if not self.accepted:
            return []
        return [v.provider_id for v in self.verdicts if v.in_majority]

    def dissenting_providers(self) -> List[str]:
        """Responded but disagreed with the majority — the reputation/
        slashing layer's input."""
        maj = self.agreed_result_hash
        return [v.provider_id for v in self.verdicts
                if v.result_hash is not None and v.result_hash != maj]


def evaluate_quorum(
    results: List[Tuple[str, Optional[str]]],
    *,
    quorum: int,
    dispatched: Optional[int] = None,
) -> QuorumOutcome:
    """Pure decision core. `results` is (provider_id, result_hash);
    result_hash None means that provider failed to produce output.
    Accept iff the most common NON-NULL hash has >= `quorum` votes.

    Ties (two hashes tied for the lead, neither reaching quorum) are a
    dispute, never a coin-flip — silent tie-breaking is how a split
    vote becomes a wrong accepted answer."""
    n_dispatched = dispatched if dispatched is not None else len(results)
    verdicts = [ProviderVerdict(pid, h) for pid, h in results]
    responded = sum(1 for _, h in results if h is not None)
    counts = Counter(h for _, h in results if h is not None)

    if not counts:
        return QuorumOutcome(
            accepted=False, agreed_result_hash=None, agreement_count=0,
            responded=0, dispatched=n_dispatched, quorum=quorum,
            dispute=True, reason="no provider returned a result",
            verdicts=verdicts)

    top_hash, top_n = counts.most_common(1)[0]
    # Genuine tie for the lead below quorum => dispute.
    leaders = [h for h, c in counts.items() if c == top_n]
    if len(leaders) > 1 and top_n < quorum:
        return QuorumOutcome(
            accepted=False, agreed_result_hash=None,
            agreement_count=top_n, responded=responded,
            dispatched=n_dispatched, quorum=quorum, dispute=True,
            reason=f"split vote: {len(leaders)} results tied at "
                   f"{top_n}, none reached quorum {quorum}",
            verdicts=verdicts)

    accepted = top_n >= quorum
    for v in verdicts:
        v.in_majority = accepted and v.result_hash == top_hash
    return QuorumOutcome(
        accepted=accepted,
        agreed_result_hash=top_hash if accepted else None,
        agreement_count=top_n, responded=responded,
        dispatched=n_dispatched, quorum=quorum,
        dispute=not accepted,
        reason=("quorum reached" if accepted else
                f"best agreement {top_n} < quorum {quorum}"),
        verdicts=verdicts)


@dataclass
class QuorumPolicy:
    """Buyer's per-job opt-in. n=1 (default) means no redundancy — the
    historical single-provider path. n>1 turns on quorum. quorum
    defaults to a strict majority of n."""
    n: int = 1
    quorum: Optional[int] = None
    # Only meaningful for deterministic jobs; the caller sets this from
    # the job (temperature==0 / seed fixed).
    deterministic: bool = True

    @property
    def enabled(self) -> bool:
        return self.n > 1 and self.deterministic

    def required_quorum(self) -> int:
        if self.quorum is not None:
            return max(1, min(self.quorum, self.n))
        return self.n // 2 + 1          # strict majority


async def run_quorum(
    dispatch: Callable[[str], Awaitable[Optional[str]]],
    provider_ids: List[str],
    policy: QuorumPolicy,
) -> QuorumOutcome:
    """Dispatch the same job to each provider concurrently via
    `dispatch(provider_id) -> result_hash` (None on failure) and
    evaluate. A dispatch that raises is treated as a None result — a
    non-responding provider must never crash the quorum, only fail to
    vote."""
    async def _one(pid: str) -> Tuple[str, Optional[str]]:
        try:
            return pid, await dispatch(pid)
        except Exception:
            return pid, None

    results = await asyncio.gather(*(_one(p) for p in provider_ids))
    return evaluate_quorum(list(results),
                           quorum=policy.required_quorum(),
                           dispatched=len(provider_ids))
