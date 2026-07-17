"""K-redundant dispatch with majority-vote result acceptance.

Defence against a single malicious provider returning a forged result.
Even if `Auction.run()` picks a winner, we don't have to trust it
unilaterally -- we can run the SAME job on K independent providers in
parallel and accept the result only if a majority agree on its hash.

Tradeoffs:
  * **Cost**: K times the compute spend per job. Use `k=1` for cheap
    inference (the result hash + sig already gates the trivial-cheat
    case), `k=3` for high-value jobs (training rounds, financial
    inference), `k>=5` only for adversarial use cases.
  * **Latency**: bounded by the SLOWEST of K providers, not the
    fastest. Use `quorum_k <= k` so we can accept a quorum-of-K once
    enough have replied (e.g. quorum=2 of k=3 is 67%).

The dispatcher returns:
  * the consensus result (the result_hash that majority of providers
    voted for),
  * the list of dissenters (providers whose hash differed -> these
    are eligible for refund + slashing once W32 is wired),
  * a "won" flag (False if no majority emerged within the deadline).

Dissenters are NOT reflected in the result itself; the broker / chain
layer is expected to translate the dissenter list into the appropriate
slashing or refund actions. This module's contract is purely
"who agreed, who didn't, what's the consensus" -- it doesn't reach
into the chain.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .providers import Bid, JobSpec, Provider

logger = logging.getLogger(__name__)


@dataclass
class DispatchVote:
    """One provider's submission for the same job."""
    provider_id: str
    result_hash_hex: Optional[str]
    result_b64: Optional[str]
    provider_sig_b64: Optional[str]
    execution_ms: float
    error: Optional[str] = None


@dataclass
class DispatchResult:
    """Outcome of a redundant dispatch round."""
    won: bool
    consensus_hash_hex: Optional[str]
    consensus_result_b64: Optional[str]
    consensus_provider_sig_b64: Optional[str]
    consensus_provider_id: Optional[str]
    votes: List[DispatchVote] = field(default_factory=list)
    dissenters: List[str] = field(default_factory=list)
    detail: Optional[str] = None

    def majority_size(self) -> int:
        """How many providers voted with the consensus."""
        if self.consensus_hash_hex is None:
            return 0
        return sum(1 for v in self.votes
                   if v.result_hash_hex == self.consensus_hash_hex)


class RedundantDispatcher:
    """Run the same JobSpec on K providers in parallel.

    Constructor takes the K (provider, bid) pairs that the auction
    has already selected. The auction's job is to pick the K best;
    this class's job is to fan-out + reconcile.

    Use `quorum_k` < k to short-circuit once a consensus is reached
    (e.g. quorum=2 of k=3 lets us accept after the second matching
    response without waiting for the third).
    """

    def __init__(
        self,
        *,
        providers_and_bids: List[tuple[Provider, Bid]],
        quorum_k: Optional[int] = None,
        per_provider_timeout_s: float = 30.0,
    ) -> None:
        if not providers_and_bids:
            raise ValueError("at least one (provider, bid) required")
        self.providers_and_bids = list(providers_and_bids)
        self.k = len(providers_and_bids)
        self.quorum_k = quorum_k if quorum_k is not None else (self.k // 2 + 1)
        if not (1 <= self.quorum_k <= self.k):
            raise ValueError(
                f"quorum_k {self.quorum_k} not in [1, {self.k}]"
            )
        self.per_provider_timeout_s = per_provider_timeout_s

    async def dispatch(self, job: JobSpec) -> DispatchResult:
        """Fan out the job, await votes, return consensus."""
        loop = asyncio.get_running_loop()

        async def _one(provider: Provider, bid: Bid) -> DispatchVote:
            t0 = time.monotonic()
            try:
                out = await asyncio.wait_for(
                    loop.run_in_executor(None, provider.execute, job, bid),
                    timeout=self.per_provider_timeout_s,
                )
            except asyncio.TimeoutError:
                return DispatchVote(
                    provider_id=getattr(provider, "provider_id", "?"),
                    result_hash_hex=None, result_b64=None,
                    provider_sig_b64=None,
                    execution_ms=(time.monotonic() - t0) * 1000.0,
                    error="timeout",
                )
            except Exception as e:
                return DispatchVote(
                    provider_id=getattr(provider, "provider_id", "?"),
                    result_hash_hex=None, result_b64=None,
                    provider_sig_b64=None,
                    execution_ms=(time.monotonic() - t0) * 1000.0,
                    error=f"{type(e).__name__}: {e}",
                )
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            if not isinstance(out, dict):
                return DispatchVote(
                    provider_id=getattr(provider, "provider_id", "?"),
                    result_hash_hex=None, result_b64=None,
                    provider_sig_b64=None, execution_ms=elapsed_ms,
                    error="non_dict_response",
                )
            if out.get("status") not in ("executed", "completed", "ok"):
                return DispatchVote(
                    provider_id=getattr(provider, "provider_id", "?"),
                    result_hash_hex=None, result_b64=None,
                    provider_sig_b64=None, execution_ms=elapsed_ms,
                    error=str(out.get("status") or "unknown_status"),
                )
            return DispatchVote(
                provider_id=getattr(provider, "provider_id", "?"),
                result_hash_hex=out.get("result_hash") or out.get("result_hash_hex"),
                result_b64=out.get("result_bytes_b64") or out.get("result_b64"),
                provider_sig_b64=out.get("provider_sig") or out.get("provider_signature_b64"),
                execution_ms=elapsed_ms,
            )

        tasks = [
            asyncio.create_task(_one(p, b))
            for p, b in self.providers_and_bids
        ]

        votes: List[DispatchVote] = []
        consensus_hash: Optional[str] = None
        try:
            while tasks:
                done, _pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED,
                )
                for t in done:
                    vote = t.result()
                    votes.append(vote)
                    tasks.remove(t)
                # Check for early consensus.
                hashes = [v.result_hash_hex for v in votes
                          if v.result_hash_hex is not None]
                if hashes:
                    counts = Counter(hashes)
                    most_common, n = counts.most_common(1)[0]
                    if n >= self.quorum_k:
                        consensus_hash = most_common
                        # Cancel any still-running tasks; we have quorum.
                        for t in tasks:
                            t.cancel()
                        # Wait briefly so cancellations settle and any
                        # last-second result still gets recorded.
                        if tasks:
                            await asyncio.gather(*tasks, return_exceptions=True)
                        tasks = []
                        break
        finally:
            for t in tasks:
                t.cancel()

        # If we never reached quorum, take the largest single bucket
        # (still useful as a "best guess" + dissenter list).
        if consensus_hash is None:
            hashes = [v.result_hash_hex for v in votes
                      if v.result_hash_hex is not None]
            if hashes:
                consensus_hash, _ = Counter(hashes).most_common(1)[0]

        won = consensus_hash is not None and \
            sum(1 for v in votes if v.result_hash_hex == consensus_hash) >= self.quorum_k

        # Find one consensus vote to ship the actual result bytes.
        consensus_vote = next(
            (v for v in votes if v.result_hash_hex == consensus_hash),
            None,
        ) if consensus_hash else None

        dissenters = [
            v.provider_id for v in votes
            if v.result_hash_hex != consensus_hash
        ]

        return DispatchResult(
            won=won,
            consensus_hash_hex=consensus_hash,
            consensus_result_b64=consensus_vote.result_b64 if consensus_vote else None,
            consensus_provider_sig_b64=consensus_vote.provider_sig_b64 if consensus_vote else None,
            consensus_provider_id=consensus_vote.provider_id if consensus_vote else None,
            votes=votes,
            dissenters=dissenters,
            detail=None if won else "no_quorum",
        )
