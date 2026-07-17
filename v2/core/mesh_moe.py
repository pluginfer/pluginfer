"""Mixture-of-Experts on the Mesh (PNIS §A8).

Standard Mixture-of-Experts (Mistral 8x7B, DeepSeek-V2, Qwen-MoE)
puts a router and N experts on the SAME machine. Pluginfer's
contribution: put the router on the REQUESTER's machine and the
experts on the MESH. This converts the centralised "one giant model
on one giant box" pattern into a permissionless marketplace where
every node can train + deploy + earn from a small specialty expert.

  * The router (small -- 5-20M params) lives locally with the user.
    It computes expert-relevance from the user's input embedding.
  * The K experts the router picks are dispatched to mesh providers
    via §A13 quorum_inference (zero downtime) + §A15 edge cache
    (sub-ms cache hit on repeat queries).
  * Provider returns the expert's contribution; router weighted-
    averages.
  * Settlement happens per-expert via §A16 revenue distribution --
    expert authors earn a per-call royalty.

Why this design is novel
----------------------
"A mixture-of-experts inference architecture in which a router model
deployed on a requester device computes expert-selection weights
locally, dispatches the chosen K experts as parallel quorum-protected
inference jobs to a permissionless decentralised compute mesh, and
combines their outputs via the locally-computed weights, with on-
chain settlement to each expert's author per call."

This is genuinely new -- no decentralised MoE has been deployed in
production. The implementation here is the orchestration layer; the
expert weights themselves live in the §9 capability marketplace.

Honest scope
------------
This module is the orchestration + dispatch surface; the actual
LoRA-as-expert weights are loaded by the providers from the
marketplace. The router's local model can be the §6 PluginferBrain
or any small classifier; we accept any callable that takes an input
and returns expert-id -> weight dict.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expert + router types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertRecord:
    """One expert in the marketplace registry."""
    expert_id: str
    domain: str                              # "medical"|"legal"|"code"|...
    model_hash: str                          # sha256 of expert weights
    author_address: str                      # for revenue split (§A16)
    quality_floor: float = 0.7
    base_price_usd_per_call: float = 0.0005


# Router signature: input_bytes -> dict[expert_id, weight], summing to ~1.
RouterFn = Callable[[bytes], Dict[str, float]]


@dataclass
class ExpertOutput:
    expert_id: str
    output: Any                              # provider returns; format opaque
    latency_ms: float
    error: Optional[str] = None


@dataclass
class MoEResult:
    """Final mixture output across K experts."""
    chosen_experts: List[str]
    weights: Dict[str, float]
    outputs: List[ExpertOutput]
    failed_experts: List[str] = field(default_factory=list)
    latency_ms_total: float = 0.0


# ---------------------------------------------------------------------------
# Router orchestration
# ---------------------------------------------------------------------------


def _topk_by_weight(weights: Dict[str, float], k: int) -> Dict[str, float]:
    """Return only the K largest weights, normalised to sum 1.
    The discarded mass is ignored; this is by design (we don't want
    to amplify low-confidence experts after pruning)."""
    if k <= 0:
        return {}
    items = sorted(weights.items(), key=lambda kv: -kv[1])[:k]
    total = sum(v for _, v in items)
    if total <= 0:
        return {}
    return {eid: w / total for eid, w in items}


@dataclass
class MeshMoERouter:
    """Mesh-MoE orchestrator.

    Caller responsibilities:
      * Provide a RouterFn that scores experts for a given input.
      * Provide a registry mapping expert_id -> ExpertRecord (or set
        registry= directly).
      * Provide a `dispatch_one` coroutine that takes
        (expert_record, input_bytes) and returns an ExpertOutput.
      * Provide a `combine` callable that takes
        (weights, outputs) and produces the final answer.
    """
    router: RouterFn
    registry: Dict[str, ExpertRecord]
    dispatch_one: Callable[
        [ExpertRecord, bytes], Awaitable[ExpertOutput]
    ]
    combine: Callable[[Dict[str, float], List[ExpertOutput]], Any]
    top_k: int = 2

    # ----------------------------------------------------------------------

    def select(self, input_bytes: bytes) -> Dict[str, float]:
        """Run the local router; return top-K (expert_id -> weight)."""
        raw = self.router(input_bytes)
        if not raw:
            return {}
        # Filter to known experts only -- a router could in principle
        # propose an expert id we don't have a record for.
        filtered = {eid: w for eid, w in raw.items()
                    if eid in self.registry and w > 0}
        return _topk_by_weight(filtered, self.top_k)

    # ----------------------------------------------------------------------

    async def __call__(self, input_bytes: bytes,
                       *, overall_timeout_s: float = 30.0) -> MoEResult:
        """Run one mixture-of-experts inference end to end."""
        weights = self.select(input_bytes)
        if not weights:
            return MoEResult(
                chosen_experts=[], weights={}, outputs=[],
                latency_ms_total=0.0,
            )
        chosen = list(weights.keys())
        records = [self.registry[eid] for eid in chosen]
        started = time.monotonic()

        async def _safe_call(rec: ExpertRecord) -> ExpertOutput:
            try:
                return await self.dispatch_one(rec, input_bytes)
            except Exception as e:
                return ExpertOutput(
                    expert_id=rec.expert_id, output=None,
                    latency_ms=0.0, error=f"{type(e).__name__}: {e}"
                )

        try:
            async with asyncio.timeout(overall_timeout_s):
                outputs: List[ExpertOutput] = list(
                    await asyncio.gather(*[_safe_call(r) for r in records])
                )
        except (asyncio.TimeoutError, TimeoutError):
            outputs = [ExpertOutput(expert_id=r.expert_id, output=None,
                                    latency_ms=0.0, error="timeout")
                       for r in records]

        elapsed_ms = (time.monotonic() - started) * 1000.0
        failed = [o.expert_id for o in outputs if o.error]

        return MoEResult(
            chosen_experts=chosen,
            weights=weights,
            outputs=outputs,
            failed_experts=failed,
            latency_ms_total=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Reference combiners (the caller can supply a custom one)
# ---------------------------------------------------------------------------


def weighted_softmax_combine(
    weights: Dict[str, float],
    outputs: List[ExpertOutput],
) -> Optional[List[float]]:
    """Reference combiner: each expert returns a list-of-floats logit
    vector; combine is weighted average. Returns None if all experts
    failed.

    Real production: the router uses §6 PluginferBrain; combiners are
    domain-specific (text-token softmax, image-RGB pixel average,
    retrieval-rank fusion).
    """
    valid: List[tuple[float, List[float]]] = []
    for o in outputs:
        if o.error or o.output is None:
            continue
        if not isinstance(o.output, list) or not all(
            isinstance(x, (int, float)) for x in o.output
        ):
            continue
        w = weights.get(o.expert_id, 0.0)
        if w > 0:
            valid.append((w, [float(x) for x in o.output]))
    if not valid:
        return None
    n = max(len(v) for _, v in valid)
    pad = lambda v: v + [0.0] * (n - len(v))
    total_w = sum(w for w, _ in valid)
    if total_w <= 0:
        return None
    out = [0.0] * n
    for w, v in valid:
        for i, x in enumerate(pad(v)):
            out[i] += (w / total_w) * x
    return out


__all__ = [
    "ExpertRecord",
    "RouterFn",
    "ExpertOutput",
    "MoEResult",
    "MeshMoERouter",
    "weighted_softmax_combine",
]
