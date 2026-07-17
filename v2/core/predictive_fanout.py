"""Predictive Request Fan-Out (PNIS §A12) -- latency preemption.

The lower bound on latency for a request is the round-trip from
client to provider plus the provider's compute time. The §A15 edge
cache shrinks this to local-network ping for *repeat* queries; this
module shrinks it to ZERO (apparent) for *first-time* queries that
the system can predict.

The mechanism: observe the user's ongoing session for behavioral
signals (typing, partial token stream, click patterns, recent
history). When a continuation reaches a probability above a
threshold, speculatively dispatch the predicted-next job to a
provider BEFORE the user finishes asking it. If the prediction
matches the actual request, the answer is already produced and the
user-visible latency is dominated by local lookup. If the
prediction misses, the speculative work is discarded (and the
provider is still paid -- the §A1 receipt and a discard-attestation
record the speculative spend transparently).

Why this design is novel
----------------------
Existing speculative computing (CPU branch prediction, autoplete
prefetch) is intra-process and lacks economic settlement. Pluginfer's
contribution:

  "A method for pre-dispatching speculative inference jobs across a
   permissionless decentralised compute mesh based on a model-
   predicted user-action probability, with cryptographic discard-
   attestation when the prediction misses, and idempotent
   substitution when it hits."

Trade-off
---------
False positives cost compute. The router exposes a `precision_floor`
parameter -- the minimum predicted probability before fan-out is
allowed -- so callers can tune for cost vs perceived latency.

This module is INTENTIONALLY storage-free: it produces the dispatch
decision, the caller wires it into the actual mesh dispatcher
(§A13 quorum_inference, or core.providers.Auction).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Predictor abstraction
# ---------------------------------------------------------------------------


@dataclass
class Prediction:
    """One predicted next request."""
    request_payload: Dict[str, Any]
    probability: float                       # in [0, 1]
    reason: str = ""                         # diagnostic / explainability


class Predictor:
    """Caller-supplied prediction function. The default is a stub --
    real implementations plug in PluginferBrain (§5 of INVENTIONS),
    or an LLM-driven user-intent classifier."""

    def __init__(self, fn: Callable[[List[Dict[str, Any]]],
                                    List[Prediction]]):
        self._fn = fn

    def predict(self, history: List[Dict[str, Any]]) -> List[Prediction]:
        """Given a list of recent requests, return predictions for what
        the user is likely to ask next, sorted highest-probability first."""
        out = self._fn(history)
        return sorted(out, key=lambda p: -p.probability)


# ---------------------------------------------------------------------------
# Speculative dispatch ledger
# ---------------------------------------------------------------------------


@dataclass
class SpeculativeJob:
    """A job dispatched ahead of the user actually asking for it."""
    spec_id: str
    payload: Dict[str, Any]
    payload_hash: str
    predicted_probability: float
    dispatched_at_ns: int = 0
    completed_at_ns: int = 0
    output_bytes: Optional[bytes] = None
    cost_paid_usd: float = 0.0
    discarded_at_ns: int = 0
    discard_reason: str = ""

    @property
    def is_pending(self) -> bool:
        return self.completed_at_ns == 0 and self.discarded_at_ns == 0

    @property
    def is_complete(self) -> bool:
        return self.completed_at_ns > 0 and self.discarded_at_ns == 0

    @property
    def is_discarded(self) -> bool:
        return self.discarded_at_ns > 0


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------


def _payload_hash(p: Dict[str, Any]) -> str:
    """Stable hash of a payload for matching purposes. Two callers
    asking the same question land on the same hash."""
    import json
    return hashlib.sha256(
        json.dumps(p, sort_keys=True, default=str).encode()
    ).hexdigest()


@dataclass
class PredictiveRouter:
    """Driver for fan-out of speculative jobs.

    Wire pattern (caller-side):

        router = PredictiveRouter(
            predictor=Predictor(fn=my_brain_predict),
            dispatch=lambda payload: actual_mesh_dispatch(payload),
            precision_floor=0.6,
            max_speculative_inflight=3,
        )

        # Each user turn:
        actual_request = ...
        spec_match = router.match_actual(actual_request)
        if spec_match is not None:
            response = spec_match.output_bytes      # zero-latency
        else:
            response = actual_mesh_dispatch(actual_request)
        router.observe(actual_request)
        # async background:
        router.run_speculation()
    """
    predictor: Predictor
    dispatch: Callable[[Dict[str, Any]], bytes]   # synchronous; tests stub
    precision_floor: float = 0.6
    max_speculative_inflight: int = 2
    history_max: int = 16
    _history: List[Dict[str, Any]] = field(default_factory=list)
    _inflight: Dict[str, SpeculativeJob] = field(default_factory=dict)
    _completed: Dict[str, SpeculativeJob] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def observe(self, actual: Dict[str, Any]) -> None:
        """Record an actual user request in history."""
        self._history.append(actual)
        if len(self._history) > self.history_max:
            self._history = self._history[-self.history_max:]

    # ------------------------------------------------------------------

    def match_actual(
        self, actual: Dict[str, Any]
    ) -> Optional[SpeculativeJob]:
        """If a speculative job matches the actual request and has
        completed, return it (caller can serve from it). Else return
        None.
        """
        h = _payload_hash(actual)
        # Prefer a completed match. Move it out of completed -> consumed.
        comp = self._completed.pop(h, None)
        if comp is not None and comp.is_complete:
            return comp
        # Pending speculation matched -- caller can wait or fall through.
        return None

    # ------------------------------------------------------------------

    def run_speculation(self) -> List[SpeculativeJob]:
        """Fire the predictor; dispatch eligible predictions; return
        the newly-launched specs.

        Eligibility:
          * predicted probability >= precision_floor
          * payload_hash not already inflight
          * inflight count < max_speculative_inflight
        """
        # Discard any entries that are pointlessly old (>= 5 min) so
        # the cache doesn't grow unbounded if the user navigates away.
        now = time.time_ns()
        dead_cutoff = now - 5 * 60 * 1_000_000_000
        for h, job in list(self._completed.items()):
            if job.completed_at_ns and job.completed_at_ns < dead_cutoff:
                self._completed.pop(h, None)

        launched: List[SpeculativeJob] = []
        slots = max(0, self.max_speculative_inflight - len(self._inflight))
        if slots == 0:
            return launched

        preds = self.predictor.predict(self._history)
        for p in preds:
            if slots <= 0:
                break
            if p.probability < self.precision_floor:
                break
            h = _payload_hash(p.request_payload)
            if h in self._inflight or h in self._completed:
                continue
            spec = SpeculativeJob(
                spec_id=h[:16],
                payload=p.request_payload,
                payload_hash=h,
                predicted_probability=p.probability,
                dispatched_at_ns=time.time_ns(),
            )
            self._inflight[h] = spec
            launched.append(spec)
            slots -= 1

            # Synchronous dispatch in this implementation; production
            # wires this to an async pool.
            try:
                out = self.dispatch(p.request_payload)
            except Exception as e:
                spec.discarded_at_ns = time.time_ns()
                spec.discard_reason = f"dispatch error: {e!r}"
                self._inflight.pop(h, None)
                continue
            spec.output_bytes = out
            spec.completed_at_ns = time.time_ns()
            self._inflight.pop(h, None)
            self._completed[h] = spec

        return launched

    # ------------------------------------------------------------------

    def discard_stale(self, max_age_seconds: int = 60) -> int:
        """Drop completed speculations older than `max_age_seconds`.
        Each drop produces a `discard_reason` audit trail (the
        cryptographic-attestation shape lives in §A1 receipts).
        Returns count discarded."""
        cutoff = time.time_ns() - max_age_seconds * 1_000_000_000
        discarded = 0
        for h, job in list(self._completed.items()):
            if job.completed_at_ns and job.completed_at_ns < cutoff:
                job.discarded_at_ns = time.time_ns()
                job.discard_reason = "stale: max_age exceeded"
                self._completed.pop(h, None)
                discarded += 1
        return discarded

    def stats(self) -> Dict[str, int]:
        return {
            "inflight": len(self._inflight),
            "completed_unconsumed": len(self._completed),
            "history": len(self._history),
        }


__all__ = [
    "Prediction",
    "Predictor",
    "PredictiveRouter",
    "SpeculativeJob",
]
