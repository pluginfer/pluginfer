"""Filum operational decision engine.

When a node receives a *situation* — a job to route, a content
class to gate, a gradient to verify, a peer to promote — it must
make a decision. Centralised AI providers route every decision
through a human operator; on a 10k-node decentralised mesh that
doesn't scale.

This module is Filum's *operational brain*: it encodes the
common decisions the substrate has to make, routes each through
the appropriate subsystem, attaches an auditable trace, and
records the outcome so the network can learn.

Decision categories handled:

* ``RouteInferenceDecision``  — pick a provider (cost / reliability / region)
* ``ElectSunDecision``        — decide if a peer should become a Sun
* ``ContentGateDecision``     — allow / deny / quarantine / rate-limit
* ``GradientAcceptDecision``  — accept / reject a peer-pushed gradient
* ``CapabilityGapDecision``   — pick the next gap to address
* ``CheckpointDecision``      — when to seal a NBGGA shard version
* ``ReceiptIssueDecision``    — when to issue a §D1 receipt

Every decision returns a ``Decision`` with:
* the chosen action
* a confidence score (0..1)
* the rules / signals that fired
* a structured trace ready for §D1 receipt anchoring

This is not a giant LLM-as-controller. It's *deterministic where
it can be deterministic, learned where it must be learned*. The
deterministic parts are the cost-optimal router, safety gate, and
sun_election. The learned parts (capability-gap selection,
gradient-trust) plug in via the recursive_improver and §C2
stability scores.

Filum the model uses this engine via two paths:
1. Direct programmatic call when reasoning about its own behaviour
   (the CLI `filum decide ...` subcommand).
2. As a tool surface inside its own inference pipeline — Filum
   *calls* the decision engine the way a programmer calls library
   functions. Its training data includes worked examples.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------- the universal Decision wrapper ---------------------------------

@dataclass
class Decision:
    kind: str                                     # category
    action: str                                   # human-readable choice
    chosen_id: Optional[str] = None               # provider_id / sun_id / etc.
    confidence: float = 0.0                       # 0..1
    rules_fired: list = field(default_factory=list)
    signals: dict = field(default_factory=dict)
    rationale: str = ""
    ts: float = 0.0
    trace_id: str = ""

    def to_receipt_payload(self) -> dict:
        """Strip to a chain-anchor-friendly dict (no objects, just primitives)."""
        return {
            "kind": self.kind,
            "action": self.action,
            "chosen_id": self.chosen_id,
            "confidence": round(float(self.confidence), 4),
            "rules_fired": list(self.rules_fired),
            "rationale": self.rationale,
            "ts": self.ts,
            "trace_id": self.trace_id,
        }


# ---------- the engine -----------------------------------------------------

class DecisionEngine:
    """Operational decision router for a Filum node.

    Constructed with optional references to existing subsystems. Each
    decision method short-circuits if the relevant subsystem isn't
    bound — this keeps the engine usable for testing one decision
    type at a time.
    """

    def __init__(
        self,
        *,
        safety_gate=None,                         # SafetyGate
        nbgga=None,                               # NonBlockingGlobalAggregator
        sun_election=None,                        # SunElection
        cost_optimizer=None,                      # core.cost_optimizer
        recursive_improver=None,                  # RecursiveImprover
        receipt_log=None,                         # ReceiptLog
    ):
        self.safety = safety_gate
        self.nbgga = nbgga
        self.sun = sun_election
        self.cost = cost_optimizer
        self.improver = recursive_improver
        self.receipt_log = receipt_log
        self._counter = 0
        self._history: list[Decision] = []

    # --- shared helpers --------------------------------------------------

    def _next_trace(self) -> str:
        self._counter += 1
        return f"d{int(time.time() * 1000):x}-{self._counter:04d}"

    def _record(self, d: Decision) -> Decision:
        self._history.append(d)
        if len(self._history) > 1024:
            self._history = self._history[-512:]
        return d

    def history(self) -> list[Decision]:
        return list(self._history)

    # --- 1) content gating ----------------------------------------------

    def gate_content(self, *, pubkey: str, content: str,
                      region: Optional[str] = None) -> Decision:
        """Allow / deny / quarantine / rate-limit. Wraps SafetyGate."""
        ts = time.time()
        if self.safety is None:
            return self._record(Decision(
                kind="content_gate", action="allow",
                confidence=0.0,
                rules_fired=["no-safety-gate-bound"],
                rationale="safety subsystem not bound; default allow",
                ts=ts, trace_id=self._next_trace(),
            ))
        sd = self.safety.check(pubkey, content, region=region)
        return self._record(Decision(
            kind="content_gate",
            action=sd.decision,
            chosen_id=sd.matched_class,
            confidence=1.0 if sd.decision != "allow" else 0.95,
            rules_fired=[sd.reason] if sd.reason else [],
            signals={"region": region, "matched_class": sd.matched_class},
            rationale=sd.reason or "passed all checks",
            ts=ts, trace_id=self._next_trace(),
        ))

    # --- 2) inference routing -------------------------------------------

    def route_inference(
        self, *,
        candidates: list,                         # list of dicts: id, price, reliability, region, energy
        max_price: float,
        min_reliability: float = 0.0,
        prefer_green: bool = False,
        deadline_s: Optional[float] = None,
    ) -> Decision:
        """Pick a provider. Cheapest reliable match wins."""
        ts = time.time()
        eligible = [
            c for c in candidates
            if c.get("price", float("inf")) <= max_price
            and c.get("reliability", 0) >= min_reliability
            and (not prefer_green or c.get("energy") == "green")
        ]
        if not eligible:
            return self._record(Decision(
                kind="route_inference", action="no_match",
                confidence=1.0,
                rules_fired=["no eligible provider"],
                rationale=f"none met max_price={max_price} "
                          f"min_reliability={min_reliability}",
                ts=ts, trace_id=self._next_trace(),
            ))
        # Cheapest first; tiebreak on reliability desc.
        eligible.sort(key=lambda c: (c.get("price", 0.0),
                                     -c.get("reliability", 0.0)))
        best = eligible[0]
        return self._record(Decision(
            kind="route_inference",
            action="route",
            chosen_id=str(best.get("id", "")),
            confidence=min(1.0, best.get("reliability", 0.0)),
            rules_fired=["cheapest_reliable"],
            signals={
                "candidates_total": len(candidates),
                "candidates_eligible": len(eligible),
                "chosen_price": best.get("price"),
                "chosen_reliability": best.get("reliability"),
            },
            rationale=f"picked {best.get('id')} at price "
                      f"{best.get('price')} reliability "
                      f"{best.get('reliability'):.2f}",
            ts=ts, trace_id=self._next_trace(),
        ))

    # --- 3) sun election promotion --------------------------------------

    def should_promote_sun(self, peer, *, current_suns: list) -> Decision:
        """Decide whether to promote a Planet to Sun."""
        ts = time.time()
        s = float(getattr(peer, "stability_score", 0.0))
        promote_thresh = 0.7
        if s < promote_thresh:
            return self._record(Decision(
                kind="elect_sun", action="keep_planet",
                confidence=1.0 - s,
                rules_fired=[f"stability {s:.2f} < {promote_thresh}"],
                rationale=f"peer stability too low",
                ts=ts, trace_id=self._next_trace(),
            ))
        # If we already have enough Suns, only promote if higher than weakest.
        if len(current_suns) >= 3:
            weakest = min(current_suns,
                           key=lambda x: getattr(x, "stability_score", 0.0))
            if s <= getattr(weakest, "stability_score", 0.0):
                return self._record(Decision(
                    kind="elect_sun", action="keep_planet",
                    confidence=0.7,
                    rules_fired=["sun set already full + not better than weakest"],
                    rationale=f"S {s:.2f} not greater than weakest sun's "
                              f"{getattr(weakest, 'stability_score', 0):.2f}",
                    ts=ts, trace_id=self._next_trace(),
                ))
        return self._record(Decision(
            kind="elect_sun", action="promote",
            chosen_id=getattr(peer, "node_id", None),
            confidence=s,
            rules_fired=[f"stability {s:.2f} >= {promote_thresh}"],
            rationale="promote to Sun based on stability",
            ts=ts, trace_id=self._next_trace(),
        ))

    # --- 4) gradient acceptance -----------------------------------------

    def accept_gradient(self, *, grain, current_version: int,
                         tau: float = 200.0) -> Decision:
        """Decide whether NBGGA should accept this grain."""
        ts = time.time()
        staleness = grain.staleness(current_version)
        decay = grain.decay_weight(current_version, tau=tau)
        rules: list[str] = []
        if staleness > 10 * tau:
            return self._record(Decision(
                kind="grad_accept", action="reject",
                confidence=1.0,
                rules_fired=[f"staleness {staleness} > 10 tau"],
                rationale="grain too stale; would dilute current model",
                signals={"staleness": staleness, "decay": decay},
                ts=ts, trace_id=self._next_trace(),
            ))
        rules.append(f"staleness {staleness} acceptable")
        if grain.meta.pressure_at_birth > 0.95:
            rules.append("contributor was at very high pressure")
        return self._record(Decision(
            kind="grad_accept", action="accept",
            confidence=decay * (1.0 - grain.meta.pressure_at_birth),
            rules_fired=rules,
            signals={"staleness": staleness, "decay": decay,
                      "contributor_pressure": grain.meta.pressure_at_birth},
            rationale=f"accept with weight {decay:.3f} * "
                      f"{1.0 - grain.meta.pressure_at_birth:.2f}",
            ts=ts, trace_id=self._next_trace(),
        ))

    # --- 5) capability gap pick -----------------------------------------

    def pick_next_capability_gap(self) -> Decision:
        """Ask the recursive improver for the next-largest gap."""
        ts = time.time()
        if self.improver is None:
            return self._record(Decision(
                kind="capability_gap", action="skip",
                rules_fired=["no recursive_improver bound"],
                rationale="no improver configured",
                ts=ts, trace_id=self._next_trace(),
            ))
        gap = self.improver.identify_gap()
        if gap is None:
            return self._record(Decision(
                kind="capability_gap", action="skip",
                rules_fired=["no unaddressed gap"],
                rationale="all current clusters have adapters",
                ts=ts, trace_id=self._next_trace(),
            ))
        return self._record(Decision(
            kind="capability_gap", action="train_adapter",
            chosen_id=gap.cluster_id,
            confidence=min(1.0, gap.weight / 100.0),
            rules_fired=[f"gap weight {gap.weight:.1f}"],
            signals={"label": gap.label, "samples": gap.sample_count,
                      "entropy": gap.avg_teacher_entropy},
            rationale=f"largest gap is '{gap.label}' "
                      f"(weight {gap.weight:.1f})",
            ts=ts, trace_id=self._next_trace(),
        ))

    # --- 6) checkpoint timing -------------------------------------------

    def should_checkpoint(self, *, shard_id: str,
                           min_norm: float = 1e-3) -> Decision:
        """Should NBGGA bump the version for this shard now?"""
        ts = time.time()
        if self.nbgga is None:
            return self._record(Decision(
                kind="checkpoint", action="skip",
                rules_fired=["no nbgga bound"],
                ts=ts, trace_id=self._next_trace(),
            ))
        # Probe internal state (best-effort).
        state = self.nbgga._state.get(shard_id)
        if state is None:
            return self._record(Decision(
                kind="checkpoint", action="skip",
                rules_fired=["unknown shard"],
                ts=ts, trace_id=self._next_trace(),
            ))
        norm = state.pending_norm()
        if norm >= min_norm:
            return self._record(Decision(
                kind="checkpoint", action="seal",
                chosen_id=shard_id,
                confidence=min(1.0, norm / max(min_norm, 1e-9)),
                rules_fired=[f"pending_norm {norm:.4f} >= {min_norm}"],
                rationale=f"shard {shard_id} accumulated enough delta",
                ts=ts, trace_id=self._next_trace(),
            ))
        return self._record(Decision(
            kind="checkpoint", action="wait",
            confidence=norm / max(min_norm, 1e-9),
            rules_fired=[f"pending_norm {norm:.4f} < {min_norm}"],
            rationale="not enough accumulated delta yet",
            ts=ts, trace_id=self._next_trace(),
        ))

    # --- 7) receipt issuance --------------------------------------------

    def should_issue_receipt(self, *, content_class: str,
                              user_requested: bool = False) -> Decision:
        """Decide whether this inference deserves a §D1 receipt now.

        Production may issue receipts on every inference; for cost-
        sensitive deployments we issue on (a) user request, (b) any
        non-'general' content class, (c) every Nth inference per node.
        """
        ts = time.time()
        if user_requested:
            return self._record(Decision(
                kind="issue_receipt", action="issue",
                confidence=1.0,
                rules_fired=["user requested"],
                ts=ts, trace_id=self._next_trace(),
            ))
        if content_class != "general":
            return self._record(Decision(
                kind="issue_receipt", action="issue",
                confidence=0.9,
                rules_fired=[f"content_class={content_class}"],
                rationale="non-general content always receipted",
                ts=ts, trace_id=self._next_trace(),
            ))
        return self._record(Decision(
            kind="issue_receipt", action="defer",
            confidence=0.6,
            rules_fired=["general content; sample N=100"],
            rationale="defer to per-100 sampling policy",
            ts=ts, trace_id=self._next_trace(),
        ))
