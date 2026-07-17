"""§F4 Adversarial Quality Verification — defense against fake-work Sybils.

The §E1 compute-as-currency primitive is vulnerable to a specific
attack: a provider claims credit for N TFLOP-hours of training,
submits a §C grain that looks plausible (signed under their key,
right shape, right version), but the gradient is *random noise* or
*recycled from an earlier real grain*. They earn the §E1 debt-
forgiveness or §C7 payment; the model takes a tiny gradient hit
that's unverifiable in isolation.

This is the single highest-impact unaddressed attack against the
mesh. Without verification, the cost of Sybiling at scale is
nothing; the expected value of fake-work is positive.

The defense is *cheap probabilistic re-execution*:

1. For every grain that lands in the §C5 NBGGA, with probability
   ``audit_rate`` (default 0.01 = 1%), the aggregator schedules a
   *re-execution* of the *same* training step on a DIFFERENT
   provider chosen uniformly from the high-stability Sun set.
2. The re-executor receives the same (model_shard_id, version_v,
   optimizer_seed, micro_batch_id) and computes its own gradient.
3. The re-executor's grain is compared against the original via
   *cosine similarity* (NOT exact equality — floating-point
   non-determinism makes that infeasible).
4. If cosine similarity < ``challenge_threshold`` (default 0.85),
   the original is *challenged*. The §C8 attestation score of the
   original provider drops; if the gap exceeds ``slash_threshold``,
   their §E1 debt is increased by a punitive multiple (default 5×
   the disputed work) — a bonded penalty.

The economics: 1% audit rate × 5× punitive multiplier = 5% expected
penalty per fake grain. Combined with the §C8 attestation score
decay (cold-start half-rate enforced harder), the expected value
of fake-work is *negative*. Sybil attack is no longer profitable.

Honest engineering note: cosine similarity at threshold 0.85 has
false-positive risk on numerically-unstable batches (e.g. very
small gradients near convergence). Two safeguards:

* The re-execution uses the same RNG seed, so deterministic ops
  produce identical results modulo non-determinism in CUDA reductions
  (~0.99 cosine typical).
* Three consecutive sub-threshold matches before slashing — single-
  batch noise doesn't slash a real provider.

novel claim §F4 (drafted in the design notes): a method of
verifying neural-network training contributions in a decentralised
mesh, comprising: (a) probabilistic re-execution at a configurable
audit rate of training steps on independently-selected provider
nodes; (b) gradient-direction comparison via cosine similarity
rather than bytewise equality, accommodating floating-point non-
determinism; (c) bonded slashing applied through a debt-multiplier
on §E1 compute-currency obligations rather than direct fund
seizure; (d) consecutive-failure threshold to suppress single-
batch numerical noise from triggering false positives.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------- audit ledger entries -------------------------------------------

@dataclass
class AuditOutcome:
    """One audit comparison's result."""
    grain_id: str
    original_provider: str
    auditor_provider: str
    cosine_similarity: float
    timestamp: float
    challenged: bool                     # True iff sim < challenge_threshold


@dataclass
class ProviderAuditRecord:
    """Per-provider audit history."""
    pubkey: str
    audits_total: int = 0
    audits_passed: int = 0
    audits_challenged: int = 0
    consecutive_challenges: int = 0
    cumulative_slash_tflop_hr: float = 0.0
    last_audit_ts: float = 0.0


@dataclass
class QualityVerificationConfig:
    audit_rate: float = 0.01                  # P(audit) per grain
    challenge_threshold: float = 0.85         # cosine below = challenged
    slash_threshold: float = 0.50             # cosine below = slash
    consecutive_for_slash: int = 3            # # of slashes in a row needed
    slash_multiplier: float = 5.0             # punitive: 5× the disputed work
    audit_pool_min_size: int = 3              # need >= this many candidates


# ---------- the verifier ---------------------------------------------------

class AdversarialQualityVerifier:
    """The §F4 audit primitive.

    Public API::

        verifier = AdversarialQualityVerifier(
            schedule_reexec_fn=task_router.schedule_reexec_for,
            slash_fn=compute_currency.add_punitive_debt,
        )
        verifier.maybe_audit(grain)                # call after NBGGA.feed()
        ...
        verifier.compare_outcome(grain_id, auditor_grain)   # later

    Constructor takes two callbacks so the verifier doesn't reach
    into transport / debt modules directly:

    * ``schedule_reexec_fn(grain, auditor_pubkey)`` — caller's
      transport ships the audit RPC.
    * ``slash_fn(provider_pubkey, tflop_hr_to_add)`` — caller's
      §E1 ledger applies the punitive debt.
    """

    def __init__(
        self,
        config: QualityVerificationConfig = QualityVerificationConfig(),
        schedule_reexec_fn: Optional[Callable] = None,
        slash_fn: Optional[Callable[[str, float], None]] = None,
        sun_pool_fn: Optional[Callable[[], list]] = None,
    ):
        self.cfg = config
        self._schedule_reexec = schedule_reexec_fn
        self._slash = slash_fn
        self._sun_pool = sun_pool_fn or (lambda: [])
        self._records: dict[str, ProviderAuditRecord] = {}
        self._pending: dict[str, dict] = {}     # grain_id -> {grain, auditor, ts}
        self._outcomes: list[AuditOutcome] = []
        self._lock = threading.RLock()
        self._rng = random.Random()

    # --- audit dispatch --------------------------------------------------

    def maybe_audit(self, grain) -> bool:
        """Call once per grain accepted by NBGGA. Returns True iff this
        grain was selected for audit and an auditor was scheduled."""
        if self._rng.random() >= self.cfg.audit_rate:
            return False
        # Pick an auditor from the sun pool, excluding the original.
        candidates = [
            p for p in self._sun_pool()
            if getattr(p, "node_id", "") != grain.meta.contributor_id
        ]
        if len(candidates) < self.cfg.audit_pool_min_size - 1:
            # Not enough independent auditors yet; skip.
            return False
        auditor = self._rng.choice(candidates)
        with self._lock:
            self._pending[grain.meta.grain_id] = {
                "grain": grain,
                "auditor_id": getattr(auditor, "node_id", "unknown"),
                "scheduled_ts": time.time(),
            }
        if self._schedule_reexec is not None:
            try:
                self._schedule_reexec(grain, auditor)
            except Exception as e:
                logger.warning("reexec schedule failed: %s", e)
        return True

    # --- audit comparison ------------------------------------------------

    def compare_outcome(self, grain_id: str, auditor_grain) -> Optional[AuditOutcome]:
        """Caller invokes when the auditor's re-executed grain arrives.

        Returns the AuditOutcome (and persists it) iff the original
        was pending audit; else None.
        """
        with self._lock:
            pending = self._pending.pop(grain_id, None)
        if pending is None:
            return None
        original = pending["grain"]
        cos = _cosine_similarity_grain(original, auditor_grain)
        challenged = cos < self.cfg.challenge_threshold
        outcome = AuditOutcome(
            grain_id=grain_id,
            original_provider=original.meta.contributor_id,
            auditor_provider=pending["auditor_id"],
            cosine_similarity=float(cos),
            timestamp=time.time(),
            challenged=challenged,
        )
        with self._lock:
            self._outcomes.append(outcome)
            rec = self._records.setdefault(
                original.meta.contributor_id,
                ProviderAuditRecord(pubkey=original.meta.contributor_id),
            )
            rec.audits_total += 1
            rec.last_audit_ts = outcome.timestamp
            if challenged:
                rec.audits_challenged += 1
                rec.consecutive_challenges += 1
            else:
                rec.audits_passed += 1
                rec.consecutive_challenges = 0
            self._maybe_slash_unlocked(rec, original, cos)
        return outcome

    def _maybe_slash_unlocked(self, rec: ProviderAuditRecord, grain,
                                cos: float) -> None:
        """Caller already holds the lock."""
        if cos >= self.cfg.slash_threshold:
            return
        if rec.consecutive_challenges < self.cfg.consecutive_for_slash:
            return
        # Slash. Use the grain's claimed TFLOP-hr (heuristic: matrix size).
        m = max(1, grain.meta.shape_m)
        n = max(1, grain.meta.shape_n)
        # 1 TFLOP-hr per 1G params — rough but stable across model sizes.
        claimed_tflop_hr = (m * n) / 1e9
        slash_amount = claimed_tflop_hr * self.cfg.slash_multiplier
        rec.cumulative_slash_tflop_hr += slash_amount
        rec.consecutive_challenges = 0
        if self._slash is not None:
            try:
                self._slash(grain.meta.contributor_id, slash_amount)
            except Exception as e:
                logger.warning("slash callback failed: %s", e)
        logger.warning(
            "slashed provider %s for %.4f TFLOP-hr (cos=%.3f, claimed=%.4f)",
            grain.meta.contributor_id, slash_amount, cos, claimed_tflop_hr,
        )

    # --- inspection -----------------------------------------------------

    def record_for(self, pubkey: str) -> Optional[ProviderAuditRecord]:
        with self._lock:
            return self._records.get(pubkey)

    def all_records(self) -> list[ProviderAuditRecord]:
        with self._lock:
            return list(self._records.values())

    def recent_outcomes(self, n: int = 100) -> list[AuditOutcome]:
        with self._lock:
            return list(self._outcomes[-n:])

    def stats(self) -> dict:
        with self._lock:
            n = len(self._records)
            challenged = sum(r.audits_challenged for r in self._records.values())
            total = sum(r.audits_total for r in self._records.values())
            slash = sum(r.cumulative_slash_tflop_hr
                         for r in self._records.values())
            return {
                "n_providers_audited": n,
                "audits_total": total,
                "audits_challenged": challenged,
                "challenge_rate":
                    challenged / total if total > 0 else 0.0,
                "cumulative_slash_tflop_hr": slash,
                "pending_audits": len(self._pending),
            }


# ---------- cosine similarity over grains ---------------------------------

def _cosine_similarity_grain(a, b) -> float:
    """Cosine similarity of two grains' gradient bytes.

    Falls back to 1.0 when grains are bytewise identical (saves the
    numpy dot for the no-op case).
    """
    if a.grad_bytes == b.grad_bytes:
        return 1.0
    try:
        import numpy as np
        ax = np.frombuffer(a.grad_bytes, dtype="<f4")
        bx = np.frombuffer(b.grad_bytes, dtype="<f4")
        if ax.size == 0 or bx.size == 0 or ax.size != bx.size:
            return 0.0
        dot = float(np.dot(ax, bx))
        na = float(np.linalg.norm(ax))
        nb = float(np.linalg.norm(bx))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
    except Exception as e:
        logger.warning("cosine similarity failed: %s", e)
        return 0.0
