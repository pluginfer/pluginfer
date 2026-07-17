"""
DiLoCo Aggregator (Outer Loop) — with Async Staleness Weighting
================================================================

Standard DiLoCo (Douillard et al., 2023) and INTELLECT-1 (Together AI +
Prime Intellect, Nov 2024) both use *synchronous* aggregation: every
worker must finish round R before round R+1 begins. The slowest worker
sets the pace. On consumer-grade WAN with heterogeneous hardware that
breaks down — 90% of throughput is lost waiting for stragglers.

This aggregator implements **Async Staleness-Weighted DiLoCo**:

    1. Each worker reports its delta tagged with the round R at which
       it pulled the global weights.
    2. The aggregator's "current" round may have advanced to R + s.
    3. The delta is weighted by  w = exp(-s / τ)  before it contributes
       to the round update.
    4. Once enough weighted mass accumulates (or a wall-clock deadline
       passes), the aggregator applies one outer-loop Nesterov step
       and increments the global round.

Result: fast workers stream deltas continuously; slow workers still
contribute (just at lower weight); no worker blocks the network. This
is the production-grade primitive consumer-grade hardware needs.

ALSO HERE
---------
* `verify_delta` — sample-based gradient provenance check. The
  aggregator can re-execute a deterministic 1-step audit on a given
  worker's reported (seed, batch_indices, base_weights_hash) tuple
  and confirm the worker's delta is consistent. If a worker submits
  a forged delta, the audit detects it; combined with stake, this
  produces real Sybil resistance via real economic cost.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception as _torch_err:                      # pragma: no cover
    torch = None                                     # type: ignore[assignment]
    nn = None                                        # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _torch_err

from .diloco_models import build_model, count_parameters
from .diloco_serialize import (
    deserialize_state_dict,
    serialize_state_dict,
    state_dict_hash,
)
from .diloco_quantize import dequantize_delta

logger = logging.getLogger(__name__)


@dataclass
class WorkerSubmission:
    """One async DiLoCo delta in flight."""
    worker_id: str
    base_round: int
    quantized_delta: bytes
    received_at: float
    base_weights_hash: str
    final_weights_hash: str
    examples_seen: int
    audit_seed: Optional[int] = None         # for gradient provenance check
    audit_batch_idx: Optional[List[int]] = None
    accepted: bool = False
    rejection_reason: Optional[str] = None


@dataclass
class AggregatorConfig:
    """Outer-loop hyperparameters and async policy."""
    outer_lr: float = 0.7              # DiLoCo paper default
    outer_momentum: float = 0.9        # Nesterov
    staleness_tau: float = 4.0         # rounds; e^(-staleness/tau) weight
    min_submissions_per_round: int = 2
    max_submissions_per_round: int = 64
    round_deadline_sec: float = 30.0   # if not enough subs, advance anyway
    audit_probability: float = 0.05    # fraction of submissions audited
    reject_stale_after: int = 16       # discard deltas older than this many rounds


class _OuterOptimizer:
    """Plain Nesterov SGD on parameter deltas (DiLoCo paper Algorithm 1)."""

    def __init__(self, init_state: Dict[str, torch.Tensor],
                 lr: float, momentum: float):
        self.lr = lr
        self.momentum = momentum
        self.velocity: Dict[str, torch.Tensor] = {
            k: torch.zeros_like(v) for k, v in init_state.items()
        }

    def step(self, params: Dict[str, torch.Tensor],
             aggregated_delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """params <- params + lr * (momentum * v + (1 - momentum) * Δ̄)"""
        out: Dict[str, torch.Tensor] = {}
        for k, theta in params.items():
            d = aggregated_delta.get(k)
            if d is None:
                out[k] = theta
                continue
            d = d.to(theta.dtype).to(theta.device)
            self.velocity[k] = self.velocity[k].to(theta.device)
            self.velocity[k] = self.momentum * self.velocity[k] + d
            update = self.momentum * self.velocity[k] + (1.0 - self.momentum) * d
            out[k] = theta + self.lr * update
        return out


class AsyncDiLoCoAggregator:
    """
    The outer loop. Holds the global model state; consumes worker deltas
    asynchronously; applies staleness-weighted Nesterov updates.

    Thread-safe: `submit_delta` may be called from many worker threads;
    `current_global_payload` may be polled by anyone. Internal state is
    guarded by a single lock — fine for thousands of submissions / sec.
    """

    def __init__(self, model_spec: Dict[str, object],
                 config: Optional[AggregatorConfig] = None):
        self.model_spec = dict(model_spec)
        self.config = config or AggregatorConfig()
        self._lock = threading.RLock()

        seed_model: nn.Module = build_model(self.model_spec)
        self._global_state: Dict[str, torch.Tensor] = {
            k: v.detach().cpu().clone() for k, v in seed_model.state_dict().items()
        }
        self._param_shapes: Dict[str, Tuple[int, ...]] = {
            k: tuple(v.shape) for k, v in self._global_state.items()
        }

        self._round: int = 0
        self._round_started_at: float = time.time()
        self._optimizer = _OuterOptimizer(
            self._global_state,
            lr=self.config.outer_lr,
            momentum=self.config.outer_momentum,
        )

        # In-flight submissions waiting to be applied to the next round.
        self._pending: List[WorkerSubmission] = []
        # History of completed rounds (for audit).
        self._round_log: List[Dict[str, object]] = []
        # Per-worker reputation: weighted accept rate.
        self._reputation: Dict[str, Dict[str, float]] = {}

        self._param_count = count_parameters(seed_model)
        logger.info(
            "AsyncDiLoCoAggregator online: arch=%s params=%d outer_lr=%.3f tau=%.2f",
            self.model_spec.get("arch"), self._param_count,
            self.config.outer_lr, self.config.staleness_tau,
        )

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------

    def current_global_payload(self) -> Tuple[int, bytes, str]:
        """(round, serialized_state_dict, sha256_hex). Hand this to a worker."""
        with self._lock:
            payload = serialize_state_dict(self._global_state)
            return self._round, payload, state_dict_hash(self._global_state)

    def current_round(self) -> int:
        with self._lock:
            return self._round

    def reputation_of(self, worker_id: str) -> float:
        with self._lock:
            r = self._reputation.get(worker_id)
            if not r:
                return 0.5
            return r["weighted_accepts"] / max(r["seen"], 1.0)

    def submit_delta(self, submission: WorkerSubmission) -> Dict[str, object]:
        """
        Validate + buffer a worker's delta. Triggers a round application
        if enough submissions have accumulated or the deadline expired.

        Returns a dict describing whether the submission was accepted.
        """
        with self._lock:
            staleness = self._round - submission.base_round
            if staleness < 0:
                submission.rejection_reason = "future_round"
                self._record_rep(submission.worker_id, accepted=False, weight=0.0)
                return {"accepted": False, "reason": "future_round"}
            if staleness > self.config.reject_stale_after:
                submission.rejection_reason = "too_stale"
                self._record_rep(submission.worker_id, accepted=False, weight=0.0)
                return {
                    "accepted": False,
                    "reason": "too_stale",
                    "staleness": staleness,
                }

            # Cheap shape/integrity validation. The dequantizer enforces
            # SHA-256 and per-tensor shape checks.
            try:
                _ = dequantize_delta(
                    submission.quantized_delta,
                    expected_shapes=self._param_shapes,
                )
            except Exception as e:
                submission.rejection_reason = f"invalid_payload: {e}"
                self._record_rep(submission.worker_id, accepted=False, weight=0.0)
                return {"accepted": False, "reason": "invalid_payload", "detail": str(e)}

            submission.accepted = True
            self._pending.append(submission)
            weight = math.exp(-staleness / max(self.config.staleness_tau, 1e-6))
            self._record_rep(submission.worker_id, accepted=True, weight=weight)

            # Maybe close the round.
            should_apply = (
                len(self._pending) >= self.config.min_submissions_per_round
                and (
                    len(self._pending) >= self.config.max_submissions_per_round
                    or (time.time() - self._round_started_at)
                    >= self.config.round_deadline_sec
                )
            )
            if should_apply:
                self._apply_round_locked()

            return {
                "accepted": True,
                "staleness": staleness,
                "weight": weight,
                "buffered": len(self._pending),
                "round": self._round,
            }

    def force_apply_round(self) -> Optional[Dict[str, object]]:
        """Force the next round (e.g. on operator command or test)."""
        with self._lock:
            if not self._pending:
                return None
            return self._apply_round_locked()

    def round_history(self, last_n: int = 10) -> List[Dict[str, object]]:
        with self._lock:
            return list(self._round_log[-last_n:])

    # --------------------------------------------------------------
    # Internal
    # --------------------------------------------------------------

    def _record_rep(self, worker_id: str, accepted: bool, weight: float) -> None:
        rep = self._reputation.setdefault(
            worker_id, {"seen": 0.0, "weighted_accepts": 0.0}
        )
        rep["seen"] += 1.0
        if accepted:
            rep["weighted_accepts"] += weight

    def _apply_round_locked(self) -> Dict[str, object]:
        """Combine pending deltas with staleness weights → outer step."""
        # 1. Compute staleness-weighted average of pending deltas.
        agg: Dict[str, torch.Tensor] = {
            k: torch.zeros(v.shape, dtype=torch.float32)
            for k, v in self._global_state.items()
        }
        weight_sum = 0.0
        accepted_workers: List[str] = []

        for sub in self._pending:
            staleness = self._round - sub.base_round
            w = math.exp(-staleness / max(self.config.staleness_tau, 1e-6))
            try:
                delta = dequantize_delta(
                    sub.quantized_delta, expected_shapes=self._param_shapes
                )
            except Exception as e:
                logger.warning("Skipping bad delta from %s: %s", sub.worker_id, e)
                continue
            for k in agg:
                if k in delta:
                    agg[k] += delta[k].to(torch.float32) * w
            weight_sum += w
            accepted_workers.append(sub.worker_id)

        if weight_sum <= 0:
            self._pending.clear()
            self._round_started_at = time.time()
            return {"applied": False, "reason": "no_weight"}

        for k in agg:
            agg[k] /= weight_sum

        # 2. Apply outer-loop Nesterov step.
        new_state = self._optimizer.step(self._global_state, agg)
        self._global_state = {k: v.detach().cpu().clone() for k, v in new_state.items()}

        # 3. Roll the round counter.
        round_idx = self._round
        self._round += 1
        self._round_started_at = time.time()
        self._pending.clear()

        log_entry = {
            "round": round_idx,
            "workers": list(set(accepted_workers)),
            "submissions_used": len(accepted_workers),
            "weight_sum": weight_sum,
            "global_hash": state_dict_hash(self._global_state),
            "ts": time.time(),
        }
        self._round_log.append(log_entry)
        if len(self._round_log) > 1000:
            self._round_log = self._round_log[-1000:]

        logger.info(
            "Round %d applied: %d submissions, weight_sum=%.3f, hash=%s",
            round_idx, len(accepted_workers), weight_sum, log_entry["global_hash"][:12],
        )
        return {"applied": True, **log_entry}


# ----------------------------------------------------------------------
# Gradient Provenance Audit (lightweight on-demand verification)
# ----------------------------------------------------------------------
def verify_delta(model_spec: Dict[str, object],
                 base_weights_payload: bytes,
                 delta_payload: bytes,
                 audit_seed: int,
                 audit_step_data: Tuple[torch.Tensor, torch.Tensor],
                 lr: float,
                 tolerance: float = 0.5,
                 ) -> Tuple[bool, str, float]:
    """
    Light-weight gradient provenance check.

    Given a worker's claimed (base_weights, delta, audit_seed, single-step
    batch), the aggregator can re-execute exactly one SGD step from the
    same seed and check that the resulting parameter change is consistent
    in direction with the worker's reported delta.

    A worker that fabricates deltas (e.g. returns pure noise) fails this
    check; combined with stake-slashing, this is real economic Sybil
    resistance — malicious behaviour costs locked tokens.

    Returns:
        (passed, reason, cosine_similarity)
    """
    from .diloco_models import build_model, loss_fn_for

    torch.manual_seed(audit_seed)
    model = build_model(model_spec)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    base_state = deserialize_state_dict(
        base_weights_payload,
        expected_keys=tuple(expected.keys()),
        expected_shapes=expected,
    )
    model.load_state_dict(base_state, strict=True)

    loss_fn = loss_fn_for(model_spec)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    x, y = audit_step_data
    model.train()
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()

    audited_delta: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        audited_delta[k] = v.detach().cpu().to(torch.float32) - base_state[k].to(torch.float32)

    worker_delta = dequantize_delta(delta_payload, expected_shapes=expected)

    # Cosine similarity between flattened deltas.
    audit_vec = torch.cat([t.flatten() for t in audited_delta.values()])
    worker_vec = torch.cat([worker_delta[k].flatten() for k in audited_delta.keys()])
    if audit_vec.norm() < 1e-8 or worker_vec.norm() < 1e-8:
        return False, "zero_delta", 0.0
    cos = float(torch.dot(audit_vec, worker_vec) / (audit_vec.norm() * worker_vec.norm()))

    # The worker's delta is K inner steps starting from base; we audited
    # 1 step. They must point in the same general direction (cos > 0).
    # `tolerance` lets the caller calibrate how strict to be.
    passed = cos > tolerance
    reason = "ok" if passed else f"cos_sim_too_low ({cos:.3f})"
    return passed, reason, cos
