"""
Pluginfer Brain — On-Device Decision Agent
==========================================
Every Pluginfer node ships with a small neural decision agent that
handles operational intelligence locally:

    * Should I accept this incoming task?
    * Should I throttle right now (overheating, gaming, low battery)?
    * What fee to bid on a transaction?
    * Is this peer trustworthy enough for a DiLoCo round?
    * When will my owner come back to use the machine?

The kicker: **this brain is itself trained via the network's own DiLoCo
infrastructure.** Each node logs (state, action, reward) tuples to a
local replay buffer. Periodically, every node spends a few inner-loop
steps fitting the brain on its own experience. The aggregator combines
those gradient deltas. Over time, the brain learns from the collective
experience of every machine in the network — millions of node-hours of
operational data nobody else has.

This is the flywheel: the network's first product is the network's own
intelligence, and forks can't replicate it without acquiring the same
collective experience.

Architecture
------------
* **Backbone**: small MLP — runs in <1ms on any device, including phones.
* **Heads**: multi-task Q-value heads (one per decision class).
* **Bootstrap policy**: hand-engineered heuristic that the brain
  improves on. Inference always works even with random init weights.
* **Confidence gating**: brain only overrides the heuristic when its
  confidence (softmax margin) exceeds threshold. Prevents bad early-
  training models from making catastrophic decisions.
* **DiLoCo integration**: `pack_for_diloco()` returns a model_spec +
  current weights compatible with `core/diloco_aggregator.py`.

Decision classes (v1 — extensible)
----------------------------------
    DECISION_ACCEPT_JOB     : {accept, defer, reject}
    DECISION_THROTTLE       : {full_speed, half_speed, pause}
    DECISION_FEE_BID        : 5 discrete bid tiers
    DECISION_PEER_TRUST     : {trust, audit, drop}

State features (v1)
-------------------
A 24-dim vector. All features are normalized to [0, 1] or standardized.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception as _torch_err:                                  # pragma: no cover
    # Soft dependency: chain/wallet/payments must work without torch.
    # PluginferBrain construction will raise a clear error; everything
    # else (NodeContext, heuristic_action) stays usable.
    torch = None                                                 # type: ignore[assignment]
    nn = None                                                    # type: ignore[assignment]
    F = None                                                     # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _torch_err

logger = logging.getLogger(__name__)


# Decision class IDs.
DECISION_ACCEPT_JOB = 0
DECISION_THROTTLE = 1
DECISION_FEE_BID = 2
DECISION_PEER_TRUST = 3
DECISIONS = ("accept_job", "throttle", "fee_bid", "peer_trust")
ACTIONS_PER_DECISION = (3, 3, 5, 3)

STATE_DIM = 24


# --------------------------------------------------------------------------
# Backbone — only defined when torch is available.
# --------------------------------------------------------------------------
if _TORCH_AVAILABLE:
    class BrainNet(nn.Module):
        """24 -> 64 -> 64 -> {3, 3, 5, 3} multi-head Q-net. ~7k params."""

        def __init__(self, state_dim: int = STATE_DIM, hidden: int = 64):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
            )
            self.heads = nn.ModuleList(
                [nn.Linear(hidden, n) for n in ACTIONS_PER_DECISION]
            )

        def forward(self, x):
            h = self.trunk(x)
            return [head(h) for head in self.heads]

        def q_values(self, state, decision_id: int):
            return self.heads[decision_id](self.trunk(state))
else:
    class BrainNet:                                              # type: ignore[no-redef]
        """Stub BrainNet — raises when instantiated without torch."""
        def __init__(self, *_, **__):
            raise NotImplementedError(
                "PluginferBrain requires torch. Install: pip install torch. "
                f"Original import error: {_TORCH_IMPORT_ERROR!r}"
            )


# --------------------------------------------------------------------------
# Featurizer: raw NodeContext -> 24-dim float tensor
# --------------------------------------------------------------------------
@dataclass
class NodeContext:
    """The state the brain sees. All callers populate from real telemetry."""
    cpu_pct: float = 0.0          # 0-100
    ram_pct: float = 0.0          # 0-100
    gpu_pct: float = 0.0          # 0-100
    gpu_temp_c: float = 50.0      # °C
    battery_pct: float = 100.0    # 0-100; 100 if AC-powered
    on_ac_power: bool = True
    is_gaming: bool = False
    network_mbps: float = 100.0   # measured down/up bandwidth
    peer_count: int = 0
    queue_depth: int = 0
    pending_tx_count: int = 0
    electricity_usd_per_kwh: float = 0.12
    plg_usd_price: float = 0.10
    hour_of_day: int = 12         # 0-23
    minute_of_hour: int = 0       # 0-59
    weekday: int = 0              # 0-6
    reputation: float = 0.5       # 0-1
    recent_audit_pass_rate: float = 1.0
    avg_round_latency_s: float = 1.0
    incoming_payload_bytes: int = 0
    incoming_peer_reputation: float = 0.5
    incoming_peer_continent_match: int = 1
    last_action_outcome: float = 0.0   # -1, 0, +1
    minutes_since_user_active: float = 60.0


def featurize(ctx: NodeContext):
    """Project NodeContext into a normalized 24-dim feature vector. Requires torch."""
    if not _TORCH_AVAILABLE:
        raise NotImplementedError(
            "featurize() requires torch. Install: pip install torch. "
            f"Original import error: {_TORCH_IMPORT_ERROR!r}"
        )
    bat = ctx.battery_pct / 100.0
    return torch.tensor([
        ctx.cpu_pct / 100.0,
        ctx.ram_pct / 100.0,
        ctx.gpu_pct / 100.0,
        max(0.0, min(1.0, (ctx.gpu_temp_c - 30.0) / 60.0)),       # 30..90 -> 0..1
        bat,
        1.0 if ctx.on_ac_power else 0.0,
        1.0 if ctx.is_gaming else 0.0,
        max(0.0, min(1.0, ctx.network_mbps / 1000.0)),             # gigabit cap
        max(0.0, min(1.0, ctx.peer_count / 200.0)),
        max(0.0, min(1.0, ctx.queue_depth / 50.0)),
        max(0.0, min(1.0, ctx.pending_tx_count / 1000.0)),
        max(0.0, min(1.0, ctx.electricity_usd_per_kwh / 0.5)),     # 50c/kWh cap
        max(0.0, min(1.0, ctx.plg_usd_price / 1.0)),               # $1 cap
        ctx.hour_of_day / 23.0,
        ctx.minute_of_hour / 59.0,
        ctx.weekday / 6.0,
        ctx.reputation,
        ctx.recent_audit_pass_rate,
        max(0.0, min(1.0, ctx.avg_round_latency_s / 30.0)),
        max(0.0, min(1.0, ctx.incoming_payload_bytes / (50 * 1024 * 1024))),
        ctx.incoming_peer_reputation,
        float(ctx.incoming_peer_continent_match),
        (ctx.last_action_outcome + 1.0) / 2.0,                     # -1..1 -> 0..1
        max(0.0, min(1.0, ctx.minutes_since_user_active / 360.0)), # 6h cap
    ], dtype=torch.float32)


# --------------------------------------------------------------------------
# Heuristic bootstrap policy: the brain's pre-training fallback.
# --------------------------------------------------------------------------
def heuristic_action(ctx: NodeContext, decision: int) -> int:
    if decision == DECISION_ACCEPT_JOB:
        if ctx.is_gaming or ctx.cpu_pct > 90 or ctx.gpu_pct > 90:
            return 2     # reject
        if ctx.gpu_temp_c > 80 or (not ctx.on_ac_power and ctx.battery_pct < 30):
            return 1     # defer
        return 0         # accept

    if decision == DECISION_THROTTLE:
        if ctx.gpu_temp_c > 85 or ctx.is_gaming:
            return 2     # pause
        if ctx.cpu_pct > 80 or ctx.ram_pct > 85:
            return 1     # half_speed
        return 0         # full_speed

    if decision == DECISION_FEE_BID:
        # Higher fee when network is congested OR PLG is cheap (we want it confirmed).
        congestion = min(1.0, ctx.pending_tx_count / 500.0)
        score = 0.6 * congestion + 0.4 * (1.0 - min(1.0, ctx.plg_usd_price / 0.5))
        return min(4, int(score * 5))

    if decision == DECISION_PEER_TRUST:
        if ctx.incoming_peer_reputation < 0.2:
            return 2     # drop
        if ctx.incoming_peer_reputation < 0.6 or ctx.recent_audit_pass_rate < 0.7:
            return 1     # audit
        return 0         # trust

    return 0


# --------------------------------------------------------------------------
# Replay buffer
# --------------------------------------------------------------------------
@dataclass
class Experience:
    state: Any                  # torch.Tensor when torch available
    decision: int
    action: int
    reward: float
    next_state: Any             # torch.Tensor when torch available
    done: bool


# --------------------------------------------------------------------------
# Pluginfer Brain
# --------------------------------------------------------------------------
class PluginferBrain:
    """
    Always-running on-device decision agent. Always available.
    Even at random-init weights, falls back to heuristic so the node
    never blocks on missing intelligence.
    """

    MODEL_SPEC = {
        "arch": "mlp",
        "config": {"in_dim": STATE_DIM, "hidden_dim": 64, "out_dim": 14, "depth": 2},
        "init_seed": 7,
    }

    def __init__(self,
                 confidence_threshold: float = 0.75,
                 epsilon: float = 0.05,
                 buffer_capacity: int = 8192,
                 weights_path: str = "user_data/brain.weights",
                 ):
        if not _TORCH_AVAILABLE:
            raise NotImplementedError(
                "PluginferBrain requires torch. Install: pip install torch. "
                f"Original import error: {_TORCH_IMPORT_ERROR!r}"
            )
        self.net = BrainNet()
        self.net.eval()
        self.weights_path = weights_path
        os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
        self._loaded = False
        self._load()

        self.confidence_threshold = confidence_threshold
        self.epsilon = epsilon
        self._buffer: Deque[Experience] = deque(maxlen=buffer_capacity)
        self._action_log: List[Dict[str, Any]] = []
        self._lock = __import__("threading").Lock()

    # ---- inference -----------------------------------------------------
    def decide(self, ctx: NodeContext, decision: int,
               explain: bool = False,
               ) -> Tuple[int, Dict[str, Any]]:
        feat = featurize(ctx).unsqueeze(0)
        with torch.no_grad():
            q = self.net.q_values(feat, decision).squeeze(0)
        probs = F.softmax(q, dim=-1)
        margin = float(probs.max() - probs.kthvalue(min(2, len(probs))).values)
        nn_action = int(q.argmax().item())
        nn_conf = float(probs.max().item())

        # Confidence gate: if model's not confident, fall back to heuristic.
        if not self._loaded or nn_conf < self.confidence_threshold:
            chosen = heuristic_action(ctx, decision)
            source = "heuristic"
        else:
            chosen = nn_action
            source = "neural"

        # ε-greedy exploration: random action small fraction of the time.
        if random.random() < self.epsilon:
            chosen = random.randrange(ACTIONS_PER_DECISION[decision])
            source = "exploration"

        info = {
            "decision": DECISIONS[decision],
            "action": chosen,
            "source": source,
            "neural_action": nn_action,
            "neural_confidence": nn_conf,
            "neural_margin": margin,
            "q_values": q.tolist() if explain else None,
        }
        self._action_log.append({"ts": time.time(), **info})
        if len(self._action_log) > 4096:
            self._action_log = self._action_log[-2048:]
        return chosen, info

    # ---- experience ----------------------------------------------------
    def record_outcome(self,
                       state,
                       decision: int,
                       action: int,
                       reward: float,
                       next_state,
                       done: bool = False) -> None:
        s = state if isinstance(state, torch.Tensor) else featurize(state)
        ns = next_state if isinstance(next_state, torch.Tensor) else featurize(next_state)
        with self._lock:
            self._buffer.append(Experience(
                state=s.detach().clone(),
                decision=decision,
                action=action,
                reward=float(reward),
                next_state=ns.detach().clone(),
                done=done,
            ))

    # ---- local fitting (single inner loop) -----------------------------
    def fit_local(self, steps: int = 64, batch_size: int = 32,
                  lr: float = 1e-3, gamma: float = 0.95) -> Dict[str, float]:
        """One DiLoCo inner-loop equivalent: K steps of Q-learning on local replay."""
        if len(self._buffer) < batch_size:
            return {"steps": 0, "skipped": "buffer too small",
                    "buffer_size": len(self._buffer)}

        self.net.train()
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr)
        losses = []
        with self._lock:
            buffer_snapshot = list(self._buffer)

        for _ in range(steps):
            batch = random.sample(buffer_snapshot, batch_size)
            states = torch.stack([e.state for e in batch])
            next_states = torch.stack([e.next_state for e in batch])
            rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32)
            dones = torch.tensor([1.0 if e.done else 0.0 for e in batch],
                                 dtype=torch.float32)

            # Compute Q(s, a) for each decision-class head independently.
            decisions = torch.tensor([e.decision for e in batch], dtype=torch.long)
            actions = torch.tensor([e.action for e in batch], dtype=torch.long)

            q_all = self.net(states)
            with torch.no_grad():
                q_next_all = self.net(next_states)

            # Gather the Q-value for the (decision, action) tuple per sample.
            q_pred = torch.stack([
                q_all[d.item()][i, actions[i]] for i, d in enumerate(decisions)
            ])
            q_target_max = torch.stack([
                q_next_all[d.item()][i].max() for i, d in enumerate(decisions)
            ])
            target = rewards + gamma * (1.0 - dones) * q_target_max

            loss = F.smooth_l1_loss(q_pred, target.detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        self.net.eval()
        self._loaded = True
        self._save()
        return {
            "steps": steps,
            "buffer_size": len(self._buffer),
            "avg_loss": sum(losses) / len(losses),
            "min_loss": min(losses),
            "max_loss": max(losses),
        }

    # ---- DiLoCo integration --------------------------------------------
    def pack_for_diloco(self) -> Tuple[Dict[str, Any], bytes]:
        """
        Returns (model_spec, serialized_state_dict) for an aggregator
        that runs DiLoCo on the brain itself. The whole network learns
        operational intelligence collectively.
        """
        from .diloco_serialize import serialize_state_dict
        return self.MODEL_SPEC, serialize_state_dict(self.net.state_dict())

    def absorb_diloco_weights(self, payload: bytes) -> None:
        """Load aggregator-blessed weights from the swarm into this brain."""
        from .diloco_serialize import deserialize_state_dict
        expected = {k: tuple(v.shape) for k, v in self.net.state_dict().items()}
        new_state = deserialize_state_dict(
            payload, expected_keys=tuple(expected.keys()),
            expected_shapes=expected,
        )
        self.net.load_state_dict(new_state, strict=True)
        self.net.eval()
        self._loaded = True
        self._save()
        logger.info("Brain absorbed DiLoCo-aggregated weights from swarm.")

    # ---- persistence ---------------------------------------------------
    def _save(self) -> None:
        try:
            from .diloco_serialize import serialize_state_dict
            with open(self.weights_path, "wb") as fh:
                fh.write(serialize_state_dict(self.net.state_dict()))
        except Exception as e:
            logger.warning("Brain save failed: %s", e)

    def _load(self) -> None:
        try:
            if not os.path.exists(self.weights_path):
                return
            from .diloco_serialize import deserialize_state_dict
            with open(self.weights_path, "rb") as fh:
                payload = fh.read()
            expected = {k: tuple(v.shape) for k, v in self.net.state_dict().items()}
            state = deserialize_state_dict(
                payload, expected_keys=tuple(expected.keys()),
                expected_shapes=expected,
            )
            self.net.load_state_dict(state, strict=True)
            self.net.eval()
            self._loaded = True
            logger.info("Brain loaded weights from %s", self.weights_path)
        except Exception as e:
            logger.warning("Brain load skipped (%s); using random init + heuristic.", e)
            self._loaded = False

    # ---- introspection -------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        recent = self._action_log[-256:]
        sources: Dict[str, int] = {"neural": 0, "heuristic": 0, "exploration": 0}
        for r in recent:
            sources[r["source"]] = sources.get(r["source"], 0) + 1
        return {
            "loaded": self._loaded,
            "buffer_size": len(self._buffer),
            "actions_total": len(self._action_log),
            "recent_action_sources": sources,
            "confidence_threshold": self.confidence_threshold,
            "epsilon": self.epsilon,
            "params": sum(p.numel() for p in self.net.parameters()),
        }
