"""TrainingGovernor → mesh broadcast wiring (gap #9).

Verifies:
  1. `broadcast_fn` is called on heartbeat (record_ok) at the throttle
     interval — not on every step.
  2. `broadcast_fn` is force-called on OOM and on illegal-address
     errors so peers quarantine the recovering node within one tick.
  3. Peers receiving the envelope route through `update_peer_health`
     so the task router penalises them.

The full controller-to-controller wire test would require a real
socket / gossip thread; we mock that boundary and prove every method
call up to it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ai.filum.training_governor import TrainingGovernor
from core.task_router import TaskRouter, HardwareProfile


class _FakeController:
    """Stands in for `CompleteMeshController` for the gossip path."""

    def __init__(self, node_id="local"):
        self.node_id = node_id
        self.nodes = {}
        self.task_router = TaskRouter(self)
        self.broadcast_calls = []

    def gossip_governor_health(self, snapshot):
        # In production this calls self.gossip.broadcast("GOVERNOR_HEALTH",
        # ...). We just record so the test can assert.
        self.broadcast_calls.append(snapshot)


def test_governor_broadcasts_on_heartbeat():
    """At default throttle (5s), the FIRST record_ok must broadcast and
    subsequent quick calls must NOT — but a forced broadcast goes through
    regardless. We use a tiny throttle to verify the first-call branch."""
    ctl = _FakeController()
    gov = TrainingGovernor.start(device="cpu", log=lambda _msg: None)
    gov.broadcast_fn = ctl.gossip_governor_health
    gov.broadcast_throttle_s = 0.01

    gov.record_ok(loss_val=1.5)
    assert len(ctl.broadcast_calls) >= 1
    snap = ctl.broadcast_calls[0]
    assert snap["device"] == "cpu"
    assert snap["loss_window_size"] == 1


def test_governor_broadcasts_throttled_under_window():
    """Calls within the throttle window must NOT trigger a broadcast."""
    ctl = _FakeController()
    gov = TrainingGovernor.start(device="cpu", log=lambda _msg: None)
    gov.broadcast_fn = ctl.gossip_governor_health
    gov.broadcast_throttle_s = 60.0

    gov.record_ok(loss_val=1.0)
    initial_count = len(ctl.broadcast_calls)
    # Five immediate follow-ups; throttle must collapse them.
    for _ in range(5):
        gov.record_ok(loss_val=0.99)
    assert len(ctl.broadcast_calls) == initial_count


def test_governor_force_broadcasts_on_oom():
    """OOM must force-broadcast regardless of throttle so peers route
    AROUND the recovering node within one tick."""
    ctl = _FakeController()
    gov = TrainingGovernor.start(device="cuda", log=lambda _msg: None)
    gov.broadcast_fn = ctl.gossip_governor_health
    gov.broadcast_throttle_s = 60.0       # huge — would normally suppress
    # Heartbeat once so the throttle bar is "set".
    gov.record_ok(loss_val=2.0)
    pre_oom_count = len(ctl.broadcast_calls)

    err = RuntimeError("CUDA out of memory: tried to allocate 4 GB")
    gov.handle_runtime_error(err, step=42)

    assert len(ctl.broadcast_calls) > pre_oom_count
    last = ctl.broadcast_calls[-1]
    assert last["oom_count"] >= 1


def test_governor_force_broadcasts_on_illegal_address():
    """cudaErrorIllegalAddress is the strongest quarantine signal — peers
    must learn within one tick."""
    ctl = _FakeController()
    gov = TrainingGovernor.start(device="cuda", log=lambda _msg: None)
    gov.broadcast_fn = ctl.gossip_governor_health
    gov.broadcast_throttle_s = 60.0
    pre_count = len(ctl.broadcast_calls)

    err = RuntimeError(
        "CUDA error: an illegal memory access was encountered")
    gov.handle_runtime_error(err, step=42)

    assert len(ctl.broadcast_calls) > pre_count
    last = ctl.broadcast_calls[-1]
    assert last["illegal_count"] >= 1


def test_received_health_quarantines_via_task_router():
    """End-to-end: a controller receiving a peer's `illegal_count > 0`
    snapshot must mark them quarantined in the task router. This is the
    full self-healing claim."""
    ctl = _FakeController()
    ctl.nodes["peer_b"] = {
        "ip": "127.0.0.1", "port": 9001, "latency": 30, "plugins": ["echo"],
    }
    ctl.task_router._peer_profiles["peer_b"] = HardwareProfile(
        plugins=["echo"], gpu_class="cuda", gpu_vram_gb=8.0,
    )
    # Direct call simulates what happens when the GOSSIP envelope
    # arrives at _handle_governor_health.
    ctl.task_router.update_peer_health("peer_b", {
        "device": "cuda",
        "oom_count": 0,
        "illegal_count": 1,         # poisoned
        "consecutive_skips": 0,
    })
    assert ctl.task_router._is_quarantined("peer_b") is True


def test_broken_broadcast_fn_does_not_break_training():
    """If the broadcast hook raises, the governor must catch + log and
    continue. Training MUST never crash because gossip is misbehaving."""
    ctl = _FakeController()

    def broken(_snapshot):
        raise RuntimeError("network down")

    log_calls = []
    gov = TrainingGovernor.start(
        device="cpu", log=lambda msg: log_calls.append(msg)
    )
    gov.broadcast_fn = broken
    gov.broadcast_throttle_s = 0.0

    # Must not raise.
    gov.record_ok(loss_val=1.0)
    # Should have logged the failure, not propagated it.
    assert any("broadcast_fn raised" in m for m in log_calls)
