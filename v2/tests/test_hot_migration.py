"""Tests for A14: Hot-Migration Mesh."""

import pytest

from core.hot_migration import (
    ContinuationEnvelope,
    HealthSignal,
    TaskCheckpoint,
    make_checkpoint,
    make_continuation,
    verify_checkpoint_chain,
    verify_handoff,
)
from core.tokenomics import Wallet


def _chain(producer: Wallet, *, task_id="t1", n: int = 3,
           start_seq: int = 0, prev_hash: str = "") -> list[TaskCheckpoint]:
    chain = []
    last_hash = prev_hash
    for i in range(n):
        cp = make_checkpoint(
            task_id=task_id,
            seq=start_seq + i,
            state_bytes=f"state-{start_seq + i}".encode(),
            prev_checkpoint_hash=last_hash,
            producer=producer,
        )
        last_hash = cp.hash_self()
        chain.append(cp)
    return chain


def test_single_checkpoint_signed_and_verifies():
    w = Wallet()
    cp = make_checkpoint(task_id="t", seq=0,
                         state_bytes=b"hello",
                         prev_checkpoint_hash="",
                         producer=w)
    assert cp.verify() is True


def test_checkpoint_signature_breaks_on_tamper():
    w = Wallet()
    cp = make_checkpoint(task_id="t", seq=0, state_bytes=b"x",
                         prev_checkpoint_hash="", producer=w)
    cp.state_b64 = "AAAA"
    assert cp.verify() is False


def test_chain_of_three_verifies_end_to_end():
    w = Wallet()
    chain = _chain(w, n=3)
    assert verify_checkpoint_chain(chain) is True


def test_chain_with_seq_gap_rejected():
    w = Wallet()
    chain = _chain(w, n=3)
    chain[1].seq = 5
    chain[1].signature = w.sign(chain[1].canonical())  # re-sign post-tamper
    assert verify_checkpoint_chain(chain) is False


def test_chain_with_broken_link_rejected():
    w = Wallet()
    chain = _chain(w, n=3)
    chain[2].prev_checkpoint_hash = "0" * 64
    chain[2].signature = w.sign(chain[2].canonical())
    assert verify_checkpoint_chain(chain) is False


def test_handoff_full_path_verifies():
    src = Wallet()
    dst = Wallet()
    src_chain = _chain(src, n=3, task_id="task42")
    last_hash = src_chain[-1].hash_self()
    cont = make_continuation(
        task_id="task42",
        from_checkpoint_hash=last_hash,
        destination=dst,
    )
    dst_chain = _chain(dst, n=2, task_id="task42",
                       start_seq=3, prev_hash=last_hash)
    assert verify_handoff(src_chain, cont, dst_chain) is True


def test_handoff_with_continuation_pointing_at_wrong_hash_rejected():
    src = Wallet()
    dst = Wallet()
    src_chain = _chain(src, n=2, task_id="task42")
    cont = make_continuation(
        task_id="task42",
        from_checkpoint_hash="0" * 64,         # wrong
        destination=dst,
    )
    dst_chain = _chain(dst, n=1, task_id="task42",
                       start_seq=2, prev_hash=src_chain[-1].hash_self())
    assert verify_handoff(src_chain, cont, dst_chain) is False


def test_handoff_with_destination_chain_starting_at_wrong_seq_rejected():
    src = Wallet()
    dst = Wallet()
    src_chain = _chain(src, n=2, task_id="task42")
    last_hash = src_chain[-1].hash_self()
    cont = make_continuation(
        task_id="task42",
        from_checkpoint_hash=last_hash,
        destination=dst,
    )
    bad_dst_chain = _chain(dst, n=1, task_id="task42",
                           start_seq=99,           # wrong; must be 2
                           prev_hash=last_hash)
    assert verify_handoff(src_chain, cont, bad_dst_chain) is False


def test_handoff_task_id_mismatch_rejected():
    src = Wallet()
    dst = Wallet()
    src_chain = _chain(src, n=2, task_id="A")
    cont = make_continuation(
        task_id="B",
        from_checkpoint_hash=src_chain[-1].hash_self(),
        destination=dst,
    )
    dst_chain = _chain(dst, n=1, task_id="B",
                       start_seq=2,
                       prev_hash=src_chain[-1].hash_self())
    assert verify_handoff(src_chain, cont, dst_chain) is False


def test_handoff_continuation_signed_by_wrong_key_rejected():
    src = Wallet()
    dst = Wallet()
    other = Wallet()
    src_chain = _chain(src, n=2, task_id="t")
    cont = make_continuation(
        task_id="t",
        from_checkpoint_hash=src_chain[-1].hash_self(),
        destination=dst,
    )
    # Tamper: substitute another node's pubkey but keep the dst's sig.
    cont.destination_pubkey_pem = other.export_keys()["public"]
    assert cont.verify() is False


# ---------------------------------------------------------------------------
# Trigger heuristic
# ---------------------------------------------------------------------------


def test_low_battery_triggers_handoff():
    sig = HealthSignal(battery_pct=10.0)
    do, why = sig.should_handoff()
    assert do
    assert "battery" in why


def test_high_network_loss_triggers():
    sig = HealthSignal(network_loss_pct=80.0)
    do, why = sig.should_handoff()
    assert do
    assert "network" in why


def test_gpu_overheat_triggers():
    sig = HealthSignal(gpu_temp_c=95.0)
    do, why = sig.should_handoff()
    assert do
    assert "GPU" in why


def test_healthy_node_does_not_handoff():
    sig = HealthSignal(battery_pct=80.0, gpu_temp_c=60.0,
                       network_loss_pct=0.5, user_active=False)
    do, why = sig.should_handoff()
    assert not do
