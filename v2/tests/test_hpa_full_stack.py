"""Full-stack §B + §C + §D smoke tests.

Covers transport, gossip, safety, observability, BPE tokenizer,
inference receipts, sun-BFT bridge, and the LR schedule.
All CPU-only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


# ---------- LR schedule ----------------------------------------------------

def test_lr_schedule_warmup_then_cosine():
    from ai.filum.lr_schedule import LRSchedule
    s = LRSchedule(target_lr=1e-3, warmup_steps=10, total_steps=100)
    # Warmup: linear ramp 0 -> target.
    assert s.lr_at(0) == pytest.approx(1e-4, abs=1e-8)   # (0+1)/10 * 1e-3
    assert s.lr_at(9) == pytest.approx(1e-3, abs=1e-9)
    # After warmup the schedule decays.
    assert s.lr_at(15) < s.lr_at(11)
    # Final step approaches min_lr.
    final = s.lr_at(99)
    assert 1e-4 <= final <= 1e-3


def test_is_finite_loss():
    from ai.filum.lr_schedule import is_finite_loss
    assert is_finite_loss(1.0)
    assert is_finite_loss(0.0)
    assert not is_finite_loss(float("nan"))
    assert not is_finite_loss(float("inf"))
    assert not is_finite_loss(1e9)


# ---------- transport ------------------------------------------------------

def test_transport_send_and_receive_roundtrip():
    from ai.filum.hpa.transport import GrainTransport, TransportConfig

    received = []

    def on_grain(blob, addr):
        received.append((blob, addr))

    a = GrainTransport(on_grain, TransportConfig(bind_host="127.0.0.1", bind_port=0)).start()
    b = GrainTransport(on_grain, TransportConfig(bind_host="127.0.0.1", bind_port=0)).start()
    try:
        payload = b"hello-from-a-" * 5
        a.send_grain(payload, peer=b.address)
        # Wait for delivery.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not received:
            time.sleep(0.02)
        assert received, "transport did not deliver the grain"
        delivered = received[0][0]
        assert delivered == payload
        assert b.stats.grains_assembled >= 1
    finally:
        a.stop(); b.stop()


def test_transport_handles_multi_fragment_grains():
    """A grain larger than MAX_PAYLOAD must be split + reassembled."""
    from ai.filum.hpa.transport import (
        GrainTransport, TransportConfig, MAX_PAYLOAD,
    )

    received = []
    a = GrainTransport(lambda b, _: None, TransportConfig(bind_host="127.0.0.1", bind_port=0)).start()
    b = GrainTransport(lambda blob, _: received.append(blob),
                        TransportConfig(bind_host="127.0.0.1", bind_port=0)).start()
    try:
        payload = b"X" * (MAX_PAYLOAD * 3 + 137)   # 3+ fragments
        a.send_grain(payload, peer=b.address)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not received:
            time.sleep(0.02)
        assert received and received[0] == payload
    finally:
        a.stop(); b.stop()


def test_transport_dedup_drops_duplicate_grain():
    from ai.filum.hpa.transport import GrainTransport, TransportConfig

    received = []
    a = GrainTransport(lambda b, _: None, TransportConfig(bind_host="127.0.0.1", bind_port=0)).start()
    b = GrainTransport(lambda blob, _: received.append(blob),
                        TransportConfig(bind_host="127.0.0.1", bind_port=0)).start()
    try:
        payload = b"deduped" * 10
        a.send_grain(payload, peer=b.address)
        a.send_grain(payload, peer=b.address)
        a.send_grain(payload, peer=b.address)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and len(received) < 1:
            time.sleep(0.02)
        time.sleep(0.3)
        assert len(received) == 1, f"expected 1 unique, got {len(received)}"
        assert b.stats.grains_duplicate >= 2
    finally:
        a.stop(); b.stop()


# ---------- gossip ---------------------------------------------------------

def test_gossip_membership_join():
    from ai.filum.hpa.transport import GrainTransport, TransportConfig
    from ai.filum.hpa.gossip import Gossip, GossipConfig, ALIVE

    tx = GrainTransport(lambda b, _: None,
                          TransportConfig(bind_host="127.0.0.1", bind_port=0)).start()
    try:
        g = Gossip("self", tx, GossipConfig(ping_period_s=10.0))
        g.start()
        try:
            g.add_seed(("127.0.0.1", 65530))
            peers = g.all_peers()
            assert len(peers) == 1
            assert peers[0].state == ALIVE
        finally:
            g.stop()
    finally:
        tx.stop()


def test_gossip_forward_excludes_sender():
    """Gossip must NOT echo a grain back to its sender."""
    from ai.filum.hpa.transport import GrainTransport, TransportConfig
    from ai.filum.hpa.gossip import Gossip, GossipConfig

    seen_at_a = []
    tx_a = GrainTransport(lambda blob, addr: seen_at_a.append((blob, addr)),
                            TransportConfig(bind_host="127.0.0.1", bind_port=0)).start()
    tx_b = GrainTransport(lambda b, _: None,
                            TransportConfig(bind_host="127.0.0.1", bind_port=0)).start()
    try:
        # B's gossip knows A as its only peer. When A sends to B,
        # B should NOT forward back to A.
        g_b = Gossip("b", tx_b, GossipConfig(ping_period_s=10.0,
                                              fanout=1))
        g_b.start()
        try:
            g_b.add_seed(tx_a.address)
            payload = b"unique-payload-" + str(time.time()).encode()
            tx_a.send_grain(payload, peer=tx_b.address)
            time.sleep(0.6)
            # A sees its own ACK (which seen_at_a may or may not capture
            # since ACKs are not handed to on_grain). It should NOT see
            # the payload echoed back.
            echoed = [blob for blob, _ in seen_at_a if blob == payload]
            assert not echoed, "gossip echoed the grain back to its sender"
        finally:
            g_b.stop()
    finally:
        tx_a.stop(); tx_b.stop()


# ---------- safety ---------------------------------------------------------

def test_safety_gate_rate_limits():
    from ai.filum.hpa.safety import SafetyGate, SafetyGateConfig, RateLimitConfig

    gate = SafetyGate(SafetyGateConfig(
        rate=RateLimitConfig(max_per_min=60, burst=2),
    ))
    pk = "alice"
    # First two pass (burst).
    assert gate.check(pk, "ok").is_allowed()
    assert gate.check(pk, "ok").is_allowed()
    # Third should rate-limit.
    res = gate.check(pk, "ok")
    assert res.decision == "rate_limited"


def test_safety_gate_rejects_credentials():
    from ai.filum.hpa.safety import SafetyGate

    gate = SafetyGate()
    res = gate.check("bob", "AKIA1234567890123456 here is a leaked key")
    assert not res.is_allowed()
    assert res.matched_class == "credentials"


def test_safety_gate_rejects_sanctioned_region():
    from ai.filum.hpa.safety import SafetyGate

    gate = SafetyGate()
    res = gate.check("c", "ok", region="IR")
    assert res.decision == "deny"
    assert "sanctioned" in res.reason.lower()


def test_safety_gate_quarantines_csam_class():
    from ai.filum.hpa.safety import (
        SafetyGate, SafetyGateConfig, ContentClassifierConfig,
    )

    def mock_classifier(text):
        return {"csam": 0.95}

    gate = SafetyGate(SafetyGateConfig(
        classifier=ContentClassifierConfig(
            pluggable_classifier=mock_classifier,
        ),
    ))
    res = gate.check("d", "anything")
    assert res.decision == "quarantined"


# ---------- observability --------------------------------------------------

def test_metrics_registry_renders_with_no_bindings():
    from ai.filum.hpa.observability import MetricsRegistry

    reg = MetricsRegistry()
    txt = reg.render()
    assert "Pluginfer" in txt


def test_metrics_registry_renders_safety_stats():
    from ai.filum.hpa.observability import MetricsRegistry
    from ai.filum.hpa.safety import SafetyGate

    gate = SafetyGate()
    gate.check("p", "ok")
    gate.check("p", "ok")

    reg = MetricsRegistry()
    reg.bind_safety_gate(gate)
    txt = reg.render()
    assert "pluginfer_safety_decisions_total" in txt
    assert 'decision="allowed"' in txt


# ---------- BPE tokenizer --------------------------------------------------

def test_bpe_train_encode_decode_roundtrip():
    from ai.filum.tokenizer_bpe import train_bpe, BPEConfig

    corpus = ["the quick brown fox jumps over the lazy dog"] * 50 + [
        "the cat sat on the mat",
        "she sells sea shells by the sea shore",
        "fox jumps over dog the cat",
    ] * 20
    tok = train_bpe(corpus, BPEConfig(vocab_size=512))
    assert tok.vocab_size <= 512
    ids = tok.encode("the quick brown fox", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    decoded = tok.decode(ids)
    assert "the" in decoded
    assert "fox" in decoded


def test_bpe_save_and_load(tmp_path: Path):
    from ai.filum.tokenizer_bpe import train_bpe, BPEConfig, BPETokenizer

    corpus = ["pluginfer mesh is the future of decentralized compute"] * 30
    tok = train_bpe(corpus, BPEConfig(vocab_size=300))
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    assert tok2.vocab_size == tok.vocab_size
    a = tok.encode("pluginfer mesh")
    b = tok2.encode("pluginfer mesh")
    assert a == b


# ---------- inference receipts (§D1) ---------------------------------------

def test_inference_receipt_sign_and_verify():
    from ai.filum.hpa.grain import fresh_keypair
    from ai.filum.hpa.inference_receipt import issue_receipt, verify_receipt

    seed, pub = fresh_keypair()
    pubkey_hex = pub.hex()
    r = issue_receipt(
        model_weights_bytes=b"fake-weights",
        input_text="hello world",
        output_text="hi back",
        model_metadata={"name": "Filum-127M", "version": "0.1"},
        node_pubkey_hex=pubkey_hex,
        node_priv_seed=seed,
    )
    assert r.receipt_id
    assert r.signature
    assert verify_receipt(r, pub)
    # Tamper -> verification fails.
    r.output_sha256 = "0" * 64
    assert not verify_receipt(r, pub)


def test_inference_receipt_log_and_merkle(tmp_path: Path):
    from ai.filum.hpa.grain import fresh_keypair
    from ai.filum.hpa.inference_receipt import (
        issue_receipt, ReceiptLog, ReceiptLogConfig,
    )

    seed, pub = fresh_keypair()
    log = ReceiptLog(ReceiptLogConfig(
        log_path=str(tmp_path / "receipts.jsonl"),
        anchor_path=str(tmp_path / "anchors.jsonl"),
        merkle_batch_size=4,
    ))
    anchors = []
    for i in range(10):
        r = issue_receipt(
            model_weights_bytes=b"w",
            input_text=f"in-{i}",
            output_text=f"out-{i}",
            node_pubkey_hex=pub.hex(),
            node_priv_seed=seed,
        )
        a = log.append(r)
        if a is not None:
            anchors.append(a)
    # 10 receipts / batch 4 = 2 sealed anchors, 2 leftover.
    assert len(anchors) == 2
    final = log.flush()
    assert final is not None
    assert final["count"] == 2
    # Merkle root is hex sha256.
    assert all(len(a["merkle_root"]) == 64 for a in anchors)


# ---------- sun-BFT bridge -------------------------------------------------

def test_sun_validator_weight_proportional():
    from ai.filum.hpa.sun_bft import sun_to_validator
    from ai.filum.hpa.sun_election import NodeMembership

    s_high = NodeMembership(node_id="hi",
                             stability_score=0.9,
                             advertised_capacity_tflops=10.0)
    s_low = NodeMembership(node_id="lo",
                            stability_score=0.4,
                            advertised_capacity_tflops=5.0)
    v_hi = sun_to_validator(s_high)
    v_lo = sun_to_validator(s_low)
    assert v_hi.weight > v_lo.weight
    # Demoted (stability == 0) -> weight 0.
    s_demoted = NodeMembership(node_id="d",
                                 stability_score=0.0,
                                 advertised_capacity_tflops=10.0)
    assert sun_to_validator(s_demoted).weight == 0


def test_sunbft_apply_election_and_propose_state():
    from ai.filum.hpa.sun_bft import SunBFTBridge
    from ai.filum.hpa.sun_election import (
        ElectionResult, NodeMembership, SunOfSunsRing,
    )

    ring = SunOfSunsRing()
    bridge = SunBFTBridge(self_id="me", ring=ring)
    suns = [
        NodeMembership(node_id="s1", stability_score=0.9,
                       advertised_capacity_tflops=10.0),
        NodeMembership(node_id="s2", stability_score=0.8,
                       advertised_capacity_tflops=12.0),
    ]
    res = ElectionResult(suns=suns)
    validators = bridge.apply_election(res)
    assert len(validators) == 2
    block = bridge.propose_training_state(
        height=1, round_n=0,
        nbgga_shard_versions={"L1": 5, "L2": 3},
        receipt_anchor_root="abc123",
    )
    assert block is not None
    assert block["kind"] == "training_state"
    assert block["nbgga"] == {"L1": 5, "L2": 3}
    assert block["receipt_anchor_root"] == "abc123"


def test_sunbft_skip_when_no_state():
    from ai.filum.hpa.sun_bft import SunBFTBridge
    from ai.filum.hpa.sun_election import SunOfSunsRing

    bridge = SunBFTBridge(self_id="me", ring=SunOfSunsRing())
    block = bridge.propose_training_state(height=1, round_n=0,
                                            nbgga_shard_versions={})
    assert block is None
