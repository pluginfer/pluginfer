"""
Gossip W24: signed-envelope regression test
============================================
Proves the LAN mesh-hijack vector is closed at the protocol layer.

Cases:
  1. Honest signed envelope from peer A is accepted by peer B.
  2. Same envelope replayed is deduped (returns False, None).
  3. Unsigned envelope is rejected when require_signed=True.
  4. Tampered envelope (origin field changed AFTER signing) is rejected
     (signature verify fails over the canonical body).
  5. Tampered payload is rejected.
  6. Re-flood: a single forwarded hop keeps signature valid because
     ttl was excluded from the canonical body.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
for parent in [_HERE.parents[1], _HERE.parents[2]]:
    if (parent / "core").is_dir():
        sys.path.insert(0, str(parent))
        break

from core.tokenomics import Wallet                                    # noqa: E402
from core.gossip import GossipProtocol                                 # noqa: E402


class _Capture:
    """Record what the broadcast callback would have sent."""
    def __init__(self):
        self.sent = []

    def __call__(self, envelope, exclude_sender):
        self.sent.append(envelope)


def test_honest_signed_envelope_accepted():
    print("\n[1] HONEST SIGNED ENVELOPE ACCEPTED")
    print("-" * 60)
    wA = Wallet()
    sender = _Capture()
    A = GossipProtocol("nodeA", sender, wallet=wA)
    A.broadcast("BLOCK", {"block_index": 1, "tx_count": 5})

    assert sender.sent, "no envelope produced"
    env = sender.sent[0]
    assert "signature" in env and "signer_pubkey" in env

    # Receiver B (different node, no wallet of its own — pure consumer).
    receiver = _Capture()
    B = GossipProtocol("nodeB", receiver, wallet=None, require_signed=True)
    is_new, payload = B.handle_gossip(env.copy())
    assert is_new, "B should accept honest signed envelope"
    assert payload["block_index"] == 1
    print("  signed envelope accepted OK")
    print("  PASS")


def test_replay_deduped():
    print("\n[2] REPLAYED SAME ENVELOPE IS DEDUPED")
    print("-" * 60)
    wA = Wallet()
    sender = _Capture()
    A = GossipProtocol("nodeA", sender, wallet=wA)
    A.broadcast("BLOCK", {"i": 2})
    env = sender.sent[0]

    receiver = _Capture()
    B = GossipProtocol("nodeB", receiver, wallet=None)
    is_new1, _ = B.handle_gossip(env.copy())
    is_new2, _ = B.handle_gossip(env.copy())
    assert is_new1 and not is_new2, "second receive should dedupe"
    print("  replay deduped OK")
    print("  PASS")


def test_unsigned_rejected():
    print("\n[3] UNSIGNED ENVELOPE REJECTED")
    print("-" * 60)
    receiver = _Capture()
    B = GossipProtocol("nodeB", receiver, wallet=None, require_signed=True)
    fake = {
        "type": "GOSSIP", "gossip_type": "BLOCK",
        "origin": "ATTACKER", "id": "abc", "payload": {"i": 99},
        "ttl": 10, "ts": time.time(),
        # no signature, no pubkey
    }
    is_new, payload = B.handle_gossip(fake)
    assert not is_new and payload is None
    assert not receiver.sent, "B re-flooded an unsigned envelope!"
    print("  unsigned envelope rejected OK")
    print("  PASS")


def test_tampered_origin_rejected():
    print("\n[4] TAMPERED ORIGIN FIELD REJECTED")
    print("-" * 60)
    wA = Wallet()
    sender = _Capture()
    A = GossipProtocol("nodeA", sender, wallet=wA)
    A.broadcast("BLOCK", {"i": 4})
    env = sender.sent[0].copy()
    env["origin"] = "EVIL-IMPOSTER"            # tamper after signing

    receiver = _Capture()
    B = GossipProtocol("nodeB", receiver, wallet=None, require_signed=True)
    is_new, _ = B.handle_gossip(env)
    assert not is_new, "tampered origin was accepted!"
    print("  tampered-origin envelope rejected OK")
    print("  PASS")


def test_tampered_payload_rejected():
    print("\n[5] TAMPERED PAYLOAD REJECTED")
    print("-" * 60)
    wA = Wallet()
    sender = _Capture()
    A = GossipProtocol("nodeA", sender, wallet=wA)
    A.broadcast("TX", {"to": "alice", "amount": "10"})
    env = sender.sent[0].copy()
    env["payload"] = {"to": "EVIL", "amount": "1000000"}   # tamper

    receiver = _Capture()
    B = GossipProtocol("nodeB", receiver, wallet=None, require_signed=True)
    is_new, _ = B.handle_gossip(env)
    assert not is_new, "tampered payload was accepted!"
    print("  tampered-payload envelope rejected OK")
    print("  PASS")


def test_reflood_keeps_signature_valid():
    print("\n[6] RE-FLOOD WITH DECREMENTED TTL KEEPS SIG VALID")
    print("-" * 60)
    wA = Wallet()
    sender = _Capture()
    A = GossipProtocol("nodeA", sender, wallet=wA)
    A.broadcast("BLOCK", {"i": 6})

    middle_send = _Capture()
    B = GossipProtocol("nodeB", middle_send, wallet=None, require_signed=True)
    B.handle_gossip(sender.sent[0].copy())

    assert middle_send.sent, "B should have re-flooded the envelope"
    forwarded = middle_send.sent[0]
    assert forwarded["ttl"] == 9, "ttl should have decremented"

    # Far-end receiver C must still accept the forwarded envelope.
    far_send = _Capture()
    C = GossipProtocol("nodeC", far_send, wallet=None, require_signed=True)
    is_new, payload = C.handle_gossip(forwarded)
    assert is_new and payload["i"] == 6, \
        "C should accept the forwarded (TTL-decremented) envelope"
    print("  forward-with-decremented-TTL still validates OK")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("GOSSIP W24 SIGNED-ENVELOPE TEST")
    print("=" * 60)
    t0 = time.time()
    test_honest_signed_envelope_accepted()
    test_replay_deduped()
    test_unsigned_rejected()
    test_tampered_origin_rejected()
    test_tampered_payload_rejected()
    test_reflood_keeps_signature_valid()
    print("\n" + "=" * 60)
    print(f"ALL GOSSIP TESTS PASSED in {time.time() - t0:.1f}s")
    print("=" * 60)
