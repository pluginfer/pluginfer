"""Replay-protection regression suite.

Focus: every signed structure that can be re-broadcast over the network
must be rejected the second time. This is the proper "sec3.3 closed"
assertion -- if a future change accidentally drops the nonce from a
tx_id pre-image (or removes the gossip dedup table), this test goes
red.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.compute_ledger import ComputeLedger  # noqa: E402
from core.gossip import GossipProtocol  # noqa: E402
from core.tokenomics import TokenMinter, Transaction, Wallet  # noqa: E402


# ---------------------------------------------------------------------------
# Tx-replay protection (sec3.3)
# ---------------------------------------------------------------------------


def _fund(ledger: ComputeLedger, wallet: Wallet, amount: float = 100.0) -> None:
    minter = TokenMinter(ledger=ledger)
    tx = minter.mint_coinbase(wallet.address, block_height=0,
                              difficulty_factor=1.0)
    assert ledger.add_transaction(tx, _internal=True)
    ledger.mine_block(wallet.address, difficulty=2)


def _build_signed_transfer(
    *, sender: Wallet, recipient: str, amount: float,
    fee: float = 0.001, nonce: int = 0,
) -> Transaction:
    tx = Transaction(
        sender=sender.address,
        recipient=recipient,
        amount=Decimal(str(amount)),
        type="transfer",
        sender_pub_key=sender.public_key_pem,
        fee=Decimal(str(fee)),
        nonce=nonce,
    )
    tx.signature = sender.sign(tx.tx_id)
    return tx


def test_signed_transfer_cannot_be_replayed():
    """The exact same signed Transaction must not be accepted twice."""
    a, b = Wallet(), Wallet()
    ledger = ComputeLedger("replay-1")
    _fund(ledger, a, 100.0)

    tx = _build_signed_transfer(
        sender=a, recipient=b.address, amount=10.0,
    )
    assert ledger.add_transaction(tx) is True
    miner_addr = Wallet().address
    ledger.mine_block(miner_addr, difficulty=2)

    # Re-submit the SAME signed body — must be rejected.
    accepted = ledger.add_transaction(tx)
    assert accepted is False, "second submission of the same tx must be rejected"


def test_replayed_tx_with_same_nonce_rejected_after_higher_lands():
    """After tx0 is confirmed, tx1 lands at higher nonce. Re-submitting
    tx0 (lower nonce) must be rejected by the per-sender nonce table
    even if its signature is still valid."""
    a, b = Wallet(), Wallet()
    ledger = ComputeLedger("replay-2")
    _fund(ledger, a, 100.0)
    miner_addr = Wallet().address

    tx0 = _build_signed_transfer(sender=a, recipient=b.address, amount=1.0, nonce=0)
    assert ledger.add_transaction(tx0)
    ledger.mine_block(miner_addr, difficulty=2)

    tx1 = _build_signed_transfer(sender=a, recipient=b.address, amount=2.0, nonce=1)
    assert ledger.add_transaction(tx1)
    ledger.mine_block(miner_addr, difficulty=2)

    # Replay of confirmed tx0 -> rejected.
    assert ledger.add_transaction(tx0) is False


def test_tampered_signature_rejected():
    """Flipping a byte of the signature must invalidate the transfer."""
    a, b = Wallet(), Wallet()
    ledger = ComputeLedger("replay-3")
    _fund(ledger, a, 100.0)

    tx = _build_signed_transfer(sender=a, recipient=b.address, amount=5.0)
    # Tamper after signing.
    if isinstance(tx.signature, (bytes, bytearray)):
        sig = bytearray(tx.signature)
    else:
        sig = bytearray(tx.signature.encode())
    sig[0] ^= 0xFF
    tx.signature = bytes(sig) if isinstance(tx.signature, (bytes, bytearray)) else sig.decode("latin1")
    assert ledger.add_transaction(tx) is False


def test_swapped_recipient_after_signing_rejected():
    """Cannot redirect a signed tx to a new recipient — sig binds it."""
    a, b, c = Wallet(), Wallet(), Wallet()
    ledger = ComputeLedger("replay-4")
    _fund(ledger, a, 100.0)

    tx = _build_signed_transfer(sender=a, recipient=b.address, amount=5.0)
    tx.recipient = c.address  # adversary swap after signing
    assert ledger.add_transaction(tx) is False


# ---------------------------------------------------------------------------
# Gossip-envelope replay
# ---------------------------------------------------------------------------


class _Capture:
    def __init__(self):
        self.sent = []
    def __call__(self, env, exclude_sender):
        self.sent.append(env)


def test_gossip_envelope_replay_rejected_by_dedup():
    """Send an honest signed envelope; replaying it through the same
    receiver returns is_new=False so the receiver does NOT re-flood."""
    wA = Wallet()
    out_a = _Capture()
    A = GossipProtocol("nodeA", out_a, wallet=wA)
    A.broadcast("BLOCK", {"i": 1})
    env = out_a.sent[0]

    out_b = _Capture()
    B = GossipProtocol("nodeB", out_b, wallet=None, require_signed=True)
    is_new1, payload1 = B.handle_gossip(env.copy())
    is_new2, payload2 = B.handle_gossip(env.copy())
    assert is_new1 is True and payload1 is not None
    assert is_new2 is False, "replay must be deduped"


def test_gossip_envelope_unsigned_rejected_in_signed_mode():
    out_b = _Capture()
    B = GossipProtocol("nodeB", out_b, wallet=None, require_signed=True)
    fake = {
        "type": "GOSSIP", "gossip_type": "BLOCK",
        "origin": "ATTACKER", "id": "abc", "payload": {"i": 99},
        "ttl": 10, "ts": 0.0,
    }
    is_new, payload = B.handle_gossip(fake)
    assert is_new is False and payload is None
    assert not out_b.sent
