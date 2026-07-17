"""Node-crash recovery fault injection.

Scenario: a node mines a few blocks, persists state, then "crashes"
(simulated by tearing down the in-memory ledger object). On restart
from the same persistent dir, the chain MUST be re-loadable and the
balance/nonce indices MUST reconverge to the same state. This is the
CP-FINAL safety property: a power loss must not corrupt accounting.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

V2 = Path(__file__).resolve().parents[2]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.compute_ledger import ComputeLedger  # noqa: E402
from core.tokenomics import TokenMinter, Transaction, Wallet  # noqa: E402


def _fund(ledger: ComputeLedger, w: Wallet) -> None:
    minter = TokenMinter(ledger=ledger)
    tx = minter.mint_coinbase(w.address, block_height=0, difficulty_factor=1.0)
    assert ledger.add_transaction(tx, _internal=True)
    ledger.mine_block(w.address, difficulty=2)


def test_balance_state_matches_after_save_and_load(tmp_path):
    """Round-trip the chain via save_chain/load_chain. Balances and
    nonces MUST reconverge to the same state."""
    a, b = Wallet(), Wallet()
    chain_path = str(tmp_path / "ledger.json")

    L1 = ComputeLedger("crash-recovery")
    _fund(L1, a)
    miner = Wallet()
    for i in range(3):
        tx = Transaction(
            sender=a.address, recipient=b.address,
            amount=Decimal("1.0"),
            type="transfer", sender_pub_key=a.public_key_pem,
            fee=Decimal("0.001"), nonce=i,
        )
        tx.signature = a.sign(tx.tx_id)
        assert L1.add_transaction(tx)
        L1.mine_block(miner.address, difficulty=2)

    height_before = len(L1.chain)
    bal_a_before = L1.get_balance(a.address)
    bal_b_before = L1.get_balance(b.address)
    nonce_a_before = L1.get_account_nonce(a.address)
    L1.save_chain(filename=chain_path)
    del L1   # simulate crash

    # Fresh ledger, then load.
    L2 = ComputeLedger("crash-recovery")
    L2.load_chain(filename=chain_path)
    assert len(L2.chain) == height_before
    # Round to milli-PLG to absorb any float<->Decimal roundoff in
    # the JSON serialization path. Accounting precision is ~1e-3 PLG.
    def _q(x):
        return float(Decimal(str(x)).quantize(Decimal("0.0001")))
    assert _q(L2.get_balance(a.address)) == _q(bal_a_before)
    assert _q(L2.get_balance(b.address)) == _q(bal_b_before)
    assert L2.get_account_nonce(a.address) == nonce_a_before


def test_pending_pool_is_volatile(tmp_path):
    """An unmined pending tx must NOT carry across save/load. Pending
    is by-design volatile; it lives in memory only."""
    a, b = Wallet(), Wallet()
    chain_path = str(tmp_path / "ledger.json")
    L1 = ComputeLedger("crash-pending")
    _fund(L1, a)
    tx = Transaction(
        sender=a.address, recipient=b.address, amount=Decimal("0.5"),
        type="transfer", sender_pub_key=a.public_key_pem,
        fee=Decimal("0.001"), nonce=0,
    )
    tx.signature = a.sign(tx.tx_id)
    assert L1.add_transaction(tx)
    assert len(L1.pending_transactions) >= 1
    L1.save_chain(filename=chain_path)
    del L1

    L2 = ComputeLedger("crash-pending")
    L2.load_chain(filename=chain_path)
    # The unmined tx is NOT persisted -- pending is volatile.
    assert L2.pending_transactions == []
