"""
Economic-security regression test
=================================
Proves the four critical economic-security gaps from TODO §1, §3
are closed:

  1. mint_coinbase clamps `difficulty_factor` to a sane range
     (was: difficulty_factor=1e9 on block 0 minted the entire 21M
     supply to the attacker).
  2. mint_coinbase rejects negative / zero / non-numeric difficulty.
  3. ComputeLedger.add_transaction REJECTS externally-submitted
     mint / coinbase / fee_reward / slash transactions (was: anyone
     could push a mint tx into pending and have it mined).
  4. Internal-marked mint txs still flow through the consensus
     path (regression check that the new lock didn't break mining).
  5. ComputeLedger.add_transaction enforces a minimum fee on
     transfers (free-tx spam protection).
"""

from __future__ import annotations

import sys
import time
from decimal import Decimal
from pathlib import Path

_HERE = Path(__file__).resolve()
for parent in [_HERE.parents[1], _HERE.parents[2]]:
    if (parent / "core").is_dir():
        sys.path.insert(0, str(parent))
        break

from core.tokenomics import Wallet, TokenMinter, Transaction         # noqa: E402
from core.compute_ledger import ComputeLedger                         # noqa: E402


def test_difficulty_factor_clamped():
    print("\n[1] DIFFICULTY_FACTOR CLAMPED to MAX_DIFFICULTY_FACTOR")
    print("-" * 60)
    w = Wallet()
    ledger = ComputeLedger("t1")
    minter = TokenMinter(ledger=ledger)
    # Attempt the supply-cap bypass: enormous difficulty_factor on block 0.
    tx = minter.mint_coinbase(w.address, block_height=0,
                              difficulty_factor=1e9)
    assert tx is not None, "expected a coinbase tx, got None"
    minted = Decimal(str(tx.amount))
    # With cap at MAX_DIFFICULTY_FACTOR=2.0 and block reward 50,
    # the absolute upper bound is 50 * 2 = 100 PLG per call.
    assert minted <= Decimal("100"), (
        f"supply-cap bypass NOT closed: minted {minted} PLG in one call"
    )
    print(f"  attempted difficulty_factor=1e9 -> minted {minted} PLG (capped) OK")
    print("  PASS")


def test_difficulty_factor_rejects_bad_inputs():
    print("\n[2] BAD difficulty_factor REJECTED")
    print("-" * 60)
    w = Wallet()
    ledger = ComputeLedger("t2")
    minter = TokenMinter(ledger=ledger)
    for bad in (0, -1, -0.5, "abc", float("nan"), float("inf")):
        tx = minter.mint_coinbase(w.address, block_height=0,
                                  difficulty_factor=bad)
        assert tx is None, f"bad difficulty_factor={bad!r} accepted"
        print(f"  difficulty_factor={bad!r} rejected OK")
    print("  PASS")


def test_external_mint_rejected():
    print("\n[3] EXTERNAL mint/coinbase/fee_reward/slash TX REJECTED")
    print("-" * 60)
    w = Wallet()
    ledger = ComputeLedger("t3")

    # Try to forge a mint tx and slip it into pending.
    forged_mint = Transaction(
        sender="COINBASE", recipient=w.address, amount=Decimal("21000000"),
        type="mint", sender_pub_key="SYSTEM",
    )
    accepted = ledger.add_transaction(forged_mint)
    assert not accepted, "external mint tx was ACCEPTED!"
    print("  forged mint tx rejected OK")

    forged_fee = Transaction(
        sender="NETWORK_FEES", recipient=w.address, amount=Decimal("999999"),
        type="fee_reward", sender_pub_key="SYSTEM",
    )
    accepted = ledger.add_transaction(forged_fee)
    assert not accepted, "external fee_reward tx was ACCEPTED!"
    print("  forged fee_reward tx rejected OK")

    forged_slash = Transaction(
        sender=w.address, recipient="0x000...dead", amount=Decimal("100"),
        type="slash", sender_pub_key="ARBITER_AUTH",
    )
    accepted = ledger.add_transaction(forged_slash)
    assert not accepted, "external slash tx was ACCEPTED!"
    print("  forged slash tx rejected OK")

    print("  PASS")


def test_internal_mint_still_works():
    print("\n[4] INTERNAL mint VIA _internal=True STILL WORKS")
    print("-" * 60)
    w = Wallet()
    ledger = ComputeLedger("t4")
    minter = TokenMinter(ledger=ledger)
    tx = minter.mint_coinbase(w.address, block_height=0,
                              difficulty_factor=1.0)
    assert tx is not None
    accepted = ledger.add_transaction(tx, _internal=True)
    assert accepted, "internal mint tx was rejected"
    block = ledger.mine_block(w.address, difficulty=2)
    assert block is not None, "mining failed after internal mint"
    print(f"  block #{block.index} mined with internal mint OK")
    print("  PASS")


def test_min_tx_fee_enforced():
    print("\n[5] MIN_TX_FEE ENFORCED ON TRANSFERS")
    print("-" * 60)
    w_a, w_b = Wallet(), Wallet()
    ledger = ComputeLedger("t5")

    # Free transfer (fee=0) should be rejected.
    free_tx = Transaction(
        sender=w_a.address, recipient=w_b.address, amount=Decimal("10"),
        type="transfer", sender_pub_key=w_a.public_key_pem,
        fee=Decimal("0"),
    )
    free_tx.signature = w_a.sign(free_tx.tx_id)
    assert not ledger.add_transaction(free_tx), \
        "free transfer was accepted (spam vector)"
    print("  fee=0 transfer rejected OK")

    # Paying transfer (fee >= MIN_TX_FEE) should be accepted.
    paid_tx = Transaction(
        sender=w_a.address, recipient=w_b.address, amount=Decimal("10"),
        type="transfer", sender_pub_key=w_a.public_key_pem,
        fee=Decimal("0.001"),
    )
    paid_tx.signature = w_a.sign(paid_tx.tx_id)
    assert ledger.add_transaction(paid_tx), "fee=0.001 transfer rejected"
    print("  fee=0.001 transfer accepted OK")

    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("ECONOMIC-SECURITY REGRESSION TEST")
    print("=" * 60)
    t0 = time.time()
    test_difficulty_factor_clamped()
    test_difficulty_factor_rejects_bad_inputs()
    test_external_mint_rejected()
    test_internal_mint_still_works()
    test_min_tx_fee_enforced()
    print("\n" + "=" * 60)
    print(f"ALL ECONOMIC TESTS PASSED in {time.time() - t0:.1f}s")
    print("=" * 60)
