"""
Account-state index + transaction nonce regression test
========================================================
Closes sec1.5 (slow get_balance walking entire chain) and sec3.3
(transaction replay protection).

Cases:
  1. Nonce monotonicity in mempool — replayed nonce rejected.
  2. Nonce monotonicity across blocks — same nonce confirmed once,
     replay rejected.
  3. Cached balance matches walk-the-chain balance after a sequence
     of mints + transfers + a fee_reward.
  4. Cached nonce table updates as transfers confirm.
  5. After reorg, both caches are recomputed correctly.
  6. tx_id includes nonce — same fields with different nonce hash
     differently and produce different signatures (no replay-via-
     identical-tx_id).
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

from core.tokenomics import Wallet, TokenMinter, Transaction          # noqa: E402
from core.compute_ledger import ComputeLedger, Block                   # noqa: E402


def _signed_transfer(sender_w, recipient, amount, fee, nonce):
    tx = Transaction(sender=sender_w.address, recipient=recipient,
                     amount=amount, type="transfer",
                     sender_pub_key=sender_w.public_key_pem,
                     fee=fee, nonce=nonce)
    tx.signature = sender_w.sign(tx.tx_id)
    return tx


def _seed_balances(ledger, wallet, n_blocks=4):
    minter = TokenMinter(ledger=ledger)
    for _ in range(n_blocks):
        tx = minter.mint_coinbase(wallet.address,
                                  block_height=ledger.get_height(),
                                  difficulty_factor=1.0)
        ledger.add_transaction(tx, _internal=True)
        ledger.mine_block(wallet.address, difficulty=1)


def test_nonce_replay_in_mempool_rejected():
    print("\n[1] NONCE REPLAY IN MEMPOOL REJECTED")
    print("-" * 60)
    ledger = ComputeLedger("t1")
    wA, wB = Wallet(), Wallet()
    _seed_balances(ledger, wA, n_blocks=2)

    tx0 = _signed_transfer(wA, wB.address, Decimal("1"),
                           Decimal("0.001"), nonce=0)
    assert ledger.add_transaction(tx0)
    print("  first nonce=0 transfer accepted OK")

    tx0_replay = _signed_transfer(wA, wB.address, Decimal("1"),
                                  Decimal("0.001"), nonce=0)
    assert not ledger.add_transaction(tx0_replay)
    print("  replayed nonce=0 transfer rejected OK")

    tx1 = _signed_transfer(wA, wB.address, Decimal("1"),
                           Decimal("0.001"), nonce=1)
    assert ledger.add_transaction(tx1)
    print("  nonce=1 transfer accepted OK")
    print("  PASS")


def test_nonce_replay_after_confirmation():
    print("\n[2] NONCE REPLAY AFTER CONFIRMATION REJECTED")
    print("-" * 60)
    ledger = ComputeLedger("t2")
    wA, wB = Wallet(), Wallet()
    _seed_balances(ledger, wA, n_blocks=2)

    tx0 = _signed_transfer(wA, wB.address, Decimal("1"),
                           Decimal("0.001"), nonce=0)
    assert ledger.add_transaction(tx0)
    block = ledger.mine_block(wA.address, difficulty=1)
    assert block is not None

    # Confirmed nonce=0 should now be in _nonces.
    assert ledger.get_account_nonce(wA.address) == 0
    print(f"  confirmed nonce=0; ledger.get_account_nonce(wA)=0 OK")

    # Replay the same nonce after confirmation.
    tx0_replay = _signed_transfer(wA, wB.address, Decimal("1"),
                                  Decimal("0.001"), nonce=0)
    assert not ledger.add_transaction(tx0_replay)
    print("  nonce=0 replay after confirmation rejected OK")
    print("  PASS")


def test_cached_balance_matches_chain_walk():
    print("\n[3] CACHED BALANCE MATCHES CHAIN WALK")
    print("-" * 60)
    ledger = ComputeLedger("t3")
    wA, wB = Wallet(), Wallet()
    _seed_balances(ledger, wA, n_blocks=4)
    # 4 blocks * 50 PLG = 200 PLG mined to wA via coinbase

    tx = _signed_transfer(wA, wB.address, Decimal("30"),
                          Decimal("0.005"), nonce=0)
    assert ledger.add_transaction(tx)
    ledger.mine_block(wA.address, difficulty=1)

    # Cached vs walk
    cached_a = ledger._balances.get(wA.address, Decimal("0"))
    cached_b = ledger._balances.get(wB.address, Decimal("0"))
    walk_a = Decimal(str(ledger.get_balance_at(wA.address, ledger.get_height())))
    walk_b = Decimal(str(ledger.get_balance_at(wB.address, ledger.get_height())))

    print(f"  wA cached={cached_a} walk={walk_a}")
    print(f"  wB cached={cached_b} walk={walk_b}")
    assert cached_a == walk_a, f"wA mismatch: {cached_a} != {walk_a}"
    assert cached_b == walk_b, f"wB mismatch: {cached_b} != {walk_b}"
    print("  PASS")


def test_nonce_table_updates_on_confirmation():
    print("\n[4] NONCE TABLE UPDATES ON CONFIRMATION")
    print("-" * 60)
    ledger = ComputeLedger("t4")
    wA, wB = Wallet(), Wallet()
    _seed_balances(ledger, wA, n_blocks=2)

    assert ledger.get_account_nonce(wA.address) == -1
    print(f"  initial nonce: {ledger.get_account_nonce(wA.address)}")

    for n in (0, 1, 2):
        tx = _signed_transfer(wA, wB.address, Decimal("1"),
                              Decimal("0.001"), nonce=n)
        assert ledger.add_transaction(tx)
    ledger.mine_block(wA.address, difficulty=1)
    assert ledger.get_account_nonce(wA.address) == 2
    print(f"  after 3 confirmed transfers: nonce={ledger.get_account_nonce(wA.address)}")
    print("  PASS")


def test_state_recomputes_after_reorg():
    print("\n[5] STATE CACHES RECOMPUTE AFTER REORG")
    print("-" * 60)
    # Build chain A: 2 mints + 1 transfer.
    led_a = ComputeLedger("nodeA")
    wA, wB = Wallet(), Wallet()
    _seed_balances(led_a, wA, n_blocks=2)
    tx = _signed_transfer(wA, wB.address, Decimal("10"),
                          Decimal("0.001"), nonce=0)
    led_a.add_transaction(tx)
    led_a.mine_block(wA.address, difficulty=1)

    pre_reorg_a = led_a._balances.get(wA.address, Decimal("0"))
    pre_reorg_b = led_a._balances.get(wB.address, Decimal("0"))
    print(f"  pre-reorg: wA={pre_reorg_a} wB={pre_reorg_b}")

    # Build a rival HARDER block at the SAME height as A's last block.
    # That's enough for fork resolution to trigger (current heuristic
    # is "greater difficulty at same height wins").
    rival_height = led_a.get_height() - 1
    rival_prev = led_a.chain[rival_height - 1].hash
    # An empty rival block (no transfers) at higher difficulty.
    rival = Block(index=rival_height, previous_hash=rival_prev,
                  transactions=[], difficulty=4)
    target = "0" * rival.difficulty
    n = 0
    while n < 2_000_000:
        rival.nonce = n
        rival.hash = rival.calculate_hash()
        if rival.hash.startswith(target):
            break
        n += 1
    assert rival.hash.startswith(target), "rival PoW failed"

    accepted = led_a.receive_remote_block({
        "index": rival.index, "previous_hash": rival.previous_hash,
        "transactions": rival.transactions, "timestamp": rival.timestamp,
        "difficulty": rival.difficulty, "nonce": rival.nonce,
        "hash": rival.hash,
    })
    assert accepted, "rival should have triggered reorg"

    # After reorg, the transfer is gone. Caches must reflect that.
    post_a = led_a._balances.get(wA.address, Decimal("0"))
    post_b = led_a._balances.get(wB.address, Decimal("0"))
    print(f"  post-reorg: wA={post_a} wB={post_b}")
    assert post_b == Decimal("0"), \
        "wB still shows the transfer in cache after reorg"

    walk_a = Decimal(str(led_a.get_balance_at(wA.address, led_a.get_height())))
    assert post_a == walk_a, "post-reorg cache != chain walk"
    print("  cache rebuilt to match chain walk OK")
    print("  PASS")


def test_tx_id_includes_nonce():
    print("\n[6] TX_ID INCLUDES NONCE (no identical-tx_id replay)")
    print("-" * 60)
    wA = Wallet()
    t1 = Transaction(sender=wA.address, recipient="X", amount=Decimal("1"),
                     type="transfer", sender_pub_key=wA.public_key_pem,
                     fee=Decimal("0.001"), nonce=0,
                     timestamp=1234567890.0)
    t2 = Transaction(sender=wA.address, recipient="X", amount=Decimal("1"),
                     type="transfer", sender_pub_key=wA.public_key_pem,
                     fee=Decimal("0.001"), nonce=1,
                     timestamp=1234567890.0)
    assert t1.tx_id != t2.tx_id, "same fields/different nonce yielded same tx_id"
    print(f"  t1.tx_id={t1.tx_id[:12]}  t2.tx_id={t2.tx_id[:12]} OK")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("STATE-INDEX + NONCE TEST")
    print("=" * 60)
    t0 = time.time()
    test_nonce_replay_in_mempool_rejected()
    test_nonce_replay_after_confirmation()
    test_cached_balance_matches_chain_walk()
    test_nonce_table_updates_on_confirmation()
    test_state_recomputes_after_reorg()
    test_tx_id_includes_nonce()
    print("\n" + "=" * 60)
    print(f"ALL STATE-INDEX/NONCE TESTS PASSED in {time.time() - t0:.1f}s")
    print("=" * 60)
