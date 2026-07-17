"""
Chain-derived reputation tests (W28 / sec6 audit)
=================================================

Cases:
  1. Initial reputation == hw_baseline only (no events yet).
  2. Each mined coinbase block grants +W_BLOCK to the miner.
  3. task_receipt system tx grants +W_TASK to the recipient.
  4. slash system tx subtracts W_SLASH.
  5. Cache hits: get_score is O(1) when chain hasn't moved.
  6. Cache invalidates when a new block is added.
  7. Score bounded below at 0 even after many slashes.
  8. ReputationManager refuses to construct without a ledger
     (no local-JSON forge possible).
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

from core.tokenomics import Wallet, TokenMinter           # noqa: E402
from core.compute_ledger import ComputeLedger, Block      # noqa: E402
from core.reputation import (                             # noqa: E402
    ReputationManager, W_BLOCK, W_TASK, W_SLASH, W_HARDWARE_BASELINE,
)


def _seed_blocks(ledger, miner_w, n=3):
    minter = TokenMinter(ledger=ledger)
    for _ in range(n):
        cb = minter.mint_coinbase(miner_w.address,
                                  block_height=ledger.get_height(),
                                  difficulty_factor=1.0)
        ledger.add_transaction(cb, _internal=True)
        ledger.mine_block(miner_w.address, difficulty=1)


def _append_synthetic_block(ledger, txs):
    """Build a block with synthetic txs (system kinds — task_receipt,
    slash) and append, bypassing PoW (we don't need real work for
    reputation accounting)."""
    prev = ledger.chain[-1]
    b = Block(index=prev.index + 1, previous_hash=prev.hash,
              transactions=txs, difficulty=1)
    n = 0
    while n < 1_000_000:
        b.nonce = n
        b.hash = b.calculate_hash()
        if b.hash.startswith("0"):
            break
        n += 1
    ledger.chain.append(b)
    ledger._apply_block_to_state(b)


def test_initial_score_is_hw_baseline():
    print("\n[1] INITIAL SCORE == HARDWARE BASELINE")
    print("-" * 60)
    led = ComputeLedger("r1")
    w = Wallet()
    rm = ReputationManager(led, w.address, hw_score=1.0)
    s = rm.get_score()
    assert s == round(1.0 * W_HARDWARE_BASELINE, 2)
    print(f"  no events; score={s} == {W_HARDWARE_BASELINE} OK")
    print("  PASS")


def test_blocks_mined_grant_reputation():
    print("\n[2] EACH BLOCK GRANTS +W_BLOCK")
    print("-" * 60)
    led = ComputeLedger("r2")
    w = Wallet()
    _seed_blocks(led, w, n=3)
    rm = ReputationManager(led, w.address, hw_score=1.0)
    comps = rm.get_components()
    assert comps.blocks_mined == 3
    expected = 3 * W_BLOCK + 1.0 * W_HARDWARE_BASELINE
    assert rm.get_score() == round(expected, 2)
    print(f"  blocks_mined={comps.blocks_mined}; score={rm.get_score()} OK")
    print("  PASS")


def test_task_receipt_grants_reputation():
    print("\n[3] task_receipt TX GRANTS +W_TASK")
    print("-" * 60)
    led = ComputeLedger("r3")
    w = Wallet()
    receipt_tx = {
        "tx_id": "t1", "sender": "SYSTEM", "recipient": w.address,
        "amount": "0", "fee": "0",
        "type": "task_receipt", "sender_pub_key": "SYSTEM",
        "timestamp": time.time(), "nonce": 0,
    }
    _append_synthetic_block(led, [receipt_tx, receipt_tx])
    rm = ReputationManager(led, w.address, hw_score=1.0)
    assert rm.get_components().tasks_completed == 2
    expected = 2 * W_TASK + W_HARDWARE_BASELINE
    assert rm.get_score() == round(expected, 2)
    print(f"  tasks_completed=2; score={rm.get_score()} OK")
    print("  PASS")


def test_slash_subtracts():
    print("\n[4] slash TX SUBTRACTS W_SLASH")
    print("-" * 60)
    led = ComputeLedger("r4")
    w = Wallet()
    _seed_blocks(led, w, n=10)
    slash_tx = {
        "tx_id": "s1", "sender": "SYSTEM", "recipient": w.address,
        "amount": "0", "fee": "0",
        "type": "slash", "sender_pub_key": "SYSTEM",
        "timestamp": time.time(), "nonce": 0,
    }
    _append_synthetic_block(led, [slash_tx])
    rm = ReputationManager(led, w.address, hw_score=1.0)
    expected = 10 * W_BLOCK - 1 * W_SLASH + W_HARDWARE_BASELINE
    assert rm.get_score() == round(max(0.0, expected), 2)
    print(f"  10 blocks - 1 slash; score={rm.get_score()} (= max(0, {expected}))")
    print("  PASS")


def test_score_floored_at_zero():
    print("\n[5] SCORE FLOORED AT 0 EVEN UNDER MANY SLASHES")
    print("-" * 60)
    led = ComputeLedger("r5")
    w = Wallet()
    # No mined blocks; just slashes.
    txs = [
        {"tx_id": f"s{i}", "sender": "SYSTEM", "recipient": w.address,
         "amount": "0", "fee": "0", "type": "slash",
         "sender_pub_key": "SYSTEM", "timestamp": time.time(), "nonce": 0}
        for i in range(50)
    ]
    _append_synthetic_block(led, txs)
    rm = ReputationManager(led, w.address, hw_score=1.0)
    s = rm.get_score()
    assert s == 0.0, f"expected floor at 0, got {s}"
    print(f"  50 slashes; floored score={s} OK")
    print("  PASS")


def test_cache_hits_when_chain_static():
    print("\n[6] CACHE: O(1) when chain hasn't moved")
    print("-" * 60)
    led = ComputeLedger("r6")
    w = Wallet()
    _seed_blocks(led, w, n=20)
    rm = ReputationManager(led, w.address, hw_score=1.0)
    # First call populates cache.
    s1 = rm.get_score()
    cached_obj = rm.cache[w.address]
    # Second call — same height, same tip — should reuse the same
    # ReputationComponents instance.
    s2 = rm.get_score()
    assert s1 == s2
    assert rm.cache[w.address] is cached_obj
    print("  same instance reused across calls OK")
    print("  PASS")


def test_cache_invalidates_on_new_block():
    print("\n[7] CACHE INVALIDATES WHEN CHAIN EXTENDS")
    print("-" * 60)
    led = ComputeLedger("r7")
    w = Wallet()
    _seed_blocks(led, w, n=5)
    rm = ReputationManager(led, w.address, hw_score=1.0)
    pre = rm.get_score()
    _seed_blocks(led, w, n=2)
    post = rm.get_score()
    assert post > pre
    print(f"  pre-extend score={pre}; post-extend score={post} OK")
    print("  PASS")


def test_construction_requires_ledger():
    print("\n[8] CANNOT CONSTRUCT WITHOUT A LEDGER")
    print("-" * 60)
    try:
        ReputationManager(None)
        assert False, "should have raised"
    except ValueError as e:
        print(f"  ledger=None rejected: {e}")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("CHAIN-DERIVED REPUTATION TEST (W28)")
    print("=" * 60)
    t0 = time.time()
    test_initial_score_is_hw_baseline()
    test_blocks_mined_grant_reputation()
    test_task_receipt_grants_reputation()
    test_slash_subtracts()
    test_score_floored_at_zero()
    test_cache_hits_when_chain_static()
    test_cache_invalidates_on_new_block()
    test_construction_requires_ledger()
    print("\n" + "=" * 60)
    print(f"ALL REPUTATION TESTS PASSED in {time.time() - t0:.1f}s")
    print("=" * 60)
