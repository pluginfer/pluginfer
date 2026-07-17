"""
Chain integrity smoke test
==========================
Proves the three Tier-1 critical fixes work:

  1. mine_block actually does proof-of-work — block hash starts with
     `difficulty` zeros after a real nonce search.
  2. receive_remote_block does longest-work fork resolution: at the
     same height, the higher-difficulty rival wins; reorgs happen.
  3. TokenMinter.total_minted is recomputed from chain (no longer
     a per-process in-memory counter).

Runs in ~5 s.
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

from core.tokenomics import Wallet, TokenMinter, Transaction  # noqa: E402
from core.compute_ledger import ComputeLedger, Block           # noqa: E402


def _add_mint(ledger: ComputeLedger, minter: TokenMinter,
              wallet: Wallet, amount: Decimal):
    """
    Helper: mint a coinbase tx and add it to pending via the internal
    consensus channel. External callers must NOT be able to do this —
    that's enforced in the W34 fix to ComputeLedger.add_transaction.
    """
    block_height = ledger.get_height()
    tx = minter.mint_coinbase(
        recipient_addr=wallet.address,
        block_height=block_height,
        difficulty_factor=float(amount) / 50.0,
    )
    if tx:
        ledger.add_transaction(tx, _internal=True)
    return tx


def test_real_pow():
    print("\n[1] REAL PROOF-OF-WORK")
    print("-" * 60)
    w = Wallet()
    ledger = ComputeLedger("test-node")
    minter = TokenMinter(ledger=ledger)

    _add_mint(ledger, minter, w, Decimal("50"))
    t0 = time.time()
    block = ledger.mine_block(w.address, difficulty=4)
    dt = time.time() - t0
    assert block is not None, "mine_block returned None"
    assert block.hash.startswith("0" * 4), f"hash {block.hash[:8]} doesn't meet target"
    assert block.nonce > 0, "nonce should be > 0 if real PoW happened"
    print(f"  block #{block.index} hash={block.hash[:16]}... nonce={block.nonce} in {dt*1000:.0f}ms")
    print("  PASS")


def test_supply_derives_from_chain():
    print("\n[2] SUPPLY DERIVED FROM CHAIN (not in-memory counter)")
    print("-" * 60)
    w = Wallet()
    ledger = ComputeLedger("test-node")
    minter = TokenMinter(ledger=ledger)

    # Mint 3 blocks worth of coinbase.
    for _ in range(3):
        _add_mint(ledger, minter, w, Decimal("50"))
        ledger.mine_block(w.address, difficulty=2)

    minted_mid = minter.total_minted
    print(f"  after 3 mints: total_minted = {minted_mid}")
    assert minted_mid > Decimal("100"), f"expected >100 PLG, got {minted_mid}"

    # Build a brand new minter against the same ledger — supply must be
    # recomputed from chain, not zeroed.
    fresh_minter = TokenMinter(ledger=ledger)
    print(f"  fresh minter: total_minted    = {fresh_minter.total_minted}")
    assert fresh_minter.total_minted == minted_mid, (
        "fresh minter must recompute supply from chain"
    )
    print("  PASS")


def test_fork_resolution():
    print("\n[3] LONGEST-WORK FORK RESOLUTION")
    print("-" * 60)
    w_a = Wallet()
    w_b = Wallet()
    chain_a = ComputeLedger("node-A")
    chain_b = ComputeLedger("node-B")

    # Both nodes mint a block at height 1.
    minter_a = TokenMinter(ledger=chain_a)
    minter_b = TokenMinter(ledger=chain_b)
    _add_mint(chain_a, minter_a, w_a, Decimal("50"))
    _add_mint(chain_b, minter_b, w_b, Decimal("50"))

    block_a = chain_a.mine_block(w_a.address, difficulty=2)   # easier
    block_b = chain_b.mine_block(w_b.address, difficulty=4)   # harder
    print(f"  A height-1 block: diff={block_a.difficulty} hash={block_a.hash[:12]}")
    print(f"  B height-1 block: diff={block_b.difficulty} hash={block_b.hash[:12]}")

    # A receives B's harder block at the same height: should reorg.
    accepted = chain_a.receive_remote_block({
        "index": block_b.index, "previous_hash": block_b.previous_hash,
        "transactions": block_b.transactions, "timestamp": block_b.timestamp,
        "difficulty": block_b.difficulty, "nonce": block_b.nonce,
        "hash": block_b.hash,
    })
    print(f"  A received B's harder block: accepted={accepted}")
    assert accepted, "A should reorg to B's harder chain"
    assert chain_a.get_latest_block().hash == block_b.hash, "A's tip should now match B"

    # The reverse: B receives A's easier block at the same height — should reject.
    rejected = chain_b.receive_remote_block({
        "index": block_a.index, "previous_hash": block_a.previous_hash,
        "transactions": block_a.transactions, "timestamp": block_a.timestamp,
        "difficulty": block_a.difficulty, "nonce": block_a.nonce,
        "hash": block_a.hash,
    })
    print(f"  B received A's easier block: accepted={rejected} (expected False)")
    assert not rejected, "B should keep its higher-work chain"
    print("  PASS")


def test_pow_rejects_invalid_block():
    print("\n[4] PoW VALIDATION REJECTS BLOCKS THAT DON'T MEET TARGET")
    print("-" * 60)
    w = Wallet()
    ledger = ComputeLedger("test-node")

    # Hand-craft a block with difficulty=4 but no PoW done.
    fake = Block(
        index=1, previous_hash=ledger.get_latest_block().hash,
        transactions=[{"tx_id": "x", "sender": "a", "recipient": "b",
                       "amount": "1", "type": "transfer", "fee": "0",
                       "sender_pub_key": "k", "timestamp": time.time()}],
        difficulty=4, nonce=1,
    )
    accepted = ledger.receive_remote_block({
        "index": fake.index, "previous_hash": fake.previous_hash,
        "transactions": fake.transactions, "timestamp": fake.timestamp,
        "difficulty": fake.difficulty, "nonce": fake.nonce,
        "hash": fake.hash,
    })
    print(f"  invalid PoW block: accepted={accepted} (expected False)")
    assert not accepted, "ledger must reject blocks that don't meet difficulty"
    print("  PASS")


def _mine_with_pow(ledger, transactions, difficulty=2):
    """Helper: build a block, mine real PoW so the test isn't testing PoW rejection."""
    new_block = Block(
        index=ledger.get_height(),
        previous_hash=ledger.get_latest_block().hash,
        transactions=transactions, difficulty=difficulty,
    )
    target = "0" * difficulty
    nonce = 0
    while nonce < 1_000_000:
        new_block.nonce = nonce
        new_block.hash = new_block.calculate_hash()
        if new_block.hash.startswith(target):
            return new_block
        nonce += 1
    raise RuntimeError("test PoW failed to find a nonce")


def test_block_rejects_forged_transfer():
    print("\n[5] BLOCK WITH FORGED TRANSFER (bad sig) REJECTED")
    print("-" * 60)
    w_a, w_b = Wallet(), Wallet()
    ledger = ComputeLedger("test-node")

    # Build a transfer tx but DO NOT sign it.
    forged = Transaction(
        sender=w_a.address, recipient=w_b.address, amount=Decimal("5"),
        type="transfer", sender_pub_key=w_a.public_key_pem,
        fee=Decimal("0.001"),
    )
    forged.signature = "bogus_signature"
    block = _mine_with_pow(ledger, [forged.to_dict()], difficulty=2)
    accepted = ledger.receive_remote_block({
        "index": block.index, "previous_hash": block.previous_hash,
        "transactions": block.transactions, "timestamp": block.timestamp,
        "difficulty": block.difficulty, "nonce": block.nonce,
        "hash": block.hash,
    })
    assert not accepted, "block with bogus transfer signature was accepted!"
    print("  forged-signature transfer block rejected OK")
    print("  PASS")


def test_block_rejects_forged_mint():
    print("\n[6] BLOCK WITH FORGED MINT (non-SYSTEM pubkey) REJECTED")
    print("-" * 60)
    w = Wallet()
    ledger = ComputeLedger("test-node")

    forged_mint = Transaction(
        sender="COINBASE", recipient=w.address, amount=Decimal("21000000"),
        type="mint", sender_pub_key="hax",
    )
    block = _mine_with_pow(ledger, [forged_mint.to_dict()], difficulty=2)
    accepted = ledger.receive_remote_block({
        "index": block.index, "previous_hash": block.previous_hash,
        "transactions": block.transactions, "timestamp": block.timestamp,
        "difficulty": block.difficulty, "nonce": block.nonce,
        "hash": block.hash,
    })
    assert not accepted, "block with forged mint (bad pubkey) was accepted!"
    print("  forged-mint block rejected OK")
    print("  PASS")


def test_block_rejects_oversized_mint():
    print("\n[7] BLOCK WITH MINT EXCEEDING PER-BLOCK CEILING REJECTED")
    print("-" * 60)
    w = Wallet()
    ledger = ComputeLedger("test-node")

    # Per-block ceiling at height=1: 50 PLG * MAX_DIFFICULTY_FACTOR=2 = 100 PLG.
    # Try to slip in 200 PLG.
    oversized = Transaction(
        sender="COINBASE", recipient=w.address, amount=Decimal("200"),
        type="mint", sender_pub_key="SYSTEM",
    )
    block = _mine_with_pow(ledger, [oversized.to_dict()], difficulty=2)
    accepted = ledger.receive_remote_block({
        "index": block.index, "previous_hash": block.previous_hash,
        "transactions": block.transactions, "timestamp": block.timestamp,
        "difficulty": block.difficulty, "nonce": block.nonce,
        "hash": block.hash,
    })
    assert not accepted, "block with oversized mint (200 PLG) was accepted!"
    print("  oversized-mint (200 PLG > 100 ceiling) block rejected OK")
    print("  PASS")


def test_block_rejects_remote_slash_until_w32():
    print("\n[8] BLOCK WITH SLASH TX REJECTED (W32 not wired yet)")
    print("-" * 60)
    w = Wallet()
    ledger = ComputeLedger("test-node")

    forged_slash = Transaction(
        sender=w.address, recipient="0x000000000000000000000000000000000000dead",
        amount=Decimal("10"), type="slash", sender_pub_key="ARBITER_AUTH",
    )
    block = _mine_with_pow(ledger, [forged_slash.to_dict()], difficulty=2)
    accepted = ledger.receive_remote_block({
        "index": block.index, "previous_hash": block.previous_hash,
        "transactions": block.transactions, "timestamp": block.timestamp,
        "difficulty": block.difficulty, "nonce": block.nonce,
        "hash": block.hash,
    })
    assert not accepted, "remote block carrying a slash tx was accepted!"
    print("  remote slash-tx block rejected OK (until W32 BFT-evidence path)")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("CHAIN INTEGRITY TEST")
    print("=" * 60)
    t0 = time.time()
    test_real_pow()
    test_supply_derives_from_chain()
    test_fork_resolution()
    test_pow_rejects_invalid_block()
    test_block_rejects_forged_transfer()
    test_block_rejects_forged_mint()
    test_block_rejects_oversized_mint()
    test_block_rejects_remote_slash_until_w32()
    print("\n" + "=" * 60)
    print(f"ALL CHAIN TESTS PASSED in {time.time() - t0:.1f}s")
    print("=" * 60)
