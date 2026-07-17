"""SQLite block-store tests (gap #1 — durable persistence)."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.storage_sqlite import BlockStore, migrate_from_json


def _block(index: int, prev_hash: str = "0", txs=None) -> dict:
    return {
        "index": index,
        "hash": f"hash_{index}",
        "previous_hash": prev_hash,
        "merkle_root": f"mr_{index}",
        "timestamp": 1700000000.0 + index,
        "difficulty": 4,
        "nonce": 0,
        "transactions": txs or [],
    }


def _tx(tx_id: str, sender: str = "alice", recipient: str = "bob",
        amount: str = "1.0", tx_type: str = "transfer", nonce: int = 0) -> dict:
    return {
        "tx_id": tx_id, "type": tx_type, "sender": sender,
        "recipient": recipient, "amount": amount, "nonce": nonce,
    }


def test_append_then_read_roundtrip(tmp_path: Path):
    store = BlockStore(tmp_path / "p.db")
    b = _block(0, txs=[_tx("t0")])
    store.append_block(b)
    assert store.get_height() == 0
    out = store.get_block(0)
    assert out is not None
    assert out["hash"] == "hash_0"
    assert out["transactions"][0]["tx_id"] == "t0"


def test_iter_blocks_streams_in_order(tmp_path: Path):
    store = BlockStore(tmp_path / "p.db")
    for i in range(50):
        store.append_block(_block(i, prev_hash=f"hash_{i-1}"))
    heights = [b["index"] for b in store.iter_blocks(start_height=10)]
    assert heights == list(range(10, 50))


def test_truncate_drops_blocks_and_transactions(tmp_path: Path):
    store = BlockStore(tmp_path / "p.db")
    for i in range(10):
        store.append_block(_block(i, prev_hash=f"hash_{i-1}",
                                   txs=[_tx(f"t{i}")]))
    assert store.total_blocks() == 10
    assert store.total_transactions() == 10
    store.truncate_to(4)
    assert store.total_blocks() == 5      # heights 0..4
    assert store.total_transactions() == 5
    assert store.get_block(7) is None


def test_tx_indexed_by_sender_and_recipient(tmp_path: Path):
    store = BlockStore(tmp_path / "p.db")
    store.append_block(_block(0, txs=[
        _tx("t0", sender="alice", recipient="bob"),
        _tx("t1", sender="alice", recipient="carol"),
        _tx("t2", sender="dave",  recipient="bob"),
    ]))
    alice_txs = store.transactions_by_sender("alice")
    assert {t["tx_id"] for t in alice_txs} == {"t0", "t1"}
    bob_received = store.transactions_by_recipient("bob")
    assert {t["tx_id"] for t in bob_received} == {"t0", "t2"}


def test_unique_hash_prevents_double_append(tmp_path: Path):
    store = BlockStore(tmp_path / "p.db")
    store.append_block(_block(0))
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        store.append_block(_block(0))   # same height + hash


def test_durable_across_reopens(tmp_path: Path):
    """Write some blocks, close, reopen, the data is still there."""
    db = tmp_path / "p.db"
    s1 = BlockStore(db)
    for i in range(5):
        s1.append_block(_block(i, prev_hash=f"hash_{i-1}"))
    s1.close()
    s2 = BlockStore(db)
    assert s2.get_height() == 4
    assert s2.total_blocks() == 5


def test_migrate_from_json_is_idempotent(tmp_path: Path):
    """Running the migration twice does not double-insert."""
    import json
    src = tmp_path / "ledger.json"
    src.write_text(json.dumps([
        _block(0), _block(1, prev_hash="hash_0"),
    ]))
    db = tmp_path / "p.db"
    n1 = migrate_from_json(src, db)
    n2 = migrate_from_json(src, db)
    assert n1 == 2
    assert n2 == 0    # already present, all skipped
    store = BlockStore(db)
    assert store.total_blocks() == 2
