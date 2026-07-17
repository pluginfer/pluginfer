"""SQLite-backed durable storage for the chain (gap #1 fix).

Why
---
Loose JSON persistence (ledger.json, governance.json, marketplace.json)
worked for the alpha but does not scale: a 1M-tx chain takes O(N)
to load on every restart, and `json.dump` rewrites the entire file
on every save (no atomic append, no crash safety beyond a single .bak
copy). For production we need:

  * append-only writes ~ O(1) per block + indexed queries,
  * crash safety via WAL journaling,
  * incremental load on restart (only fetch what's needed),
  * room for 100M+ rows without rewriting the world.

SQLite gives us all four with zero external service. Production
deployments at >100k TPS swap in PostgreSQL or RocksDB by reimplementing
the same `BlockStore` interface; everything above this layer stays
unchanged.

Schema
------
    blocks
        height          INTEGER PRIMARY KEY
        hash            TEXT    NOT NULL UNIQUE
        previous_hash   TEXT    NOT NULL
        merkle_root     TEXT    NOT NULL
        timestamp       REAL    NOT NULL
        difficulty      INTEGER NOT NULL
        nonce           INTEGER NOT NULL
        body_json       TEXT    NOT NULL    -- full Block.to_dict()

    transactions
        tx_id           TEXT    PRIMARY KEY
        block_height    INTEGER NOT NULL
        tx_type         TEXT    NOT NULL
        sender          TEXT
        recipient       TEXT
        amount          TEXT             -- Decimal as string
        nonce           INTEGER
        body_json       TEXT    NOT NULL
        FOREIGN KEY (block_height) REFERENCES blocks(height)

    INDEX  ix_transactions_sender    ON transactions(sender)
    INDEX  ix_transactions_recipient ON transactions(recipient)
    INDEX  ix_transactions_block     ON transactions(block_height)

The full block body is also kept as JSON for forward-compatibility:
when we add new tx fields, we don't need a schema migration to keep
loading old blocks.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_SCHEMA_BLOCKS = """
CREATE TABLE IF NOT EXISTS blocks (
    height        INTEGER PRIMARY KEY,
    hash          TEXT    NOT NULL UNIQUE,
    previous_hash TEXT    NOT NULL,
    merkle_root   TEXT    NOT NULL,
    timestamp     REAL    NOT NULL,
    difficulty    INTEGER NOT NULL,
    nonce         INTEGER NOT NULL,
    body_json     TEXT    NOT NULL
);
"""

_SCHEMA_TRANSACTIONS = """
CREATE TABLE IF NOT EXISTS transactions (
    tx_id        TEXT    PRIMARY KEY,
    block_height INTEGER NOT NULL,
    tx_type      TEXT    NOT NULL,
    sender       TEXT,
    recipient    TEXT,
    amount       TEXT,
    nonce        INTEGER,
    body_json    TEXT    NOT NULL,
    FOREIGN KEY (block_height) REFERENCES blocks(height)
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_transactions_sender ON transactions(sender);",
    "CREATE INDEX IF NOT EXISTS ix_transactions_recipient ON transactions(recipient);",
    "CREATE INDEX IF NOT EXISTS ix_transactions_block ON transactions(block_height);",
    "CREATE INDEX IF NOT EXISTS ix_transactions_type ON transactions(tx_type);",
]


class BlockStore:
    """SQLite-backed durable block + transaction store.

    Thread-safe via a serialised connection (one `_conn` guarded by a
    re-entrant lock). For higher concurrency split the connection per
    thread or move to PostgreSQL. The chain layer's writes are
    inherently serialised (one block at a time) so a single shared
    connection is fine.
    """

    def __init__(self, db_path: str | Path = "pluginfer.db"):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,            # autocommit; we batch via transactions
        )
        self._conn.row_factory = sqlite3.Row
        # Crash safety: WAL journal + synchronous=NORMAL is the
        # standard durable-but-fast SQLite combo.
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute("PRAGMA synchronous = NORMAL;")
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(_SCHEMA_BLOCKS)
            self._conn.execute(_SCHEMA_TRANSACTIONS)
            for stmt in _INDEXES:
                self._conn.execute(stmt)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # ---- writes ----------------------------------------------------------
    def append_block(self, block_dict: Dict[str, Any]) -> None:
        """Append one block + its transactions atomically."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO blocks "
                "(height, hash, previous_hash, merkle_root, timestamp, "
                " difficulty, nonce, body_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(block_dict["index"]),
                    block_dict["hash"],
                    block_dict["previous_hash"],
                    block_dict["merkle_root"],
                    float(block_dict.get("timestamp", 0.0)),
                    int(block_dict.get("difficulty", 0)),
                    int(block_dict.get("nonce", 0)),
                    json.dumps(block_dict, default=str),
                ),
            )
            for tx in block_dict.get("transactions") or []:
                self._conn.execute(
                    "INSERT OR REPLACE INTO transactions "
                    "(tx_id, block_height, tx_type, sender, recipient, "
                    " amount, nonce, body_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        tx.get("tx_id"),
                        int(block_dict["index"]),
                        tx.get("type"),
                        tx.get("sender"),
                        tx.get("recipient"),
                        str(tx.get("amount", "0")),
                        int(tx.get("nonce", 0)),
                        json.dumps(tx, default=str),
                    ),
                )

    def truncate_to(self, height: int) -> None:
        """Drop all blocks (and their txs) at height > `height`. Used
        on reorg. Foreign-key cascade ensures transactions go with the
        blocks they belonged to."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM transactions WHERE block_height > ?",
                               (height,))
            self._conn.execute("DELETE FROM blocks WHERE height > ?", (height,))

    # ---- reads -----------------------------------------------------------
    def get_height(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(height) FROM blocks"
            ).fetchone()
            return -1 if row[0] is None else int(row[0])

    def get_block(self, height: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT body_json FROM blocks WHERE height = ?", (height,)
            ).fetchone()
            if row is None:
                return None
            return json.loads(row["body_json"])

    def iter_blocks(self, start_height: int = 0) -> Iterable[Dict[str, Any]]:
        """Stream blocks in order. O(1) memory."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT body_json FROM blocks WHERE height >= ? "
                "ORDER BY height ASC", (start_height,),
            )
            while True:
                rows = cur.fetchmany(256)
                if not rows:
                    break
                for row in rows:
                    yield json.loads(row["body_json"])

    def get_tx(self, tx_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT body_json FROM transactions WHERE tx_id = ?", (tx_id,)
            ).fetchone()
            if row is None:
                return None
            return json.loads(row["body_json"])

    def transactions_by_sender(self, address: str,
                               limit: int = 1000) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT body_json FROM transactions WHERE sender = ? "
                "ORDER BY block_height ASC LIMIT ?", (address, limit),
            )
            return [json.loads(r["body_json"]) for r in cur.fetchall()]

    def transactions_by_recipient(self, address: str,
                                  limit: int = 1000) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT body_json FROM transactions WHERE recipient = ? "
                "ORDER BY block_height ASC LIMIT ?", (address, limit),
            )
            return [json.loads(r["body_json"]) for r in cur.fetchall()]

    def total_blocks(self) -> int:
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM blocks"
            ).fetchone()[0])

    def total_transactions(self) -> int:
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM transactions"
            ).fetchone()[0])


# ---------------------------------------------------------------------------
# JSON -> SQLite migration helper
# ---------------------------------------------------------------------------


def migrate_from_json(json_path: str | Path,
                      db_path: str | Path = "pluginfer.db",
                      *, replace_existing: bool = False) -> int:
    """One-shot migration: load `ledger.json` and append every block to
    the SQLite store. Returns the number of blocks migrated. Idempotent
    when `replace_existing=False` (already-present blocks are skipped
    by the UNIQUE(hash) constraint, which raises sqlite3.IntegrityError
    we silently swallow)."""
    src = Path(json_path)
    if not src.exists():
        return 0
    with src.open("r", encoding="utf-8") as f:
        chain_data = json.load(f)
    store = BlockStore(db_path)
    if replace_existing:
        store.truncate_to(-1)
    migrated = 0
    for block_dict in chain_data:
        try:
            store.append_block(block_dict)
            migrated += 1
        except sqlite3.IntegrityError:
            # Already in the DB — skip.
            pass
    return migrated
