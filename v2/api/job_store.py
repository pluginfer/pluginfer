"""JobStore — durable persistence for `JobRecord` state transitions.

The previous JobsService kept jobs in an in-memory dict. That works
for the demo and short inference jobs but loses everything on
restart — fatal for hours-long training runs, batch jobs, anything
the buyer expects to survive a gateway crash.

This module turns persistence into a pluggable backend:

  * `JobStore` — abstract write-through API. Every JobsService state
    transition calls `put(rec)`; reads go through `get(job_id)`.
  * `InMemoryJobStore` — the original dict-backed behaviour. Fast,
    zero-dependency, ideal for tests + ephemeral nodes.
  * `SQLiteJobStore` — durable single-file backend. SQLite handles
    crash recovery via WAL; we re-hydrate JobRecords on
    `restore_open_jobs()` and demote anything still in `running`
    state to `interrupted` so the gateway can resume or refuse.

The serialisation surface is intentionally narrow — we persist only
the JSON-safe fields. Background asyncio.Queue watchers and any
provider object references stay in memory; on restore they are
rebuilt as empty queues so SSE consumers can re-subscribe.

Innovation worth filing: §A24 "Crash-safe escrow ledger for
auction-routed compute." A signed AIReceipt is enough to prove
payment was settled even after the executing gateway dies; the
buyer can present the receipt to ANY mesh node and either replay
or refund the job. This module is the substrate that makes that
guarantee mechanical.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

OPEN_STATES = ("queued", "matched", "running", "paused_funding")
TERMINAL_STATES = (
    "completed", "completed_partial", "failed",
    "timeout", "cancelled", "interrupted",
    "abandoned", "abandoned_partial",
    "disputed_refunded",
)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _rec_to_row(rec: Any) -> Dict[str, Any]:
    """Snapshot a JobRecord (or duck-typed equivalent) into a JSON-safe
    row. We intentionally ignore the asyncio.Queue watchers + delta_queue
    — those are restored as fresh queues on the next process boot."""
    payload = rec.payload if isinstance(rec.payload, dict) else {}
    energy = None
    if rec.energy_report is not None:
        try:
            energy = rec.energy_report.to_receipt_fields()
        except Exception:
            energy = None
    return {
        "job_id": rec.job_id,
        "kind": rec.kind,
        "payload_json": json.dumps(payload, default=str),
        "cost_ceiling_usd": float(rec.cost_ceiling_usd),
        "latency_ceiling_ms": int(rec.latency_ceiling_ms),
        "privacy_class": str(rec.privacy_class),
        "quality_floor": float(rec.quality_floor),
        "submitted_at_unix": float(rec.submitted_at_unix),
        "requester_identity": str(rec.requester_identity),
        "state": str(rec.state),
        "detail": rec.detail,
        "matched_provider_pubkey": rec.matched_provider_pubkey,
        "price_locked_usd": (
            float(rec.price_locked_usd) if rec.price_locked_usd is not None else None
        ),
        "result_b64": rec.result_b64,
        "result_hash_hex": rec.result_hash_hex,
        "provider_signature_b64": rec.provider_signature_b64,
        "execution_ms": (
            float(rec.execution_ms) if rec.execution_ms is not None else None
        ),
        "completed_at_unix": (
            float(rec.completed_at_unix) if rec.completed_at_unix is not None else None
        ),
        "energy_report_json": json.dumps(energy) if energy is not None else None,
    }


def _row_to_rec(row: Dict[str, Any]) -> Any:
    """Rebuild a bare JobRecord from a row. The asyncio.Queue + watchers
    are not restored — callers re-subscribe via SSE if they want
    live deltas on a resumed job."""
    from api.jobs_service import JobRecord
    payload = {}
    try:
        payload = json.loads(row.get("payload_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        payload = {}
    rec = JobRecord(
        job_id=row["job_id"],
        kind=row["kind"],
        payload=payload,
        cost_ceiling_usd=row["cost_ceiling_usd"],
        latency_ceiling_ms=row["latency_ceiling_ms"],
        privacy_class=row["privacy_class"],
        quality_floor=row["quality_floor"],
        submitted_at_unix=row["submitted_at_unix"],
        requester_identity=row["requester_identity"],
    )
    rec.state = row.get("state") or "queued"
    rec.detail = row.get("detail")
    rec.matched_provider_pubkey = row.get("matched_provider_pubkey")
    rec.price_locked_usd = row.get("price_locked_usd")
    rec.result_b64 = row.get("result_b64")
    rec.result_hash_hex = row.get("result_hash_hex")
    rec.provider_signature_b64 = row.get("provider_signature_b64")
    rec.execution_ms = row.get("execution_ms")
    rec.completed_at_unix = row.get("completed_at_unix")
    return rec


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class JobStore(abc.ABC):
    """Write-through job persistence. Implementations must be
    thread-safe — JobsService schedules executions on the default
    thread-pool executor."""

    @abc.abstractmethod
    def put(self, rec: Any) -> None: ...

    @abc.abstractmethod
    def get(self, job_id: str) -> Optional[Any]: ...

    @abc.abstractmethod
    def all_open(self) -> List[Any]: ...

    @abc.abstractmethod
    def all(self) -> List[Any]: ...

    def restore_open_jobs(self) -> List[Any]:
        """Mark every still-open job as `interrupted` and return them.
        Called by JobsService at boot so the buyer's GET /v1/jobs/{id}
        sees a terminal state instead of a stuck-running record."""
        survivors: List[Any] = []
        for rec in self.all_open():
            rec.state = "interrupted"
            rec.detail = "gateway_restart_before_completion"
            rec.completed_at_unix = time.time()
            self.put(rec)
            survivors.append(rec)
        return survivors


# ---------------------------------------------------------------------------
# In-memory backend (the prior behaviour, now isolated)
# ---------------------------------------------------------------------------

class InMemoryJobStore(JobStore):
    def __init__(self) -> None:
        self._jobs: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def put(self, rec: Any) -> None:
        with self._lock:
            self._jobs[rec.job_id] = rec

    def get(self, job_id: str) -> Optional[Any]:
        with self._lock:
            return self._jobs.get(job_id)

    def all_open(self) -> List[Any]:
        with self._lock:
            return [r for r in self._jobs.values() if r.state in OPEN_STATES]

    def all(self) -> List[Any]:
        with self._lock:
            return list(self._jobs.values())


# ---------------------------------------------------------------------------
# SQLite backend — single-file, crash-safe (WAL), zero-config
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id                  TEXT PRIMARY KEY,
    kind                    TEXT NOT NULL,
    payload_json            TEXT,
    cost_ceiling_usd        REAL NOT NULL,
    latency_ceiling_ms      INTEGER NOT NULL,
    privacy_class           TEXT NOT NULL,
    quality_floor           REAL NOT NULL,
    submitted_at_unix       REAL NOT NULL,
    requester_identity      TEXT NOT NULL,
    state                   TEXT NOT NULL,
    detail                  TEXT,
    matched_provider_pubkey TEXT,
    price_locked_usd        REAL,
    result_b64              TEXT,
    result_hash_hex         TEXT,
    provider_signature_b64  TEXT,
    execution_ms            REAL,
    completed_at_unix       REAL,
    energy_report_json      TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_submitted ON jobs(submitted_at_unix);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_SCHEMA_VERSION = "1"


class SQLiteJobStore(JobStore):
    """Single-file SQLite persistence. WAL mode for concurrent reads
    during writes. Connections are created per-call (thread-safe in
    sqlite3 across threads only if check_same_thread=False AND the
    connection is serialised — we hold a single lock instead).

    Path: defaults to `~/.pluginfer/jobs.db`, override via
    `PLUGINFER_JOBS_DB` env. In-memory path `:memory:` is supported
    for tests.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or os.environ.get(
            "PLUGINFER_JOBS_DB",
            str(Path.home() / ".pluginfer" / "jobs.db"),
        )
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because JobsService dispatches to the
        # default thread pool executor; we serialise all access through
        # `_lock` so SQLite still sees one writer at a time.
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None,
        )
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            # WAL gives us crash-safe atomic commits + concurrent reads
            # during writes. Doesn't apply to :memory:.
            if self.db_path != ":memory:":
                try:
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.execute("PRAGMA synchronous=NORMAL")
                except sqlite3.OperationalError:
                    pass
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                ("schema_version", _SCHEMA_VERSION),
            )

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
            finally:
                cur.close()

    def put(self, rec: Any) -> None:
        row = _rec_to_row(rec)
        cols = list(row.keys())
        placeholders = ",".join("?" for _ in cols)
        updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "job_id")
        sql = (
            f"INSERT INTO jobs({','.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(job_id) DO UPDATE SET {updates}"
        )
        try:
            with self._cursor() as cur:
                cur.execute(sql, [row[c] for c in cols])
        except sqlite3.Error as e:
            logger.error("SQLiteJobStore put failed for %s: %s", rec.job_id, e)
            raise

    def get(self, job_id: str) -> Optional[Any]:
        with self._cursor() as cur:
            cur.row_factory = sqlite3.Row
            cur.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            r = cur.fetchone()
        return _row_to_rec(dict(r)) if r is not None else None

    def all_open(self) -> List[Any]:
        with self._cursor() as cur:
            cur.row_factory = sqlite3.Row
            placeholders = ",".join("?" for _ in OPEN_STATES)
            cur.execute(
                f"SELECT * FROM jobs WHERE state IN ({placeholders}) "
                f"ORDER BY submitted_at_unix",
                tuple(OPEN_STATES),
            )
            rows = cur.fetchall()
        return [_row_to_rec(dict(r)) for r in rows]

    def all(self) -> List[Any]:
        with self._cursor() as cur:
            cur.row_factory = sqlite3.Row
            cur.execute("SELECT * FROM jobs ORDER BY submitted_at_unix DESC")
            rows = cur.fetchall()
        return [_row_to_rec(dict(r)) for r in rows]

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


__all__ = [
    "InMemoryJobStore",
    "JobStore",
    "OPEN_STATES",
    "SQLiteJobStore",
    "TERMINAL_STATES",
]
