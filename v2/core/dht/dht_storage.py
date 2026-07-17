"""Persistent LRU + TTL key-value store for DHT records.

Used by Kademlia STORE/FIND_VALUE to hold signed records on this
node. JSON file on disk; memory-resident dict; LRU eviction when
the in-memory size exceeds capacity; TTL expires per-record.

Design:
  - keys are sha256 hex strings (160 or 256 bits; we don't enforce here)
  - values are SignedRecord dicts (signature verified BEFORE put)
  - capacity = 10_000 entries by default
  - default TTL = 24 hours
  - persistence file is rewritten atomically on `flush()`
  - `clean_expired()` is called on every put + on a periodic sweep
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .signed_records import SignedRecord, verify_record

logger = logging.getLogger(__name__)


@dataclass
class DHTRecord:
    record: SignedRecord
    expires_at: float

    def to_dict(self) -> dict:
        return {
            "record": self.record.to_dict(),
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, body: dict) -> "DHTRecord":
        return cls(
            record=SignedRecord.from_dict(body["record"]),
            expires_at=float(body["expires_at"]),
        )


class DHTStorage:
    """Thread-safe LRU+TTL store for SignedRecords."""

    def __init__(
        self,
        path: str | Path = "dht_storage.json",
        *,
        capacity: int = 10_000,
        default_ttl_s: float = 24 * 3600,
    ) -> None:
        self.path = Path(path)
        self.capacity = int(capacity)
        self.default_ttl_s = float(default_ttl_s)
        self._lock = threading.RLock()
        self._items: OrderedDict[str, DHTRecord] = OrderedDict()
        self.load()

    # ------------------------------------------------------------------

    def put(
        self, key: str, record: SignedRecord,
        *, ttl_s: Optional[float] = None, verify: bool = True,
    ) -> bool:
        """Store `record` under `key`. Returns True on success.

        Verification: if `verify=True` (default), the signature is
        checked. Records that fail verification are rejected and
        return False (no exception, since DHT STORE is best-effort
        and a bad record is just one to drop).
        """
        if verify and not verify_record(record):
            logger.warning("DHT.put rejected unverifiable record at %s", key[:16])
            return False
        ttl = float(ttl_s if ttl_s is not None else self.default_ttl_s)
        with self._lock:
            self._evict_expired_locked()
            if key in self._items:
                # Last-writer-wins by timestamp; reject staler records.
                existing = self._items[key].record
                if existing.timestamp > record.timestamp:
                    return False
            self._items[key] = DHTRecord(
                record=record, expires_at=time.time() + ttl,
            )
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)  # LRU eviction
        return True

    def get(self, key: str) -> Optional[SignedRecord]:
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            if entry.expires_at <= time.time():
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return entry.record

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._items.pop(key, None) is not None

    def __len__(self) -> int:
        with self._lock:
            self._evict_expired_locked()
            return len(self._items)

    def keys(self) -> list[str]:
        with self._lock:
            self._evict_expired_locked()
            return list(self._items.keys())

    def clean_expired(self) -> int:
        with self._lock:
            return self._evict_expired_locked()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def flush(self) -> None:
        with self._lock:
            self._evict_expired_locked()
            body = {k: r.to_dict() for k, r in self._items.items()}
        # Atomic write: write to .tmp, rename.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(body), encoding="utf-8")
        os.replace(tmp, self.path)

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            body = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("DHTStorage.load failed: %s", e)
            return
        now = time.time()
        with self._lock:
            for k, raw in body.items():
                try:
                    rec = DHTRecord.from_dict(raw)
                except Exception:
                    continue
                if rec.expires_at > now:
                    self._items[k] = rec
            # LRU order is by file order; trim if over capacity.
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    # ------------------------------------------------------------------

    def _evict_expired_locked(self) -> int:
        now = time.time()
        expired = [k for k, r in self._items.items() if r.expires_at <= now]
        for k in expired:
            del self._items[k]
        return len(expired)
