"""CP-2 Task 2.4: tests for DHT signed records + persistent storage.

The Kademlia routing layer itself lives in `core.kademlia` (already
audited and fixed in W6). This file pins:
  - SignedRecord round-trip (sign -> verify)
  - Tampered records rejected
  - DHTStorage put/get/expire/persist round-trip
  - LRU eviction at capacity
  - Stale record rejected on put (last-writer-wins by timestamp)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest  # noqa: E402

from core.dht import (  # noqa: E402
    DHTRecord,
    DHTStorage,
    SignedRecord,
    SignedRecordError,
    sign_record,
    verify_record,
)
from core.tokenomics import Wallet  # noqa: E402


# ---------------------------------------------------------------------------
# Signed records
# ---------------------------------------------------------------------------

def test_sign_and_verify_round_trip() -> None:
    w = Wallet()
    rec = sign_record({"hello": "world", "n": 7}, wallet=w)
    assert isinstance(rec, SignedRecord)
    assert verify_record(rec) is True


def test_tampered_value_fails_verify() -> None:
    w = Wallet()
    rec = sign_record({"hello": "world"}, wallet=w)
    rec.value = {"hello": "evil"}
    assert verify_record(rec) is False


def test_tampered_publisher_fails_verify() -> None:
    w = Wallet()
    other = Wallet()
    rec = sign_record({"x": 1}, wallet=w)
    rec.publisher_pem = other.public_key_pem
    assert verify_record(rec) is False


def test_signed_record_dict_round_trip() -> None:
    w = Wallet()
    rec = sign_record({"a": 1}, wallet=w)
    body = rec.to_dict()
    assert verify_record(SignedRecord.from_dict(body)) is True


def test_signed_record_from_dict_rejects_malformed() -> None:
    with pytest.raises(SignedRecordError):
        SignedRecord.from_dict({"value": 1})  # missing keys


# ---------------------------------------------------------------------------
# DHTStorage
# ---------------------------------------------------------------------------

def test_dht_storage_put_get_round_trip(tmp_path: Path) -> None:
    storage = DHTStorage(path=tmp_path / "dht.json", capacity=10)
    w = Wallet()
    rec = sign_record({"foo": 42}, wallet=w)
    assert storage.put("k1", rec) is True
    out = storage.get("k1")
    assert out is not None
    assert out.value == {"foo": 42}


def test_dht_storage_rejects_unverifiable(tmp_path: Path) -> None:
    storage = DHTStorage(path=tmp_path / "dht.json", capacity=10)
    w = Wallet()
    rec = sign_record({"x": 1}, wallet=w)
    rec.value = {"x": 999}  # tamper
    assert storage.put("k1", rec) is False
    assert storage.get("k1") is None


def test_dht_storage_lru_eviction(tmp_path: Path) -> None:
    storage = DHTStorage(path=tmp_path / "dht.json", capacity=3)
    w = Wallet()
    for i in range(5):
        rec = sign_record({"i": i}, wallet=w)
        assert storage.put(f"k{i}", rec)
    # Capacity 3: only k2, k3, k4 remain
    assert len(storage) == 3
    for i in range(2):
        assert storage.get(f"k{i}") is None
    for i in range(2, 5):
        assert storage.get(f"k{i}") is not None


def test_dht_storage_ttl_expiry(tmp_path: Path) -> None:
    storage = DHTStorage(path=tmp_path / "dht.json", capacity=10)
    w = Wallet()
    rec = sign_record({"y": 1}, wallet=w)
    storage.put("kshort", rec, ttl_s=0.05)
    time.sleep(0.1)
    assert storage.get("kshort") is None
    assert storage.clean_expired() == 0  # already cleaned by get


def test_dht_storage_last_writer_wins(tmp_path: Path) -> None:
    storage = DHTStorage(path=tmp_path / "dht.json", capacity=10)
    w = Wallet()
    rec_old = sign_record({"v": 1}, wallet=w)
    time.sleep(0.01)
    rec_new = sign_record({"v": 2}, wallet=w)
    storage.put("k", rec_new)
    # Putting the older record (timestamp < new) should be rejected
    assert storage.put("k", rec_old) is False
    assert storage.get("k").value == {"v": 2}


def test_dht_storage_persists_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "dht.json"
    storage = DHTStorage(path=path, capacity=10)
    w = Wallet()
    storage.put("k1", sign_record({"v": "hello"}, wallet=w))
    storage.flush()

    storage2 = DHTStorage(path=path, capacity=10)
    out = storage2.get("k1")
    assert out is not None
    assert out.value == {"v": "hello"}
    assert verify_record(out)


def test_dht_storage_load_drops_expired_records(tmp_path: Path) -> None:
    path = tmp_path / "dht.json"
    storage = DHTStorage(path=path, capacity=10)
    w = Wallet()
    storage.put("kshort", sign_record({"v": 1}, wallet=w), ttl_s=0.01)
    storage.put("klong", sign_record({"v": 2}, wallet=w), ttl_s=3600)
    storage.flush()
    time.sleep(0.05)
    # On reload, only the long-TTL record survives
    storage2 = DHTStorage(path=path, capacity=10)
    assert storage2.get("kshort") is None
    assert storage2.get("klong") is not None
