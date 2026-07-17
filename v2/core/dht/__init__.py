"""Pluginfer DHT toolkit.

Built on top of the existing `core.kademlia` module (160-bit XOR,
K=20 buckets, ALPHA-parallel iterative lookup). This package adds
signed-record wrapping, persistent LRU+TTL storage, and a bootstrap
helper that wires Kademlia to the seed_node infrastructure.
"""

from .signed_records import (
    SignedRecord,
    SignedRecordError,
    sign_record,
    verify_record,
)
from .dht_storage import DHTStorage, DHTRecord
from .dht_bootstrap import bootstrap_dht_from_seeds

__all__ = [
    "SignedRecord",
    "SignedRecordError",
    "sign_record",
    "verify_record",
    "DHTStorage",
    "DHTRecord",
    "bootstrap_dht_from_seeds",
]
