"""Kademlia LRU ping-and-evict path (W20-partial → closed).

The Kademlia paper §2.2 says: when a k-bucket is full, ping the
least-recently-seen contact; if it responds, refresh it and put
the new contact on the replacement list; if it does NOT respond,
evict the dead LRU and admit the new contact.

These tests pin both branches.
"""

from __future__ import annotations

from core.kademlia import KBucket, Peer


def _peer(i: int) -> Peer:
    return Peer(node_id=i, host="127.0.0.1", port=9000 + i)


def test_full_bucket_with_alive_lru_keeps_lru_and_queues_new():
    bucket = KBucket(capacity=3)
    for i in range(3):
        bucket.add(_peer(i))
    assert [p.node_id for p in bucket.peers] == [0, 1, 2]

    # ping_fn says LRU (peer 0) is alive: it should be refreshed to
    # the back, the new peer queued in replacement.
    bucket.add(_peer(99), ping_fn=lambda _peer_arg: True)
    assert 99 not in [p.node_id for p in bucket.peers]
    assert 99 in [p.node_id for p in bucket.replacement]
    assert bucket.peers[-1].node_id == 0       # LRU bumped to MRU


def test_full_bucket_with_dead_lru_evicts_and_admits_new():
    bucket = KBucket(capacity=3)
    for i in range(3):
        bucket.add(_peer(i))
    # ping_fn says LRU is dead: evict + admit new.
    bucket.add(_peer(99), ping_fn=lambda _peer_arg: False)
    ids = [p.node_id for p in bucket.peers]
    assert 0 not in ids                         # evicted
    assert 99 in ids                            # admitted
    assert ids[-1] == 99                        # at MRU end
    assert ids == [1, 2, 99]


def test_full_bucket_without_ping_fn_uses_replacement_list():
    bucket = KBucket(capacity=3)
    for i in range(3):
        bucket.add(_peer(i))
    # No ping_fn — fall through to replacement-only fallback.
    bucket.add(_peer(99))
    assert 99 in [p.node_id for p in bucket.replacement]
    assert 99 not in [p.node_id for p in bucket.peers]


def test_existing_peer_refresh_does_not_trigger_eviction():
    bucket = KBucket(capacity=3)
    for i in range(3):
        bucket.add(_peer(i))
    # Re-adding peer 0 should just bump it to MRU; ping_fn must NOT fire.
    pings = []

    def _ping(p):
        pings.append(p.node_id)
        return True

    bucket.add(_peer(0), ping_fn=_ping)
    assert pings == []                          # no ping for refresh
    assert bucket.peers[-1].node_id == 0        # bumped to MRU
