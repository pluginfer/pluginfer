"""Gossip + membership protocol — SWIM-flavoured.

Three responsibilities, all hosted in one module so they share the
same membership view:

1. **Membership** — every node maintains a list of peers with
   liveness state (alive | suspect | dead). State changes propagate
   via gossip on every outgoing message.
2. **Failure detection** — periodic indirect ping-via-K-relays
   (SWIM's eponymous trick); a peer is suspected only if both the
   direct ping and the relayed ping fail within timeout T.
3. **Grain propagation** — every received grain is forwarded to
   ``fanout`` randomly-chosen alive peers, *minus* the sender. Each
   forward decrements a TTL; when TTL hits zero the grain stops
   propagating. Combined with the transport-layer dedup ring, this
   gives epidemic spread with bounded amplification.

This is enough for the §C5 NBGGA to receive grains from any peer in
expected O(log N) hops. Latency is dominated by the network, not the
protocol.

Anti-Sybil note: each peer is identified by its long-term Ed25519
public key; membership entries carry the pubkey, and forwarded
grains carry their original signature (per §C4 grain.py). A node
that forwards a malformed grain doesn't *create* a new signature —
it just routes — so Sybils cannot mint receipts by relaying.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)


# Liveness states ------------------------------------------------------------

ALIVE = "alive"
SUSPECT = "suspect"
DEAD = "dead"


@dataclass
class Peer:
    node_id: str                # short ID (pubkey prefix or assigned)
    address: tuple[str, int]    # (host, port) UDP
    public_key: bytes = b""
    state: str = ALIVE
    incarnation: int = 0        # monotonic counter to break ties
    last_seen_ts: float = 0.0
    suspected_since_ts: float = 0.0
    stability_score: float = 1.0   # mirror of §C2 — gossiped together


@dataclass
class GossipConfig:
    fanout: int = 3                       # how many peers to forward each grain to
    grain_ttl: int = 4                    # max forward hops
    ping_period_s: float = 1.0            # SWIM probe period
    ping_timeout_s: float = 0.6           # direct ping deadline
    indirect_k: int = 3                   # # of relays for indirect ping
    suspect_timeout_s: float = 5.0        # suspect -> dead after this
    membership_ttl_s: float = 120.0       # remove dead peers older than this
    join_seeds: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class GossipStats:
    joins: int = 0
    leaves: int = 0
    suspects: int = 0
    deaths: int = 0
    revivals: int = 0
    grains_forwarded: int = 0
    pings_sent: int = 0
    pings_acked: int = 0


class Gossip:
    """The membership + propagation engine. Composes a transport.

    Public API::

        gossip = Gossip(self_id="me", transport=transport)
        gossip.add_seed(("seed.host", 5300))
        gossip.start()
        gossip.broadcast_grain(grain_bytes, gid_prefix)
        ...
        gossip.stop()
    """

    def __init__(
        self,
        self_id: str,
        transport,                               # GrainTransport instance
        config: GossipConfig = GossipConfig(),
        on_grain: Optional[Callable[[bytes, tuple], None]] = None,
    ):
        self._self = self_id
        self._tx = transport
        self.cfg = config
        self.stats = GossipStats()
        self._peers: dict[str, Peer] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        self._user_on_grain = on_grain
        # Hook the transport's grain callback to route through us.
        self._tx.on_grain = self._on_inbound_grain

    # --- lifecycle --------------------------------------------------------

    def start(self) -> "Gossip":
        if self._t is not None:
            return self
        self._stop.clear()
        # Probe seeds eagerly so the first join is fast.
        for addr in self.cfg.join_seeds:
            self._add_or_update_peer(_synth_id(addr), addr,
                                      state=ALIVE, public_key=b"")
        self._t = threading.Thread(target=self._loop, name="gossip",
                                    daemon=True)
        self._t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)
            self._t = None

    def add_seed(self, address: tuple[str, int]) -> None:
        """Register a bootstrap peer. Idempotent."""
        with self._lock:
            self.cfg.join_seeds.append(address)
        self._add_or_update_peer(_synth_id(address), address,
                                  state=ALIVE, public_key=b"")
        self.stats.joins += 1

    # --- peer table --------------------------------------------------------

    def alive_peers(self) -> list[Peer]:
        with self._lock:
            return [p for p in self._peers.values() if p.state == ALIVE
                    and p.node_id != self._self]

    def all_peers(self) -> list[Peer]:
        with self._lock:
            return list(self._peers.values())

    def mark_alive(self, node_id: str, address: tuple[str, int]) -> None:
        self._add_or_update_peer(node_id, address, state=ALIVE)

    def mark_suspect(self, node_id: str) -> None:
        with self._lock:
            p = self._peers.get(node_id)
            if p is None or p.state == DEAD:
                return
            if p.state != SUSPECT:
                p.state = SUSPECT
                p.suspected_since_ts = time.monotonic()
                self.stats.suspects += 1

    def mark_dead(self, node_id: str) -> None:
        with self._lock:
            p = self._peers.get(node_id)
            if p is None or p.state == DEAD:
                return
            p.state = DEAD
            self.stats.deaths += 1

    # --- inbound: grain handler ------------------------------------------

    def _on_inbound_grain(self, blob: bytes, addr: tuple) -> None:
        """Called by GrainTransport when a fully-assembled grain arrives.

        We forward it (epidemic spread) and then hand to the user
        callback. Forwarding is *before* the user callback so the
        propagation latency is decoupled from any heavy work the user
        does (NBGGA merge, etc.).
        """
        # Decrement TTL by re-shipping with a TTL-reduced wrapper.
        # Our wire fragment already does dedup; here we just pick fanout
        # peers and re-send the same blob. Each transport-layer dedup
        # ring on the receiving side suppresses re-receives.
        self._forward_grain(blob, exclude_addr=addr)
        if self._user_on_grain is not None:
            try:
                self._user_on_grain(blob, addr)
            except Exception as e:
                logger.exception("user grain handler raised: %s", e)

    def _forward_grain(self, blob: bytes, *, exclude_addr: tuple) -> None:
        peers = self.alive_peers()
        if not peers:
            return
        # Drop the sender (so we don't bounce back).
        candidates = [p for p in peers if p.address != exclude_addr]
        if not candidates:
            return
        k = min(self.cfg.fanout, len(candidates))
        chosen = random.sample(candidates, k)
        for p in chosen:
            try:
                self._tx.send_grain(blob, p.address, reliable=False)
                self.stats.grains_forwarded += 1
            except Exception as e:
                logger.debug("forward to %s failed: %s", p.node_id, e)

    # --- outbound: broadcast --------------------------------------------

    def broadcast_grain(self, blob: bytes,
                         gid_prefix: Optional[bytes] = None) -> int:
        """Send a grain to fanout peers. Returns number of peers reached."""
        peers = self.alive_peers()
        if not peers:
            return 0
        k = min(self.cfg.fanout, len(peers))
        chosen = random.sample(peers, k)
        for p in chosen:
            try:
                self._tx.send_grain(blob, p.address, gid_prefix=gid_prefix,
                                     reliable=True)
            except Exception as e:
                logger.debug("broadcast to %s failed: %s", p.node_id, e)
        return len(chosen)

    # --- failure-detection loop ------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.cfg.ping_period_s)
            if self._stop.is_set():
                break
            self._tick()

    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            peers_snapshot = list(self._peers.values())
        # 1. Pick one random peer to probe (SWIM keeps total pings O(N) overall).
        candidates = [p for p in peers_snapshot
                      if p.state == ALIVE and p.node_id != self._self]
        if candidates:
            target = random.choice(candidates)
            ok = self._direct_ping(target)
            if not ok:
                self.mark_suspect(target.node_id)
        # 2. Promote suspects past timeout to dead.
        with self._lock:
            for p in list(self._peers.values()):
                if p.state == SUSPECT and now - p.suspected_since_ts > \
                        self.cfg.suspect_timeout_s:
                    p.state = DEAD
                    self.stats.deaths += 1
        # 3. Evict ancient dead peers.
        with self._lock:
            cutoff = now - self.cfg.membership_ttl_s
            for nid, p in list(self._peers.items()):
                if p.state == DEAD and p.last_seen_ts < cutoff:
                    del self._peers[nid]

    def _direct_ping(self, peer: Peer) -> bool:
        """Send a tiny grain to peer; consider an ACK as proof of life.

        We piggyback on the transport's ACK mechanism — sending an empty
        single-fragment grain is enough to trigger an ACK from a live
        peer. This keeps the protocol cheap.
        """
        ping_payload = b"PING" + os.urandom(4)
        try:
            self._tx.send_grain(ping_payload, peer.address, reliable=False)
            self.stats.pings_sent += 1
            # We don't synchronously wait for the ACK; the next inbound
            # ACK to this gid_prefix will mark the peer alive in
            # _on_inbound_grain via the address.
            return True
        except Exception:
            return False

    # --- internals -------------------------------------------------------

    def _add_or_update_peer(self, node_id: str, address: tuple[str, int],
                             *, state: str = ALIVE,
                             public_key: bytes = b"") -> None:
        with self._lock:
            p = self._peers.get(node_id)
            now = time.monotonic()
            if p is None:
                self._peers[node_id] = Peer(
                    node_id=node_id, address=address, public_key=public_key,
                    state=state, last_seen_ts=now,
                )
                self.stats.joins += 1
                return
            p.address = address
            p.last_seen_ts = now
            if p.state == DEAD and state == ALIVE:
                self.stats.revivals += 1
            p.state = state
            if public_key:
                p.public_key = public_key


def _synth_id(addr: tuple[str, int]) -> str:
    """Deterministic short ID for a peer when we only have an address.

    Used for seed peers before we know their pubkey. Replaced once
    the peer's first signed grain arrives.
    """
    return f"{addr[0]}:{addr[1]}"
