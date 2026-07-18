"""Gossip-propagated peer discovery — find one, find all.

The seed-based path (`infrastructure.seed_node`) scales to ~10k
registrations per seed before the TCP listener becomes the
bottleneck. For the user's stated target of 100s–millions of nodes,
the seed is the bootstrap layer only; **steady-state membership
propagates peer-to-peer** through an epidemic protocol:

  1. Every node maintains a local `MembershipView` — a dict keyed by
     pubkey, holding the peer's last-known (ip, port, score,
     last_seen).
  2. Every `GOSSIP_INTERVAL_S`, the node picks `FANOUT_K` random
     known peers and asks each one for ITS membership view (`GET
     /peers`). The local view becomes the union of itself and the
     replies.
  3. New peers added via gossip get their `/v1/hardware` fetched
     once on first sight so the auction sees real GPU info, not
     a flat template.
  4. Stale peers (`last_seen` older than `STALE_TTL_S`) get pruned;
     a node that vanishes from one peer's view ages out
     mesh-wide within `STALE_TTL_S` of its last heartbeat.

Properties at scale:
  * **O(log N) convergence** — well-mixed gossip propagates a new
    join across the whole mesh in ~log_K(N) rounds at fanout K.
  * **No central authority** — once any node has even ONE neighbour
    via the seed, gossip can carry it to the rest of the mesh
    without the seed ever being contacted again.
  * **Partition-tolerant** — partitions converge separately;
    healing the partition (one node in each side meets one node
    on the other) re-merges them in O(log N) more rounds.
  * **Self-healing** — a peer going offline is removed locally
    when its TTL expires; the rest of the mesh re-routes around
    it via the auction's bid abstention.

This is layer 2 of the three-layer topology described in the
WORKLOG entry for this commit. Layer 3 (consortium auction) lives
in `core.consortium_auction`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

GOSSIP_INTERVAL_S = float(__import__("os").environ.get(
    "PLUGINFER_GOSSIP_INTERVAL_S", "8.0"
))
GOSSIP_FANOUT_K = int(__import__("os").environ.get(
    "PLUGINFER_GOSSIP_FANOUT_K", "4"
))
STALE_TTL_S = float(__import__("os").environ.get(
    "PLUGINFER_GOSSIP_STALE_TTL_S", "180.0"
))


# ---------------------------------------------------------------------------
# Membership view
# ---------------------------------------------------------------------------

def peer_base_url(ip: str, port: int) -> str:
    """Base URL for reaching a peer. Port 443 means the peer sits
    behind TLS (a reverse proxy, load balancer, or tunnel like
    Cloudflare/ngrok), so we speak https; every other port is plain
    http as before. This is what lets a node advertise a public
    hostname instead of only a raw LAN ip:port."""
    if int(port) == 443:
        return f"https://{ip}"
    return f"http://{ip}:{port}"


@dataclass
class PeerEntry:
    pubkey_pem: str
    ip: str
    port: int
    node_version: str = "1.0.0"
    last_seen_unix: float = field(default_factory=time.time)
    # Optional advertised score from /v1/hardware. None = not fetched yet.
    score: Optional[float] = None
    device_type: Optional[str] = None

    def url(self) -> str:
        return peer_base_url(self.ip, self.port)

    def to_wire(self) -> Dict[str, Any]:
        return {
            "pubkey_pem": self.pubkey_pem,
            "ip": self.ip,
            "port": self.port,
            "node_version": self.node_version,
            "last_seen_unix": self.last_seen_unix,
            "score": self.score,
            "device_type": self.device_type,
        }

    @classmethod
    def from_wire(cls, d: Dict[str, Any]) -> "PeerEntry":
        return cls(
            pubkey_pem=str(d.get("pubkey_pem") or ""),
            ip=str(d.get("ip") or ""),
            port=int(d.get("port") or 0),
            node_version=str(d.get("node_version") or "1.0.0"),
            last_seen_unix=float(d.get("last_seen_unix") or time.time()),
            score=(float(d["score"]) if d.get("score") is not None else None),
            device_type=d.get("device_type"),
        )


@dataclass
class MembershipView:
    """Each node's local view of the mesh. Keyed by pubkey so a peer
    moving IP/port doesn't fork into two entries."""
    own_pubkey: str
    peers: Dict[str, PeerEntry] = field(default_factory=dict)

    def add_or_update(self, entry: PeerEntry) -> bool:
        """Merge `entry` into the view. Returns True iff this was a
        first sight (caller can probe hardware on first sight)."""
        if not entry.pubkey_pem or entry.pubkey_pem == self.own_pubkey:
            return False
        existing = self.peers.get(entry.pubkey_pem)
        if existing is None:
            self.peers[entry.pubkey_pem] = entry
            return True
        # Take the newer last_seen (we trust our own clock over the
        # remote's — gossip's lamport-style would be over-engineered).
        existing.last_seen_unix = max(existing.last_seen_unix, entry.last_seen_unix)
        # IP/port may have changed (NAT shift, restart on new port).
        existing.ip = entry.ip
        existing.port = entry.port
        if entry.score is not None:
            existing.score = entry.score
        if entry.device_type is not None:
            existing.device_type = entry.device_type
        return False

    def reap_stale(self, ttl_s: float = STALE_TTL_S) -> int:
        """Drop peers whose last_seen is older than `ttl_s`. Returns
        the count removed."""
        cutoff = time.time() - ttl_s
        stale = [pk for pk, p in self.peers.items() if p.last_seen_unix < cutoff]
        for pk in stale:
            del self.peers[pk]
        return len(stale)

    def random_sample(self, k: int) -> List[PeerEntry]:
        """Pick `k` random peers from the current view (without
        replacement). Used by the gossip loop to pick its outbound
        targets each round."""
        if not self.peers:
            return []
        ks = list(self.peers.keys())
        random.shuffle(ks)
        return [self.peers[pk] for pk in ks[:max(0, k)]]

    def to_wire_list(self, exclude_pubkey: Optional[str] = None) -> List[Dict[str, Any]]:
        """Snapshot for an outbound `/peers` reply. Excludes the
        querier (no need to echo their own entry back)."""
        out = []
        for p in self.peers.values():
            if exclude_pubkey and p.pubkey_pem == exclude_pubkey:
                continue
            out.append(p.to_wire())
        return out

    def __len__(self) -> int:
        return len(self.peers)


# ---------------------------------------------------------------------------
# Wire IO — defaults to plain HTTP; injectable for tests
# ---------------------------------------------------------------------------

def _http_get_json(url: str, *, timeout: float = 3.0) -> Optional[dict]:
    try:
        # Private-swarm key rides on every gossip/announce call so keyed
        # peers accept us; a no-op (empty headers) on public meshes.
        from core.swarm_auth import auth_headers
        req = urllib.request.Request(url, headers=auth_headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


PeerFetcher = Callable[[str], Optional[dict]]


# ---------------------------------------------------------------------------
# Self-announce — every gossip round carries the querier's identity so
# the responder can fold the querier into its own view. Without this,
# a node that nobody else knows about (e.g. a fresh join via a single
# bootstrap peer) stays invisible to the rest of the mesh.
# ---------------------------------------------------------------------------

def _announce_query_string(own_pubkey: str, own_ip: str, own_port: int,
                           own_node_version: str = "1.0.0") -> str:
    import urllib.parse
    if not (own_pubkey and own_ip and own_port):
        return ""
    qs = urllib.parse.urlencode({
        "from_pubkey": own_pubkey,
        "from_ip": own_ip,
        "from_port": str(int(own_port)),
        "from_version": own_node_version,
    })
    return "?" + qs


# ---------------------------------------------------------------------------
# Gossip round — one pass over the fanout sample
# ---------------------------------------------------------------------------

async def gossip_round(
    view: MembershipView,
    *,
    fanout_k: int = GOSSIP_FANOUT_K,
    fetcher: PeerFetcher = _http_get_json,
    on_new_peer: Optional[Callable[[PeerEntry], None]] = None,
    own_ip: str = "",
    own_port: int = 0,
    own_node_version: str = "1.0.0",
) -> int:
    """Execute one gossip round. Returns the count of new peers
    learned this round (zero when the view is stable).

    When `own_ip` and `own_port` are set, each /peers query carries
    the local node's identity as query params so the responder can
    fold the querier into its own view. This is what makes
    "find one, find all" work: a node bootstrapping via a single
    known peer becomes visible to the rest of the mesh on its
    very next outbound gossip query, without needing a separate
    /announce roundtrip."""
    targets = view.random_sample(fanout_k)
    if not targets:
        return 0
    new_count = 0
    qs = _announce_query_string(
        view.own_pubkey, own_ip, own_port, own_node_version,
    )
    for peer in targets:
        url = f"{peer.url()}/peers{qs}"
        payload = await asyncio.get_running_loop().run_in_executor(
            None, fetcher, url,
        )
        if not payload:
            continue
        peer.last_seen_unix = time.time()
        # The auto_mesh /peers endpoint returns
        # {"me": {...}, "discovered_peers": [<wire>...], ...}
        discovered = payload.get("discovered_peers") or payload.get("peers") or []
        # Include the responder's own /me block so we record their
        # advertised metadata too — gossip propagates by both
        # "tell me who you know" AND "your own existence."
        me_block = payload.get("me") or {}
        if isinstance(me_block, dict) and me_block.get("pubkey"):
            try:
                me_entry = PeerEntry(
                    pubkey_pem=str(me_block["pubkey"]),
                    ip=peer.ip, port=peer.port,
                )
                if view.add_or_update(me_entry) and on_new_peer is not None:
                    on_new_peer(me_entry)
                    new_count += 1
            except Exception:
                pass
        for raw in discovered:
            if not isinstance(raw, dict):
                continue
            try:
                e = PeerEntry.from_wire(raw)
            except Exception:
                continue
            if not e.pubkey_pem:
                continue
            was_new = view.add_or_update(e)
            if was_new:
                new_count += 1
                if on_new_peer is not None:
                    try:
                        on_new_peer(e)
                    except Exception:
                        logger.debug("on_new_peer callback raised", exc_info=True)
    view.reap_stale()
    return new_count


async def gossip_loop(
    view: MembershipView,
    *,
    interval_s: float = GOSSIP_INTERVAL_S,
    fanout_k: int = GOSSIP_FANOUT_K,
    fetcher: PeerFetcher = _http_get_json,
    on_new_peer: Optional[Callable[[PeerEntry], None]] = None,
    stop: Optional[asyncio.Event] = None,
) -> None:
    """Long-running gossip loop. `stop` (asyncio.Event) cancels gracefully."""
    while True:
        if stop is not None and stop.is_set():
            return
        try:
            await gossip_round(
                view, fanout_k=fanout_k, fetcher=fetcher,
                on_new_peer=on_new_peer,
            )
        except Exception as e:
            logger.warning("gossip_round raised: %s", e)
        await asyncio.sleep(interval_s)


__all__ = [
    "GOSSIP_FANOUT_K",
    "GOSSIP_INTERVAL_S",
    "MembershipView",
    "PeerEntry",
    "STALE_TTL_S",
    "gossip_loop",
    "gossip_round",
]
