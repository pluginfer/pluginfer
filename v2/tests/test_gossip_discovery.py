"""Gossip-propagated peer discovery — A finds C via B without ever
asking the seed about C. The "find one, find all" property.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.gossip_discovery import (  # noqa: E402
    MembershipView,
    PeerEntry,
    gossip_round,
)


# ---------------------------------------------------------------------------
# Membership view unit tests
# ---------------------------------------------------------------------------

def test_view_dedupes_by_pubkey():
    view = MembershipView(own_pubkey="me")
    e1 = PeerEntry(pubkey_pem="A", ip="1.1.1.1", port=8100)
    e2 = PeerEntry(pubkey_pem="A", ip="1.1.1.1", port=8100)
    assert view.add_or_update(e1) is True       # first sight
    assert view.add_or_update(e2) is False      # not new — same pubkey
    assert len(view) == 1


def test_view_updates_ip_port_on_restart():
    view = MembershipView(own_pubkey="me")
    view.add_or_update(PeerEntry(pubkey_pem="A", ip="1.1.1.1", port=8100))
    view.add_or_update(PeerEntry(pubkey_pem="A", ip="2.2.2.2", port=8200))
    assert view.peers["A"].ip == "2.2.2.2"
    assert view.peers["A"].port == 8200


def test_view_drops_self_silently():
    view = MembershipView(own_pubkey="me")
    assert view.add_or_update(PeerEntry(pubkey_pem="me", ip="1.1.1.1", port=8100)) is False
    assert len(view) == 0


def test_reap_stale_drops_old_peers():
    view = MembershipView(own_pubkey="me")
    old = PeerEntry(pubkey_pem="A", ip="1.1.1.1", port=8100,
                    last_seen_unix=time.time() - 10_000)
    fresh = PeerEntry(pubkey_pem="B", ip="2.2.2.2", port=8200)
    view.add_or_update(old)
    view.add_or_update(fresh)
    removed = view.reap_stale(ttl_s=180.0)
    assert removed == 1
    assert "B" in view.peers and "A" not in view.peers


# ---------------------------------------------------------------------------
# Gossip round — synthetic three-node mesh
# ---------------------------------------------------------------------------

class _FakeNetwork:
    """In-memory directory of node URLs -> /peers responses. The gossip
    fetcher reads from this dict instead of issuing HTTP."""

    def __init__(self):
        self.endpoints: dict[str, dict] = {}

    def serve(self, url: str, body: dict) -> None:
        self.endpoints[url.rstrip("/")] = body

    def fetch(self, url: str):
        return self.endpoints.get(url.rstrip("/"))


def test_one_round_picks_up_transitive_peer():
    """Node A starts with only Node B in its view (got it from the
    seed). Node B's /peers reports Node C. After one gossip round,
    Node A also knows Node C — without ever asking the seed."""
    net = _FakeNetwork()
    # B advertises itself + C as discovered.
    net.serve("http://1.1.1.2:8100/peers", {
        "me": {"pubkey": "B"},
        "discovered_peers": [
            {"pubkey_pem": "C", "ip": "1.1.1.3", "port": 8100},
        ],
    })

    view = MembershipView(own_pubkey="A")
    view.add_or_update(PeerEntry(pubkey_pem="B", ip="1.1.1.2", port=8100))

    async def _run():
        return await gossip_round(view, fanout_k=4, fetcher=net.fetch)

    new = asyncio.run(_run())
    assert new >= 1
    assert "C" in view.peers
    assert view.peers["C"].ip == "1.1.1.3"


def test_three_rounds_converge_on_full_mesh():
    """Simulate 8 nodes, each known to only one neighbour at start.
    After enough gossip rounds, every node knows every other node."""
    net = _FakeNetwork()
    # Build a ring A->B->C->D->E->F->G->H->A. Each node's /peers
    # only reports its immediate successor (the worst-case for
    # convergence).
    ring = ["A", "B", "C", "D", "E", "F", "G", "H"]
    ips = {n: f"10.0.0.{i + 1}" for i, n in enumerate(ring)}

    # We'll iterate the simulation by mutating per-node views, but
    # the /peers responses must also evolve. Build them dynamically.
    views = {n: MembershipView(own_pubkey=n) for n in ring}
    for i, n in enumerate(ring):
        succ = ring[(i + 1) % len(ring)]
        views[n].add_or_update(
            PeerEntry(pubkey_pem=succ, ip=ips[succ], port=8100)
        )

    def install_endpoints():
        for n in ring:
            net.serve(
                f"http://{ips[n]}:8100/peers",
                {
                    "me": {"pubkey": n},
                    "discovered_peers": views[n].to_wire_list(),
                },
            )

    # Multiple rounds with shuffled order.
    for _ in range(6):
        install_endpoints()
        for n in ring:
            asyncio.run(gossip_round(
                views[n], fanout_k=len(ring),    # max fanout for fastest convergence
                fetcher=net.fetch,
            ))

    # Each view should now contain every other node.
    for n in ring:
        seen = set(views[n].peers.keys())
        expected = set(ring) - {n}
        missing = expected - seen
        assert not missing, f"{n} missing {missing}"


def test_on_new_peer_callback_fires_for_each_first_sight():
    """The hook the live auto-mesh script uses to fetch
    /v1/hardware on first sight of a new peer."""
    net = _FakeNetwork()
    net.serve("http://1.1.1.2:8100/peers", {
        "me": {"pubkey": "B"},
        "discovered_peers": [
            {"pubkey_pem": "C", "ip": "1.1.1.3", "port": 8100},
            {"pubkey_pem": "D", "ip": "1.1.1.4", "port": 8100},
        ],
    })
    view = MembershipView(own_pubkey="A")
    view.add_or_update(PeerEntry(pubkey_pem="B", ip="1.1.1.2", port=8100))

    seen: list = []
    asyncio.run(gossip_round(
        view, fanout_k=2, fetcher=net.fetch,
        on_new_peer=lambda e: seen.append(e.pubkey_pem),
    ))
    assert set(seen) >= {"C", "D"}


def test_offline_peer_eventually_pruned():
    """A peer that stops emitting heartbeats ages out without any
    central coordinator marking it dead."""
    view = MembershipView(own_pubkey="me")
    view.add_or_update(PeerEntry(
        pubkey_pem="dead-node", ip="1.1.1.5", port=8100,
        last_seen_unix=time.time() - 1000,
    ))
    view.add_or_update(PeerEntry(
        pubkey_pem="live-node", ip="1.1.1.6", port=8100,
    ))
    reaped = view.reap_stale(ttl_s=180.0)
    assert reaped == 1
    assert "live-node" in view.peers
    assert "dead-node" not in view.peers


def test_gossip_round_does_not_crash_on_unreachable_peer():
    """A peer that's down returns None from the fetcher; gossip
    skips it cleanly."""
    net = _FakeNetwork()
    # No endpoint for B -> fetch returns None.
    view = MembershipView(own_pubkey="A")
    view.add_or_update(PeerEntry(pubkey_pem="B", ip="1.1.1.2", port=8100))
    new = asyncio.run(gossip_round(view, fanout_k=4, fetcher=net.fetch))
    assert new == 0     # nothing learned, but no exception
