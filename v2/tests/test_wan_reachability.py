"""WAN reachability — the mesh must work across the web, not just WiFi.

Regression class (2026-07-17): `discover_local_ip()` self-reports the
LAN address, the seed stored it verbatim, and remote peers were handed
192.168.x.x — so two strangers on different networks could never dial
each other even though registration "worked". The fix has two halves,
each pinned here:

  1. The seed records + returns the OBSERVED source IP of every
     registration (free STUN; the signed self-reported ip is kept, so
     the ECDSA registration contract is untouched).
  2. auto_mesh adopts the observed public address (and re-registers,
     freshly signed) unless the operator pinned one — with guards so
     localhost test meshes and LAN-seed meshes are never re-pointed.

What is deliberately NOT claimed here: inbound reachability through a
NAT that won't route the node's port (needs port-forward, the §A24
mesh-native HTTP relay, or the §F2 UDP hole-punch path), and real-WAN
end-to-end proof, which requires two physical networks (FIRST_PROOF /
MH1 — off-keyboard).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.tokenomics import Wallet  # noqa: E402
from infrastructure.seed_node.seed_server import (  # noqa: E402
    SeedServer,
    _signed_bytes,
)
from tools.auto_mesh import _should_adopt_observed  # noqa: E402


def _register_msg(wallet: Wallet, ip: str = "192.168.1.50",
                  port: int = 8100) -> dict:
    ts = time.time()
    signed = _signed_bytes(wallet.public_key_pem, ip, port, "1.0.0", ts)
    return {
        "op": "REGISTER",
        "pubkey_pem": wallet.public_key_pem,
        "ip": ip, "port": port, "node_version": "1.0.0",
        "timestamp": ts,
        "signature": wallet.sign(signed),
    }


# ---------------------------------------------------------------------------
# Half 1 — the seed tells the truth about where it saw you.
# ---------------------------------------------------------------------------

def test_register_response_carries_observed_ip():
    s = SeedServer()
    w = Wallet()
    r = s.handle(_register_msg(w, ip="192.168.1.50"),
                 client_ip="203.0.113.7")
    assert r["status"] == "ok"
    assert r["observed_ip"] == "203.0.113.7"


def test_peers_wire_carries_observed_ip():
    s = SeedServer()
    w = Wallet()
    s.handle(_register_msg(w, ip="192.168.1.50"), client_ip="203.0.113.7")
    r = s.handle({"op": "PEERS", "max": 10}, client_ip="198.51.100.9")
    assert r["status"] == "ok"
    (peer,) = r["peers"]
    # Self-reported (signed) ip preserved; observed truth alongside.
    assert peer["ip"] == "192.168.1.50"
    assert peer["observed_ip"] == "203.0.113.7"


# ---------------------------------------------------------------------------
# Half 2 — the node adopts its public address, and ONLY then.
# ---------------------------------------------------------------------------

def test_adopts_public_ip_when_advertising_lan_address():
    # A genuinely global IP (pure predicate — no traffic ever leaves).
    # RFC-5737 doc ranges are deliberately NOT used here: ipaddress
    # classifies them as non-global, and the predicate must refuse
    # them (same class as CGNAT 100.64/10 — not dialable from the
    # WAN, so adopting would advertise a dead address).
    assert _should_adopt_observed("192.168.1.50", "8.8.8.8",
                                  pinned=False)


def test_doc_range_and_cgnat_observed_ips_refused():
    assert not _should_adopt_observed("192.168.1.50", "203.0.113.7",
                                      pinned=False)
    assert not _should_adopt_observed("192.168.1.50", "100.64.0.9",
                                      pinned=False)


def test_never_overrides_operator_pinned_address():
    assert not _should_adopt_observed("192.168.1.50", "8.8.8.8",
                                      pinned=True)


def test_localhost_test_meshes_unchanged():
    # The hermetic two/three-stranger tests run everything on loopback:
    # observed 127.0.0.1 is not global -> never adopted.
    assert not _should_adopt_observed("127.0.0.1", "127.0.0.1",
                                      pinned=False)
    assert not _should_adopt_observed("192.168.1.50", "127.0.0.1",
                                      pinned=False)


def test_lan_seed_meshes_unchanged():
    # Private mesh with the seed on the same LAN: observed is RFC1918,
    # the self-reported LAN ip already works -> no adoption.
    assert not _should_adopt_observed("192.168.1.50", "192.168.1.9",
                                      pinned=False)


def test_already_public_node_not_repointed():
    # A Hetzner box behind a corporate proxy must not be re-pointed to
    # the proxy's egress address.
    assert not _should_adopt_observed("198.51.100.20", "203.0.113.7",
                                      pinned=False)


def test_garbage_observed_ip_ignored():
    assert not _should_adopt_observed("192.168.1.50", "not-an-ip",
                                      pinned=False)
    assert not _should_adopt_observed("192.168.1.50", "",
                                      pinned=False)
