"""
Global Mesh Bootstrap (Cross-Continent Auto-Discovery)
======================================================
Why this file exists
--------------------
`core/discovery.py` only works on a /24 LAN (UDP broadcast). To form
a mesh across continents, a node needs:

    1. A list of always-on **bootstrap peers** to connect to first
       (operator-run; published).
    2. A way to **discover bootstrap peers without hard-coding** them
       (DNS TXT records — survives operator key rotation).
    3. **NAT traversal** so two consumer nodes behind home routers
       can talk directly: UPnP for IGD-capable routers, STUN-based
       hole punching as fallback.
    4. **Peer Exchange (PEX)** — once you've talked to *one* node in
       the mesh you can ask it for its peer list, transitively
       walking the network. Already implemented in
       `complete_mesh_controller.connect_to_peer`.
    5. **Geo-aware preference** — prefer peers on the same continent
       to keep DiLoCo round latency low.

Strategy (run all in parallel, accept the first that wires up):

    Strategy 0  : Local LAN broadcast (existing MeshDiscovery)
    Strategy 1  : Hard-coded community bootstrap list (compiled in)
    Strategy 2  : DNS TXT  bootstrap.pluginfer.network -> peers
    Strategy 3  : Cached peers.json from previous sessions
    Strategy 4  : Optional libp2p / IPFS rendezvous (future)

Whichever strategy returns peers first wins; the others are not
cancelled (they keep populating PEX in the background).

This module deliberately depends on **only the Python stdlib + requests**
so it can run on a fresh consumer laptop without compiling C extensions.
"""

from __future__ import annotations

import json
import logging
import os
import random
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compile-time community bootstrap. Populate before first public release.
# ---------------------------------------------------------------------------
COMMUNITY_BOOTSTRAP_PEERS: List[dict] = [
    # {'host': 'na-east.bootstrap.pluginfer.network', 'port': 9000, 'continent': 'NA'},
    # {'host': 'eu-west.bootstrap.pluginfer.network', 'port': 9000, 'continent': 'EU'},
    # {'host': 'asia.bootstrap.pluginfer.network',    'port': 9000, 'continent': 'AS'},
]

DNS_BOOTSTRAP_DOMAIN = "_pluginfer-seed.bootstrap.pluginfer.network"

# Public STUN servers (RFC-compliant; Google + Cloudflare).
STUN_SERVERS: List[tuple[str, int]] = [
    ("stun.l.google.com", 19302),
    ("stun.cloudflare.com", 3478),
    ("stun1.l.google.com", 19302),
]


@dataclass
class Peer:
    host: str
    port: int
    continent: Optional[str] = None
    last_seen: float = 0.0
    source: str = "unknown"           # 'lan' | 'community' | 'dns' | 'cached' | 'pex'
    rtt_ms: Optional[float] = None
    reputation: float = 0.5


@dataclass
class GlobalBootstrapResult:
    peers: List[Peer] = field(default_factory=list)
    public_endpoint: Optional[tuple[str, int]] = None
    nat_type: str = "unknown"          # 'open' | 'cone' | 'symmetric' | 'unknown'
    strategies_used: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategy 1: Community list (in-process constant)
# ---------------------------------------------------------------------------
def _community_peers() -> List[Peer]:
    return [
        Peer(host=p["host"], port=p["port"],
             continent=p.get("continent"), source="community")
        for p in COMMUNITY_BOOTSTRAP_PEERS
    ]


# ---------------------------------------------------------------------------
# Strategy 2: DNS TXT records
# ---------------------------------------------------------------------------
def _dns_peers(domain: str = DNS_BOOTSTRAP_DOMAIN, timeout: float = 3.0) -> List[Peer]:
    """
    Use DNS TXT records as a Sybil-resistant bootstrap channel.
    TXT format we expect (RFC 1464-flavoured):
        "peer=na-east.bootstrap.pluginfer.network:9000;c=NA"
        "peer=eu-west.bootstrap.pluginfer.network:9000;c=EU"

    No third-party dependency: uses stdlib `socket.getaddrinfo` for A
    fallback if no `dnspython`. If `dnspython` is available, we use it
    properly for TXT lookups.
    """
    peers: List[Peer] = []
    try:
        import dns.resolver  # type: ignore
        resolver = dns.resolver.Resolver()
        resolver.timeout = timeout
        resolver.lifetime = timeout
        try:
            answer = resolver.resolve(domain, "TXT")
            for rdata in answer:
                txt = b"".join(rdata.strings).decode("utf-8", errors="ignore")
                p = _parse_txt_record(txt)
                if p:
                    peers.append(p)
        except Exception as e:
            logger.debug("DNS TXT resolve failed for %s: %s", domain, e)
    except ImportError:
        # No dnspython. Best-effort fall-back: try resolving the bare
        # domain for A records, treat the result as a single bootstrap
        # peer at port 9000.
        try:
            for info in socket.getaddrinfo(domain.lstrip("_").split(".", 1)[1],
                                           9000, socket.AF_UNSPEC,
                                           socket.SOCK_STREAM):
                ip = info[4][0]
                peers.append(Peer(host=ip, port=9000, source="dns"))
        except Exception:
            pass
    return peers


def _parse_txt_record(txt: str) -> Optional[Peer]:
    fields = {}
    for token in txt.split(";"):
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        fields[k.strip()] = v.strip()
    if "peer" not in fields:
        return None
    host_port = fields["peer"]
    if ":" not in host_port:
        return None
    host, port_s = host_port.rsplit(":", 1)
    try:
        return Peer(host=host, port=int(port_s),
                    continent=fields.get("c"), source="dns")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Strategy 3: Cached peers.json
# ---------------------------------------------------------------------------
def _cached_peers(path: str = "peers.json") -> List[Peer]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        logger.debug("peers.json read failed: %s", e)
        return []
    out: List[Peer] = []
    for p in data:
        try:
            out.append(Peer(
                host=p["ip"], port=int(p.get("port", 9000)),
                continent=p.get("continent"),
                last_seen=float(p.get("last_seen", 0.0)),
                reputation=float(p.get("reputation", 0.5)),
                source="cached",
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# STUN: discover (public_ip, public_port, NAT type)
#
# This is a minimal RFC-5389 implementation. We only need a Binding
# Request + the XOR-MAPPED-ADDRESS response. Two probes against
# different STUN servers tell us if the NAT is cone (public_port the
# same for both) or symmetric (public_port differs).
# ---------------------------------------------------------------------------
def _stun_probe(server: tuple[str, int],
                local_port: int,
                timeout: float = 2.0) -> Optional[tuple[str, int]]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("", local_port))
        s.settimeout(timeout)

        # 20-byte STUN binding request: type=0x0001, len=0,
        # magic cookie 0x2112A442, 12-byte transaction ID.
        magic = 0x2112A442
        tx_id = bytes(random.randint(0, 255) for _ in range(12))
        msg = struct.pack("!HHI", 0x0001, 0, magic) + tx_id
        s.sendto(msg, server)
        data, _ = s.recvfrom(2048)
        s.close()

        # Parse XOR-MAPPED-ADDRESS attribute (type 0x0020).
        if len(data) < 20:
            return None
        msg_type, msg_len, _, _ = struct.unpack("!HHI12s", data[:20])
        if msg_type != 0x0101:
            return None
        pos = 20
        end = 20 + msg_len
        while pos + 4 <= end:
            attr_type, attr_len = struct.unpack("!HH", data[pos:pos + 4])
            pos += 4
            if attr_type == 0x0020 and attr_len >= 8:
                _, family, x_port = struct.unpack("!BBH", data[pos:pos + 4])
                port = x_port ^ (magic >> 16)
                if family == 0x01:                      # IPv4
                    x_addr = struct.unpack("!I", data[pos + 4:pos + 8])[0]
                    addr = x_addr ^ magic
                    ip = socket.inet_ntoa(struct.pack("!I", addr))
                    return ip, port
            pos += attr_len + (-attr_len % 4)           # 4-byte alignment
        return None
    except Exception as e:
        logger.debug("STUN probe to %s failed: %s", server, e)
        return None


def discover_public_endpoint(local_port: int = 0,
                             servers: Sequence[tuple[str, int]] = STUN_SERVERS,
                             ) -> tuple[Optional[tuple[str, int]], str]:
    """
    Returns ((public_ip, public_port), nat_type).

    NAT type heuristic (deliberately simple):
      * No response at all       -> 'unknown'
      * One response, port == local -> 'open' or 'cone' (we say 'cone')
      * Different ports per probe  -> 'symmetric'
      * Otherwise                  -> 'cone'
    """
    results = []
    for srv in servers:
        if local_port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tmp:
                tmp.bind(("", 0))
                local_port = tmp.getsockname()[1]
        probe = _stun_probe(srv, local_port)
        if probe is not None:
            results.append(probe)
        if len(results) >= 2:
            break

    if not results:
        return None, "unknown"
    public = results[0]
    if len(results) >= 2 and results[0][1] != results[1][1]:
        return public, "symmetric"
    return public, "cone"


# ---------------------------------------------------------------------------
# Continent-aware ranking
# ---------------------------------------------------------------------------
def _rank_peers(peers: List[Peer], my_continent: Optional[str] = None,
                ) -> List[Peer]:
    def key(p: Peer):
        same_cont = 0 if (my_continent and p.continent == my_continent) else 1
        rtt = p.rtt_ms if p.rtt_ms is not None else 9999
        return (same_cont, -p.reputation, rtt, p.host)
    return sorted(peers, key=key)


def _rtt_probe(host: str, port: int, timeout: float = 2.0) -> Optional[float]:
    t0 = time.time()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, port))
        return (time.time() - t0) * 1000.0
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API: bootstrap_global_mesh
# ---------------------------------------------------------------------------
def bootstrap_global_mesh(my_continent: Optional[str] = None,
                          local_port: int = 0,
                          peers_json_path: str = "peers.json",
                          dns_domain: str = DNS_BOOTSTRAP_DOMAIN,
                          run_stun: bool = True,
                          rtt_top_k: int = 8,
                          ) -> GlobalBootstrapResult:
    """
    Run all bootstrap strategies in parallel, dedupe, RTT-probe the
    best candidates, and return ranked peers + public endpoint info.

    This is what `pluginfer_node.py` should call once at startup,
    then hand the result over to the existing connect_to_peer loop.
    """
    result = GlobalBootstrapResult()
    workers: List[threading.Thread] = []
    bag: List[Peer] = []
    bag_lock = threading.Lock()

    def _add(strategy: str, peers: List[Peer]) -> None:
        with bag_lock:
            for p in peers:
                p.source = strategy
                bag.append(p)
            if peers:
                result.strategies_used.append(strategy)

    workers.append(threading.Thread(
        target=lambda: _add("community", _community_peers()), daemon=True))
    workers.append(threading.Thread(
        target=lambda: _add("dns", _dns_peers(dns_domain)), daemon=True))
    workers.append(threading.Thread(
        target=lambda: _add("cached", _cached_peers(peers_json_path)), daemon=True))

    if run_stun:
        def _stun_thread():
            endpoint, nat = discover_public_endpoint(local_port=local_port)
            with bag_lock:
                result.public_endpoint = endpoint
                result.nat_type = nat
        workers.append(threading.Thread(target=_stun_thread, daemon=True))

    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=5.0)

    # Dedupe by (host, port).
    seen = set()
    unique: List[Peer] = []
    for p in bag:
        key = (p.host, p.port)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)

    # RTT-probe the top candidates so ranking has signal.
    candidates = unique[:max(rtt_top_k * 2, len(unique))]
    rtt_threads = []

    def _probe(p: Peer) -> None:
        p.rtt_ms = _rtt_probe(p.host, p.port)

    for p in candidates:
        t = threading.Thread(target=_probe, args=(p,), daemon=True)
        t.start()
        rtt_threads.append(t)
    for t in rtt_threads:
        t.join(timeout=3.0)

    result.peers = _rank_peers(unique, my_continent=my_continent)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Bootstrap probe (cross-continent)...")
    r = bootstrap_global_mesh(my_continent="NA")
    print(f"  strategies used    : {r.strategies_used}")
    print(f"  public endpoint    : {r.public_endpoint}")
    print(f"  nat type           : {r.nat_type}")
    print(f"  peers (top 5):")
    for p in r.peers[:5]:
        rtt = f"{p.rtt_ms:.1f}ms" if p.rtt_ms else "?"
        print(f"    {p.host}:{p.port} [{p.source}] cont={p.continent} rtt={rtt}")
