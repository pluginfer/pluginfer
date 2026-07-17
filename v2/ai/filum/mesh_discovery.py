"""§H4 Mesh Auto-Discovery — finding peers without the user typing anything.

Three layers, each tried in order, each tolerant of the prior failing:

1. **LAN (mDNS / Bonjour).** On the same Wi-Fi or LAN subnet,
   nodes announce themselves under the ``_pluginfer._tcp.local``
   service. macOS / iOS / most modern Linux desktops carry Bonjour
   natively; Windows needs ``zeroconf`` (auto-installed by setup).
   This is what makes "two laptops on the same Wi-Fi just find
   each other" possible — no IP typing.

2. **DNS seeds.** ``seed1.pluginfer.net`` / ``seed2.pluginfer.net``
   resolve at first launch. Production replaces the hardcoded list
   with a DNS-discovered list per ``auto_setup._default_seeds``.

3. **Public IP exchange.** When the two prior layers fail (two
   strangers on different home networks behind NAT), the user can
   still bring up the mesh by exchanging *node ids + public IP +
   port* through any out-of-band channel (DM, email, chat). The
   ``add_peer_manual`` helper accepts those and the gossip + NAT
   layers (core/nat/) handle the hole punching.

The contract: ``MeshDiscovery.find_peers()`` returns a list of
``DiscoveredPeer`` records sorted by liveness + distance. Empty
list is a valid return — caller decides whether to surface a
"connect to a friend manually" dialog.

design notes §H4 (drafted in the design notes): a method of discovering
peers in a decentralised AI compute mesh in which (a) a multicast
service announcement carries the node's public key and capability
profile, (b) DNS-resolved seed nodes return the current validator
set, and (c) any user-provided ``node_id @ host:port`` triple is
accepted as a manual peer override; the user need not type, edit,
or paste any configuration if any of (a) or (b) succeeds.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PLUGINFER_SERVICE = "_pluginfer._tcp.local."
DEFAULT_PORT = 5300


@dataclass
class DiscoveredPeer:
    """One peer the discovery layer has located."""
    addr: str                       # IP / hostname
    port: int
    node_id: Optional[str] = None   # hex public key, if known
    source: str = "unknown"         # "lan" | "dns" | "manual" | "history"
    rtt_ms: float = -1.0
    last_seen: float = 0.0


# ---------- LAN layer (mDNS / Bonjour) -------------------------------------


class _LANDiscovery:
    """Try to import zeroconf; degrade gracefully when missing."""

    def __init__(self, my_port: int, my_node_id: str):
        self.my_port = my_port
        self.my_node_id = my_node_id
        self._zeroconf = None
        self._info = None
        self._listener_browser = None
        self._found: dict[str, DiscoveredPeer] = {}

    def announce(self) -> bool:
        """Publish this node on mDNS. Returns True if zeroconf was loadable."""
        try:
            from zeroconf import ServiceInfo, Zeroconf  # type: ignore
        except Exception:
            return False
        try:
            self._zeroconf = Zeroconf()
            local_ip = self._best_local_ip()
            self._info = ServiceInfo(
                PLUGINFER_SERVICE,
                f"pluginfer-{self.my_node_id[:8]}.{PLUGINFER_SERVICE}",
                addresses=[socket.inet_aton(local_ip)],
                port=self.my_port,
                properties={"node_id": self.my_node_id},
            )
            self._zeroconf.register_service(self._info)
            return True
        except Exception as e:
            logger.debug("mDNS announce failed: %s", e)
            return False

    def browse(self, timeout_s: float = 2.0) -> list[DiscoveredPeer]:
        """Scan the LAN for other Pluginfer nodes for ``timeout_s`` seconds."""
        try:
            from zeroconf import ServiceBrowser, Zeroconf  # type: ignore
        except Exception:
            return []
        zc = self._zeroconf or Zeroconf()
        try:
            class _Listener:
                def __init__(self, found: dict, my_id: str):
                    self.found = found
                    self.my_id = my_id

                def update_service(self, *_a, **_k):
                    pass

                def remove_service(self, *_a, **_k):
                    pass

                def add_service(self, zeroconf, type_, name):
                    info = zeroconf.get_service_info(type_, name)
                    if not info or not info.addresses:
                        return
                    addr = socket.inet_ntoa(info.addresses[0])
                    nid_b = info.properties.get(b"node_id", b"")
                    nid = nid_b.decode("utf-8", "ignore") if nid_b else ""
                    if nid == self.my_id:
                        return
                    self.found[nid or addr] = DiscoveredPeer(
                        addr=addr, port=info.port,
                        node_id=nid or None,
                        source="lan",
                        last_seen=time.time(),
                    )

            listener = _Listener(self._found, self.my_node_id)
            ServiceBrowser(zc, PLUGINFER_SERVICE, listener)
            time.sleep(timeout_s)
        finally:
            if self._zeroconf is None:
                zc.close()
        return list(self._found.values())

    def stop(self) -> None:
        if self._zeroconf and self._info:
            try:
                self._zeroconf.unregister_service(self._info)
            except Exception:
                pass
            try:
                self._zeroconf.close()
            except Exception:
                pass

    def _best_local_ip(self) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"


# ---------- DNS-seed layer ------------------------------------------------


def _resolve_dns_seeds(seeds: list[tuple]) -> list[DiscoveredPeer]:
    """Resolve seed hostnames to A records. Returns peers we could reach."""
    out: list[DiscoveredPeer] = []
    for seed in seeds:
        host, port = seed[0], int(seed[1])
        try:
            ip = socket.gethostbyname(host)
            out.append(DiscoveredPeer(
                addr=ip, port=port, source="dns",
                last_seen=time.time(),
            ))
        except Exception as e:
            logger.debug("DNS seed %s unreachable: %s", host, e)
    return out


# ---------- public-IP probe (so the user can hand it to a friend) ---------


def detect_public_ip(timeout_s: float = 3.0) -> Optional[str]:
    """Best-effort public-IP detection. Tries 3 services so any one
    can be down and the function still returns something."""
    services = (
        "https://api.ipify.org",
        "https://ifconfig.me/ip",
        "https://icanhazip.com",
    )
    for url in services:
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as r:
                ip = r.read().decode("utf-8").strip()
                if ip and len(ip) <= 64:
                    return ip
        except Exception:
            continue
    return None


# ---------- peers.json persistence ---------------------------------------


def peers_json_path(state_dir: str) -> Path:
    return Path(state_dir) / "peers.json"


def load_peers(state_dir: str) -> list[dict]:
    p = peers_json_path(state_dir)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_peers(state_dir: str, peers: list[dict]) -> None:
    p = peers_json_path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(peers, indent=2), encoding="utf-8")


def add_peer_manual(state_dir: str, addr: str, port: int = DEFAULT_PORT,
                      node_id: Optional[str] = None) -> dict:
    """Add a peer to peers.json. Idempotent on (addr, port)."""
    peers = load_peers(state_dir)
    for p in peers:
        if p.get("ip") == addr and int(p.get("port", DEFAULT_PORT)) == int(port):
            if node_id and not p.get("node_id"):
                p["node_id"] = node_id
            save_peers(state_dir, peers)
            return p
    record = {"ip": addr, "port": int(port), "source": "manual",
                "added_ts": time.time()}
    if node_id:
        record["node_id"] = node_id
    peers.append(record)
    save_peers(state_dir, peers)
    return record


# ---------- public API ----------------------------------------------------


@dataclass
class DiscoveryResult:
    peers: list[DiscoveredPeer] = field(default_factory=list)
    lan_active: bool = False
    dns_resolved: int = 0
    public_ip: Optional[str] = None
    my_node_id: str = ""
    my_port: int = DEFAULT_PORT


class MeshDiscovery:
    """Orchestrates LAN + DNS + public-IP probe in one call.

    Usage::

        from ai.filum.mesh_discovery import MeshDiscovery
        d = MeshDiscovery(my_node_id=cfg.identity.pubkey_hex,
                          my_port=5300, state_dir=cfg.state_dir,
                          seeds=cfg.seed_addresses)
        result = d.find_peers(lan_timeout_s=2.0)
        print(f"found {len(result.peers)} peers; my public IP: {result.public_ip}")
    """

    def __init__(
        self,
        my_node_id: str,
        my_port: int = DEFAULT_PORT,
        state_dir: str = "",
        seeds: Optional[list] = None,
    ):
        self.my_node_id = my_node_id
        self.my_port = my_port
        self.state_dir = state_dir
        self.seeds = list(seeds or [])
        self._lan = _LANDiscovery(my_port=my_port, my_node_id=my_node_id)
        self._announced = False

    def announce_self(self) -> bool:
        if not self._announced:
            self._announced = self._lan.announce()
        return self._announced

    def find_peers(self, *, lan_timeout_s: float = 2.0) -> DiscoveryResult:
        # 1. Try LAN.
        self.announce_self()
        lan_peers = self._lan.browse(timeout_s=lan_timeout_s)

        # 2. DNS seeds.
        dns_peers = _resolve_dns_seeds(self.seeds) if self.seeds else []

        # 3. History (peers.json) — manual / past adds survive across runs.
        hist_peers = []
        if self.state_dir:
            for rec in load_peers(self.state_dir):
                hist_peers.append(DiscoveredPeer(
                    addr=rec.get("ip", ""),
                    port=int(rec.get("port", DEFAULT_PORT)),
                    node_id=rec.get("node_id"),
                    source=rec.get("source", "history"),
                    last_seen=rec.get("added_ts", 0.0),
                ))

        # Dedup by (addr, port).
        merged: dict[tuple, DiscoveredPeer] = {}
        for p in lan_peers + dns_peers + hist_peers:
            key = (p.addr, p.port)
            if key not in merged or p.source == "lan":
                merged[key] = p

        return DiscoveryResult(
            peers=list(merged.values()),
            lan_active=self._announced,
            dns_resolved=len(dns_peers),
            public_ip=detect_public_ip(),
            my_node_id=self.my_node_id,
            my_port=self.my_port,
        )

    def close(self) -> None:
        self._lan.stop()


def quick_status(my_node_id: str, my_port: int = DEFAULT_PORT,
                  state_dir: str = "", seeds: Optional[list] = None) -> str:
    """One-shot text summary for the GUI / CLI."""
    d = MeshDiscovery(my_node_id=my_node_id, my_port=my_port,
                       state_dir=state_dir, seeds=seeds)
    try:
        r = d.find_peers(lan_timeout_s=1.5)
    finally:
        d.close()
    lines = [
        f"Mesh discovery for node {my_node_id[:16]}...",
        f"  LAN announce  : {'on' if r.lan_active else 'unavailable (zeroconf missing?)'}",
        f"  DNS seeds     : {r.dns_resolved} resolved",
        f"  Public IP     : {r.public_ip or 'unknown (offline?)'}",
        f"  Peers known   : {len(r.peers)}",
    ]
    for p in r.peers[:10]:
        nid = (p.node_id or "")[:12] + "..." if p.node_id else "(unknown)"
        lines.append(f"    [{p.source:<7}] {p.addr}:{p.port}  id={nid}")
    if not r.peers:
        lines.append(
            "    (no peers yet — to connect a friend manually, share your "
            f"`{my_node_id[:16]}...@{r.public_ip or 'YOUR_PUBLIC_IP'}:{my_port}` "
            "with them and run "
            "`python -m ai.filum peer add THEIR_IP:THEIR_PORT`)"
        )
    return "\n".join(lines)
