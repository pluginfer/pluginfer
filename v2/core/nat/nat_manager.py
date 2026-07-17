"""NAT manager: orchestrates direct / UPnP / STUN / TURN strategies.

Strategy order (try in sequence, use first that works):

  1. Direct -- no NAT or full-cone NAT. Just bind and listen.
  2. UPnP   -- request a port mapping from the home router.
               Uses `core.networking.NetworkManager.enable_upnp`.
  3. STUN   -- query a public STUN server for the external IP+port.
               Provides reachability info even when UPnP fails (the
               other peer can reach us via UDP hole-punching).
  4. TURN   -- relay through the seed node when symmetric NAT blocks
               direct UDP. Documented but the relay channel is a
               CP-FINAL+ deliverable (depends on a deployed seed).

Returns a `NATDiscovery` describing what worked and the resulting
externally-visible (ip, port).

This module is intentionally small. The actual UPnP code already
lives in core/networking.py (rebuilt in c41c86e to remove the 8.8.8.8
phone-home); we DI it here rather than duplicating it.
"""

from __future__ import annotations

import enum
import logging
import socket
from dataclasses import dataclass
from typing import Optional

from . import stun_client

logger = logging.getLogger(__name__)


class NATStrategy(enum.Enum):
    DIRECT = "direct"
    UPNP = "upnp"
    STUN = "stun"
    TURN = "turn"


@dataclass(frozen=True)
class NATDiscovery:
    strategy: NATStrategy
    external_ip: str
    external_port: int
    detail: str = ""


def discover_local_ip() -> str:
    """Best-effort local IP. RFC 5737 doc range; no DNS leak."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("198.51.100.1", 65535))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return "127.0.0.1"


class NATManager:
    """Discover the externally-reachable (ip, port) for a local UDP/TCP port.

    Usage:

        mgr = NATManager(local_port=8100)
        info = mgr.discover()
        print(info.strategy, info.external_ip, info.external_port)
    """

    def __init__(
        self,
        local_port: int,
        *,
        upnp_manager=None,                 # core.networking.NetworkManager-shaped
        prefer_strategy: Optional[NATStrategy] = None,
    ) -> None:
        self.local_port = int(local_port)
        self.upnp_manager = upnp_manager
        self.prefer_strategy = prefer_strategy

    # ------------------------------------------------------------------

    def discover(self) -> NATDiscovery:
        local_ip = discover_local_ip()

        # If a strategy is forced (testing/explicit), try that first.
        if self.prefer_strategy is not None:
            try:
                return self._try(self.prefer_strategy, local_ip)
            except Exception as e:
                logger.warning(
                    "preferred strategy %s failed: %s",
                    self.prefer_strategy.value, e,
                )

        # 1. Direct - non-RFC1918 local IP suggests no NAT
        if not _is_rfc1918(local_ip):
            return NATDiscovery(
                strategy=NATStrategy.DIRECT,
                external_ip=local_ip,
                external_port=self.local_port,
                detail="public IP detected; no NAT traversal needed",
            )

        # 2. UPnP
        try:
            return self._try(NATStrategy.UPNP, local_ip)
        except Exception as e:
            logger.info("UPnP failed: %s", e)

        # 3. STUN
        try:
            return self._try(NATStrategy.STUN, local_ip)
        except Exception as e:
            logger.info("STUN failed: %s", e)

        # 4. TURN -- requires the seed-node relay protocol; not yet
        # implemented (CP-FINAL+). Return DIRECT with the local IP and
        # mark detail so the caller knows reachability is uncertain.
        return NATDiscovery(
            strategy=NATStrategy.DIRECT,
            external_ip=local_ip,
            external_port=self.local_port,
            detail=("no traversal succeeded; advertising local IP "
                    "(symmetric NAT? deploy seed-relay to enable TURN)"),
        )

    # ------------------------------------------------------------------

    def _try(self, strategy: NATStrategy, local_ip: str) -> NATDiscovery:
        if strategy == NATStrategy.DIRECT:
            return NATDiscovery(
                strategy=NATStrategy.DIRECT,
                external_ip=local_ip,
                external_port=self.local_port,
            )
        if strategy == NATStrategy.UPNP:
            return self._try_upnp(local_ip)
        if strategy == NATStrategy.STUN:
            return self._try_stun(local_ip)
        if strategy == NATStrategy.TURN:
            # The TURN client (`core.nat.turn_client.TurnRelayClient`)
            # is async (it speaks UDP and lives on the asyncio loop),
            # so it doesn't fit this sync `discover()` shape -- callers
            # that want a relay session call TurnRelayClient.start()
            # directly. `discover()` is for binding-time external-IP
            # discovery only; the TURN path lives at message time.
            raise NotImplementedError(
                "TURN allocation is per-message, not per-bind. Use "
                "core.nat.turn_client.TurnRelayClient.open() at the "
                "moment a peer wants to send and direct UDP failed."
            )
        raise ValueError(f"unknown strategy: {strategy}")

    def _try_upnp(self, local_ip: str) -> NATDiscovery:
        if self.upnp_manager is None:
            raise RuntimeError(
                "no UPnP manager wired; pass `upnp_manager=` to NATManager()"
            )
        # core.networking.NetworkManager.enable_upnp() returns the
        # external IP on success (per c41c86e). We pre-request a mapping
        # for self.local_port -> self.local_port.
        try:
            external_ip = self.upnp_manager.enable_upnp(self.local_port)
        except TypeError:
            # Older signature: enable_upnp() with no args
            external_ip = self.upnp_manager.enable_upnp()
        if not external_ip:
            raise RuntimeError("UPnP returned no external IP")
        return NATDiscovery(
            strategy=NATStrategy.UPNP,
            external_ip=external_ip,
            external_port=self.local_port,
            detail="UPnP IGD AddPortMapping",
        )

    def _try_stun(self, local_ip: str) -> NATDiscovery:
        result = stun_client.discover_external_address_sync(
            bind_port=self.local_port,
        )
        return NATDiscovery(
            strategy=NATStrategy.STUN,
            external_ip=result.external_ip,
            external_port=result.external_port,
            detail=f"STUN via {result.server}",
        )


def _is_rfc1918(ip: str) -> bool:
    """True for 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, plus loopback."""
    if ip.startswith("10.") or ip.startswith("192.168."):
        return True
    if ip.startswith("127."):
        return True
    if ip.startswith("172."):
        try:
            second = int(ip.split(".", 2)[1])
            if 16 <= second <= 31:
                return True
        except (ValueError, IndexError):
            pass
    if ip.startswith("169.254."):  # link-local
        return True
    return False
