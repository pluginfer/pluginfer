"""NAT traversal toolkit for Pluginfer nodes.

Surfaces:
  - stun_client    -- RFC 5389 STUN client (CP-2)
  - nat_manager    -- DIRECT/UPnP/STUN strategy chain (CP-2)
  - hole_punch     -- seed-brokered UDP hole-punch coordinator
                      (closes ~80% of consumer-router NAT cases)
  - turn_client    -- TURN-relay fallback for symmetric NAT
                      (covers the remaining ~15-20%)
"""

from .stun_client import (
    STUNError,
    STUNResult,
    discover_external_address_async,
    discover_external_address_sync,
)
from .nat_manager import NATManager, NATDiscovery, NATStrategy
from .hole_punch import (
    HolePunchClient,
    HolePunchNotImplementedError,
    PunchOutcome,
)
from .turn_client import RelaySession, TurnRelayClient

__all__ = [
    "STUNError",
    "STUNResult",
    "discover_external_address_async",
    "discover_external_address_sync",
    "NATManager",
    "NATDiscovery",
    "NATStrategy",
    "HolePunchClient",
    "HolePunchNotImplementedError",
    "PunchOutcome",
    "RelaySession",
    "TurnRelayClient",
]
