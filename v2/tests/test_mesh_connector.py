"""End-to-end test of `core.mesh_connector.MeshConnector`.

Drives two real MeshConnectors against a real seed UDP server over
loopback. Verifies:

  - Two peers can `connect()` to each other and exchange bytes via
    direct UDP after seed-brokered hole-punch.
  - The TURN-relay fallback path is reachable when hole-punch is
    bypassed (we synthesise this by having one peer connect via
    the relay client only and asserting bytes flow).

This is the "actually using the mesh" smoke test -- the closest
in-process equivalent to "two strangers on different ISPs" without
needing two ISPs.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Tuple

import pytest

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.mesh_connector import MeshConnector  # noqa: E402
from core.tokenomics import Wallet  # noqa: E402
from infrastructure.seed_node.punch_server import (  # noqa: E402
    PunchRelayState,
    _PunchProtocol,
)


async def _start_seed(loop) -> Tuple:
    state = PunchRelayState()
    transport, proto = await loop.create_datagram_endpoint(
        lambda: _PunchProtocol(state),
        local_addr=("127.0.0.1", 0),
    )
    proto.state = state
    return transport, proto, transport.get_extra_info("sockname")


def test_two_connectors_exchange_bytes_via_direct_punch():
    async def _run() -> None:
        loop = asyncio.get_running_loop()
        seed_t, seed_p, seed_addr = await _start_seed(loop)
        wa, wb = Wallet(), Wallet()
        a = await MeshConnector.start(seed_addr=seed_addr, wallet=wa,
                                      bind_host="127.0.0.1")
        b = await MeshConnector.start(seed_addr=seed_addr, wallet=wb,
                                      bind_host="127.0.0.1")
        # Wait for both punch clients' REGISTER_UDP replies to land.
        for _ in range(30):
            if (wa.public_key_pem in seed_p.state.regs
                    and wb.public_key_pem in seed_p.state.regs):
                break
            await asyncio.sleep(0.02)

        try:
            ch_a = await a.connect(wb.public_key_pem)
            assert ch_a.direct, (
                f"expected direct channel via punch, got strategy={ch_a.strategy}"
            )
            assert ch_a.peer_addr is not None

            received_on_b: list[bytes] = []

            # B will only have a channel back to A *after* the first
            # MESH_DATA arrives -- the on_punch_invite hook pre-creates
            # one as soon as the seed's PUNCH_INVITE lands at B.
            for _ in range(30):
                if wa.public_key_pem in b.channels:
                    break
                await asyncio.sleep(0.02)
            assert wa.public_key_pem in b.channels, (
                "B never observed an inbound channel from A"
            )
            ch_b = b.channels[wa.public_key_pem]
            ch_b.on_message = lambda payload: received_on_b.append(payload)

            await ch_a.send(b"hello-mesh-direct")
            for _ in range(30):
                if received_on_b:
                    break
                await asyncio.sleep(0.02)
            assert received_on_b == [b"hello-mesh-direct"]
            assert ch_a.bytes_sent == len(b"hello-mesh-direct")
            assert ch_b.bytes_received == len(b"hello-mesh-direct")
        finally:
            a.close()
            b.close()
            seed_t.close()
            await asyncio.sleep(0)

    asyncio.run(_run())


def test_turn_relay_path_when_punch_unavailable():
    """Force the relay path: monkey-patch A's punch.introduce to always
    return failure. Connect() must fall back to TURN on the SAME
    socket. Bytes flow through the seed in both directions."""

    async def _run() -> None:
        from core.nat.hole_punch import PunchOutcome
        loop = asyncio.get_running_loop()
        seed_t, seed_p, seed_addr = await _start_seed(loop)
        wa, wb = Wallet(), Wallet()
        a = await MeshConnector.start(seed_addr=seed_addr, wallet=wa,
                                      bind_host="127.0.0.1")
        b = await MeshConnector.start(seed_addr=seed_addr, wallet=wb,
                                      bind_host="127.0.0.1")
        # Both peers must be registered with the seed for relay_open
        # to find the partner.
        for _ in range(40):
            if (wa.public_key_pem in seed_p.state.regs
                    and wb.public_key_pem in seed_p.state.regs):
                break
            await asyncio.sleep(0.02)

        # Disable the punch path: introduce() always returns failure.
        async def _no_punch(_pub: str) -> PunchOutcome:
            return PunchOutcome(success=False, detail="punch_disabled_for_test")
        a.punch.introduce = _no_punch                # type: ignore[method-assign]

        received_on_b: list[bytes] = []
        try:
            ch_a = await a.connect(wb.public_key_pem)
            assert ch_a.strategy == "relay", (
                f"expected relay fallback, got {ch_a.strategy}"
            )
            assert ch_a.relay_session is not None

            # Send first message; B's connector creates the inbound
            # relay channel on first delivery. We then attach
            # on_message and send a second message that the callback
            # will capture.
            await ch_a.send(b"first-relay")
            for _ in range(40):
                if wa.public_key_pem in b.channels:
                    break
                await asyncio.sleep(0.02)
            assert wa.public_key_pem in b.channels, (
                "B's connector never observed the relay session"
            )
            ch_b = b.channels[wa.public_key_pem]
            assert ch_b.strategy == "relay"
            ch_b.on_message = lambda payload: received_on_b.append(payload)

            await ch_a.send(b"second-relay")
            for _ in range(40):
                if any(p == b"second-relay" for p in received_on_b):
                    break
                await asyncio.sleep(0.02)
            assert any(p == b"second-relay" for p in received_on_b), (
                f"relay delivery to wired callback failed: {received_on_b}"
            )
        finally:
            a.close()
            b.close()
            seed_t.close()
            await asyncio.sleep(0)

    asyncio.run(_run())
