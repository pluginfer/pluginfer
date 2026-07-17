"""W38 — India ↔ Singapore peer-connect rendezvous (hermetic).

The end-to-end claim being tested:

    Two peers on different home networks, neither knowing the other's
    address, can find each other through a public seed and exchange
    bytes over a punched UDP path — without either peer typing IP
    addresses, opening firewall ports, or running anything but the
    Pluginfer client.

Hermetic version of that claim:

  * One in-process punch_server bound on 127.0.0.1 (the "global seed").
  * Two PeerConnectClients each bound on their own ephemeral 127.0.0.1
    UDP port (the "India peer" and the "Singapore peer"). Both register
    with the seed under their own ECDSA pubkey.
  * India calls connect_to_peer(singapore_pubkey). Seed brokers the
    introduce. Both peers fire PUNCH_HELLO at each other. India's
    connect_to_peer returns ConnectResult(method="direct", ...).
  * India sends a payload over the punched path; Singapore receives it
    on its inbound handler.

Note on "different networks": there is no real NAT in a hermetic test.
The protocol's correctness for cross-NAT is established by the
introduce flow itself (the seed observes each peer's external addr;
peers use the seed's reported addr, not any DNS or hardcoded value).
That mechanic IS exercised here — the peers do NOT know each other's
ports up front; the seed teaches them. Real-NAT verification is
covered by the off-keyboard MH1 task (two-stranger test).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest  # noqa: E402

from core.peer_connect import (  # noqa: E402
    ConnectResult,
    PeerConnectClient,
    SeedAddress,
    connect_to_peer,
)
from core.tokenomics import Wallet  # noqa: E402
from infrastructure.seed_node.punch_server import (  # noqa: E402
    PunchRelayState,
    _PunchProtocol,
)


async def _start_seed(host: str = "127.0.0.1", port: int = 0):
    """Spin up a punch_server bound on (host, port). Returns
    (transport, state, bound_addr) — caller closes when done."""
    state = PunchRelayState()
    loop = asyncio.get_running_loop()
    transport, _proto = await loop.create_datagram_endpoint(
        lambda: _PunchProtocol(state),
        local_addr=(host, port),
    )
    bound_addr = transport.get_extra_info("sockname")
    return transport, state, bound_addr


# ---------------------------------------------------------------------------


def test_two_strangers_punch_via_seed_and_exchange_payload():
    """India peer + Singapore peer + global seed, all on 127.0.0.1.
    Introduce + payload exchange.

    The two peers do NOT know each other's port at start. They each
    register their own pubkey with the seed; India calls
    connect_to_peer(singapore_pubkey); seed brokers the introduce;
    PUNCH_HELLO exchange opens the path; India sends a payload over
    the punched UDP path; Singapore receives it on its inbound handler.
    """

    async def _run():
        # 1. Spin up the global seed.
        seed_transport, _seed_state, seed_addr = await _start_seed()
        try:
            # 2. Two wallets — one for each "regional" peer.
            wallet_in = Wallet()
            wallet_sg = Wallet()

            # 3. Two PeerConnectClients, each bound on an ephemeral
            #    127.0.0.1 port (simulating two different home networks
            #    behind the same seed).
            india = await PeerConnectClient.start(
                seeds=[SeedAddress(host=seed_addr[0], port=seed_addr[1])],
                local_pubkey_pem=wallet_in.public_key_pem,
                sign=wallet_in.sign,
                bind_host="127.0.0.1",
            )
            singapore = await PeerConnectClient.start(
                seeds=[SeedAddress(host=seed_addr[0], port=seed_addr[1])],
                local_pubkey_pem=wallet_sg.public_key_pem,
                sign=wallet_sg.sign,
                bind_host="127.0.0.1",
            )
            try:
                # 4. Wire up Singapore's inbound application handler.
                received_app: list[tuple[bytes, tuple]] = []

                def _on_app(data, addr):
                    received_app.append((data, addr))

                singapore.set_inbound_handler(_on_app)

                # 5. Wait briefly for both REGISTER_UDP replies to land
                #    so the seed has both peers' addrs.
                t_deadline = time.monotonic() + 1.0
                while time.monotonic() < t_deadline and (
                    india.external_addr is None
                    or singapore.external_addr is None
                ):
                    await asyncio.sleep(0.02)

                assert india.external_addr is not None, (
                    "India never received REGISTER_UDP reply; seed unreachable?"
                )
                assert singapore.external_addr is not None, (
                    "Singapore never received REGISTER_UDP reply"
                )

                # 6. India calls connect_to_peer(singapore.pubkey).
                result: ConnectResult = await india.connect_to_peer(
                    target_pubkey_pem=wallet_sg.public_key_pem,
                )
                assert result.success, f"introduce failed: {result.detail}"
                assert result.method == "direct"
                assert result.peer_addr is not None
                # The peer addr we got should match Singapore's
                # externally-observed addr (modulo localhost).
                assert (
                    result.peer_addr[1] == singapore.external_addr[1]
                ), (f"peer port mismatch: got {result.peer_addr}, "
                    f"sg ext {singapore.external_addr}")

                # 7. India sends a non-protocol payload along the
                #    punched path. The handler dispatches it to
                #    Singapore's inbound application handler because
                #    it lacks an "op" key.
                payload = b'{"hello":"singapore","from":"india"}'
                await india.send_to_punched_peer(result.peer_addr, payload)

                # 8. Wait for Singapore to receive.
                t_deadline = time.monotonic() + 1.0
                while time.monotonic() < t_deadline and not received_app:
                    await asyncio.sleep(0.02)

                assert received_app, (
                    "Singapore never received the application payload"
                )
                got, src_addr = received_app[0]
                assert got == payload
                # Source addr should be India's externally-observed addr.
                assert src_addr[1] == india.external_addr[1]
            finally:
                india.close()
                singapore.close()
        finally:
            seed_transport.close()
            # Allow asyncio to settle the close.
            await asyncio.sleep(0.05)

    asyncio.run(_run())


def test_introduce_rejects_unregistered_target():
    """If the dialer asks the seed to introduce a peer that hasn't
    registered, the seed returns peer_not_registered and the
    PeerConnectClient surfaces a clean failure (no traceback)."""

    async def _run():
        seed_transport, _state, seed_addr = await _start_seed()
        try:
            wallet_a = Wallet()
            wallet_b = Wallet()
            client = await PeerConnectClient.start(
                seeds=[SeedAddress(host=seed_addr[0], port=seed_addr[1])],
                local_pubkey_pem=wallet_a.public_key_pem,
                sign=wallet_a.sign,
                bind_host="127.0.0.1",
            )
            try:
                # B never registered.
                result = await client.connect_to_peer(
                    target_pubkey_pem=wallet_b.public_key_pem,
                )
                # Direct fails (peer_not_registered -> punch_timeout
                # because the seed answers our INTRODUCE with an error
                # that has no PUNCH_INVITE follow-up).
                # The introduce-future then times out cleanly.
                assert not result.success
                assert result.method in ("failed",)
                assert result.detail
            finally:
                client.close()
        finally:
            seed_transport.close()
            await asyncio.sleep(0.05)

    asyncio.run(_run())


def test_seed_picker_with_one_seed_short_circuits():
    """Single-seed PeerConnectClient.start does NOT run the RTT probe
    (which would burn a UDP roundtrip on every cold start)."""

    async def _run():
        seed_transport, _state, seed_addr = await _start_seed()
        try:
            wallet = Wallet()
            t0 = time.monotonic()
            client = await PeerConnectClient.start(
                seeds=[SeedAddress(host=seed_addr[0], port=seed_addr[1])],
                local_pubkey_pem=wallet.public_key_pem,
                sign=wallet.sign,
                bind_host="127.0.0.1",
            )
            elapsed = time.monotonic() - t0
            try:
                # Single-seed start should be very fast (no probe RTT).
                # Allow generous slack for slow CI.
                assert elapsed < 1.0, f"single-seed start took {elapsed:.3f}s"
                assert client.active_seed is not None
                assert client.active_seed.host == seed_addr[0]
                assert client.active_seed.port == seed_addr[1]
            finally:
                client.close()
        finally:
            seed_transport.close()
            await asyncio.sleep(0.05)

    asyncio.run(_run())


def test_top_level_connect_to_peer_helper():
    """The convenience `connect_to_peer(...)` returns (client, result).
    Useful for one-off scripts where the user doesn't need the long-
    lived client surface."""

    async def _run():
        seed_transport, _state, seed_addr = await _start_seed()
        try:
            wallet_in = Wallet()
            wallet_sg = Wallet()
            # Pre-register Singapore so the top-level helper has a
            # peer to introduce to. (Without this Singapore wouldn't
            # be in the seed table when India calls.)
            sg = await PeerConnectClient.start(
                seeds=[SeedAddress(host=seed_addr[0], port=seed_addr[1])],
                local_pubkey_pem=wallet_sg.public_key_pem,
                sign=wallet_sg.sign,
                bind_host="127.0.0.1",
            )
            try:
                # Wait for sg to register.
                t_deadline = time.monotonic() + 1.0
                while time.monotonic() < t_deadline and sg.external_addr is None:
                    await asyncio.sleep(0.02)

                client, result = await connect_to_peer(
                    seed_addrs=[(seed_addr[0], seed_addr[1])],
                    local_pubkey_pem=wallet_in.public_key_pem,
                    sign=wallet_in.sign,
                    target_pubkey_pem=wallet_sg.public_key_pem,
                    bind_host="127.0.0.1",
                )
                try:
                    assert result.success
                    assert result.method == "direct"
                    assert result.peer_addr is not None
                finally:
                    client.close()
            finally:
                sg.close()
        finally:
            seed_transport.close()
            await asyncio.sleep(0.05)

    asyncio.run(_run())
