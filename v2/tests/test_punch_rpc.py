"""§HG6 — jobs over the NAT-traversal path, hermetically.

The claim: two nodes that cannot reach each other by HTTP in EITHER
direction (symmetric NAT both sides, no reachable third peer) can
still trade a chat-completion request/response, because the punch
seed either (a) brokers a direct UDP pinhole, or (b) TURN-relays the
datagrams itself.

Hermetic setup mirrors test_peer_connect_cross_region: one in-process
punch_server on 127.0.0.1 as the "global seed", two PeerConnectClients
as the strangers. Real NAT boxes can't exist in-process; what IS
exercised end-to-end here:

  * the punched path: introduce → PUNCH_HELLO exchange → PunchRPC
    request/response with fragmentation (test 1),
  * the TURN path: introduce force-failed (that's what symmetric NAT
    does) → RELAY_OPEN on the registered socket → RELAY / RELAY_DELIVER
    both directions, responder replying on a session it never opened
    (test 2),
  * both paths through the SAME unified surface auto_mesh uses
    (send_to_peer / set_inbound_handler / PunchRPC.call).

Real-WAN symmetric-NAT verification remains the off-keyboard MH1 run.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.nat.hole_punch import PunchOutcome  # noqa: E402
from core.peer_connect import (  # noqa: E402
    PeerConnectClient,
    SeedAddress,
)
from core.punch_rpc import PunchRPC  # noqa: E402
from core.tokenomics import Wallet  # noqa: E402
from infrastructure.seed_node.punch_server import (  # noqa: E402
    PunchRelayState,
    _PunchProtocol,
)


async def _start_seed(host: str = "127.0.0.1", port: int = 0):
    state = PunchRelayState()
    loop = asyncio.get_running_loop()
    transport, _proto = await loop.create_datagram_endpoint(
        lambda: _PunchProtocol(state),
        local_addr=(host, port),
    )
    return transport, state, transport.get_extra_info("sockname")


async def _bring_up_pair(seed_addr):
    """Two strangers, each with a wallet, a punch client, and a
    PunchRPC whose handler echoes the request body back."""
    w_a, w_b = Wallet(), Wallet()
    a = await PeerConnectClient.start(
        seeds=[SeedAddress(host=seed_addr[0], port=seed_addr[1])],
        local_pubkey_pem=w_a.public_key_pem, sign=w_a.sign,
        bind_host="127.0.0.1",
    )
    b = await PeerConnectClient.start(
        seeds=[SeedAddress(host=seed_addr[0], port=seed_addr[1])],
        local_pubkey_pem=w_b.public_key_pem, sign=w_b.sign,
        bind_host="127.0.0.1",
    )

    async def _echo_handler(body: dict):
        return 200, {"X-Pluginfer-Receipt-Signed": "1"}, {
            "echo": body, "served_by": "b"}

    rpc_a = PunchRPC(a, _echo_handler, my_pubkey_pem=w_a.public_key_pem)
    rpc_b = PunchRPC(b, _echo_handler, my_pubkey_pem=w_b.public_key_pem)
    # Give both REGISTER_UDP replies a beat to land at the seed.
    await asyncio.sleep(0.3)
    return (w_a, a, rpc_a), (w_b, b, rpc_b)


def test_rpc_roundtrip_over_punched_path_with_fragmentation():
    async def _run():
        seed_t, _state, seed_addr = await _start_seed()
        try:
            (w_a, a, rpc_a), (w_b, b, rpc_b) = await _bring_up_pair(
                seed_addr)
            try:
                # >8 kB body forces multi-fragment request AND response
                # (chunk = 900 raw bytes -> ~10 fragments each way).
                big = "x" * 8_500
                status, headers, body = await rpc_a.call(
                    w_b.public_key_pem,
                    {"prompt": big, "max_tokens": 8},
                    timeout_s=15.0,
                )
                assert status == 200
                assert headers.get("X-Pluginfer-Receipt-Signed") == "1"
                assert body["echo"]["prompt"] == big
                # The path used was a punched direct addr, not TURN.
                assert w_b.public_key_pem in a._punched
            finally:
                a.close()
                b.close()
        finally:
            seed_t.close()
        await asyncio.sleep(0.1)

    asyncio.run(_run())


def test_rpc_roundtrip_over_turn_when_punch_fails():
    """Symmetric-NAT simulation: the punch NEVER succeeds (that is
    what symmetric NAT does to the introduce flow), so connect falls
    back to a seed-relayed TURN session — and the full RPC still
    completes, with the responder replying on a session it never
    opened."""
    async def _run():
        seed_t, state, seed_addr = await _start_seed()
        try:
            (w_a, a, rpc_a), (w_b, b, rpc_b) = await _bring_up_pair(
                seed_addr)
            try:
                async def _no_punch(_target):
                    return PunchOutcome(
                        success=False, peer_addr=None,
                        detail="simulated symmetric NAT",
                    )
                a._hp.introduce = _no_punch  # force the TURN rung

                res = await a.connect_to_peer(w_b.public_key_pem)
                assert res.success, res.detail
                assert res.method == "turn"

                status, headers, body = await rpc_a.call(
                    w_b.public_key_pem,
                    {"prompt": "hello through the relay",
                     "max_tokens": 8},
                    timeout_s=15.0,
                )
                assert status == 200
                assert (body["echo"]["prompt"]
                        == "hello through the relay")
                # A has no punched addr for B — this went via TURN.
                assert w_b.public_key_pem not in a._punched
                # B learned the session from RELAY_DELIVER and replied
                # on it without ever calling connect_to_peer.
                assert w_a.public_key_pem in b._turn_sessions
            finally:
                a.close()
                b.close()
        finally:
            seed_t.close()
        await asyncio.sleep(0.1)

    asyncio.run(_run())


def test_call_sync_bridges_from_worker_thread():
    """CrossNodeProvider.execute runs in an executor thread; call_sync
    must complete the coroutine on the node's loop from there."""
    async def _run():
        seed_t, _state, seed_addr = await _start_seed()
        try:
            (w_a, a, rpc_a), (w_b, b, rpc_b) = await _bring_up_pair(
                seed_addr)
            try:
                def _worker():
                    return rpc_a.call_sync(
                        w_b.public_key_pem,
                        {"prompt": "from-a-thread"},
                        timeout_s=15.0,
                    )
                status, _headers, body = (
                    await asyncio.get_running_loop().run_in_executor(
                        None, _worker))
                assert status == 200
                assert body["echo"]["prompt"] == "from-a-thread"
            finally:
                a.close()
                b.close()
        finally:
            seed_t.close()
        await asyncio.sleep(0.1)

    asyncio.run(_run())
