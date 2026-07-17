"""End-to-end tests for the seed-brokered UDP hole-punch coordinator
and TURN relay fallback.

Drives the SHIPPING server (`punch_server.PunchRelayState` +
`_PunchProtocol`) and the SHIPPING client (`HolePunchClient`,
`TurnRelayClient`) over loopback UDP. No mocks, no monkey-patches --
real cryptography for REGISTER / INTRODUCE / RELAY_OPEN signatures,
real asyncio DatagramProtocols on the same loop.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import pytest

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.nat.hole_punch import HolePunchClient, PunchOutcome  # noqa: E402
from core.nat.turn_client import TurnRelayClient  # noqa: E402
from core.tokenomics import Wallet  # noqa: E402
from infrastructure.seed_node.punch_server import (  # noqa: E402
    PunchRelayState,
    _PunchProtocol,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _start_seed(loop: asyncio.AbstractEventLoop) -> Tuple[
    asyncio.DatagramTransport, _PunchProtocol, Tuple[str, int],
]:
    """Bind the punch+relay server on a free loopback port. Returns
    (transport, protocol, addr)."""
    state = PunchRelayState()
    transport, proto = await loop.create_datagram_endpoint(
        lambda: _PunchProtocol(state),
        local_addr=("127.0.0.1", 0),
    )
    addr = transport.get_extra_info("sockname")
    # The state object lives on the protocol; expose it for assertions.
    proto.state = state
    return transport, proto, addr


# ---------------------------------------------------------------------------
# unit: PunchRelayState dispatch (no socket)
# ---------------------------------------------------------------------------


def test_state_register_then_introduce_emits_two_invites():
    """The pure state machine: A and B register, then A INTRODUCEs.
    State should emit TWO outbound packets (one to A, one to B), each
    containing the OTHER's external addr + the same nonce."""
    state = PunchRelayState()
    a, b = Wallet(), Wallet()
    now = time.time()

    # REGISTER_UDP for A from 198.51.100.5:50000
    a_addr = ("198.51.100.5", 50000)
    sig = a.sign(f"REGISTER_UDP|{a.public_key_pem}|{now}")
    out = state.register_udp(
        {"pubkey_pem": a.public_key_pem, "timestamp": now,
         "signature": sig},
        a_addr, now,
    )
    assert out["status"] == "ok"
    assert out["external_ip"] == "198.51.100.5"
    assert out["external_port"] == 50000

    # REGISTER_UDP for B
    b_addr = ("203.0.113.7", 60123)
    sig_b = b.sign(f"REGISTER_UDP|{b.public_key_pem}|{now}")
    out_b = state.register_udp(
        {"pubkey_pem": b.public_key_pem, "timestamp": now,
         "signature": sig_b},
        b_addr, now,
    )
    assert out_b["status"] == "ok"

    # A INTRODUCE -> B
    sig_intr = a.sign(f"INTRODUCE|{a.public_key_pem}|{b.public_key_pem}|{now}")
    reply, outbound = state.introduce(
        {"from_pubkey": a.public_key_pem,
         "target_pubkey": b.public_key_pem,
         "timestamp": now, "signature": sig_intr},
        a_addr, now,
    )
    assert reply["status"] == "ok"
    nonce = reply["nonce"]
    assert isinstance(nonce, str) and len(nonce) >= 16
    assert len(outbound) == 2
    addrs = {dst for dst, _ in outbound}
    assert addrs == {a_addr, b_addr}
    for dst, pkt in outbound:
        assert pkt["op"] == "PUNCH_INVITE"
        assert pkt["nonce"] == nonce
        if dst == a_addr:
            assert (pkt["peer_ip"], pkt["peer_port"]) == b_addr
            assert pkt["peer_pubkey"] == b.public_key_pem
        else:
            assert (pkt["peer_ip"], pkt["peer_port"]) == a_addr
            assert pkt["peer_pubkey"] == a.public_key_pem


def test_state_introduce_rejects_src_mismatch():
    """If the INTRODUCE packet arrives from a different addr than the
    one A registered with, reject -- otherwise an attacker who sniffs
    a single INTRODUCE could replay it from anywhere and reflect punch
    packets at arbitrary victims."""
    state = PunchRelayState()
    a, b = Wallet(), Wallet()
    now = time.time()
    state.register_udp(
        {"pubkey_pem": a.public_key_pem, "timestamp": now,
         "signature": a.sign(f"REGISTER_UDP|{a.public_key_pem}|{now}")},
        ("198.51.100.5", 50000), now,
    )
    state.register_udp(
        {"pubkey_pem": b.public_key_pem, "timestamp": now,
         "signature": b.sign(f"REGISTER_UDP|{b.public_key_pem}|{now}")},
        ("203.0.113.7", 60123), now,
    )
    sig = a.sign(f"INTRODUCE|{a.public_key_pem}|{b.public_key_pem}|{now}")
    reply, outbound = state.introduce(
        {"from_pubkey": a.public_key_pem,
         "target_pubkey": b.public_key_pem,
         "timestamp": now, "signature": sig},
        ("203.0.113.42", 1111),    # DIFFERENT addr from the one A registered
        now,
    )
    assert reply["status"] == "error"
    assert reply["code"] == "src_mismatch"
    assert outbound == []


def test_state_introduce_self_rejected():
    """Self-introduce is filtered (would otherwise be a packet
    amplification primitive)."""
    state = PunchRelayState()
    w = Wallet()
    now = time.time()
    state.register_udp(
        {"pubkey_pem": w.public_key_pem, "timestamp": now,
         "signature": w.sign(f"REGISTER_UDP|{w.public_key_pem}|{now}")},
        ("198.51.100.5", 50000), now,
    )
    sig = w.sign(f"INTRODUCE|{w.public_key_pem}|{w.public_key_pem}|{now}")
    reply, outbound = state.introduce(
        {"from_pubkey": w.public_key_pem,
         "target_pubkey": w.public_key_pem,
         "timestamp": now, "signature": sig},
        ("198.51.100.5", 50000), now,
    )
    assert reply["status"] == "error"
    assert reply["code"] == "self_introduce_forbidden"


def test_state_introduce_rejects_unsigned():
    state = PunchRelayState()
    a, b = Wallet(), Wallet()
    now = time.time()
    reply, _ = state.introduce(
        {"from_pubkey": a.public_key_pem,
         "target_pubkey": b.public_key_pem,
         "timestamp": now, "signature": "AAAA"},
        ("198.51.100.5", 50000), now,
    )
    assert reply["status"] == "error"
    assert reply["code"] in ("bad_signature", "peer_not_registered")


def test_state_relay_round_trip_meters_bytes():
    """RELAY_OPEN allocates a session; subsequent RELAY packets are
    forwarded to the OTHER party with the sender's pubkey. Server
    bandwidth metering increments correctly."""
    state = PunchRelayState()
    a, b = Wallet(), Wallet()
    now = time.time()
    a_addr, b_addr = ("198.51.100.5", 50000), ("203.0.113.7", 60123)
    state.register_udp(
        {"pubkey_pem": a.public_key_pem, "timestamp": now,
         "signature": a.sign(f"REGISTER_UDP|{a.public_key_pem}|{now}")},
        a_addr, now,
    )
    state.register_udp(
        {"pubkey_pem": b.public_key_pem, "timestamp": now,
         "signature": b.sign(f"REGISTER_UDP|{b.public_key_pem}|{now}")},
        b_addr, now,
    )
    sig = a.sign(f"RELAY_OPEN|{a.public_key_pem}|{b.public_key_pem}|{now}")
    open_reply = state.relay_open(
        {"from_pubkey": a.public_key_pem,
         "target_pubkey": b.public_key_pem,
         "timestamp": now, "signature": sig},
        a_addr, now,
    )
    assert open_reply["status"] == "ok"
    sid = open_reply["session"]

    # A sends to seed -> seed forwards to B
    import base64 as _b64
    payload = b"hello-from-A-via-relay"
    reply, outbound = state.relay(
        {"session": sid, "payload_b64": _b64.b64encode(payload).decode()},
        a_addr, now,
    )
    assert reply["status"] == "ok"
    assert len(outbound) == 1
    dst, pkt = outbound[0]
    assert dst == b_addr
    assert pkt["op"] == "RELAY_DELIVER"
    assert _b64.b64decode(pkt["payload_b64"]) == payload
    assert pkt["from_pubkey"] == a.public_key_pem
    assert state.metrics_relay_bytes_total == len(payload)


def test_state_relay_rejects_session_unknown():
    state = PunchRelayState()
    now = time.time()
    reply, outbound = state.relay(
        {"session": "deadbeef", "payload_b64": ""},
        ("127.0.0.1", 1), now,
    )
    assert reply["status"] == "error"
    assert reply["code"] == "session_unknown_or_expired"
    assert outbound == []


def test_state_relay_quota_enforced():
    """A session that has already eaten its byte budget is refused."""
    from infrastructure.seed_node.punch_server import (
        RELAY_PER_SESSION_BANDWIDTH_BYTES,
    )
    state = PunchRelayState()
    a, b = Wallet(), Wallet()
    now = time.time()
    a_addr = ("198.51.100.5", 50000)
    state.register_udp(
        {"pubkey_pem": a.public_key_pem, "timestamp": now,
         "signature": a.sign(f"REGISTER_UDP|{a.public_key_pem}|{now}")},
        a_addr, now,
    )
    state.register_udp(
        {"pubkey_pem": b.public_key_pem, "timestamp": now,
         "signature": b.sign(f"REGISTER_UDP|{b.public_key_pem}|{now}")},
        ("203.0.113.7", 60123), now,
    )
    open_reply = state.relay_open(
        {"from_pubkey": a.public_key_pem,
         "target_pubkey": b.public_key_pem,
         "timestamp": now,
         "signature": a.sign(
             f"RELAY_OPEN|{a.public_key_pem}|{b.public_key_pem}|{now}"
         )},
        a_addr, now,
    )
    sid = open_reply["session"]
    # Exhaust the budget directly on the session.
    state.sessions[sid].bytes_used = RELAY_PER_SESSION_BANDWIDTH_BYTES - 10
    import base64 as _b64
    reply, outbound = state.relay(
        {"session": sid,
         "payload_b64": _b64.b64encode(b"x" * 100).decode()},
        a_addr, now,
    )
    assert reply["status"] == "error"
    assert reply["code"] == "session_quota_exceeded"


# ---------------------------------------------------------------------------
# integration: real loopback UDP, real cryptography, real asyncio
# ---------------------------------------------------------------------------


def test_two_clients_punch_through_loopback():
    """Two HolePunchClients on different loopback ports register with a
    real seed UDP server, then A INTRODUCEs to B. Both clients fire
    PUNCH_HELLOs at each other; the introduce future resolves with B's
    addr; the receive table on each side has the OTHER side's addr.
    This is the autonomous-cross-internet path running end-to-end."""

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        seed_transport, seed_proto, seed_addr = await _start_seed(loop)
        a_wallet, b_wallet = Wallet(), Wallet()

        a = await HolePunchClient.start(
            seed_addr=seed_addr,
            local_pubkey_pem=a_wallet.public_key_pem,
            sign=a_wallet.sign,
            bind_host="127.0.0.1", bind_port=0,
        )
        b = await HolePunchClient.start(
            seed_addr=seed_addr,
            local_pubkey_pem=b_wallet.public_key_pem,
            sign=b_wallet.sign,
            bind_host="127.0.0.1", bind_port=0,
        )
        # Let REGISTER replies land + state populate.
        for _ in range(20):
            await asyncio.sleep(0.01)
            if a.external_addr and b.external_addr and len(seed_proto.state.regs) >= 2:
                break

        try:
            outcome: PunchOutcome = await a.introduce(b_wallet.public_key_pem)
            assert outcome.success, f"introduce did not succeed: {outcome.detail}"
            assert outcome.peer_addr is not None
            # The peer addr we converged on must match B's actual UDP
            # endpoint as observed by the seed.
            seen_b_addr = seed_proto.state.regs[b_wallet.public_key_pem].addr
            assert outcome.peer_addr == seen_b_addr
            assert outcome.nonce
        finally:
            a.close()
            b.close()
            seed_transport.close()
            await asyncio.sleep(0)

    asyncio.run(_run())


def test_two_clients_relay_round_trip_loopback():
    """A and B can't punch (we don't drive PUNCH_HELLO at all). Use
    TurnRelayClient.open() then send(); B's `on_message` callback
    receives the bytes."""

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        seed_transport, seed_proto, seed_addr = await _start_seed(loop)
        a_wallet, b_wallet = Wallet(), Wallet()

        a = await TurnRelayClient.start(
            seed_addr=seed_addr,
            local_pubkey_pem=a_wallet.public_key_pem,
            sign=a_wallet.sign,
            bind_host="127.0.0.1", bind_port=0,
        )
        b = await TurnRelayClient.start(
            seed_addr=seed_addr,
            local_pubkey_pem=b_wallet.public_key_pem,
            sign=b_wallet.sign,
            bind_host="127.0.0.1", bind_port=0,
        )
        # Both peers also need to be REGISTER_UDP'd. The TurnRelayClient
        # doesn't do registration itself (HolePunchClient does); for
        # the test, we send REGISTER_UDP via the same protocol's
        # transport so the seed learns each peer's addr.
        async def _register(client: TurnRelayClient, w: Wallet) -> None:
            ts = time.time()
            sig = w.sign(f"REGISTER_UDP|{w.public_key_pem}|{ts}")
            client.transport.sendto(
                json.dumps({
                    "op": "REGISTER_UDP",
                    "pubkey_pem": w.public_key_pem,
                    "timestamp": ts,
                    "signature": sig,
                }).encode(),
                seed_addr,
            )
            # Tiny grace period so the seed processes and replies.
            await asyncio.sleep(0.05)

        await _register(a, a_wallet)
        await _register(b, b_wallet)
        # Wait for state to populate (UDP is asynchronous).
        for _ in range(30):
            if (a_wallet.public_key_pem in seed_proto.state.regs
                    and b_wallet.public_key_pem in seed_proto.state.regs):
                break
            await asyncio.sleep(0.02)

        received: list[tuple[str, bytes]] = []
        b.on_message = lambda from_pub, payload, sid: received.append((from_pub, payload))

        try:
            session = await a.open(b_wallet.public_key_pem)
            assert session.session_id
            await a.send(session, b"hello-via-turn")
            # Wait for B's protocol to receive the forwarded packet.
            for _ in range(50):
                if received:
                    break
                await asyncio.sleep(0.02)
            assert received, "B never received the relayed message"
            from_pub, payload = received[0]
            assert from_pub == a_wallet.public_key_pem
            assert payload == b"hello-via-turn"
        finally:
            a.close()
            b.close()
            seed_transport.close()
            await asyncio.sleep(0)

    asyncio.run(_run())
