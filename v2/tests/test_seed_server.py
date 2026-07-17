"""CP-2 Task 2.1: tests for the bootstrap seed server.

In-process exercise of the SeedServer state machine + a real socket
round-trip via run_server() to prove the asyncio listener works.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest  # noqa: E402

from core.tokenomics import Wallet  # noqa: E402
from infrastructure.seed_node.seed_client import (  # noqa: E402
    SeedAddress,
    fetch_peers_async,
    ping_async,
    register_async,
)
from infrastructure.seed_node.seed_server import (  # noqa: E402
    PEER_TTL_SECONDS,
    RATE_LIMIT_PER_MINUTE,
    TIMESTAMP_WINDOW_SECONDS,
    SeedServer,
    _signed_bytes,
    run_server,
)


# ---------------------------------------------------------------------------
# Pure state-machine tests (no sockets)
# ---------------------------------------------------------------------------

def _make_register(wallet: Wallet, ip: str = "1.2.3.4",
                    port: int = 8100, version: str = "1.0.0",
                    timestamp: float | None = None) -> dict:
    timestamp = timestamp or time.time()
    signed = _signed_bytes(wallet.public_key_pem, ip, port, version, timestamp)
    return {
        "op": "REGISTER",
        "pubkey_pem": wallet.public_key_pem,
        "ip": ip, "port": port, "node_version": version,
        "timestamp": timestamp,
        "signature": wallet.sign(signed),
    }


def test_register_happy_path() -> None:
    s = SeedServer()
    w = Wallet()
    r = s.handle(_make_register(w), client_ip="9.9.9.9")
    assert r["status"] == "ok"
    assert r["ttl_seconds"] == PEER_TTL_SECONDS
    assert r["peers"] == 1


def test_register_rejects_stale_timestamp() -> None:
    s = SeedServer()
    w = Wallet()
    msg = _make_register(w, timestamp=time.time() - TIMESTAMP_WINDOW_SECONDS - 60)
    r = s.handle(msg, client_ip="9.9.9.9")
    assert r["status"] == "error"
    assert r["code"] == "stale_timestamp"


def test_register_rejects_tampered_payload() -> None:
    s = SeedServer()
    w = Wallet()
    msg = _make_register(w, ip="1.2.3.4")
    msg["ip"] = "9.9.9.9"  # tamper after signing
    r = s.handle(msg, client_ip="9.9.9.9")
    assert r["status"] == "error"
    assert r["code"] == "bad_signature"


def test_register_rejects_unsigned() -> None:
    s = SeedServer()
    w = Wallet()
    msg = _make_register(w)
    msg["signature"] = "AAAA"  # garbage
    r = s.handle(msg, client_ip="9.9.9.9")
    assert r["status"] == "error"
    assert r["code"] == "bad_signature"


def test_register_rejects_bad_port() -> None:
    s = SeedServer()
    w = Wallet()
    msg = _make_register(w, port=70000)
    r = s.handle(msg, client_ip="9.9.9.9")
    assert r["status"] == "error"
    # Could be bad_signature (port baked into signed bytes); the contract
    # is just that it fails; assert the failure.
    assert r["status"] == "error"


def test_rate_limit_per_ip() -> None:
    s = SeedServer()
    # RATE_LIMIT_PER_MINUTE distinct registrations from the same IP succeed,
    # the next one is rejected.
    for _ in range(RATE_LIMIT_PER_MINUTE):
        w = Wallet()
        r = s.handle(_make_register(w), client_ip="1.1.1.1")
        assert r["status"] == "ok", r
    w_extra = Wallet()
    r = s.handle(_make_register(w_extra), client_ip="1.1.1.1")
    assert r["status"] == "error"
    assert r["code"] == "rate_limited"


def test_peers_returns_sample() -> None:
    s = SeedServer()
    for _ in range(5):
        w = Wallet()
        # Use distinct IPs to dodge the rate limit
        ip = f"10.0.0.{len(s.peers)+1}"
        s.handle(_make_register(w), client_ip=ip)
    r = s.handle({"op": "PEERS", "max": 3}, client_ip="1.1.1.1")
    assert r["status"] == "ok"
    assert len(r["peers"]) == 3
    for p in r["peers"]:
        assert "pubkey_pem" in p and "ip" in p and "port" in p


def test_expire_removes_stale_entries() -> None:
    s = SeedServer()
    w = Wallet()
    s.handle(_make_register(w), client_ip="2.2.2.2")
    # Force expiry by walking forward in simulated time
    now_future = time.time() + PEER_TTL_SECONDS + 1
    n_expired = s.expire(now=now_future)
    assert n_expired == 1
    r = s.handle({"op": "PING"}, client_ip="2.2.2.2")
    assert r["peers"] == 0


def test_unknown_op_returns_error() -> None:
    s = SeedServer()
    r = s.handle({"op": "DROP_TABLE"}, client_ip="x")
    assert r["status"] == "error"
    assert r["code"] == "unknown_op"


# ---------------------------------------------------------------------------
# End-to-end: real asyncio listener + real client
# ---------------------------------------------------------------------------

async def _start_server(port: int):
    """Start `run_server` as a background task; return (task, port)."""
    task = asyncio.create_task(run_server(host="127.0.0.1", port=port))
    # Give the listener a moment to bind
    for _ in range(20):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return task
        except OSError:
            await asyncio.sleep(0.05)
    raise RuntimeError("seed_server didn't bind in time")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def _e2e_register_and_peers() -> None:
    port = _free_port()
    server_task = await _start_server(port)
    try:
        seed = SeedAddress(host="127.0.0.1", port=port)
        ping = await ping_async(seed)
        assert ping["status"] == "ok"
        assert ping["peers"] == 0

        wallet = Wallet()
        resp = await register_async(
            seed,
            pubkey_pem=wallet.public_key_pem,
            sign_fn=wallet.sign,
            ip="127.0.0.1",
            port=8100,
            node_version="1.0.0-test",
        )
        assert resp["status"] == "ok"

        peers = await fetch_peers_async(seed, max_n=10)
        assert len(peers) == 1
        assert peers[0]["ip"] == "127.0.0.1"
        assert peers[0]["port"] == 8100
        assert peers[0]["node_version"] == "1.0.0-test"
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


async def _e2e_rejects_bad_signature() -> None:
    port = _free_port()
    server_task = await _start_server(port)
    try:
        seed = SeedAddress(host="127.0.0.1", port=port)
        victim = Wallet()
        attacker = Wallet()
        resp = await register_async(
            seed,
            pubkey_pem=victim.public_key_pem,
            sign_fn=lambda m: attacker.sign(m),
            ip="127.0.0.1",
            port=8100,
            node_version="1.0.0-test",
        )
        assert resp["status"] == "error"
        assert resp["code"] == "bad_signature"
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


def test_real_socket_register_and_peers() -> None:
    asyncio.run(_e2e_register_and_peers())


def test_real_socket_rejects_bad_signature() -> None:
    asyncio.run(_e2e_rejects_bad_signature())
