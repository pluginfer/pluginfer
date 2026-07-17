"""CP-2 Task 2.3: tests for STUN client + NAT manager.

The STUN parsing/encoding is exercised against a synthetic Binding
Success Response so we don't need internet access in CI. NATManager
strategy selection is tested with mock UPnP managers.
"""

from __future__ import annotations

import asyncio
import socket
import struct
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest  # noqa: E402

from core.nat import (  # noqa: E402
    NATDiscovery,
    NATManager,
    NATStrategy,
    STUNError,
)
from core.nat import nat_manager as _nm  # noqa: E402
from core.nat import stun_client as _sc  # noqa: E402
from core.nat.hole_punch import (  # noqa: E402
    HolePunchNotImplementedError,
    coordinate,
)


# ---------------------------------------------------------------------------
# STUN wire format
# ---------------------------------------------------------------------------

def _build_xor_mapped_response(
    txn_id: bytes, ext_ip: str, ext_port: int,
) -> bytes:
    # XOR-MAPPED-ADDRESS attribute body (IPv4)
    xport = ext_port ^ (_sc._STUN_MAGIC_COOKIE >> 16)
    ip_int = struct.unpack(">I", socket.inet_aton(ext_ip))[0]
    xip = ip_int ^ _sc._STUN_MAGIC_COOKIE
    attr_val = struct.pack(">BBH", 0, 0x01, xport) + struct.pack(">I", xip)
    attr_hdr = struct.pack(">HH", _sc._ATTR_XOR_MAPPED_ADDRESS, len(attr_val))
    attrs = attr_hdr + attr_val
    header = struct.pack(
        ">HHI12s",
        _sc._STUN_BINDING_SUCCESS_RESPONSE,
        len(attrs),
        _sc._STUN_MAGIC_COOKIE,
        txn_id,
    )
    return header + attrs


def test_stun_response_parses_correctly() -> None:
    request, txn_id = _sc._build_binding_request()
    fake_resp = _build_xor_mapped_response(txn_id, "203.0.113.5", 41234)
    ip, port = _sc._parse_response(fake_resp, txn_id)
    assert ip == "203.0.113.5"
    assert port == 41234


def test_stun_response_rejects_wrong_txn_id() -> None:
    request, txn_id = _sc._build_binding_request()
    fake_resp = _build_xor_mapped_response(txn_id, "203.0.113.5", 41234)
    with pytest.raises(STUNError, match="txn_id"):
        _sc._parse_response(fake_resp, b"\x00" * 12)


def test_stun_response_rejects_wrong_magic() -> None:
    request, txn_id = _sc._build_binding_request()
    fake = _build_xor_mapped_response(txn_id, "203.0.113.5", 41234)
    # Tamper magic cookie
    tampered = fake[:4] + b"\xff\xff\xff\xff" + fake[8:]
    with pytest.raises(STUNError, match="magic"):
        _sc._parse_response(tampered, txn_id)


def test_stun_async_round_trip_through_loopback_udp() -> None:
    """End-to-end: spin up a UDP echo server that pretends to be STUN."""
    async def _harness():
        loop = asyncio.get_event_loop()

        # Bind a UDP socket; reply to whatever Binding Request comes in.
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        sock.setblocking(False)
        srv_port = sock.getsockname()[1]

        async def fake_server():
            data, addr = await loop.sock_recvfrom(sock, 4096)
            # Extract txn_id from request header (offset 8, 12 bytes)
            txn_id = data[8:20]
            resp = _build_xor_mapped_response(txn_id, "198.51.100.7", 9999)
            await loop.sock_sendto(sock, resp, addr)

        task = asyncio.create_task(fake_server())
        try:
            result = await _sc.discover_external_address_async(
                servers=[("127.0.0.1", srv_port)], timeout_per_server_s=2.0,
            )
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            sock.close()
        return result

    result = asyncio.run(_harness())
    assert result.external_ip == "198.51.100.7"
    assert result.external_port == 9999


# ---------------------------------------------------------------------------
# NATManager strategy selection
# ---------------------------------------------------------------------------

def test_nat_manager_picks_direct_for_public_ip(monkeypatch) -> None:
    monkeypatch.setattr(_nm, "discover_local_ip", lambda: "203.0.113.10")
    mgr = NATManager(local_port=8100)
    info = mgr.discover()
    assert info.strategy == NATStrategy.DIRECT
    assert info.external_ip == "203.0.113.10"


def test_nat_manager_falls_through_to_stun(monkeypatch) -> None:
    monkeypatch.setattr(_nm, "discover_local_ip", lambda: "192.168.1.5")

    class _BadUPnP:
        def enable_upnp(self, _port=None):
            raise RuntimeError("UPnP refused")

    def _fake_stun(*, bind_port):
        return _sc.STUNResult(
            external_ip="198.51.100.42",
            external_port=8100,
            server="fake:1234",
        )
    monkeypatch.setattr(_sc, "discover_external_address_sync", _fake_stun)

    mgr = NATManager(local_port=8100, upnp_manager=_BadUPnP())
    info = mgr.discover()
    assert info.strategy == NATStrategy.STUN
    assert info.external_ip == "198.51.100.42"


def test_nat_manager_uses_upnp_when_available(monkeypatch) -> None:
    monkeypatch.setattr(_nm, "discover_local_ip", lambda: "192.168.1.5")

    class _GoodUPnP:
        def enable_upnp(self, _port=None):
            return "198.51.100.99"

    mgr = NATManager(local_port=8100, upnp_manager=_GoodUPnP())
    info = mgr.discover()
    assert info.strategy == NATStrategy.UPNP
    assert info.external_ip == "198.51.100.99"


def test_nat_manager_total_failure_returns_local_with_warning(
    monkeypatch,
) -> None:
    monkeypatch.setattr(_nm, "discover_local_ip", lambda: "10.0.0.5")

    def _fail_stun(**_kw):
        raise STUNError("no stun server reachable")
    monkeypatch.setattr(_sc, "discover_external_address_sync", _fail_stun)

    mgr = NATManager(local_port=8100)  # no upnp_manager
    info = mgr.discover()
    assert info.strategy == NATStrategy.DIRECT
    assert info.external_ip == "10.0.0.5"
    assert "no traversal succeeded" in info.detail


# ---------------------------------------------------------------------------
# Hole punch is honest stub
# ---------------------------------------------------------------------------

def test_hole_punch_is_honest_stub() -> None:
    with pytest.raises(HolePunchNotImplementedError):
        coordinate()
