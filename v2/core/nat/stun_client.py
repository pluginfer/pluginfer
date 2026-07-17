"""STUN client (RFC 5389 / RFC 8489).

Sends a Binding Request to a public STUN server, reads the response,
and returns the XOR-MAPPED-ADDRESS as the node's externally-visible
IP and port.

Default servers: stun.l.google.com:19302, stun1.l.google.com:19302,
stun.cloudflare.com:3478. The client tries them in order, first
working response wins.

Why we need this: the 8.8.8.8 phone-home in `core/networking.py` and
`complete_mesh_controller.py` was replaced with the RFC 5737 doc-range
trick which gives the LOCAL outgoing IP, not the EXTERNAL post-NAT IP.
For the bootstrap REGISTER message to advertise a reachable address,
we need the external one. STUN gives that.

Pure asyncio + struct + secrets; no third-party deps.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import socket
import struct
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

# RFC 5389 wire constants
_STUN_MAGIC_COOKIE = 0x2112A442
_STUN_BINDING_REQUEST = 0x0001
_STUN_BINDING_SUCCESS_RESPONSE = 0x0101
_ATTR_XOR_MAPPED_ADDRESS = 0x0020
_ATTR_MAPPED_ADDRESS = 0x0001  # legacy, fallback

DEFAULT_STUN_SERVERS: tuple[tuple[str, int], ...] = (
    ("stun.l.google.com", 19302),
    ("stun1.l.google.com", 19302),
    ("stun.cloudflare.com", 3478),
)


class STUNError(RuntimeError):
    pass


@dataclass(frozen=True)
class STUNResult:
    external_ip: str
    external_port: int
    server: str   # which STUN server gave us this (for logging)


def _build_binding_request() -> tuple[bytes, bytes]:
    """Return (request_bytes, transaction_id) for a fresh Binding Request."""
    txn_id = secrets.token_bytes(12)
    header = struct.pack(
        ">HHI12s",
        _STUN_BINDING_REQUEST,
        0,  # message length (no attributes)
        _STUN_MAGIC_COOKIE,
        txn_id,
    )
    return header, txn_id


def _parse_response(data: bytes, expected_txn: bytes) -> Tuple[str, int]:
    """Parse a STUN Binding Success Response. Returns (ip, port).

    Raises STUNError on any structural problem.
    """
    if len(data) < 20:
        raise STUNError(f"response too short: {len(data)}")
    msg_type, msg_len, magic, txn_id = struct.unpack(">HHI12s", data[:20])
    if msg_type != _STUN_BINDING_SUCCESS_RESPONSE:
        raise STUNError(f"unexpected msg_type 0x{msg_type:04x}")
    if magic != _STUN_MAGIC_COOKIE:
        raise STUNError(f"bad magic cookie 0x{magic:08x}")
    if txn_id != expected_txn:
        raise STUNError("txn_id mismatch")
    if len(data) < 20 + msg_len:
        raise STUNError(
            f"truncated body: claimed {msg_len}, got {len(data) - 20}"
        )
    body = data[20:20 + msg_len]
    cursor = 0
    while cursor + 4 <= len(body):
        a_type, a_len = struct.unpack(">HH", body[cursor:cursor + 4])
        a_val = body[cursor + 4:cursor + 4 + a_len]
        cursor += 4 + ((a_len + 3) // 4) * 4  # 32-bit padded
        if a_type == _ATTR_XOR_MAPPED_ADDRESS:
            return _parse_xor_mapped(a_val)
        if a_type == _ATTR_MAPPED_ADDRESS:
            return _parse_mapped(a_val)
    raise STUNError("no MAPPED-ADDRESS attribute in response")


def _parse_xor_mapped(val: bytes) -> Tuple[str, int]:
    if len(val) < 8:
        raise STUNError("xor-mapped-address too short")
    family = val[1]
    xport = struct.unpack(">H", val[2:4])[0]
    port = xport ^ (_STUN_MAGIC_COOKIE >> 16)
    if family == 0x01:  # IPv4
        if len(val) < 8:
            raise STUNError("xor-mapped IPv4 truncated")
        xip_bytes = val[4:8]
        ip_int = struct.unpack(">I", xip_bytes)[0] ^ _STUN_MAGIC_COOKIE
        ip = socket.inet_ntoa(struct.pack(">I", ip_int))
        return ip, port
    if family == 0x02:  # IPv6
        if len(val) < 20:
            raise STUNError("xor-mapped IPv6 truncated")
        # XOR with magic-cookie || txn-id; for simplicity we use the
        # first 16 bytes of (magic_cookie + magic_cookie + magic_cookie + magic_cookie)
        # which only works when txn_id is folded in. For our bootstrap
        # use case IPv4 is what the seeds care about; IPv6 path is a
        # follow-up. Surface a clear error so callers fall back.
        raise STUNError("IPv6 STUN responses not yet supported")
    raise STUNError(f"unknown family 0x{family:02x}")


def _parse_mapped(val: bytes) -> Tuple[str, int]:
    if len(val) < 8:
        raise STUNError("mapped-address too short")
    family = val[1]
    port = struct.unpack(">H", val[2:4])[0]
    if family == 0x01:
        ip = socket.inet_ntoa(val[4:8])
        return ip, port
    raise STUNError(f"non-IPv4 family 0x{family:02x} in MAPPED-ADDRESS")


# ---------------------------------------------------------------------------
# Async UDP transport
# ---------------------------------------------------------------------------

class _STUNProtocol(asyncio.DatagramProtocol):
    def __init__(self, future: asyncio.Future) -> None:
        self.future = future

    def datagram_received(self, data: bytes, addr) -> None:
        if not self.future.done():
            self.future.set_result(data)

    def error_received(self, exc: Exception) -> None:
        if not self.future.done():
            self.future.set_exception(exc)


async def _query_one(
    server: str, port: int, *, timeout_s: float,
    bind_port: Optional[int] = None,
) -> STUNResult:
    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()
    bind_addr = ("0.0.0.0", bind_port) if bind_port is not None else ("0.0.0.0", 0)
    transport, _ = await loop.create_datagram_endpoint(
        lambda: _STUNProtocol(fut), local_addr=bind_addr,
    )
    try:
        request, txn_id = _build_binding_request()
        transport.sendto(request, (server, port))
        try:
            data = await asyncio.wait_for(fut, timeout=timeout_s)
        except asyncio.TimeoutError as e:
            raise STUNError(f"STUN timeout to {server}:{port}") from e
        ip, ext_port = _parse_response(data, txn_id)
        return STUNResult(
            external_ip=ip,
            external_port=ext_port,
            server=f"{server}:{port}",
        )
    finally:
        transport.close()


async def discover_external_address_async(
    servers: Iterable[Tuple[str, int]] = DEFAULT_STUN_SERVERS,
    *,
    timeout_per_server_s: float = 2.0,
    bind_port: Optional[int] = None,
) -> STUNResult:
    """Try each STUN server in order; return first success.

    `bind_port`: bind the local UDP socket to this port (so callers can
    learn the external mapping for a SPECIFIC local port, not a fresh
    ephemeral one). Pass None to use any free ephemeral port.
    """
    last_err: Exception | None = None
    for host, port in servers:
        try:
            return await _query_one(
                host, port, timeout_s=timeout_per_server_s, bind_port=bind_port
            )
        except (STUNError, OSError, asyncio.TimeoutError) as e:
            logger.debug("STUN %s:%s failed: %s", host, port, e)
            last_err = e
            continue
    raise STUNError(
        f"all STUN servers failed; last error: {last_err!r}"
    )


def discover_external_address_sync(
    servers: Iterable[Tuple[str, int]] = DEFAULT_STUN_SERVERS,
    *,
    timeout_per_server_s: float = 2.0,
    bind_port: Optional[int] = None,
) -> STUNResult:
    """Sync wrapper for non-async callers (CompleteMeshController etc.)."""
    return asyncio.run(
        discover_external_address_async(
            servers=servers,
            timeout_per_server_s=timeout_per_server_s,
            bind_port=bind_port,
        )
    )
