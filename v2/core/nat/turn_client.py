"""Client-side TURN relay (symmetric-NAT fallback).

When `HolePunchClient` fails to punch through (`PunchOutcome.success ==
False`), the caller falls back to this relay: the seed becomes a
post office between the two peers. Each `send()` call serializes a
RELAY packet to the seed; the seed forwards it to the partner; the
partner's protocol callback fires `on_message`.

Bandwidth is metered server-side (50 MB / session by default in
`punch_server.py`). For high-throughput workloads, the application
should fall back to direct UDP after the first message succeeds via
relay (the relay confirmation tells both peers they CAN see the seed,
which means a TURN-style allocation could work too -- but that's a
v2 optimization).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

RELAY_OPEN_TIMEOUT_S: float = 5.0


@dataclass
class RelaySession:
    session_id: str
    target_pubkey_pem: str
    expires_at: float


class TurnRelayClient(asyncio.DatagramProtocol):
    """Client-side TURN relay protocol.

    Usage:

        relay = await TurnRelayClient.start(
            seed_addr=("203.0.113.10", 9000),
            local_pubkey_pem=wallet.public_key_pem,
            sign=wallet.sign,
        )
        relay.on_message = lambda from_pub, payload_bytes: ...
        session = await relay.open(target_pubkey_pem)
        await relay.send(session, b"hello over relay")
    """

    def __init__(
        self,
        *,
        seed_addr: Tuple[str, int],
        local_pubkey_pem: str,
        sign: Callable[[str], str],
    ) -> None:
        self.seed_addr = seed_addr
        self.local_pubkey_pem = local_pubkey_pem
        self.sign = sign
        self.transport: Optional[asyncio.DatagramTransport] = None
        # Pending RELAY_OPEN responses, keyed by target pubkey.
        self._pending_open: dict[str, asyncio.Future] = {}
        # Application-level inbound callback.
        self.on_message: Optional[Callable[[str, bytes, str], None]] = None

    @classmethod
    async def start(
        cls,
        *,
        seed_addr: Tuple[str, int],
        local_pubkey_pem: str,
        sign: Callable[[str], str],
        bind_host: str = "0.0.0.0",
        bind_port: int = 0,
    ) -> "TurnRelayClient":
        loop = asyncio.get_running_loop()
        proto = cls(seed_addr=seed_addr,
                    local_pubkey_pem=local_pubkey_pem,
                    sign=sign)
        transport, _ = await loop.create_datagram_endpoint(
            lambda: proto,
            local_addr=(bind_host, bind_port),
        )
        proto.transport = transport
        return proto

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()

    def _send(self, msg: dict, addr: Tuple[str, int]) -> None:
        if self.transport is None:
            raise RuntimeError("transport not open")
        self.transport.sendto(json.dumps(msg).encode("utf-8"), addr)

    async def open(self, target_pubkey_pem: str) -> RelaySession:
        ts = time.time()
        signed = f"RELAY_OPEN|{self.local_pubkey_pem}|{target_pubkey_pem}|{ts}"
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_open[target_pubkey_pem] = fut
        self._send({
            "op": "RELAY_OPEN",
            "from_pubkey": self.local_pubkey_pem,
            "target_pubkey": target_pubkey_pem,
            "timestamp": ts,
            "signature": self.sign(signed),
        }, self.seed_addr)
        try:
            session = await asyncio.wait_for(fut, timeout=RELAY_OPEN_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._pending_open.pop(target_pubkey_pem, None)
            raise TimeoutError("RELAY_OPEN timed out")
        return session

    async def send(self, session: RelaySession, payload: bytes) -> None:
        if time.time() >= session.expires_at:
            raise RuntimeError("relay session expired -- re-open")
        self._send({
            "op": "RELAY",
            "session": session.session_id,
            "payload_b64": base64.b64encode(payload).decode("ascii"),
        }, self.seed_addr)

    # ------------------------------------------------------------------
    # asyncio.DatagramProtocol callbacks
    # ------------------------------------------------------------------

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:  # type: ignore[override]
        self.transport = transport

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        try:
            msg = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(msg, dict):
            return

        # RELAY_OPEN reply
        if msg.get("status") == "ok" and "session" in msg:
            sid = str(msg["session"])
            ttl = float(msg.get("ttl_s", 300.0))
            # Resolve the FIRST pending open. (We don't echo the target
            # pubkey in the reply currently, so we resolve in FIFO order
            # -- one outstanding open per target by convention.)
            for target, fut in list(self._pending_open.items()):
                if not fut.done():
                    fut.set_result(RelaySession(
                        session_id=sid,
                        target_pubkey_pem=target,
                        expires_at=time.time() + ttl,
                    ))
                    self._pending_open.pop(target, None)
                    return
            return

        # Inbound RELAY_DELIVER
        if msg.get("op") == "RELAY_DELIVER":
            try:
                from_pub = str(msg["from_pubkey"])
                sid = str(msg["session"])
                payload_b64 = str(msg["payload_b64"])
            except (KeyError, TypeError, ValueError):
                return
            try:
                payload = base64.b64decode(payload_b64)
            except Exception:
                return
            if self.on_message is not None:
                try:
                    self.on_message(from_pub, payload, sid)
                except Exception as e:                              # pragma: no cover
                    logger.warning("on_message callback raised: %s", e)
            return

        # Error replies are returned for debugging; tests can inspect.
        # We don't surface them via futures (only RELAY_OPEN's ok case
        # has a future). For a production-grade client we'd add a
        # negative path here; tracked as a follow-up.
