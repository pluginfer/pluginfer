"""UDP hole-punching coordinator (client side).

Pairs with `infrastructure/seed_node/punch_server.py`. The seed maintains
a UDP registration table for each peer (their external (ip, port) AS
SEEN BY THE SEED), and brokers introductions: when peer A asks for B,
the seed sends a PUNCH_INVITE to BOTH peers. They then fire PUNCH_HELLO
at each other simultaneously; the first packet on each side opens the
NAT pinhole.

Works for full-cone, restricted-cone, and port-restricted-cone NAT
(>80% of consumer routers per RFC 4787 surveys). Symmetric NAT (where
the external port differs per destination) breaks this scheme; those
peers fall back to the TURN relay (`turn_client.py`).

This file is the client side. The server side is in
`infrastructure/seed_node/punch_server.py`. Both use the same wire
format documented in the server module's docstring.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# How often the client refreshes its UDP registration with the seed.
# Must be < punch_server.UDP_TTL_SECONDS so the registration never
# expires from the seed's table.
KEEPALIVE_INTERVAL_SECONDS: int = 60
PUNCH_HELLO_BURST_COUNT: int = 5         # send 5 packets to defeat 1-2 lost UDPs
PUNCH_HELLO_BURST_INTERVAL_S: float = 0.05
PUNCH_RESULT_TIMEOUT_S: float = 8.0


@dataclass
class PunchOutcome:
    """What the coordinator gives back to the caller."""
    success: bool
    peer_addr: Optional[Tuple[str, int]] = None
    nonce: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class _Pending:
    """An in-flight introduction, keyed by nonce."""
    nonce: str
    target_pubkey: str
    future: asyncio.Future


class HolePunchClient(asyncio.DatagramProtocol):
    """A long-lived asyncio DatagramProtocol that:

      1. Keeps a UDP registration alive with the seed.
      2. Translates local "introduce me to <pubkey>" calls into wire
         INTRODUCEs and waits for the matching PUNCH_INVITE.
      3. Fires PUNCH_HELLO packets at peers when invited.
      4. Acks incoming PUNCH_HELLO so the inviter knows the pinhole
         is open from THEIR side.

    Usage (typical):

        client = await HolePunchClient.start(
            seed_addr=("203.0.113.10", 9000),
            local_pubkey_pem=wallet.public_key_pem,
            sign=wallet.sign,
        )
        outcome = await client.introduce("<peer pubkey pem>")
        if outcome.success:
            ...peer-to-peer UDP from now on at outcome.peer_addr...
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
        # nonce -> _Pending (set by introduce())
        self._pending: Dict[str, _Pending] = {}
        # nonce -> peer_addr we've heard from with PUNCH_HELLO/_ACK
        self._heard_from: Dict[str, Tuple[str, int]] = {}
        # external addr the seed reports (last REGISTER_UDP reply)
        self.external_addr: Optional[Tuple[str, int]] = None
        # caller-supplied callback when an INTRODUCE *for us* arrives
        # (i.e. someone else asked the seed to introduce us). Lets a
        # higher-level mesh code respond by, e.g., immediately starting
        # an application-level handshake on the now-open UDP pinhole.
        self.on_invite: Optional[Callable[[Tuple[str, int], str], None]] = None
        self._keepalive_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # construction / shutdown
    # ------------------------------------------------------------------

    @classmethod
    async def start(
        cls,
        *,
        seed_addr: Tuple[str, int],
        local_pubkey_pem: str,
        sign: Callable[[str], str],
        bind_host: str = "0.0.0.0",
        bind_port: int = 0,
    ) -> "HolePunchClient":
        loop = asyncio.get_running_loop()
        proto = cls(
            seed_addr=seed_addr,
            local_pubkey_pem=local_pubkey_pem,
            sign=sign,
        )
        transport, _ = await loop.create_datagram_endpoint(
            lambda: proto,
            local_addr=(bind_host, bind_port),
        )
        proto.transport = transport
        # Initial registration + start keep-alive loop.
        await proto.register_with_seed()
        proto._keepalive_task = asyncio.create_task(proto._keepalive_loop())
        return proto

    def close(self) -> None:
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
        if self.transport is not None:
            self.transport.close()

    # ------------------------------------------------------------------
    # protocol primitives
    # ------------------------------------------------------------------

    def _send(self, msg: dict, addr: Tuple[str, int]) -> None:
        if self.transport is None:
            raise RuntimeError("transport not yet open")
        self.transport.sendto(json.dumps(msg).encode("utf-8"), addr)

    async def register_with_seed(self) -> None:
        """REGISTER_UDP. Reply (handled in datagram_received) updates
        `self.external_addr` so the local node knows what address the
        seed sees."""
        ts = time.time()
        signed = f"REGISTER_UDP|{self.local_pubkey_pem}|{ts}"
        self._send({
            "op": "REGISTER_UDP",
            "pubkey_pem": self.local_pubkey_pem,
            "timestamp": ts,
            "signature": self.sign(signed),
        }, self.seed_addr)

    async def _keepalive_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(KEEPALIVE_INTERVAL_SECONDS)
                try:
                    await self.register_with_seed()
                except Exception as e:
                    logger.warning("keep-alive register failed: %s", e)
        except asyncio.CancelledError:
            return

    # ------------------------------------------------------------------
    # introduce flow
    # ------------------------------------------------------------------

    async def introduce(self, target_pubkey_pem: str) -> PunchOutcome:
        """Ask the seed to introduce us to a peer; punch a hole; return
        the (now-reachable) peer addr. Times out after
        PUNCH_RESULT_TIMEOUT_S if we never hear back."""
        ts = time.time()
        signed = f"INTRODUCE|{self.local_pubkey_pem}|{target_pubkey_pem}|{ts}"
        # We don't know the nonce yet; the seed picks it. We register a
        # placeholder fut keyed by `target_pubkey` and migrate it when
        # PUNCH_INVITE arrives with the assigned nonce. Simpler: future
        # keyed by target_pubkey only -- one outstanding introduce per
        # target at a time.
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        # Arm one shot keyed by target (replace any prior).
        old = self._pending.get(target_pubkey_pem)
        if old is not None and not old.future.done():
            old.future.cancel()
        self._pending[target_pubkey_pem] = _Pending(
            nonce="", target_pubkey=target_pubkey_pem, future=fut,
        )
        self._send({
            "op": "INTRODUCE",
            "from_pubkey": self.local_pubkey_pem,
            "target_pubkey": target_pubkey_pem,
            "timestamp": ts,
            "signature": self.sign(signed),
        }, self.seed_addr)
        try:
            outcome = await asyncio.wait_for(fut, timeout=PUNCH_RESULT_TIMEOUT_S)
            return outcome
        except asyncio.TimeoutError:
            self._pending.pop(target_pubkey_pem, None)
            return PunchOutcome(success=False, detail="punch_timeout")

    async def _do_hello_burst(
        self, peer_addr: Tuple[str, int], nonce: str,
    ) -> None:
        """Fire PUNCH_HELLO at the peer N times, ~50ms apart. The first
        packet creates the local NAT pinhole; the burst defeats 1-2
        dropped UDPs without giving up too quickly."""
        for _ in range(PUNCH_HELLO_BURST_COUNT):
            self._send({
                "op": "PUNCH_HELLO",
                "nonce": nonce,
                "from_pubkey": self.local_pubkey_pem,
            }, peer_addr)
            await asyncio.sleep(PUNCH_HELLO_BURST_INTERVAL_S)

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

        # 1. REGISTER_UDP reply -- update external_addr.
        if msg.get("status") == "ok" and "external_ip" in msg:
            self.external_addr = (str(msg["external_ip"]),
                                  int(msg["external_port"]))
            return

        op = msg.get("op")
        # 2. PUNCH_INVITE -- from seed. Two cases:
        #    (a) We INTRODUCEd; this is the seed mirroring our request.
        #    (b) Someone else INTRODUCEd to us.
        # In either case, fire PUNCH_HELLO at the indicated peer addr
        # to open OUR side of the pinhole.
        if op == "PUNCH_INVITE":
            try:
                peer_pub = str(msg["peer_pubkey"])
                peer_addr = (str(msg["peer_ip"]), int(msg["peer_port"]))
                nonce = str(msg["nonce"])
            except (KeyError, TypeError, ValueError):
                return
            asyncio.create_task(self._do_hello_burst(peer_addr, nonce))
            # Migrate the pending future from target-keyed to nonce-keyed.
            pending = self._pending.get(peer_pub)
            if pending is not None and pending.future and not pending.future.done():
                pending.nonce = nonce
            # Inbound (case b) -- run callback if registered.
            if self.on_invite is not None:
                try:
                    self.on_invite(peer_addr, peer_pub)
                except Exception as e:                              # pragma: no cover
                    logger.warning("on_invite callback raised: %s", e)
            return

        # 3. PUNCH_HELLO from a peer (could be the inviter or the
        #    invitee; protocol is symmetric).
        if op == "PUNCH_HELLO":
            try:
                nonce = str(msg["nonce"])
                from_pub = str(msg["from_pubkey"])
            except (KeyError, TypeError, ValueError):
                return
            # ACK so the sender knows their pinhole punched through.
            self._send({
                "op": "PUNCH_ACK",
                "nonce": nonce,
                "from_pubkey": self.local_pubkey_pem,
            }, addr)
            self._heard_from[nonce] = addr
            self._maybe_resolve(from_pub, nonce, addr)
            return

        # 4. PUNCH_ACK from a peer -- both sides have punched through.
        if op == "PUNCH_ACK":
            try:
                nonce = str(msg["nonce"])
                from_pub = str(msg["from_pubkey"])
            except (KeyError, TypeError, ValueError):
                return
            self._heard_from[nonce] = addr
            self._maybe_resolve(from_pub, nonce, addr)
            return

    def _maybe_resolve(
        self, peer_pub: str, nonce: str, peer_addr: Tuple[str, int],
    ) -> None:
        """If we have a pending future for this peer, resolve it."""
        pending = self._pending.get(peer_pub)
        if pending is None or pending.future.done():
            return
        if pending.nonce and pending.nonce != nonce:
            # Wrong nonce -- ignore (could be a stale invite).
            return
        pending.future.set_result(PunchOutcome(
            success=True, peer_addr=peer_addr, nonce=nonce,
        ))
        self._pending.pop(peer_pub, None)


class HolePunchNotImplementedError(NotImplementedError):
    """Kept as a typed exception for callers that pre-date the real
    implementation. Raise it explicitly only when the seed can't be
    reached AND the TURN-relay fallback hasn't been wired."""


def coordinate(*args, **kwargs):
    """Backwards-compatible shim. Real call surface is HolePunchClient
    (instance, not a free function). The CP-2 stub accepted any args
    and raised; we keep that surface so existing code that catches
    HolePunchNotImplementedError still compiles, but the actual
    coordination path is HolePunchClient.start() + .introduce()."""
    raise HolePunchNotImplementedError(
        "coordinate() is the legacy shim -- use "
        "core.nat.hole_punch.HolePunchClient.start() instead."
    )
