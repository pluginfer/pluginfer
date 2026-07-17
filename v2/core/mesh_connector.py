"""MeshConnector: one call, one channel, transparent NAT survival.

Owns ONE UDP socket per local node and multiplexes four classes of
traffic on it:

  * REGISTER_UDP        -- our own keep-alive registration with the seed
                           (handled inside HolePunchClient).
  * PUNCH_INVITE/HELLO  -- seed-brokered hole-punch (HolePunchClient).
  * RELAY_OPEN/RELAY    -- TURN relay through the seed (added here).
  * MESH_DATA           -- our application-level bytes once a peer is
                           connected (added here).

Why one socket: the seed's `PunchRelayState` keys registrations by
peer pubkey -> external (ip, port). Two sockets per node would mean
two REGISTER_UDP entries fighting for the same pubkey -- only the
last write wins, breaking either punch or relay depending on order.

Strategy:

  1. Hole-punch via HolePunchClient.introduce(). 4-second budget.
  2. On punch failure or timeout, fall back to TURN relay using the
     same socket: send RELAY_OPEN to seed, then RELAY packets.

Both paths produce a `MeshChannel` with identical `await ch.send(b"...")`
+ `ch.on_message(payload)` semantics. Caller doesn't know or care.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Tuple

from .nat.hole_punch import HolePunchClient, PunchOutcome

logger = logging.getLogger(__name__)

PUNCH_FALLBACK_BUDGET_S: float = 4.0
RELAY_OPEN_TIMEOUT_S: float = 5.0


@dataclass
class _RelaySession:
    session_id: str
    target_pubkey: str
    expires_at: float


@dataclass
class MeshChannel:
    """Duplex bytes channel to one peer.

      * `direct=True`  -- post-hole-punch UDP. send() writes via the
                          shared socket directly to peer_addr.
      * `direct=False` -- TURN relay. send() wraps the payload in a
                          RELAY packet forwarded by the seed.
    """
    peer_pubkey: str
    direct: bool
    peer_addr: Optional[Tuple[str, int]] = None
    relay_session: Optional[_RelaySession] = None
    _send_fn: Optional[Callable[[bytes], Awaitable[None]]] = None
    on_message: Optional[Callable[[bytes], None]] = None
    bytes_sent: int = 0
    bytes_received: int = 0

    @property
    def strategy(self) -> str:
        return "direct" if self.direct else "relay"

    async def send(self, payload: bytes) -> None:
        if self._send_fn is None:
            raise RuntimeError("channel send fn not bound")
        await self._send_fn(payload)
        self.bytes_sent += len(payload)


class MeshConnector:
    """One coordinator per local node."""

    def __init__(self, *, punch: HolePunchClient, wallet) -> None:
        self.punch = punch
        self.wallet = wallet
        # peer_pubkey -> MeshChannel
        self.channels: dict[str, MeshChannel] = {}
        # session_id -> peer_pubkey
        self._session_index: dict[str, str] = {}
        # Pending RELAY_OPEN futures, keyed by target pubkey (one
        # outstanding open per target).
        self._pending_open: dict[str, asyncio.Future] = {}

        # Hook the punch protocol's datagram_received so we can also
        # handle RELAY_DELIVER + MESH_DATA + RELAY_OPEN replies.
        self._install_dispatch_hook()
        # Inbound INTRODUCEs from peers (when SOMEONE ELSE wanted us)
        # show up as PUNCH_INVITEs; the punch client's on_invite hook
        # tells us so we can pre-create a channel.
        self.punch.on_invite = self._on_punch_invite

    @classmethod
    async def start(
        cls,
        *,
        seed_addr: Tuple[str, int],
        wallet,
        bind_host: str = "0.0.0.0",
    ) -> "MeshConnector":
        punch = await HolePunchClient.start(
            seed_addr=seed_addr,
            local_pubkey_pem=wallet.public_key_pem,
            sign=wallet.sign,
            bind_host=bind_host,
        )
        return cls(punch=punch, wallet=wallet)

    def close(self) -> None:
        self.punch.close()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def connect(self, peer_pubkey_pem: str) -> MeshChannel:
        existing = self.channels.get(peer_pubkey_pem)
        if existing is not None:
            return existing

        # 1. Hole-punch attempt.
        try:
            outcome: PunchOutcome = await asyncio.wait_for(
                self.punch.introduce(peer_pubkey_pem),
                timeout=PUNCH_FALLBACK_BUDGET_S,
            )
        except asyncio.TimeoutError:
            outcome = PunchOutcome(success=False, detail="punch_timeout_local")
        if outcome.success and outcome.peer_addr is not None:
            ch = self._make_direct_channel(peer_pubkey_pem, outcome.peer_addr)
            self.channels[peer_pubkey_pem] = ch
            logger.info("mesh_connector direct: pubkey=%s addr=%s",
                        peer_pubkey_pem[-32:], outcome.peer_addr)
            return ch

        # 2. TURN relay fallback (via the same socket as the punch).
        logger.info("mesh_connector relay (punch failed: %s)",
                    outcome.detail)
        session = await self._open_relay(peer_pubkey_pem)
        ch = self._make_relay_channel(peer_pubkey_pem, session)
        self.channels[peer_pubkey_pem] = ch
        self._session_index[session.session_id] = peer_pubkey_pem
        return ch

    # ------------------------------------------------------------------
    # relay open over the SHARED socket
    # ------------------------------------------------------------------

    async def _open_relay(self, target_pubkey: str) -> _RelaySession:
        ts = time.time()
        signed = (
            f"RELAY_OPEN|{self.wallet.public_key_pem}|{target_pubkey}|{ts}"
        )
        if self.punch.transport is None:
            raise RuntimeError("punch transport closed; cannot open relay")
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_open[target_pubkey] = fut
        self.punch.transport.sendto(
            json.dumps({
                "op": "RELAY_OPEN",
                "from_pubkey": self.wallet.public_key_pem,
                "target_pubkey": target_pubkey,
                "timestamp": ts,
                "signature": self.wallet.sign(signed),
            }).encode("utf-8"),
            self.punch.seed_addr,
        )
        try:
            session = await asyncio.wait_for(fut, timeout=RELAY_OPEN_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._pending_open.pop(target_pubkey, None)
            raise TimeoutError("RELAY_OPEN timed out")
        return session

    # ------------------------------------------------------------------
    # channel constructors
    # ------------------------------------------------------------------

    def _make_direct_channel(
        self, peer_pubkey: str, peer_addr: Tuple[str, int],
    ) -> MeshChannel:
        ch = MeshChannel(peer_pubkey=peer_pubkey, direct=True,
                         peer_addr=peer_addr)

        async def _send(payload: bytes) -> None:
            self.punch.transport.sendto(
                json.dumps({
                    "op": "MESH_DATA",
                    "from_pubkey": self.wallet.public_key_pem,
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                }).encode("utf-8"),
                peer_addr,
            )

        ch._send_fn = _send
        return ch

    def _make_relay_channel(
        self, peer_pubkey: str, session: _RelaySession,
    ) -> MeshChannel:
        ch = MeshChannel(peer_pubkey=peer_pubkey, direct=False,
                         relay_session=session)

        async def _send(payload: bytes) -> None:
            if time.time() >= session.expires_at:
                raise RuntimeError("relay session expired -- re-open")
            self.punch.transport.sendto(
                json.dumps({
                    "op": "RELAY",
                    "session": session.session_id,
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                }).encode("utf-8"),
                self.punch.seed_addr,
            )

        ch._send_fn = _send
        return ch

    # ------------------------------------------------------------------
    # inbound dispatch (wraps the punch protocol's datagram_received)
    # ------------------------------------------------------------------

    def _install_dispatch_hook(self) -> None:
        original = self.punch.datagram_received

        def _wrapped(data: bytes, addr: Tuple[str, int]) -> None:
            try:
                msg = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return original(data, addr)
            if not isinstance(msg, dict):
                return original(data, addr)
            op = msg.get("op")

            # MESH_DATA: application bytes from a peer over the punch
            # NAT pinhole. Route to the channel's on_message.
            if op == "MESH_DATA":
                from_pub = str(msg.get("from_pubkey", ""))
                try:
                    payload = base64.b64decode(msg["payload_b64"])
                except Exception:
                    return
                ch = self.channels.get(from_pub)
                if ch is not None:
                    ch.bytes_received += len(payload)
                    if ch.on_message is not None:
                        try:
                            ch.on_message(payload)
                        except Exception as e:                          # pragma: no cover
                            logger.warning("on_message raised: %s", e)
                return

            # RELAY_DELIVER: forwarded by the seed. Find or create the
            # inbound relay channel and route.
            if op == "RELAY_DELIVER":
                from_pub = str(msg.get("from_pubkey", ""))
                sid = str(msg.get("session", ""))
                try:
                    payload = base64.b64decode(msg["payload_b64"])
                except Exception:
                    return
                ch = self.channels.get(from_pub)
                if ch is None:
                    # Inbound relay from a peer we haven't connected
                    # to yet -- create a channel keyed by from_pub.
                    sess = _RelaySession(
                        session_id=sid,
                        target_pubkey=from_pub,
                        expires_at=time.time() + 300.0,
                    )
                    ch = self._make_relay_channel(from_pub, sess)
                    self.channels[from_pub] = ch
                    self._session_index[sid] = from_pub
                ch.bytes_received += len(payload)
                if ch.on_message is not None:
                    try:
                        ch.on_message(payload)
                    except Exception as e:                              # pragma: no cover
                        logger.warning("on_message raised: %s", e)
                return

            # RELAY_OPEN reply (status=ok, session=...). Resolve the
            # FIRST pending open (one outstanding per target by
            # convention).
            if msg.get("status") == "ok" and "session" in msg:
                sid = str(msg["session"])
                ttl = float(msg.get("ttl_s", 300.0))
                for target, fut in list(self._pending_open.items()):
                    if not fut.done():
                        fut.set_result(_RelaySession(
                            session_id=sid,
                            target_pubkey=target,
                            expires_at=time.time() + ttl,
                        ))
                        self._pending_open.pop(target, None)
                        return
                # Fall through if no pending -- not for us.

            # Anything else: let the punch protocol handle it
            # (REGISTER_UDP reply, PUNCH_INVITE, PUNCH_HELLO,
            # PUNCH_ACK).
            return original(data, addr)

        self.punch.datagram_received = _wrapped  # type: ignore[method-assign]

    def _on_punch_invite(
        self, peer_addr: Tuple[str, int], peer_pubkey: str,
    ) -> None:
        """Pre-create a direct channel when SOMEONE ELSE is INTRODUCing
        themselves to us so the first MESH_DATA after the punch lands
        somewhere."""
        if peer_pubkey in self.channels:
            # Already have a channel; if it was the relay path, prefer
            # upgrading to direct now that punch succeeded -- but for
            # v1 we leave the existing channel as is to avoid in-
            # flight reordering.
            return
        ch = self._make_direct_channel(peer_pubkey, peer_addr)
        self.channels[peer_pubkey] = ch
