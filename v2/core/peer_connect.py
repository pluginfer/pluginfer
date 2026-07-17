"""Cross-region peer-connect protocol — India ↔ Singapore (or any two
peers behind home-grade NAT, regardless of geography).

This module is the *integration* the existing pieces have been waiting
for. The pieces:
  * ``core/nat/stun_client.py`` — discovers the local node's external
    (ip, port) via STUN.
  * ``infrastructure/seed_node/punch_server.py`` — the public seed
    that brokers UDP introductions + acts as TURN relay for symmetric
    NAT.
  * ``core/nat/hole_punch.py:HolePunchClient`` — the asyncio client
    that registers with the seed, calls INTRODUCE, fires PUNCH_HELLO,
    and returns the punched-through peer addr.
  * ``core/nat/turn_client.py`` — the TURN-relay fallback when direct
    UDP can't be punched (symmetric NAT both sides).

What was missing: the **all-in-one** flow that a node operator actually
calls. A user in India who wants to reach a friend in Singapore should
not need to read four module docstrings + assemble five steps. They
should call::

    session = await connect_to_peer(
        seed_addrs=[("seed.in.pluginfer.net", 9000),
                    ("seed.sg.pluginfer.net", 9000)],
        local_pubkey_pem=wallet.public_key_pem,
        sign=wallet.sign,
        target_pubkey_pem=friend_pubkey,
    )
    await session.send(b"hello from india")

That's it. This module owns:
  1. Picking the closest seed (round-robin RTT probe across regions).
  2. STUN-discovering the local external address (best-effort, used
     only as a hint for diagnostics — the seed observes it directly).
  3. Registering with the seed via UDP.
  4. Calling INTRODUCE for the target peer.
  5. On success: returning a PunchedSession bound to the punched UDP
     endpoint. The caller exchanges signed grains directly.
  6. On failure (symmetric NAT both sides): falling back to TURN
     relay via the same seed, transparently, so the caller's API
     doesn't change.

design rationale §H4-extended: a method of establishing a direct UDP
data path between two peers in different geographic regions in a
decentralised compute mesh, comprising: each peer registering with
one or more region-distributed seed servers; the dialing peer issuing
a signed INTRODUCE message naming the callee by public key; the seed
emitting symmetric PUNCH_INVITE messages to both peers carrying each
other's externally-observed (ip, port); each peer firing a burst of
PUNCH_HELLO datagrams at the other to open the corresponding NAT
pinhole; on first successful round-trip, returning a punched session
to the caller; on punch failure, transparently falling back to a
seed-mediated TURN relay session — without the user supplying any
configuration beyond the target peer's public key.

Hermetic-test target: a single in-process punch_server bound on
localhost serving as a "global seed", two `PeerConnectClient`s on
distinct localhost ports simulating geographically-separated peers
(behind imaginary NATs that only the seed sees through), an
INTRODUCE round-trip, a payload exchange.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from .nat.hole_punch import HolePunchClient, PunchOutcome

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class SeedAddress:
    """A region-distributed seed (e.g. seed.sg.pluginfer.net:9000)."""

    host: str
    port: int = 9000

    def as_tuple(self) -> Tuple[str, int]:
        return (self.host, int(self.port))


@dataclass
class ConnectResult:
    """Outcome of `connect_to_peer`."""

    success: bool
    peer_pubkey_pem: str
    method: str = "direct"        # "direct" | "turn" | "failed"
    peer_addr: Optional[Tuple[str, int]] = None
    seed_used: Optional[Tuple[str, int]] = None
    detail: Optional[str] = None
    rtt_ms: Optional[float] = None


@dataclass
class _TurnSession:
    """An open seed-mediated relay toward one peer. Lives on the
    punched socket — see `_try_turn_fallback` for why there is no
    second socket."""

    session_id: str
    peer_pubkey_pem: str
    expires_at: float


# ---------------------------------------------------------------------------
# PeerConnectClient — the all-in-one
# ---------------------------------------------------------------------------


class PeerConnectClient:
    """Long-lived client that owns the seed registration + introduce flow.

    Construct with ``await PeerConnectClient.start(...)``. Reuse one
    instance for the lifetime of the node — the keep-alive loop
    inherited from HolePunchClient maintains the seed registration.

    The class is intentionally thin: HolePunchClient already handles
    REGISTER_UDP, INTRODUCE, PUNCH_HELLO/_ACK, and timeouts. This
    wrapper adds:
      * Multi-seed selection. If multiple seeds are configured (one in
        each region), pick the lowest-RTT seed at start-up. On a punch
        failure, retry against the next-best seed before falling back
        to TURN.
      * TURN fallback. When INTRODUCE returns timeout (symmetric NAT
        both sides), open a relay session and surface a session-shaped
        send/recv interface so the caller doesn't care which transport
        was used.
      * Diagnostic structured result (`ConnectResult`).
    """

    def __init__(
        self,
        *,
        seeds: List[SeedAddress],
        local_pubkey_pem: str,
        sign: Callable[[str], str],
    ) -> None:
        if not seeds:
            raise ValueError("at least one seed required")
        self.seeds = list(seeds)
        self.local_pubkey_pem = local_pubkey_pem
        self.sign = sign
        # Active HolePunchClient bound to the chosen seed.
        self._hp: Optional[HolePunchClient] = None
        # The seed we're currently registered with.
        self._active_seed: Optional[SeedAddress] = None
        # Transport paths per peer pubkey. `send_to_peer` prefers the
        # punched (ip, port); falls back to an open TURN session.
        self._punched: Dict[str, Tuple[str, int]] = {}
        self._turn_sessions: Dict[str, _TurnSession] = {}
        # FIFO of (target_pubkey, future) awaiting a RELAY_OPEN reply
        # (the seed's reply doesn't echo the target, so order matters —
        # same convention as turn_client).
        self._pending_relay_open: List[Tuple[str, asyncio.Future]] = []
        # Application inbound handler (set via set_inbound_handler).
        self._app_handler: Optional[
            Callable[[bytes, Tuple[str, int]], None]] = None

    # ------------------------------------------------------------------
    @classmethod
    async def start(
        cls,
        *,
        seeds: List[SeedAddress],
        local_pubkey_pem: str,
        sign: Callable[[str], str],
        bind_host: str = "0.0.0.0",
        bind_port: int = 0,
    ) -> "PeerConnectClient":
        """Pick a seed (lowest RTT) and bring up the punch client."""
        self = cls(
            seeds=seeds,
            local_pubkey_pem=local_pubkey_pem,
            sign=sign,
        )
        await self._bring_up(bind_host=bind_host, bind_port=bind_port)
        return self

    async def _bring_up(self, *, bind_host: str, bind_port: int) -> None:
        chosen = await self._pick_seed()
        self._active_seed = chosen
        self._hp = await HolePunchClient.start(
            seed_addr=chosen.as_tuple(),
            local_pubkey_pem=self.local_pubkey_pem,
            sign=self.sign,
            bind_host=bind_host,
            bind_port=bind_port,
        )
        self._install_dispatcher()

    def _install_dispatcher(self) -> None:
        """Route every datagram on the punched socket to its consumer.

        ONE UDP socket carries four traffics — punch protocol, TURN
        control replies, TURN payload delivery, and direct application
        payloads — because the punch server pins RELAY security to the
        REGISTERED source address: a RELAY_OPEN from any second socket
        is rejected with src_mismatch by design. Everything must ride
        the socket that REGISTER_UDP'd."""
        original = self._hp.datagram_received

        def _dispatched(data: bytes, addr: Tuple[str, int]) -> None:
            try:
                msg = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                msg = None
            if isinstance(msg, dict):
                if msg.get("op") == "RELAY_DELIVER":
                    self._on_relay_deliver(msg)
                    return
                if msg.get("status") == "ok" and "session" in msg:
                    self._resolve_relay_open(msg)
                    return
                if "op" in msg or "status" in msg:
                    return original(data, addr)
                # JSON without protocol markers falls through to app.
            if self._app_handler is not None:
                try:
                    self._app_handler(data, addr)
                except Exception as e:
                    logger.warning("inbound handler raised: %s", e)

        self._hp.datagram_received = _dispatched  # type: ignore[method-assign]

    def _on_relay_deliver(self, msg: dict) -> None:
        """TURN payload from the seed. Deliver to the app handler AND
        remember the session — the server accepts either end of a
        session as sender, so the responder can reply on the same
        relay without ever calling connect_to_peer."""
        try:
            from_pub = str(msg["from_pubkey"])
            sid = str(msg["session"])
            payload = base64.b64decode(str(msg["payload_b64"]))
        except (KeyError, TypeError, ValueError):
            return
        known = self._turn_sessions.get(from_pub)
        if known is None or known.session_id != sid:
            self._turn_sessions[from_pub] = _TurnSession(
                session_id=sid, peer_pubkey_pem=from_pub,
                expires_at=time.time() + 240.0,
            )
        else:
            known.expires_at = time.time() + 240.0
        if self._app_handler is not None:
            try:
                self._app_handler(payload, ("turn", from_pub))
            except Exception as e:
                logger.warning("inbound handler raised: %s", e)

    def _resolve_relay_open(self, msg: dict) -> None:
        sid = str(msg["session"])
        ttl = float(msg.get("ttl_s", 300.0))
        while self._pending_relay_open:
            target, fut = self._pending_relay_open.pop(0)
            if fut.done():
                continue
            sess = _TurnSession(
                session_id=sid, peer_pubkey_pem=target,
                expires_at=time.time() + ttl,
            )
            self._turn_sessions[target] = sess
            fut.set_result(sess)
            return

    async def _pick_seed(self) -> SeedAddress:
        """RTT probe each seed; choose the lowest. Single-shot UDP echo
        against REGISTER_UDP. With one seed, just return it."""
        if len(self.seeds) == 1:
            return self.seeds[0]
        # Tiny probe: open a dgram socket per seed, send a no-op
        # JSON, time the first packet back. Probes that fail return
        # +inf so they're sorted last.
        async def _probe(s: SeedAddress) -> Tuple[float, SeedAddress]:
            t0 = time.monotonic()
            try:
                loop = asyncio.get_running_loop()
                fut: asyncio.Future = loop.create_future()

                class _Probe(asyncio.DatagramProtocol):
                    def datagram_received(self, _data, _addr):
                        if not fut.done():
                            fut.set_result(time.monotonic() - t0)

                transport, _proto = await loop.create_datagram_endpoint(
                    _Probe, remote_addr=s.as_tuple(),
                )
                # Send something the server will safely reject -- it
                # still emits a reply, which is what we time.
                ts = time.time()
                signed = f"REGISTER_UDP|{self.local_pubkey_pem}|{ts}"
                transport.sendto(json.dumps({
                    "op": "REGISTER_UDP",
                    "pubkey_pem": self.local_pubkey_pem,
                    "timestamp": ts,
                    "signature": self.sign(signed),
                }).encode("utf-8"))
                try:
                    rtt = await asyncio.wait_for(fut, timeout=1.5)
                except asyncio.TimeoutError:
                    rtt = float("inf")
                transport.close()
                return rtt, s
            except Exception:
                return float("inf"), s

        results = await asyncio.gather(*(_probe(s) for s in self.seeds))
        results.sort(key=lambda r: r[0])
        return results[0][1]

    # ------------------------------------------------------------------
    @property
    def external_addr(self) -> Optional[Tuple[str, int]]:
        """The (ip, port) the seed reports back. May be None until the
        first REGISTER_UDP reply arrives."""
        return self._hp.external_addr if self._hp else None

    @property
    def active_seed(self) -> Optional[SeedAddress]:
        return self._active_seed

    # ------------------------------------------------------------------
    async def connect_to_peer(
        self, target_pubkey_pem: str,
        *,
        retry_seeds: bool = True,
    ) -> ConnectResult:
        """Try to establish a direct UDP path to ``target_pubkey_pem``.

        Strategy:
          1. INTRODUCE via the currently-active seed.
          2. If INTRODUCE times out and ``retry_seeds=True``, rotate to
             the next-best seed and try once more.
          3. On all introduce failures, fall back to TURN relay
             (`PunchedSession.via_turn=True`).
          4. On TURN failure, surface a structured failure.
        """
        if self._hp is None:
            return ConnectResult(
                success=False,
                peer_pubkey_pem=target_pubkey_pem,
                method="failed",
                detail="punch client not started",
            )
        t0 = time.monotonic()
        outcome: PunchOutcome = await self._hp.introduce(target_pubkey_pem)
        if outcome.success:
            self._punched[target_pubkey_pem] = outcome.peer_addr
            return ConnectResult(
                success=True,
                peer_pubkey_pem=target_pubkey_pem,
                method="direct",
                peer_addr=outcome.peer_addr,
                seed_used=self._active_seed.as_tuple() if self._active_seed else None,
                rtt_ms=(time.monotonic() - t0) * 1000.0,
                detail="direct UDP path opened via punch",
            )
        # Try the next seed if we have more than one.
        if retry_seeds and len(self.seeds) > 1:
            others = [s for s in self.seeds if s != self._active_seed]
            for s in others:
                try:
                    self._hp.close()
                except Exception:
                    pass
                self._hp = None
                self._active_seed = s
                self._hp = await HolePunchClient.start(
                    seed_addr=s.as_tuple(),
                    local_pubkey_pem=self.local_pubkey_pem,
                    sign=self.sign,
                )
                self._install_dispatcher()
                outcome = await self._hp.introduce(target_pubkey_pem)
                if outcome.success:
                    self._punched[target_pubkey_pem] = outcome.peer_addr
                    return ConnectResult(
                        success=True,
                        peer_pubkey_pem=target_pubkey_pem,
                        method="direct",
                        peer_addr=outcome.peer_addr,
                        seed_used=s.as_tuple(),
                        rtt_ms=(time.monotonic() - t0) * 1000.0,
                        detail="direct UDP path via fallback seed",
                    )
        # Direct punch failed end-to-end. Try TURN.
        return await self._try_turn_fallback(target_pubkey_pem, t0)

    # ------------------------------------------------------------------
    async def _try_turn_fallback(
        self,
        target_pubkey_pem: str,
        t0: float,
    ) -> ConnectResult:
        """Open a seed-mediated TURN relay session ON THE PUNCHED SOCKET.

        Two hard-won constraints shape this:
          * An earlier revision constructed a standalone TurnRelayClient
            here. Dead on arrival twice over: the call never matched
            turn_client's surface (instance method invoked like a
            constructor → TypeError, swallowed), AND the punch server
            pins RELAY_OPEN/RELAY to the socket that REGISTER_UDP'd —
            any second socket gets src_mismatch by design.
          * Therefore the relay must ride the already-registered
            hole-punch socket; replies are resolved by the dispatcher
            (`_resolve_relay_open`) and the session lands in
            `_turn_sessions` where `send_to_peer` finds it.

        The caller's contract is unchanged: on success, `send_to_peer`
        + the inbound handler work exactly as they do for a punched
        path — just slower (extra hop through the seed, 50 MB/session
        server-side quota).
        """
        if self._hp is None or self._active_seed is None:
            return ConnectResult(
                success=False,
                peer_pubkey_pem=target_pubkey_pem,
                method="failed",
                detail="no punch client / active seed; cannot TURN-relay",
            )
        ts = time.time()
        signed = (
            f"RELAY_OPEN|{self.local_pubkey_pem}|{target_pubkey_pem}|{ts}"
        )
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_relay_open.append((target_pubkey_pem, fut))
        try:
            self._hp._send({
                "op": "RELAY_OPEN",
                "from_pubkey": self.local_pubkey_pem,
                "target_pubkey": target_pubkey_pem,
                "timestamp": ts,
                "signature": self.sign(signed),
            }, self._active_seed.as_tuple())
            sess = await asyncio.wait_for(fut, timeout=5.0)
        except Exception as e:
            self._pending_relay_open = [
                (t, f) for (t, f) in self._pending_relay_open if f is not fut
            ]
            return ConnectResult(
                success=False,
                peer_pubkey_pem=target_pubkey_pem,
                method="failed",
                detail=f"turn open failed: {e!r}",
                seed_used=self._active_seed.as_tuple(),
            )
        return ConnectResult(
            success=True,
            peer_pubkey_pem=target_pubkey_pem,
            method="turn",
            seed_used=self._active_seed.as_tuple(),
            rtt_ms=(time.monotonic() - t0) * 1000.0,
            detail=f"turn-relay session {sess.session_id}",
        )

    # ------------------------------------------------------------------
    # Unified transport surface — callers (punch_rpc, auto_mesh) never
    # care whether the path is punched UDP or a TURN relay.
    # ------------------------------------------------------------------

    def note_punched_addr(
        self, peer_pubkey_pem: str, addr: Tuple[str, int],
    ) -> None:
        """Record a live return path learned from INBOUND traffic. The
        responder side of an RPC never called connect_to_peer — the
        requester's datagrams teach it where to reply."""
        self._punched[peer_pubkey_pem] = (str(addr[0]), int(addr[1]))

    def has_path(self, peer_pubkey_pem: str) -> bool:
        if peer_pubkey_pem in self._punched:
            return True
        sess = self._turn_sessions.get(peer_pubkey_pem)
        return bool(sess and time.time() < sess.expires_at)

    async def send_to_peer(
        self, peer_pubkey_pem: str, payload: bytes,
    ) -> bool:
        """Send over whichever path exists: punched direct UDP first,
        TURN relay second. False = no path (connect_to_peer first)."""
        if self._hp is None or self._hp.transport is None:
            return False
        addr = self._punched.get(peer_pubkey_pem)
        if addr is not None:
            self._hp.transport.sendto(payload, addr)
            return True
        sess = self._turn_sessions.get(peer_pubkey_pem)
        if (sess and time.time() < sess.expires_at
                and self._active_seed is not None):
            self._hp._send({
                "op": "RELAY",
                "session": sess.session_id,
                "payload_b64": base64.b64encode(payload).decode("ascii"),
            }, self._active_seed.as_tuple())
            return True
        return False

    # ------------------------------------------------------------------
    async def send_to_punched_peer(
        self,
        peer_addr: Tuple[str, int],
        payload: bytes,
    ) -> None:
        """Send a single UDP datagram on the already-punched path.

        Caller is responsible for fragmenting if `payload` exceeds the
        path MTU. Production grain transport (`core/grain`) handles
        fragmentation natively once the path is up.
        """
        if self._hp is None or self._hp.transport is None:
            raise RuntimeError("punch client not connected")
        self._hp.transport.sendto(payload, peer_addr)

    # ------------------------------------------------------------------
    def set_inbound_handler(
        self,
        handler: Callable[[bytes, Tuple[str, int]], None],
    ) -> None:
        """Register a handler for application-level datagrams.

        Protocol packets (PUNCH_HELLO/_ACK, REGISTER_UDP replies,
        PUNCH_INVITE, RELAY control) are consumed by the dispatcher
        installed at bring-up. Application payloads arrive here from
        BOTH transports: `addr` is the real (ip, port) for punched
        datagrams and ``("turn", peer_pubkey_pem)`` for TURN-relayed
        ones. The handler runs synchronously inside
        ``datagram_received``; keep it fast (queue +
        asyncio.create_task for slow work).
        """
        if self._hp is None:
            raise RuntimeError("punch client not connected")
        self._app_handler = handler

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._hp is not None:
            try:
                self._hp.close()
            except Exception:
                pass
            self._hp = None


# ---------------------------------------------------------------------------
# Convenience top-level entry point
# ---------------------------------------------------------------------------


async def connect_to_peer(
    *,
    seed_addrs: List[Tuple[str, int]],
    local_pubkey_pem: str,
    sign: Callable[[str], str],
    target_pubkey_pem: str,
    bind_host: str = "0.0.0.0",
    bind_port: int = 0,
) -> Tuple[PeerConnectClient, ConnectResult]:
    """One-call cross-region rendezvous. Returns the long-lived client
    AND the connect result.

    Caller keeps the client to drive subsequent sends + future
    introduces; the result describes what happened on this attempt.
    """
    seeds = [SeedAddress(host=h, port=p) for (h, p) in seed_addrs]
    client = await PeerConnectClient.start(
        seeds=seeds,
        local_pubkey_pem=local_pubkey_pem,
        sign=sign,
        bind_host=bind_host,
        bind_port=bind_port,
    )
    result = await client.connect_to_peer(target_pubkey_pem)
    return client, result
