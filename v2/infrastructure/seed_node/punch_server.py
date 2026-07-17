"""Seed-node UDP hole-punch + TURN relay.

Runs alongside the existing TCP REGISTER/PEERS server (`seed_server.py`)
on the same VPS, listening on the same port over UDP. Three jobs:

  1. **UDP keep-alive registration** -- peers send a periodic
     REGISTER_UDP so the seed learns each peer's external UDP
     (ip, port) as observed FROM THE SEED'S NETWORK PERSPECTIVE.
     This is what STUN gives you, but folded into the same channel.

  2. **Hole-punch introduction** -- peer A sends INTRODUCE(target=B's
     pubkey). The seed sends a PUNCH_INVITE to B with A's external
     (ip, port) + a fresh nonce, AND a mirror PUNCH_INVITE to A with
     B's external (ip, port) + the same nonce. Both peers then fire
     PUNCH_HELLO at each other simultaneously; the first packet on
     each side opens the NAT pinhole. Works for full-cone /
     restricted-cone / port-restricted-cone NATs.

  3. **TURN relay** -- for symmetric-NAT peers (different external
     port per destination, ~5-15% of consumer routers), the same
     seed acts as a relay. RELAY_OPEN allocates a session id;
     subsequent RELAY packets are forwarded to the session partner.
     Per-peer bandwidth is metered so we can bill or throttle.

Wire format
-----------

JSON, one packet per UDP datagram. MAX 8 KiB per datagram (UDP-friendly).

  Client -> Seed REGISTER_UDP:
    {"op":"REGISTER_UDP","pubkey_pem":"...","timestamp":<unix>,
     "signature":"<b64>"}
    Signed bytes: f"REGISTER_UDP|{pubkey_pem}|{timestamp}"
  Seed -> Client:
    {"status":"ok","external_ip":"...","external_port":N,"ttl_s":120}

  Client -> Seed INTRODUCE:
    {"op":"INTRODUCE","from_pubkey":"...","target_pubkey":"...",
     "timestamp":<unix>,"signature":"<b64>"}
    Signed bytes: f"INTRODUCE|{from_pubkey}|{target_pubkey}|{ts}"
  Seed -> both peers (asynchronously):
    {"op":"PUNCH_INVITE","peer_pubkey":"...","peer_ip":"...",
     "peer_port":N,"nonce":"<hex>"}

  Peer-to-peer PUNCH_HELLO (no seed in the path):
    {"op":"PUNCH_HELLO","nonce":"<hex>","from_pubkey":"..."}
    Reply:
    {"op":"PUNCH_ACK","nonce":"<hex>","from_pubkey":"..."}

  Client -> Seed RELAY_OPEN (TURN fallback):
    {"op":"RELAY_OPEN","from_pubkey":"...","target_pubkey":"...",
     "timestamp":<unix>,"signature":"<b64>"}
    Signed bytes: f"RELAY_OPEN|{from_pubkey}|{target_pubkey}|{ts}"
  Seed -> Client:
    {"status":"ok","session":"<hex>","ttl_s":300}

  Client -> Seed RELAY (within an open session):
    {"op":"RELAY","session":"...","payload_b64":"..."}
  Seed -> Other end of session:
    {"op":"RELAY_DELIVER","session":"...","payload_b64":"...",
     "from_pubkey":"..."}
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger("punch_server")

UDP_TTL_SECONDS: int = 120              # peer must REGISTER_UDP every <= 2 min
TIMESTAMP_WINDOW: int = 30
RELAY_TTL_SECONDS: int = 300
MAX_DATAGRAM_BYTES: int = 8 * 1024
RELAY_PER_SESSION_BANDWIDTH_BYTES: int = 50 * 1024 * 1024   # 50 MB / session


@dataclass
class _UDPRegistration:
    pubkey_pem: str
    addr: Tuple[str, int]                # (ip, port) AS SEEN BY THE SEED
    expires_at: float


@dataclass
class _RelaySession:
    session_id: str
    a_pubkey: str
    b_pubkey: str
    expires_at: float
    bytes_used: int = 0


def _signed_bytes(parts: list[str]) -> str:
    return "|".join(parts)


def _verify(pubkey_pem: str, signed_str: str, signature_b64: str) -> bool:
    """Lazy-imports the crypto path so the punch_server can deploy
    without dragging the full v2/core tree (mirrors seed_server's
    pattern)."""
    try:
        from core.tokenomics import Wallet
        return Wallet.verify(pubkey_pem, signed_str, signature_b64)
    except Exception:
        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import ec
            pub = serialization.load_pem_public_key(pubkey_pem.encode())
            try:
                pub.verify(
                    base64.b64decode(signature_b64),
                    signed_str.encode(),
                    ec.ECDSA(hashes.SHA256()),
                )
                return True
            except InvalidSignature:
                return False
        except Exception as e:           # pragma: no cover - defensive
            logger.error("inline verify failed: %s", e)
            return False


class PunchRelayState:
    """Pure in-memory state. Asyncio protocol calls into this; tests
    drive it directly. Separating state from transport lets us unit-
    test the introduce + relay logic without spinning a real socket."""

    def __init__(self) -> None:
        # pubkey_pem -> _UDPRegistration
        self.regs: dict[str, _UDPRegistration] = {}
        # session_id -> _RelaySession
        self.sessions: dict[str, _RelaySession] = {}
        # bookkeeping
        self.metrics_introduce_ok: int = 0
        self.metrics_introduce_fail: int = 0
        self.metrics_relay_bytes_total: int = 0

    # ------------------------------------------------------------------
    def expire(self, now: Optional[float] = None) -> int:
        now = now or time.time()
        gone = 0
        for pk in [p for p, r in self.regs.items() if r.expires_at <= now]:
            del self.regs[pk]
            gone += 1
        for sid in [s for s, r in self.sessions.items() if r.expires_at <= now]:
            del self.sessions[sid]
        return gone

    # ------------------------------------------------------------------
    # REGISTER_UDP -- the peer learns its external (ip, port) here.
    def register_udp(
        self, msg: dict, src_addr: Tuple[str, int], now: float,
    ) -> dict:
        try:
            pub = str(msg["pubkey_pem"])
            ts = float(msg["timestamp"])
            sig = str(msg["signature"])
        except (KeyError, TypeError, ValueError) as e:
            return {"status": "error", "code": "bad_request",
                    "reason": f"missing/invalid field: {e!r}"}
        if abs(now - ts) > TIMESTAMP_WINDOW:
            return {"status": "error", "code": "stale_timestamp"}
        if not _verify(pub, _signed_bytes(["REGISTER_UDP", pub, str(ts)]), sig):
            return {"status": "error", "code": "bad_signature"}
        self.regs[pub] = _UDPRegistration(
            pubkey_pem=pub,
            addr=src_addr,
            expires_at=now + UDP_TTL_SECONDS,
        )
        return {
            "status": "ok",
            "external_ip": src_addr[0],
            "external_port": src_addr[1],
            "ttl_s": UDP_TTL_SECONDS,
        }

    # ------------------------------------------------------------------
    # INTRODUCE -- emits TWO outgoing packets (to A, to B) for the caller
    # to actually send. Returns (reply_to_caller, [(addr, packet), ...]).
    def introduce(
        self, msg: dict, src_addr: Tuple[str, int], now: float,
    ) -> Tuple[dict, list[Tuple[Tuple[str, int], dict]]]:
        try:
            from_pub = str(msg["from_pubkey"])
            target_pub = str(msg["target_pubkey"])
            ts = float(msg["timestamp"])
            sig = str(msg["signature"])
        except (KeyError, TypeError, ValueError) as e:
            self.metrics_introduce_fail += 1
            return {"status": "error", "code": "bad_request",
                    "reason": f"missing/invalid field: {e!r}"}, []
        if abs(now - ts) > TIMESTAMP_WINDOW:
            self.metrics_introduce_fail += 1
            return {"status": "error", "code": "stale_timestamp"}, []
        if not _verify(
            from_pub,
            _signed_bytes(["INTRODUCE", from_pub, target_pub, str(ts)]),
            sig,
        ):
            self.metrics_introduce_fail += 1
            return {"status": "error", "code": "bad_signature"}, []

        # Self-introduce is nonsense (and can be used to amplify packets
        # back to a victim if we don't filter it).
        if from_pub == target_pub:
            self.metrics_introduce_fail += 1
            return {"status": "error", "code": "self_introduce_forbidden"}, []

        # Both peers must be registered for us to know where to send.
        a = self.regs.get(from_pub)
        b = self.regs.get(target_pub)
        if a is None or b is None:
            self.metrics_introduce_fail += 1
            return {"status": "error", "code": "peer_not_registered"}, []

        # Anti-spoof: the registered addr for `from_pub` must match the
        # source addr of THIS packet. Otherwise an attacker who sniffs a
        # single signed INTRODUCE could replay it from anywhere and
        # reflect punch packets at arbitrary victims.
        if a.addr != src_addr:
            self.metrics_introduce_fail += 1
            return {"status": "error", "code": "src_mismatch"}, []

        nonce = secrets.token_hex(16)
        invite_to_b = {
            "op": "PUNCH_INVITE",
            "peer_pubkey": from_pub,
            "peer_ip": a.addr[0],
            "peer_port": a.addr[1],
            "nonce": nonce,
        }
        invite_to_a = {
            "op": "PUNCH_INVITE",
            "peer_pubkey": target_pub,
            "peer_ip": b.addr[0],
            "peer_port": b.addr[1],
            "nonce": nonce,
        }
        self.metrics_introduce_ok += 1
        return {"status": "ok", "nonce": nonce}, [
            (b.addr, invite_to_b),
            (a.addr, invite_to_a),
        ]

    # ------------------------------------------------------------------
    # RELAY_OPEN -- allocate a session id both peers can use
    def relay_open(
        self, msg: dict, src_addr: Tuple[str, int], now: float,
    ) -> dict:
        try:
            from_pub = str(msg["from_pubkey"])
            target_pub = str(msg["target_pubkey"])
            ts = float(msg["timestamp"])
            sig = str(msg["signature"])
        except (KeyError, TypeError, ValueError) as e:
            return {"status": "error", "code": "bad_request",
                    "reason": f"missing/invalid field: {e!r}"}
        if abs(now - ts) > TIMESTAMP_WINDOW:
            return {"status": "error", "code": "stale_timestamp"}
        if not _verify(
            from_pub,
            _signed_bytes(["RELAY_OPEN", from_pub, target_pub, str(ts)]),
            sig,
        ):
            return {"status": "error", "code": "bad_signature"}
        if from_pub == target_pub:
            return {"status": "error", "code": "self_relay_forbidden"}
        a = self.regs.get(from_pub)
        if a is None or a.addr != src_addr:
            return {"status": "error", "code": "src_mismatch"}
        if target_pub not in self.regs:
            return {"status": "error", "code": "peer_not_registered"}
        sid = secrets.token_hex(16)
        self.sessions[sid] = _RelaySession(
            session_id=sid,
            a_pubkey=from_pub,
            b_pubkey=target_pub,
            expires_at=now + RELAY_TTL_SECONDS,
        )
        return {"status": "ok", "session": sid, "ttl_s": RELAY_TTL_SECONDS}

    # ------------------------------------------------------------------
    # RELAY -- forward bytes to the OTHER side of the session
    def relay(
        self, msg: dict, src_addr: Tuple[str, int], now: float,
    ) -> Tuple[dict, list[Tuple[Tuple[str, int], dict]]]:
        sid = msg.get("session")
        payload = msg.get("payload_b64")
        if not isinstance(sid, str) or not isinstance(payload, str):
            return {"status": "error", "code": "bad_request"}, []
        sess = self.sessions.get(sid)
        if sess is None or sess.expires_at <= now:
            return {"status": "error", "code": "session_unknown_or_expired"}, []

        # Identify the sender: must be one of the two pubkeys in the
        # session AND its UDP addr must match the registered one.
        sender_pub: Optional[str] = None
        for cand in (sess.a_pubkey, sess.b_pubkey):
            r = self.regs.get(cand)
            if r is not None and r.addr == src_addr:
                sender_pub = cand
                break
        if sender_pub is None:
            return {"status": "error", "code": "src_not_in_session"}, []

        # Bandwidth ceiling per session (cheap DoS guard).
        try:
            blen = len(base64.b64decode(payload))
        except Exception:
            return {"status": "error", "code": "bad_payload_b64"}, []
        if sess.bytes_used + blen > RELAY_PER_SESSION_BANDWIDTH_BYTES:
            return {"status": "error", "code": "session_quota_exceeded"}, []
        sess.bytes_used += blen
        self.metrics_relay_bytes_total += blen

        partner_pub = (
            sess.b_pubkey if sender_pub == sess.a_pubkey else sess.a_pubkey
        )
        partner = self.regs.get(partner_pub)
        if partner is None:
            return {"status": "error", "code": "partner_offline"}, []
        forward = {
            "op": "RELAY_DELIVER",
            "session": sid,
            "from_pubkey": sender_pub,
            "payload_b64": payload,
        }
        return {"status": "ok"}, [(partner.addr, forward)]

    # ------------------------------------------------------------------
    # Single dispatch: caller hands us a parsed JSON dict + src; we
    # return (reply_to_caller_or_None, list_of_outgoing_packets).
    def dispatch(
        self, msg: dict, src_addr: Tuple[str, int], now: float,
    ) -> Tuple[Optional[dict], list[Tuple[Tuple[str, int], dict]]]:
        op = msg.get("op")
        if op == "REGISTER_UDP":
            return self.register_udp(msg, src_addr, now), []
        if op == "INTRODUCE":
            reply, out = self.introduce(msg, src_addr, now)
            return reply, out
        if op == "RELAY_OPEN":
            return self.relay_open(msg, src_addr, now), []
        if op == "RELAY":
            reply, out = self.relay(msg, src_addr, now)
            return reply, out
        return {"status": "error", "code": "unknown_op",
                "reason": f"unknown op: {op!r}"}, []


# ---------------------------------------------------------------------------
# asyncio UDP transport
# ---------------------------------------------------------------------------


class _PunchProtocol(asyncio.DatagramProtocol):
    """Thin asyncio wrapper around PunchRelayState."""

    def __init__(self, state: PunchRelayState) -> None:
        self.state = state
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:  # type: ignore[override]
        self.transport = transport

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        if len(data) > MAX_DATAGRAM_BYTES:
            return
        try:
            msg = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(msg, dict):
            return
        now = time.time()
        reply, outbound = self.state.dispatch(msg, addr, now)
        if reply is not None and self.transport is not None:
            self.transport.sendto(json.dumps(reply).encode("utf-8"), addr)
        for dst, pkt in outbound:
            if self.transport is not None:
                self.transport.sendto(json.dumps(pkt).encode("utf-8"), dst)


async def run_punch_server(host: str = "0.0.0.0", port: int = 9000) -> None:
    state = PunchRelayState()
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: _PunchProtocol(state),
        local_addr=(host, port),
    )
    expire_task = asyncio.create_task(_expire_loop(state))
    logger.info(json.dumps({"event": "PUNCH_STARTED", "host": host, "port": port}))
    try:
        # Run until cancelled.
        while True:
            await asyncio.sleep(3600)
    finally:
        expire_task.cancel()
        transport.close()


async def _expire_loop(state: PunchRelayState, interval_s: int = 30) -> None:
    while True:
        await asyncio.sleep(interval_s)
        state.expire()
