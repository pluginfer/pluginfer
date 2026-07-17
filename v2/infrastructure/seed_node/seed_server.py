"""Pluginfer Bootstrap Seed Server.

Lightweight asyncio TCP listener that:
  - Accepts peer registration messages signed with the peer's wallet pubkey.
  - Rejects registrations whose signature does not verify (ECDSA SECP256K1).
  - Rejects registrations whose timestamp is more than 30 seconds stale
    (replay protection).
  - Maintains an in-memory peer list with a 10-minute TTL per entry.
  - Responds to peer-list requests with up to 50 random live peers.
  - Exposes a /health-style PING that returns peer count.
  - Token-bucket rate limit: max 10 registrations per minute per source IP.
  - Logs registrations / expiries / rejections as structured JSON.

Wire protocol (newline-delimited JSON, one message per line):

  Client -> Server REGISTER:
    {"op": "REGISTER", "pubkey_pem": "...", "ip": "...", "port": 8100,
     "node_version": "1.0.0", "timestamp": <unix>, "signature": "<b64>"}

    Signed bytes: f"{pubkey_pem}|{ip}|{port}|{node_version}|{timestamp}"

  Server -> Client REGISTER ack:
    {"status": "ok", "ttl_seconds": 600, "peers": <n>}
    {"status": "error", "code": "...", "reason": "..."}

  Client -> Server PEERS:
    {"op": "PEERS", "max": 50}
  Server -> Client PEERS resp:
    {"status": "ok", "peers": [{"ip","port","pubkey_pem","node_version"}, ...]}

  Client -> Server PING:
    {"op": "PING"}
  Server -> Client PING resp:
    {"status": "ok", "peers": <n>, "uptime_s": <int>}

The server is intentionally small: ~250 lines, no external deps beyond
stdlib + cryptography (already in Pluginfer's runtime). It is designed to
run as a Linux container on a $5/month VPS and survive without state
beyond the in-memory peer list (which expires every restart and rebuilds
within a few minutes from registrations).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import random
import secrets
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any

# We rely on the project's tokenomics.Wallet.verify; if we ever want to
# run the seed independently of the rest of v2/, the verify function is
# small enough to lift out. For now, import lazily inside verify_signature
# so the script can be packaged into a slim container without dragging
# the whole v2/core/ tree.
logger = logging.getLogger("seed_server")

PEER_TTL_SECONDS: int = 600                # 10 minutes
TIMESTAMP_WINDOW_SECONDS: int = 30         # accept +/- 30s drift
RATE_LIMIT_PER_MINUTE: int = 10            # registrations per source IP
PEERS_PER_RESPONSE_DEFAULT: int = 50
MAX_LINE_BYTES: int = 16 * 1024            # protect against giant payloads
BANNER_VERSION: str = "pluginfer-seed/1.0.0"


@dataclass
class PeerRecord:
    pubkey_pem: str
    ip: str
    port: int
    node_version: str
    registered_at: float
    expires_at: float
    # The source IP the seed actually SAW the registration come from.
    # Behind NAT the node self-reports its LAN address (192.168.x.x),
    # which is unroutable across the WAN — observed_ip is the truth a
    # remote peer can dial. Kept alongside (not replacing) the signed
    # self-reported ip so signature verification stays intact.
    observed_ip: str = ""

    def to_wire(self) -> dict:
        return {
            "pubkey_pem": self.pubkey_pem,
            "ip": self.ip,
            "port": self.port,
            "node_version": self.node_version,
            "observed_ip": self.observed_ip,
        }


@dataclass
class _RateBucket:
    """Token bucket: 10 tokens, refill rate 10 / 60s = 0.1666 / s."""

    tokens: float = float(RATE_LIMIT_PER_MINUTE)
    last_refill: float = field(default_factory=time.time)

    def consume(self, now: float) -> bool:
        elapsed = max(0.0, now - self.last_refill)
        self.tokens = min(
            float(RATE_LIMIT_PER_MINUTE),
            self.tokens + elapsed * (RATE_LIMIT_PER_MINUTE / 60.0),
        )
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


def _signed_bytes(
    pubkey_pem: str, ip: str, port: int,
    node_version: str, timestamp: float,
) -> str:
    """Canonical signing payload. ORDER IS LOAD-BEARING; do not reorder."""
    return f"{pubkey_pem}|{ip}|{int(port)}|{node_version}|{timestamp}"


def verify_signature(
    pubkey_pem: str, signature_b64: str, signed_bytes: str,
) -> bool:
    """ECDSA verify using the project's Wallet helper.

    Lazy import so the seed package can be deployed without the full
    v2/core/ tree (only `core.tokenomics.Wallet.verify` is reached).
    """
    try:
        from core.tokenomics import Wallet
    except Exception:  # pragma: no cover - fallback if core/ isn't available
        # Inline minimal verify if running in a slim container. We rely on
        # `cryptography` which IS in Pluginfer's pinned deps.
        try:
            import base64
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import ec
            pub = serialization.load_pem_public_key(pubkey_pem.encode())
            try:
                pub.verify(
                    base64.b64decode(signature_b64),
                    signed_bytes.encode(),
                    ec.ECDSA(hashes.SHA256()),
                )
                return True
            except InvalidSignature:
                return False
        except Exception as e:  # pragma: no cover
            logger.error("inline verify failed: %s", e)
            return False
    return Wallet.verify(pubkey_pem, signed_bytes, signature_b64)


class SeedServer:
    """The Pluginfer bootstrap seed server (in-process state only)."""

    def __init__(self) -> None:
        self.peers: dict[str, PeerRecord] = {}   # pubkey -> record
        self.rate_buckets: dict[str, _RateBucket] = defaultdict(_RateBucket)
        self.started_at: float = time.time()
        self.recent_log: deque[dict] = deque(maxlen=1000)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def handle(self, msg: dict, client_ip: str) -> dict:
        op = msg.get("op")
        if op == "PING":
            return {
                "status": "ok",
                "peers": self._live_count(),
                "uptime_s": int(time.time() - self.started_at),
                "version": BANNER_VERSION,
            }
        if op == "REGISTER":
            return self._register(msg, client_ip)
        if op == "PEERS":
            return self._peers(msg)
        return {"status": "error", "code": "unknown_op",
                "reason": f"unknown op: {op!r}"}

    def expire(self, now: float | None = None) -> int:
        now = now or time.time()
        expired = [
            pk for pk, rec in self.peers.items() if rec.expires_at <= now
        ]
        for pk in expired:
            self._log("EXPIRE", pubkey_short=pk[:32])
            del self.peers[pk]
        return len(expired)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _register(self, msg: dict, client_ip: str) -> dict:
        now = time.time()
        # Rate limit BEFORE any expensive crypto work
        bucket = self.rate_buckets[client_ip]
        if not bucket.consume(now):
            self._log("RATE_LIMIT", client_ip=client_ip)
            return {"status": "error", "code": "rate_limited",
                    "reason": f"max {RATE_LIMIT_PER_MINUTE}/min per IP"}

        # Validate required fields
        try:
            pubkey_pem = str(msg["pubkey_pem"])
            ip = str(msg["ip"])
            port = int(msg["port"])
            node_version = str(msg["node_version"])
            timestamp = float(msg["timestamp"])
            signature = str(msg["signature"])
        except (KeyError, TypeError, ValueError) as e:
            return {"status": "error", "code": "bad_request",
                    "reason": f"missing/invalid field: {e!r}"}

        # Replay protection: timestamp must be within +/- window
        if abs(now - timestamp) > TIMESTAMP_WINDOW_SECONDS:
            self._log("STALE_TS", client_ip=client_ip,
                      drift=now - timestamp)
            return {"status": "error", "code": "stale_timestamp",
                    "reason": (
                        f"timestamp drift > {TIMESTAMP_WINDOW_SECONDS}s "
                        f"(client {timestamp:.0f}, server {now:.0f})"
                    )}

        # Signature verify
        signed = _signed_bytes(pubkey_pem, ip, port, node_version, timestamp)
        if not verify_signature(pubkey_pem, signature, signed):
            self._log("BAD_SIG", client_ip=client_ip,
                      pubkey_short=pubkey_pem[-32:])
            return {"status": "error", "code": "bad_signature",
                    "reason": "ECDSA verify failed"}

        # Sanity: ip looks IPv4/IPv6-ish; port in user-facing range
        if not (1 <= port <= 65_535):
            return {"status": "error", "code": "bad_port",
                    "reason": f"port out of range: {port}"}

        rec = PeerRecord(
            pubkey_pem=pubkey_pem, ip=ip, port=port,
            node_version=node_version,
            registered_at=now,
            expires_at=now + PEER_TTL_SECONDS,
            observed_ip=client_ip,
        )
        self.peers[pubkey_pem] = rec
        self._log("REGISTER", client_ip=client_ip,
                  pubkey_short=pubkey_pem[-32:],
                  ip=ip, port=port, node_version=node_version)
        # observed_ip doubles as free STUN: a NAT'd node that self-
        # reported its LAN address learns the public IP the world sees
        # and can re-register with it (signed) — that's what makes the
        # mesh work across the web, not just inside one WiFi.
        return {"status": "ok",
                "ttl_seconds": PEER_TTL_SECONDS,
                "peers": self._live_count(),
                "observed_ip": client_ip}

    def _peers(self, msg: dict) -> dict:
        max_n = int(msg.get("max", PEERS_PER_RESPONSE_DEFAULT))
        max_n = max(1, min(max_n, PEERS_PER_RESPONSE_DEFAULT))
        # Filter out expired entries on the read path too (don't depend
        # only on the periodic sweep).
        now = time.time()
        live = [
            r for r in self.peers.values() if r.expires_at > now
        ]
        random.shuffle(live)
        sample = live[:max_n]
        return {"status": "ok", "peers": [r.to_wire() for r in sample]}

    def _live_count(self) -> int:
        now = time.time()
        return sum(1 for r in self.peers.values() if r.expires_at > now)

    def _log(self, event: str, **fields: Any) -> None:
        entry = {"ts": time.time(), "event": event, **fields}
        self.recent_log.append(entry)
        logger.info(json.dumps(entry, default=str))


# ---------------------------------------------------------------------------
# asyncio TCP listener
# ---------------------------------------------------------------------------

async def _handle_client(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    server: SeedServer,
) -> None:
    peer = writer.get_extra_info("peername")
    client_ip = (peer[0] if peer else "?")
    try:
        # Single-message-per-connection protocol (cheap; no long-lived state).
        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        if not line:
            return
        if len(line) > MAX_LINE_BYTES:
            await _write_json(writer, {"status": "error",
                                       "code": "too_large",
                                       "reason": f"max {MAX_LINE_BYTES}"})
            return
        try:
            msg = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            await _write_json(writer, {"status": "error",
                                       "code": "bad_json"})
            return
        if not isinstance(msg, dict):
            await _write_json(writer, {"status": "error",
                                       "code": "bad_request"})
            return
        resp = server.handle(msg, client_ip=client_ip)
        await _write_json(writer, resp)
    except asyncio.TimeoutError:
        pass
    except Exception as e:  # pragma: no cover - defensive
        logger.error("handler error: %s", e)
    finally:
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()


async def _write_json(writer: asyncio.StreamWriter, body: dict) -> None:
    line = json.dumps(body).encode("utf-8") + b"\n"
    writer.write(line)
    await writer.drain()


async def _expire_loop(server: SeedServer, interval_s: int = 30) -> None:
    while True:
        await asyncio.sleep(interval_s)
        server.expire()


async def run_server(host: str = "0.0.0.0", port: int = 9000) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    server = SeedServer()
    tcp_server = await asyncio.start_server(
        lambda r, w: _handle_client(r, w, server), host=host, port=port,
    )
    expire_task = asyncio.create_task(_expire_loop(server))
    addrs = ", ".join(str(s.getsockname()) for s in tcp_server.sockets)
    logger.info(json.dumps({"event": "STARTED", "version": BANNER_VERSION,
                            "addrs": addrs}))
    try:
        async with tcp_server:
            await tcp_server.serve_forever()
    finally:
        expire_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await expire_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Pluginfer bootstrap seed.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9000, type=int)
    args = parser.parse_args()
    try:
        asyncio.run(run_server(host=args.host, port=args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
