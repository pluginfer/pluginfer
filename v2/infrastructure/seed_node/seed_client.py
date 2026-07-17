"""Pluginfer seed-node client.

Used by `core.complete_mesh_controller._bootstrap_from_seeds()` to:
  - Register THIS node with one or more public seed servers
  - Pull a peer list to start gossip / DHT bootstrap

The wire format matches `seed_server.py`. All registrations are signed
by the local wallet (ECDSA SECP256K1) and rejected by the seed if the
timestamp is more than 30 seconds stale (replay protection).
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("seed_client")

# Match the server constants.
PEER_TTL_SECONDS: int = 600
DEFAULT_REGISTER_TIMEOUT_S: float = 5.0


@dataclass
class SeedAddress:
    host: str
    port: int = 9000


def _signed_bytes(
    pubkey_pem: str, ip: str, port: int,
    node_version: str, timestamp: float,
) -> str:
    return f"{pubkey_pem}|{ip}|{int(port)}|{node_version}|{timestamp}"


async def _send_one(
    seed: SeedAddress, msg: dict, timeout_s: float,
) -> Optional[dict]:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(seed.host, seed.port),
            timeout=timeout_s,
        )
    except (asyncio.TimeoutError, OSError) as e:
        logger.warning("seed %s:%s unreachable: %s", seed.host, seed.port, e)
        return None
    try:
        writer.write(json.dumps(msg).encode("utf-8") + b"\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
        if not line:
            return None
        try:
            return json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def register_async(
    seed: SeedAddress,
    *,
    pubkey_pem: str,
    sign_fn,
    ip: str,
    port: int,
    node_version: str,
    timeout_s: float = DEFAULT_REGISTER_TIMEOUT_S,
) -> Optional[dict]:
    """Register THIS node with `seed`. Returns the server response dict.

    `sign_fn(message: str) -> str` is the wallet sign function (returns
    base64 ECDSA signature).
    """
    timestamp = time.time()
    signed = _signed_bytes(pubkey_pem, ip, port, node_version, timestamp)
    msg = {
        "op": "REGISTER",
        "pubkey_pem": pubkey_pem,
        "ip": ip,
        "port": port,
        "node_version": node_version,
        "timestamp": timestamp,
        "signature": sign_fn(signed),
    }
    return await _send_one(seed, msg, timeout_s)


async def fetch_peers_async(
    seed: SeedAddress, max_n: int = 50,
    timeout_s: float = DEFAULT_REGISTER_TIMEOUT_S,
) -> list[dict]:
    """Pull a sample of live peers from `seed`. Returns [] on failure."""
    msg = {"op": "PEERS", "max": int(max_n)}
    resp = await _send_one(seed, msg, timeout_s)
    if not resp or resp.get("status") != "ok":
        return []
    return list(resp.get("peers", []))


async def ping_async(
    seed: SeedAddress, timeout_s: float = DEFAULT_REGISTER_TIMEOUT_S,
) -> Optional[dict]:
    return await _send_one(seed, {"op": "PING"}, timeout_s)


# ---------------------------------------------------------------------------
# sync wrappers (used by complete_mesh_controller, which is non-async)
# ---------------------------------------------------------------------------

def _run(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        # Allow re-entrancy from within an existing loop (Jupyter, etc.).
        if "already running" not in str(e):
            raise
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def register_sync(seed: SeedAddress, **kw) -> Optional[dict]:
    return _run(register_async(seed, **kw))


def fetch_peers_sync(seed: SeedAddress, **kw) -> list[dict]:
    return _run(fetch_peers_async(seed, **kw))


def ping_sync(seed: SeedAddress, **kw) -> Optional[dict]:
    return _run(ping_async(seed, **kw))


# Local IP discovery utility (replaces the previous # 8.8.8.8 phone-home,
# uses RFC 5737 documentation prefix; ifconfig route-table gives the
# correct outgoing local IP without DNS leak). Re-exported here for the
# seed-client caller's convenience.
def discover_local_ip() -> str:
    """Best-effort local IP. Falls back to 127.0.0.1 if no route exists."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # RFC 5737 doc range; no DNS query, no traffic actually sent.
            s.connect(("198.51.100.1", 65535))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return "127.0.0.1"
