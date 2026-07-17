"""Request/response RPC over the punched/TURN UDP path (§HG6).

`core.peer_connect` gives the mesh a raw byte pipe to a peer behind
symmetric NAT. Jobs need more than a pipe — a request that finds its
response. This module frames JSON messages over the pipe with:

  * **rid correlation** (uuid4) — concurrent calls never cross,
  * **fragmentation/reassembly** — a UDP datagram across a real WAN
    can't safely exceed ~1.2 kB (path MTU); chat bodies and receipted
    responses often do,
  * **a sync bridge** (`call_sync`) — CrossNodeProvider.execute runs
    in an executor thread, while the punched socket lives on the
    node's asyncio loop.

Envelope (one per datagram). Deliberately carries NO "op" and NO
"status" key, so peer_connect's dispatcher classifies it as an
application payload and never confuses it with punch/TURN protocol
traffic:

    {"pfr": 1, "rid": "<uuid>", "k": "q"|"r", "seq": i, "tot": n,
     "data": "<b64 chunk>"}

Reassembled request:  {"rid": ..., "from": <pubkey_pem>, "body": {...}}
Reassembled response: {"rid": ..., "resp": 1, "status": int,
                       "headers": {...}, "body": {...}}

The serving side answers by pushing response fragments back through
`send_to_peer(from_pubkey)`: for punched requests the return path was
learned from the inbound datagram's source address
(`note_punched_addr`); for TURN requests peer_connect remembered the
relay session on delivery. The responder NEVER needs its own
connect_to_peer — being reachable is free.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Raw bytes per fragment. 900 raw → ~1.2 kB envelope after b64 + JSON,
# under every sane WAN path MTU (IPv6 minimum 1280).
_CHUNK_RAW = 900
_REASSEMBLY_TTL_S = 120.0


class PunchRPC:
    """One instance per node, wrapping its PeerConnectClient.

    ``handler`` is the serving side: ``async (body: dict) ->
    (status: int, headers: dict, body: dict)``. auto_mesh points it at
    the node's own local gateway so punched jobs run the exact same
    auction/receipt pipeline as HTTP jobs.
    """

    def __init__(
        self,
        client: Any,
        handler: Callable[[dict], Any],
        *,
        my_pubkey_pem: str,
    ) -> None:
        self.client = client
        self.handler = handler
        self.my_pubkey_pem = my_pubkey_pem
        self._loop = asyncio.get_running_loop()
        self._pending: Dict[str, asyncio.Future] = {}
        # (rid, kind) -> {"chunks": {seq: bytes}, "tot": n, "ts": t}
        self._buffers: Dict[Tuple[str, str], dict] = {}
        client.set_inbound_handler(self._on_datagram)

    # ------------------------------------------------------------------
    # framing
    # ------------------------------------------------------------------

    def _fragments(self, rid: str, kind: str, message: dict):
        raw = json.dumps(message).encode("utf-8")
        chunks = [raw[i:i + _CHUNK_RAW]
                  for i in range(0, len(raw), _CHUNK_RAW)] or [b""]
        tot = len(chunks)
        for i, c in enumerate(chunks):
            yield json.dumps({
                "pfr": 1, "rid": rid, "k": kind, "seq": i, "tot": tot,
                "data": base64.b64encode(c).decode("ascii"),
            }).encode("utf-8")

    def _on_datagram(self, data: bytes, addr: Tuple[str, Any]) -> None:
        try:
            env = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(env, dict) or env.get("pfr") != 1:
            return
        try:
            rid = str(env["rid"])
            kind = str(env["k"])
            seq = int(env["seq"])
            tot = int(env["tot"])
            chunk = base64.b64decode(str(env["data"]))
        except (KeyError, TypeError, ValueError):
            return
        now = time.time()
        # GC stale partial reassemblies so a lossy path can't leak RAM.
        for key in [k for k, b in self._buffers.items()
                    if now - b["ts"] > _REASSEMBLY_TTL_S]:
            self._buffers.pop(key, None)
        buf = self._buffers.setdefault(
            (rid, kind), {"chunks": {}, "tot": tot, "ts": now})
        buf["chunks"][seq] = chunk
        buf["ts"] = now
        if len(buf["chunks"]) < buf["tot"]:
            return
        self._buffers.pop((rid, kind), None)
        try:
            raw = b"".join(buf["chunks"][i] for i in range(buf["tot"]))
            message = json.loads(raw.decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(message, dict):
            return

        if kind == "r":
            fut = self._pending.pop(rid, None)
            if fut is not None and not fut.done():
                fut.set_result(message)
            return

        # kind == "q": serve it.
        from_pub = str(message.get("from") or "")
        if not from_pub:
            return
        # Punched requests teach us the live return path; TURN requests
        # already registered their session inside peer_connect.
        if (isinstance(addr, tuple) and len(addr) == 2
                and addr[0] != "turn"):
            try:
                self.client.note_punched_addr(from_pub, addr)
            except Exception:
                pass
        self._loop.create_task(
            self._serve(rid, from_pub, message.get("body") or {}))

    # ------------------------------------------------------------------
    # serving side
    # ------------------------------------------------------------------

    async def _serve(self, rid: str, from_pub: str, body: dict) -> None:
        try:
            status, headers, resp_body = await self.handler(body)
        except Exception as e:
            logger.warning("punch_rpc handler failed: %s", e)
            status, headers, resp_body = 500, {}, {
                "error": f"handler failed: {type(e).__name__}"}
        message = {
            "rid": rid, "resp": 1, "status": int(status),
            "headers": dict(headers or {}), "body": resp_body,
        }
        for frag in self._fragments(rid, "r", message):
            ok = await self.client.send_to_peer(from_pub, frag)
            if not ok:
                logger.warning(
                    "punch_rpc: no return path to %.24s… — response "
                    "dropped (requester will time out and retry).",
                    from_pub,
                )
                return

    # ------------------------------------------------------------------
    # calling side
    # ------------------------------------------------------------------

    async def call(
        self, target_pubkey_pem: str, body: dict,
        *, timeout_s: float = 60.0,
    ) -> Tuple[int, dict, Optional[dict]]:
        """Send `body` to the peer's handler; return (status, headers,
        body). Establishes the NAT-traversal path first when none is
        open (punch, then TURN — peer_connect's ladder)."""
        if not self.client.has_path(target_pubkey_pem):
            res = await self.client.connect_to_peer(target_pubkey_pem)
            if not res.success:
                raise RuntimeError(
                    f"no NAT-traversal path to peer: {res.detail}")
        rid = uuid.uuid4().hex
        fut: asyncio.Future = self._loop.create_future()
        self._pending[rid] = fut
        try:
            message = {"rid": rid, "from": self.my_pubkey_pem,
                       "body": body}
            for frag in self._fragments(rid, "q", message):
                if not await self.client.send_to_peer(
                        target_pubkey_pem, frag):
                    raise RuntimeError("punched path lost mid-send")
            reply = await asyncio.wait_for(fut, timeout=timeout_s)
        finally:
            self._pending.pop(rid, None)
        return (
            int(reply.get("status", 500)),
            dict(reply.get("headers") or {}),
            reply.get("body"),
        )

    def call_sync(
        self, target_pubkey_pem: str, body: dict,
        *, timeout_s: float = 60.0,
    ) -> Tuple[int, dict, Optional[dict]]:
        """Thread-safe bridge for executor-thread callers
        (CrossNodeProvider.execute). Never call from the event loop
        thread — it would deadlock waiting on itself."""
        cfut = asyncio.run_coroutine_threadsafe(
            self.call(target_pubkey_pem, body, timeout_s=timeout_s),
            self._loop,
        )
        return cfut.result(timeout=timeout_s + 10.0)
