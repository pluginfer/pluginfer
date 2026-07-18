"""Chunked Merkle-tree blob transfer over the mesh.

Pluginfer routinely needs to ship large blobs between peers (model
checkpoints, dataset shards, batched inference results > 1 MB). The
single-datagram MESH_DATA envelope in `mesh_connector.py` caps at
~8 KiB; anything bigger needs chunking, integrity, and resumability.

This module provides the layer:

  * **Sender** -- splits a blob into fixed-size chunks (default 4 KiB),
    builds a binary Merkle tree over chunk hashes, ships an OFFER
    envelope (blob_id + total_chunks + merkle_root + total_size) then
    streams chunks. Each chunk carries its index + per-chunk hash, so
    the receiver can verify in isolation as bytes arrive.
  * **Receiver** -- on OFFER, allocates an N-chunk buffer, verifies
    each incoming chunk's hash, slots it into the buffer. After
    `recv_timeout_s`, requests missing chunks by index via a NACK
    envelope. On full assembly, recomputes the Merkle root and
    confirms it matches the OFFER. ANY mismatch -> the assembled
    blob is REJECTED, the future resolves with `BlobReceiveError`.

Why a real Merkle tree (not just a single sha256 over the assembled
blob): the receiver wants to reject a tampered chunk *immediately* so
the sender stops wasting bandwidth, AND a future protocol upgrade can
serve partial blobs (subscribe to chunk indices [a, b]) under the same
integrity story.

Wire format (JSON-line over MeshConnector bytes):

  Sender -> Receiver OFFER:
    {"op":"BLOB_OFFER","blob_id":"<hex>","chunk_size":4096,
     "total_chunks":N,"total_bytes":B,"merkle_root":"<hex>"}

  Sender -> Receiver CHUNK:
    {"op":"BLOB_CHUNK","blob_id":"<hex>","index":I,
     "hash_hex":"<sha256(chunk)>","data_b64":"..."}

  Receiver -> Sender NACK (request retransmit of missing indices):
    {"op":"BLOB_NACK","blob_id":"<hex>","missing":[I, I, ...]}

  Receiver -> Sender DONE:
    {"op":"BLOB_DONE","blob_id":"<hex>","ok":true}
    or {"op":"BLOB_DONE","blob_id":"<hex>","ok":false,"reason":"..."}
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from .mesh_connector import MeshChannel

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE: int = 4096
NACK_INTERVAL_S: float = 0.5
RECV_TIMEOUT_S: float = 30.0


class BlobReceiveError(RuntimeError):
    """Raised when a blob cannot be assembled (Merkle mismatch,
    timeout, or sender abort)."""


# ---------------------------------------------------------------------------
# Merkle tree (binary, sha256, duplicated last leaf for odd counts)
# ---------------------------------------------------------------------------


def _merkle_root(leaf_hashes: List[bytes]) -> bytes:
    """Bitcoin-style binary Merkle tree: pair adjacent hashes and
    sha256(left || right). For odd levels, duplicate the last hash.
    `leaf_hashes` must be the per-chunk sha256 digests in order."""
    if not leaf_hashes:
        return b"\x00" * 32
    layer = list(leaf_hashes)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        next_layer = []
        for i in range(0, len(layer), 2):
            h = hashlib.sha256(layer[i] + layer[i + 1]).digest()
            next_layer.append(h)
        layer = next_layer
    return layer[0]


def _chunk_blob(blob: bytes, chunk_size: int) -> List[bytes]:
    return [blob[i:i + chunk_size] for i in range(0, max(1, len(blob)), chunk_size)] \
        if blob else [b""]


# ---------------------------------------------------------------------------
# Sender
# ---------------------------------------------------------------------------


@dataclass
class _PendingSend:
    blob_id: str
    chunks: List[bytes]
    chunk_hashes: List[bytes]
    merkle_root: bytes
    chunk_size: int
    sent_count: int = 0
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    ok: bool = False
    detail: Optional[str] = None


class BlobSender:
    """Per-channel sender. Maintains in-flight blob state so the
    receiver's NACKs can drive retransmission."""

    def __init__(self, channel: MeshChannel) -> None:
        self._bg_tasks: set = set()
        self.channel = channel
        # blob_id -> _PendingSend
        self._inflight: Dict[str, _PendingSend] = {}
        # Wrap the channel's on_message so we see NACKs / DONEs.
        self._wrap_channel()

    def _wrap_channel(self) -> None:
        prior = self.channel.on_message

        def _on(payload: bytes) -> None:
            if prior is not None:
                try:
                    prior(payload)
                except Exception as e:                          # pragma: no cover
                    logger.warning("prior on_message raised: %s", e)
            try:
                msg = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return
            if not isinstance(msg, dict):
                return
            op = msg.get("op")
            if op == "BLOB_NACK":
                blob_id = str(msg.get("blob_id", ""))
                missing = msg.get("missing") or []
                rec = self._inflight.get(blob_id)
                if rec is None or not isinstance(missing, list):
                    return
                # Strong ref — a GC'd task would silently drop the
                # retransmit and stall the transfer.
                t = asyncio.create_task(
                    self._retransmit(rec, [int(i) for i in missing]))
                self._bg_tasks.add(t)
                t.add_done_callback(self._bg_tasks.discard)
            elif op == "BLOB_DONE":
                blob_id = str(msg.get("blob_id", ""))
                rec = self._inflight.get(blob_id)
                if rec is None:
                    return
                rec.ok = bool(msg.get("ok", False))
                rec.detail = msg.get("reason")
                rec.done_event.set()

        self.channel.on_message = _on

    async def send_blob(
        self,
        blob: bytes,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        timeout_s: float = 60.0,
    ) -> bool:
        """Ship `blob` to the channel's peer with Merkle integrity.
        Returns True iff the receiver confirmed BLOB_DONE ok=true."""
        blob_id = secrets.token_hex(8)
        chunks = _chunk_blob(blob, chunk_size)
        chunk_hashes = [hashlib.sha256(c).digest() for c in chunks]
        root = _merkle_root(chunk_hashes)
        rec = _PendingSend(
            blob_id=blob_id, chunks=chunks, chunk_hashes=chunk_hashes,
            merkle_root=root, chunk_size=chunk_size,
        )
        self._inflight[blob_id] = rec

        # OFFER then stream all chunks.
        await self.channel.send(json.dumps({
            "op": "BLOB_OFFER",
            "blob_id": blob_id,
            "chunk_size": chunk_size,
            "total_chunks": len(chunks),
            "total_bytes": len(blob),
            "merkle_root": root.hex(),
        }).encode("utf-8"))
        for i, chunk in enumerate(chunks):
            await self._send_chunk(rec, i)
            # tiny yield so the receiver gets a chance to process
            await asyncio.sleep(0)

        try:
            await asyncio.wait_for(rec.done_event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            self._inflight.pop(blob_id, None)
            return False
        finally:
            # Keep the record around briefly in case of late NACK arrivals,
            # but small TTL is fine; we drop on next send anyway.
            pass
        ok = rec.ok
        self._inflight.pop(blob_id, None)
        return ok

    async def _send_chunk(self, rec: _PendingSend, index: int) -> None:
        chunk = rec.chunks[index]
        await self.channel.send(json.dumps({
            "op": "BLOB_CHUNK",
            "blob_id": rec.blob_id,
            "index": index,
            "hash_hex": rec.chunk_hashes[index].hex(),
            "data_b64": base64.b64encode(chunk).decode("ascii"),
        }).encode("utf-8"))
        rec.sent_count += 1

    async def _retransmit(self, rec: _PendingSend, indices: List[int]) -> None:
        for i in indices:
            if 0 <= i < len(rec.chunks):
                await self._send_chunk(rec, i)


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------


@dataclass
class _PendingRecv:
    blob_id: str
    chunk_size: int
    total_chunks: int
    total_bytes: int
    merkle_root: bytes
    chunks: List[Optional[bytes]] = field(default_factory=list)
    chunk_hashes: List[Optional[bytes]] = field(default_factory=list)
    received_count: int = 0
    last_chunk_at: float = 0.0
    future: asyncio.Future = field(default_factory=lambda: None)  # type: ignore[arg-type]


class BlobReceiver:
    """Per-channel receiver. Reassembles chunked blobs, verifies the
    Merkle root, surfaces the assembled bytes via `await receive(...)`."""

    def __init__(self, channel: MeshChannel) -> None:
        self.channel = channel
        # blob_id -> _PendingRecv
        self._inflight: Dict[str, _PendingRecv] = {}
        # blob_id -> Future (resolved with assembled bytes or exception)
        self._futures: Dict[str, asyncio.Future] = {}
        self._wrap_channel()
        self._nack_task: Optional[asyncio.Task] = None

    def _wrap_channel(self) -> None:
        prior = self.channel.on_message

        def _on(payload: bytes) -> None:
            if prior is not None:
                try:
                    prior(payload)
                except Exception as e:                          # pragma: no cover
                    logger.warning("prior on_message raised: %s", e)
            try:
                msg = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return
            if not isinstance(msg, dict):
                return
            op = msg.get("op")
            if op == "BLOB_OFFER":
                self._handle_offer(msg)
            elif op == "BLOB_CHUNK":
                self._handle_chunk(msg)

        self.channel.on_message = _on

    # ------------------------------------------------------------------

    def _handle_offer(self, msg: dict) -> None:
        try:
            blob_id = str(msg["blob_id"])
            chunk_size = int(msg["chunk_size"])
            total_chunks = int(msg["total_chunks"])
            total_bytes = int(msg["total_bytes"])
            merkle_root = bytes.fromhex(str(msg["merkle_root"]))
        except (KeyError, ValueError, TypeError):
            return
        if blob_id in self._inflight:
            return  # duplicate offer
        rec = _PendingRecv(
            blob_id=blob_id, chunk_size=chunk_size,
            total_chunks=total_chunks, total_bytes=total_bytes,
            merkle_root=merkle_root,
            chunks=[None] * total_chunks,
            chunk_hashes=[None] * total_chunks,
            last_chunk_at=time.monotonic(),
        )
        rec.future = self._futures.get(blob_id) or asyncio.get_event_loop().create_future()
        self._inflight[blob_id] = rec
        self._futures[blob_id] = rec.future
        if self._nack_task is None or self._nack_task.done():
            self._nack_task = asyncio.create_task(self._nack_loop())

    def _handle_chunk(self, msg: dict) -> None:
        try:
            blob_id = str(msg["blob_id"])
            index = int(msg["index"])
            declared_hash = bytes.fromhex(str(msg["hash_hex"]))
            data = base64.b64decode(str(msg["data_b64"]))
        except (KeyError, ValueError, TypeError):
            return
        rec = self._inflight.get(blob_id)
        if rec is None:
            return
        if not (0 <= index < rec.total_chunks):
            return
        # Verify the per-chunk hash BEFORE storing. A tampered chunk
        # is dropped silently; the NACK loop will re-request it.
        actual = hashlib.sha256(data).digest()
        if actual != declared_hash:
            logger.warning("blob %s chunk %d hash mismatch; dropped",
                           blob_id, index)
            return
        if rec.chunks[index] is None:
            rec.chunks[index] = data
            rec.chunk_hashes[index] = actual
            rec.received_count += 1
            rec.last_chunk_at = time.monotonic()

        # Done?
        if rec.received_count == rec.total_chunks:
            self._finalize(rec)

    def _finalize(self, rec: _PendingRecv) -> None:
        # Recompute the Merkle root over received hashes.
        hashes: List[bytes] = []
        for h in rec.chunk_hashes:
            if h is None:
                # Shouldn't happen if received_count == total_chunks,
                # but guard anyway.
                if not rec.future.done():
                    rec.future.set_exception(BlobReceiveError(
                        "internal: assembled with missing chunks"
                    ))
                self._inflight.pop(rec.blob_id, None)
                return
            hashes.append(h)
        root = _merkle_root(hashes)
        if root != rec.merkle_root:
            asyncio.create_task(self._send_done(rec.blob_id, ok=False,
                                                reason="merkle_mismatch"))
            if not rec.future.done():
                rec.future.set_exception(BlobReceiveError(
                    f"Merkle root mismatch (got {root.hex()}, "
                    f"expected {rec.merkle_root.hex()})"
                ))
            self._inflight.pop(rec.blob_id, None)
            return
        # Assemble.
        assembled = b"".join(rec.chunks)  # type: ignore[arg-type]
        if len(assembled) != rec.total_bytes:
            # Trim if last chunk had padding (shouldn't happen with
            # our chunker, but be defensive).
            assembled = assembled[:rec.total_bytes]
        asyncio.create_task(self._send_done(rec.blob_id, ok=True))
        if not rec.future.done():
            rec.future.set_result(assembled)
        self._inflight.pop(rec.blob_id, None)

    async def _send_done(self, blob_id: str, *, ok: bool,
                         reason: Optional[str] = None) -> None:
        body: dict = {"op": "BLOB_DONE", "blob_id": blob_id, "ok": ok}
        if reason is not None:
            body["reason"] = reason
        try:
            await self.channel.send(json.dumps(body).encode("utf-8"))
        except Exception as e:                                  # pragma: no cover
            logger.warning("send DONE failed: %s", e)

    async def _nack_loop(self) -> None:
        """Periodically scan for in-flight blobs missing chunks and
        send NACKs."""
        try:
            while self._inflight:
                await asyncio.sleep(NACK_INTERVAL_S)
                now = time.monotonic()
                for blob_id, rec in list(self._inflight.items()):
                    if rec.received_count >= rec.total_chunks:
                        continue
                    if now - rec.last_chunk_at > RECV_TIMEOUT_S:
                        if not rec.future.done():
                            rec.future.set_exception(BlobReceiveError(
                                f"recv timeout (got {rec.received_count}"
                                f"/{rec.total_chunks} chunks)"
                            ))
                        self._inflight.pop(blob_id, None)
                        continue
                    missing = [i for i, c in enumerate(rec.chunks) if c is None]
                    if missing:
                        try:
                            await self.channel.send(json.dumps({
                                "op": "BLOB_NACK",
                                "blob_id": blob_id,
                                "missing": missing,
                            }).encode("utf-8"))
                        except Exception as e:                  # pragma: no cover
                            logger.warning("send NACK failed: %s", e)
        except asyncio.CancelledError:
            return

    # ------------------------------------------------------------------

    async def receive(self, blob_id: str,
                      timeout_s: float = 60.0) -> bytes:
        """Wait for a specific blob to arrive in full. If the offer
        hasn't landed yet, the future is created here and resolved
        when the offer + chunks complete."""
        fut = self._futures.get(blob_id)
        if fut is None:
            fut = asyncio.get_event_loop().create_future()
            self._futures[blob_id] = fut
        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        finally:
            self._futures.pop(blob_id, None)

    async def receive_any(self, timeout_s: float = 60.0) -> bytes:
        """Wait for the next assembled blob on this channel,
        whatever blob_id it has."""
        deadline = time.monotonic() + timeout_s
        while True:
            for blob_id, fut in list(self._futures.items()):
                if fut.done():
                    self._futures.pop(blob_id, None)
                    return fut.result()
            for blob_id in list(self._inflight.keys()):
                fut = self._futures.get(blob_id)
                if fut is None:
                    continue
                if fut.done():
                    self._futures.pop(blob_id, None)
                    return fut.result()
            if time.monotonic() >= deadline:
                raise BlobReceiveError("receive_any timeout")
            await asyncio.sleep(0.02)
