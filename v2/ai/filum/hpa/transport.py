"""Grain transport over UDP.

The §C bundle is protocol-pure but transport-naive — `Grain.to_bytes`
is fine, but nothing actually ships those bytes between machines.
This module is the thinnest possible transport: UDP datagrams with
multi-fragment reassembly, content-addressed dedup, and a bounded
in-flight retry buffer.

UDP is chosen deliberately:

* Connectionless — we do not want one TCP connection per peer in a
  10k-node mesh.
* Datagram-shaped — a Grain serialises to ~200 bytes - 64 KiB; UDP
  fragments ride that range cleanly with a small header.
* Loss-tolerant — the staleness-decay merge in §C5 NBGGA absorbs
  packet loss as just another form of delay; we don't need TCP's
  reliability guarantee.
* Cheap on consumer hardware — no kernel TCP buffer per peer.

What we add on top:

* 16-byte fragment header (magic + grain_id_prefix + frag_idx + frag_count
  + payload_len) so larger grains split cleanly across MTU.
* A reassembly cache keyed by ``(sender_pubkey_prefix, grain_id_prefix)``
  with a 30-second TTL.
* Content-addressed dedup: a 4096-slot ring of recently-seen grain_ids
  drops re-receives from gossip storms in O(1).
* Retry buffer: outbound grains are kept for ``retry_window_s``; if
  no ACK arrives we re-send up to ``max_retries`` times. ACKs are
  also tiny UDP datagrams.

NAT traversal is out of scope here — that's a hole-punching layer
above (production: STUN + symmetric-NAT relay via a Sun). For now,
the assumption is direct reachability or a simple relay through a
publicly-reachable Sun.
"""

from __future__ import annotations

import logging
import os
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# Wire format constants ------------------------------------------------------

MAGIC = b"PLGN"                     # 4 bytes — Pluginfer Grain
HDR_FMT = ">4s8sHHI"                # magic, gid_prefix(8), frag_idx, frag_count, payload_len
HDR_SIZE = struct.calcsize(HDR_FMT)  # 20 bytes
MAX_PAYLOAD = 1200                  # safe under typical 1500-byte MTU minus IP+UDP+hdr
ACK_MAGIC = b"PLGA"                 # 4 bytes — Pluginfer Grain ACK
ACK_FMT = ">4s8s"                   # magic, gid_prefix(8)
ACK_SIZE = struct.calcsize(ACK_FMT)


@dataclass
class TransportConfig:
    bind_host: str = "0.0.0.0"
    bind_port: int = 0                  # 0 = pick an ephemeral
    reasm_ttl_s: float = 30.0
    seen_ring_size: int = 4096
    retry_window_s: float = 5.0
    max_retries: int = 3
    recv_buf_bytes: int = 1 << 20       # 1 MiB SO_RCVBUF
    send_buf_bytes: int = 1 << 20


@dataclass
class TransportStats:
    packets_sent: int = 0
    packets_received: int = 0
    grains_assembled: int = 0
    grains_duplicate: int = 0
    grains_dropped_invalid: int = 0
    acks_sent: int = 0
    acks_received: int = 0
    retries_sent: int = 0


class GrainTransport:
    """UDP transport for grains. Thread-safe; one socket per instance.

    Public API::

        transport = GrainTransport(on_grain=lambda g, addr: ...)
        transport.start()
        transport.send_grain(grain, peer=("10.0.0.7", 5301))
        ...
        transport.stop()
    """

    def __init__(
        self,
        on_grain: Callable[[bytes, tuple], None],
        config: TransportConfig = TransportConfig(),
    ):
        self.cfg = config
        self.on_grain = on_grain
        self.stats = TransportStats()
        self._sock: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._rx_thread: Optional[threading.Thread] = None
        self._retry_thread: Optional[threading.Thread] = None
        # Reassembly cache: key=(sender, gid_prefix) -> (frags_dict, started_ts, total_count)
        self._reasm: dict[tuple, list] = {}
        self._reasm_lock = threading.Lock()
        # Dedup ring: recently-seen gid_prefix bytes.
        self._seen: deque = deque(maxlen=self.cfg.seen_ring_size)
        self._seen_set: set[bytes] = set()
        self._seen_lock = threading.Lock()
        # Outbound retry buffer: gid_prefix -> (raw_packets, peer, ts, attempts)
        self._outflight: dict[bytes, tuple] = {}
        self._outflight_lock = threading.Lock()

    # --- lifecycle --------------------------------------------------------

    def start(self) -> "GrainTransport":
        if self._sock is not None:
            return self
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                                   self.cfg.recv_buf_bytes)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF,
                                   self.cfg.send_buf_bytes)
        except OSError:
            pass
        self._sock.bind((self.cfg.bind_host, self.cfg.bind_port))
        self._stop.clear()
        self._rx_thread = threading.Thread(
            target=self._recv_loop, name="grain-rx", daemon=True,
        )
        self._retry_thread = threading.Thread(
            target=self._retry_loop, name="grain-retry", daemon=True,
        )
        self._rx_thread.start()
        self._retry_thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._sock is not None:
                # Wake up the recv loop.
                try:
                    self._sock.sendto(b"", self._sock.getsockname())
                except OSError:
                    pass
                self._sock.close()
        except Exception:
            pass
        for t in (self._rx_thread, self._retry_thread):
            if t is not None:
                t.join(timeout=2.0)
        self._sock = None

    @property
    def address(self) -> tuple[str, int]:
        if self._sock is None:
            return (self.cfg.bind_host, self.cfg.bind_port)
        return self._sock.getsockname()[:2]

    # --- send -------------------------------------------------------------

    def send_grain(self, grain_bytes: bytes, peer: tuple[str, int],
                   gid_prefix: Optional[bytes] = None,
                   reliable: bool = True) -> None:
        """Send a grain to one peer. Splits across fragments as needed."""
        if self._sock is None:
            raise RuntimeError("transport not started")
        gid = gid_prefix or _gid_prefix_of(grain_bytes)
        # Split into fragments.
        n = max(1, (len(grain_bytes) + MAX_PAYLOAD - 1) // MAX_PAYLOAD)
        packets: list[bytes] = []
        for i in range(n):
            chunk = grain_bytes[i * MAX_PAYLOAD:(i + 1) * MAX_PAYLOAD]
            hdr = struct.pack(HDR_FMT, MAGIC, gid, i, n, len(chunk))
            packets.append(hdr + chunk)
        for p in packets:
            try:
                self._sock.sendto(p, peer)
                self.stats.packets_sent += 1
            except OSError as e:
                logger.debug("sendto failed: %s", e)
        if reliable:
            with self._outflight_lock:
                self._outflight[gid] = (packets, peer, time.monotonic(), 0)

    # --- receive loop ----------------------------------------------------

    def _recv_loop(self) -> None:
        assert self._sock is not None
        sock = self._sock
        sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(MAX_PAYLOAD + HDR_SIZE + 64)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                continue
            self.stats.packets_received += 1
            # ACK?
            if len(data) >= ACK_SIZE and data[:4] == ACK_MAGIC:
                try:
                    _, gid = struct.unpack(ACK_FMT, data[:ACK_SIZE])
                    with self._outflight_lock:
                        self._outflight.pop(gid, None)
                    self.stats.acks_received += 1
                except struct.error:
                    pass
                continue
            # Grain fragment?
            if len(data) < HDR_SIZE or data[:4] != MAGIC:
                self.stats.grains_dropped_invalid += 1
                continue
            try:
                magic, gid, frag_idx, frag_count, payload_len = struct.unpack(
                    HDR_FMT, data[:HDR_SIZE],
                )
            except struct.error:
                self.stats.grains_dropped_invalid += 1
                continue
            if magic != MAGIC:
                self.stats.grains_dropped_invalid += 1
                continue
            payload = data[HDR_SIZE:HDR_SIZE + payload_len]
            if len(payload) != payload_len:
                self.stats.grains_dropped_invalid += 1
                continue
            self._handle_fragment(gid, frag_idx, frag_count, payload, addr)

    def _handle_fragment(self, gid: bytes, idx: int, count: int,
                         payload: bytes, addr: tuple) -> None:
        # Quick dedup at the *grain* level (for fully-assembled grains).
        with self._seen_lock:
            if gid in self._seen_set:
                self.stats.grains_duplicate += 1
                # Re-ack so the sender stops retrying.
                self._send_ack(gid, addr)
                return
        # Single-fragment fast path.
        if count == 1:
            self._on_complete(gid, payload, addr)
            return
        # Multi-fragment: store in reassembly buffer.
        key = (addr[0], gid)
        with self._reasm_lock:
            entry = self._reasm.get(key)
            if entry is None:
                entry = ([None] * count, time.monotonic())
                self._reasm[key] = entry
            frags, _ts = entry
            if 0 <= idx < count and frags[idx] is None:
                frags[idx] = payload
            if all(f is not None for f in frags):
                full = b"".join(frags)
                del self._reasm[key]
                self._on_complete(gid, full, addr)

    def _on_complete(self, gid: bytes, blob: bytes, addr: tuple) -> None:
        with self._seen_lock:
            if gid in self._seen_set:
                self.stats.grains_duplicate += 1
                self._send_ack(gid, addr)
                return
            self._seen.append(gid)
            self._seen_set.add(gid)
            # Trim set to ring size.
            if len(self._seen_set) > self.cfg.seen_ring_size:
                old = self._seen.popleft()
                self._seen_set.discard(old)
        self.stats.grains_assembled += 1
        self._send_ack(gid, addr)
        try:
            self.on_grain(blob, addr)
        except Exception as e:
            logger.exception("on_grain handler raised: %s", e)

    def _send_ack(self, gid: bytes, addr: tuple) -> None:
        if self._sock is None:
            return
        try:
            self._sock.sendto(struct.pack(ACK_FMT, ACK_MAGIC, gid), addr)
            self.stats.acks_sent += 1
        except OSError:
            pass

    # --- retry loop ------------------------------------------------------

    def _retry_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(0.5)
            if self._sock is None:
                continue
            now = time.monotonic()
            to_resend: list = []
            to_drop: list = []
            with self._outflight_lock:
                for gid, (packets, peer, ts, attempts) in list(self._outflight.items()):
                    if now - ts < self.cfg.retry_window_s:
                        continue
                    if attempts >= self.cfg.max_retries:
                        to_drop.append(gid)
                        continue
                    to_resend.append((gid, packets, peer, attempts + 1))
                for gid in to_drop:
                    self._outflight.pop(gid, None)
                for gid, packets, peer, attempts in to_resend:
                    self._outflight[gid] = (packets, peer, now, attempts)
            for _gid, packets, peer, _att in to_resend:
                for p in packets:
                    try:
                        self._sock.sendto(p, peer)
                        self.stats.retries_sent += 1
                    except OSError:
                        pass
            # Reassembly buffer GC.
            cutoff = now - self.cfg.reasm_ttl_s
            with self._reasm_lock:
                for k, (_frags, ts) in list(self._reasm.items()):
                    if ts < cutoff:
                        del self._reasm[k]


def _gid_prefix_of(grain_bytes: bytes) -> bytes:
    """Compute the 8-byte gid prefix used in headers + ACKs.

    sha256(grain_bytes)[:8]. Cheap and content-addressed.
    """
    import hashlib
    return hashlib.sha256(grain_bytes).digest()[:8]
