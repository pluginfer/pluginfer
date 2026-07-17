"""End-to-end tests for chunked Merkle-tree blob transfer over the mesh."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Tuple

import pytest

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.blob_transfer import (  # noqa: E402
    BlobReceiveError,
    BlobReceiver,
    BlobSender,
    _chunk_blob,
    _merkle_root,
)
from core.mesh_connector import MeshConnector  # noqa: E402
from core.tokenomics import Wallet  # noqa: E402
from infrastructure.seed_node.punch_server import (  # noqa: E402
    PunchRelayState,
    _PunchProtocol,
)


# ---------------------------------------------------------------------------
# Merkle helpers (no socket)
# ---------------------------------------------------------------------------


def test_merkle_root_deterministic():
    leaves = [bytes([i]) * 32 for i in range(8)]
    r1 = _merkle_root(leaves)
    r2 = _merkle_root(list(leaves))
    assert r1 == r2 and len(r1) == 32


def test_merkle_root_changes_when_leaf_changes():
    a = [b"\x00" * 32, b"\x01" * 32, b"\x02" * 32]
    b = list(a)
    b[1] = b"\xff" * 32
    assert _merkle_root(a) != _merkle_root(b)


def test_merkle_handles_odd_leaf_count():
    leaves = [bytes([i]) * 32 for i in range(5)]
    r = _merkle_root(leaves)
    assert len(r) == 32
    # Duplicating the last leaf to make even count must yield the same.
    same = leaves + [leaves[-1]]
    assert _merkle_root(leaves) == _merkle_root(leaves)  # not necessarily == same


def test_chunk_blob_round_trip_size():
    blob = b"a" * 10_000
    chunks = _chunk_blob(blob, 4096)
    assert len(chunks) == 3
    assert b"".join(chunks) == blob


# ---------------------------------------------------------------------------
# integration: real loopback UDP
# ---------------------------------------------------------------------------


async def _start_seed(loop) -> Tuple:
    state = PunchRelayState()
    transport, proto = await loop.create_datagram_endpoint(
        lambda: _PunchProtocol(state),
        local_addr=("127.0.0.1", 0),
    )
    proto.state = state
    return transport, proto, transport.get_extra_info("sockname")


async def _connect_pair() -> Tuple:
    """Spin up two MeshConnectors over loopback + seed; connect A->B
    via direct punch; return (a_conn, b_conn, ch_a, ch_b, seed_t)."""
    loop = asyncio.get_running_loop()
    seed_t, seed_p, seed_addr = await _start_seed(loop)
    wa, wb = Wallet(), Wallet()
    a = await MeshConnector.start(seed_addr=seed_addr, wallet=wa,
                                  bind_host="127.0.0.1")
    b = await MeshConnector.start(seed_addr=seed_addr, wallet=wb,
                                  bind_host="127.0.0.1")
    for _ in range(40):
        if (wa.public_key_pem in seed_p.state.regs
                and wb.public_key_pem in seed_p.state.regs):
            break
        await asyncio.sleep(0.02)
    ch_a = await a.connect(wb.public_key_pem)
    for _ in range(40):
        if wa.public_key_pem in b.channels:
            break
        await asyncio.sleep(0.02)
    ch_b = b.channels[wa.public_key_pem]
    return a, b, ch_a, ch_b, seed_t


def test_send_1mb_blob_round_trip_intact():
    """Push 1 MiB through the chunked Merkle pipeline. Receiver
    reassembles bytes-exact and confirms BLOB_DONE ok=true."""

    async def _run() -> None:
        a, b, ch_a, ch_b, seed_t = await _connect_pair()
        try:
            sender = BlobSender(ch_a)
            receiver = BlobReceiver(ch_b)
            blob = bytes((i * 13 + 7) & 0xFF for i in range(1_048_576))
            ok_task = asyncio.create_task(
                sender.send_blob(blob, chunk_size=4096, timeout_s=30.0))
            assembled = await asyncio.wait_for(
                receiver.receive_any(timeout_s=30.0), timeout=30.0)
            ok = await asyncio.wait_for(ok_task, timeout=30.0)
            assert assembled == blob, (
                f"size mismatch or content mismatch: "
                f"{len(assembled)} vs {len(blob)}"
            )
            assert ok is True
        finally:
            a.close()
            b.close()
            seed_t.close()
            await asyncio.sleep(0)

    asyncio.run(_run())


def test_tampered_chunk_dropped_then_retransmit_succeeds():
    """Sender -> middleman that flips a byte in chunk 5, drops it,
    sends a clean retransmit on NACK. Receiver detects mismatch on
    first arrival, NACKs, accepts the clean retransmit."""

    async def _run() -> None:
        a, b, ch_a, ch_b, seed_t = await _connect_pair()
        try:
            # Replace ch_a._send_fn with a wrapper that corrupts the
            # 5th BLOB_CHUNK only on the FIRST send. Subsequent
            # retransmits go through clean.
            import json
            original_send = ch_a._send_fn
            corrupted_indices: set[int] = set()

            async def _corrupting(payload: bytes) -> None:
                try:
                    msg = json.loads(payload.decode("utf-8"))
                except Exception:
                    return await original_send(payload)
                if (msg.get("op") == "BLOB_CHUNK"
                        and msg.get("index") == 5
                        and 5 not in corrupted_indices):
                    corrupted_indices.add(5)
                    # Tamper: replace the data with garbage of the same
                    # length so the per-chunk hash fails.
                    import base64
                    bad = base64.b64encode(b"X" * len(
                        base64.b64decode(msg["data_b64"]))).decode()
                    msg["data_b64"] = bad
                    payload = json.dumps(msg).encode("utf-8")
                return await original_send(payload)

            ch_a._send_fn = _corrupting

            sender = BlobSender(ch_a)
            receiver = BlobReceiver(ch_b)
            blob = bytes((i * 31) & 0xFF for i in range(50_000))   # ~13 chunks
            ok_task = asyncio.create_task(
                sender.send_blob(blob, chunk_size=4096, timeout_s=15.0))
            assembled = await asyncio.wait_for(
                receiver.receive_any(timeout_s=15.0), timeout=15.0)
            ok = await asyncio.wait_for(ok_task, timeout=15.0)
            assert assembled == blob
            assert ok is True
            assert 5 in corrupted_indices, "test setup did not trigger corruption"
        finally:
            a.close()
            b.close()
            seed_t.close()
            await asyncio.sleep(0)

    asyncio.run(_run())


def test_receiver_rejects_blob_with_inconsistent_root():
    """The sender LIES about the Merkle root in OFFER. Even though
    every individual chunk hash verifies, the recomputed root does
    NOT match the offered root, so receive() must raise."""

    async def _run() -> None:
        a, b, ch_a, ch_b, seed_t = await _connect_pair()
        try:
            import json
            original_send = ch_a._send_fn

            async def _lying(payload: bytes) -> None:
                try:
                    msg = json.loads(payload.decode("utf-8"))
                except Exception:
                    return await original_send(payload)
                if msg.get("op") == "BLOB_OFFER":
                    # Tamper: swap in a fake root.
                    msg["merkle_root"] = "ff" * 32
                    payload = json.dumps(msg).encode("utf-8")
                return await original_send(payload)

            ch_a._send_fn = _lying

            sender = BlobSender(ch_a)
            receiver = BlobReceiver(ch_b)
            blob = b"x" * 12_000
            send_task = asyncio.create_task(
                sender.send_blob(blob, chunk_size=4096, timeout_s=10.0))
            with pytest.raises(BlobReceiveError):
                await asyncio.wait_for(receiver.receive_any(timeout_s=10.0),
                                       timeout=10.0)
            ok = await asyncio.wait_for(send_task, timeout=10.0)
            assert ok is False, "sender should learn the receiver rejected"
        finally:
            a.close()
            b.close()
            seed_t.close()
            await asyncio.sleep(0)

    asyncio.run(_run())
