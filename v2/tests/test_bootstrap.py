"""CP-2 Task 2.2: end-to-end bootstrap test.

Spins up a real seed_server on a free port, monkey-patches
BOOTSTRAP_SEEDS in `core.complete_mesh_controller` to point at it,
and asserts that `_bootstrap_from_seeds()` registers self and
returns a peer list. Persistence to peers.json is also exercised.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import threading
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest  # noqa: E402

import core.complete_mesh_controller as cmc  # noqa: E402
from core.tokenomics import Wallet  # noqa: E402
from infrastructure.seed_node.seed_client import (  # noqa: E402
    SeedAddress,
    register_async,
)
from infrastructure.seed_node.seed_server import run_server  # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _SeedHarness:
    """Run an asyncio seed_server on a background thread for a test."""

    def __init__(self) -> None:
        self.port = _free_port()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._task = None

    def start(self) -> None:
        ready = threading.Event()

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._task = self._loop.create_task(
                run_server(host="127.0.0.1", port=self.port)
            )
            ready.set()
            try:
                self._loop.run_forever()
            finally:
                self._loop.close()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        ready.wait(timeout=2)
        # Wait for socket to actually bind
        for _ in range(40):
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("seed harness failed to bind")

    def stop(self) -> None:
        if self._loop and self._task:
            self._loop.call_soon_threadsafe(self._task.cancel)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2)


@pytest.fixture()
def seed():
    h = _SeedHarness()
    h.start()
    try:
        yield h
    finally:
        h.stop()


def test_bootstrap_from_seeds_registers_and_returns_peers(
    seed: _SeedHarness, tmp_path: Path, monkeypatch
) -> None:
    # Pre-populate the seed with a peer so we have something to fetch.
    other = Wallet()

    async def _register_other():
        await register_async(
            SeedAddress(host="127.0.0.1", port=seed.port),
            pubkey_pem=other.public_key_pem,
            sign_fn=other.sign,
            ip="10.0.0.42",
            port=8200,
            node_version="1.0.0",
        )

    asyncio.run(_register_other())

    # Point BOOTSTRAP_SEEDS at the harness
    monkeypatch.setattr(cmc, "BOOTSTRAP_SEEDS",
                        [{"host": "127.0.0.1", "port": seed.port,
                          "region": "test"}])

    # Run inside tmp_path so peers.json doesn't pollute repo cwd
    monkeypatch.chdir(tmp_path)

    # Build a minimal stand-in for the controller that has just enough to
    # call _bootstrap_from_seeds without spinning up the full mesh.
    class _Stub:
        port = 8100
        wallet = Wallet()
        # Re-bind methods from CompleteMeshController class.
        _bootstrap_from_seeds = cmc.CompleteMeshController._bootstrap_from_seeds
        _persist_peers = cmc.CompleteMeshController._persist_peers

    stub = _Stub()
    discovered = stub._bootstrap_from_seeds()

    # Should have learned about the OTHER peer (not our own pubkey).
    assert any(p["ip"] == "10.0.0.42" for p in discovered)
    assert all(
        p.get("pubkey_pem") != stub.wallet.public_key_pem for p in discovered
    )
    # peers.json was persisted in tmp_path
    assert (tmp_path / "peers.json").exists()


def test_bootstrap_with_empty_seed_list_returns_empty(
    seed: _SeedHarness, tmp_path: Path, monkeypatch
) -> None:
    """If BOOTSTRAP_SEEDS is empty, the helper returns [] without raising."""
    monkeypatch.setattr(cmc, "BOOTSTRAP_SEEDS", [])
    monkeypatch.chdir(tmp_path)

    class _Stub:
        port = 8100
        wallet = Wallet()
        _bootstrap_from_seeds = cmc.CompleteMeshController._bootstrap_from_seeds
        _persist_peers = cmc.CompleteMeshController._persist_peers

    out = _Stub()._bootstrap_from_seeds()
    assert out == []


def test_bootstrap_handles_unreachable_seed_gracefully(
    tmp_path: Path, monkeypatch
) -> None:
    """A seed at a closed port should be skipped, not raise."""
    closed = _free_port()  # nothing listening on this one
    monkeypatch.setattr(cmc, "BOOTSTRAP_SEEDS",
                        [{"host": "127.0.0.1", "port": closed,
                          "region": "test-unreachable"}])
    monkeypatch.chdir(tmp_path)

    class _Stub:
        port = 8100
        wallet = Wallet()
        _bootstrap_from_seeds = cmc.CompleteMeshController._bootstrap_from_seeds
        _persist_peers = cmc.CompleteMeshController._persist_peers

    out = _Stub()._bootstrap_from_seeds()
    assert out == []
