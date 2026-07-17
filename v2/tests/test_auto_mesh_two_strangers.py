"""THE product test — two strangers, zero config, auto-mesh forms.

Three OS processes, no pre-shared state beyond the seed URL:

  * **Seed** — `infrastructure.seed_node.seed_server.run_server`
    (TCP REGISTER/PEERS protocol with ECDSA signature verify).
  * **Stranger A** — `tools.auto_mesh` on `:<port-a>`.
  * **Stranger B** — `tools.auto_mesh` on `:<port-b>`.

After ~5 seconds:

  1. The seed has both A and B in its peer table.
  2. A's `/peers` endpoint lists B (and vice versa).
  3. A's auction has registered B as a `_CrossNodeProvider`; B's
     auction has registered A.
  4. Submitting `POST /v1/chat/completions` to A's gateway routes
     through the auction (A's local flagship OR B's cross-node
     wrapper, whichever wins the Pareto score). A receipt-signed
     header comes back.

This is the auto-mesh-on-install product test from TODO §1.3 — the
thing that ends the "but does anyone actually find anyone" objection.

The test soft-skips when uvicorn isn't installed (rare CI matrices).
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parents[1]


def _have_uvicorn() -> bool:
    try:
        import uvicorn  # noqa: F401
        return True
    except ImportError:
        return False


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# Deadlines are LOAD budgets, not expectations: every wait below is a
# poll loop that exits the moment the condition holds, so the isolated
# run stays ~29 s. Under a full-suite run (dozens of server tests, the
# whole tree at below-normal priority since host_guard) node boot can
# take several times longer — the old 30 s ceilings were the recorded
# flake (2026-07-17 audit).
TCP_DEADLINE_S = 20.0
BOOT_DEADLINE_S = 90.0
DISCOVERY_DEADLINE_S = 90.0
NODE_BOOT_ATTEMPTS = 3


def _tail_output(proc, limit: int = 2000) -> str:
    """Best-effort last output of a dead/dying child for assert
    messages. Terminates the child if it is still running."""
    try:
        if proc.poll() is None:
            proc.terminate()
        out = proc.communicate(timeout=5)[0]
        return out.decode("utf-8", "replace")[-limit:]
    except Exception:
        return "<no output captured>"


def _boot_node(*, node_id: str, wallet_path: Path, env: dict,
               seed_port: int, procs: list) -> "tuple[object, int]":
    """Boot one auto_mesh node; retry on a fresh port when the child
    lost a bind race. `_free_port()` is inherently check-then-use —
    under full-suite load another test can claim the port between the
    pick and uvicorn's bind, and the child exits immediately. A child
    that is still ALIVE but unhealthy is a genuine boot failure and is
    not retried."""
    last_out = ""
    for _ in range(NODE_BOOT_ATTEMPTS):
        port = _free_port()
        proc = _spawn(
            [sys.executable, "-m", "tools.auto_mesh",
             "--seed-host", "127.0.0.1", "--seed-port", str(seed_port),
             "--node-port", str(port), "--bind-ip", "127.0.0.1",
             "--node-id", node_id,
             "--wallet-path", str(wallet_path)],
            env, node_id,
        )
        procs.append(proc)
        if _wait_for_http_200(
                f"http://127.0.0.1:{port}/healthz", BOOT_DEADLINE_S):
            return proc, port
        alive = proc.poll() is None
        last_out = _tail_output(proc)
        procs.remove(proc)
        if alive:
            break   # unhealthy-but-running: retrying won't help
    raise AssertionError(
        f"stranger {node_id} never came up; last output:\n{last_out}"
    )


def _wait_for_tcp(host: str, port: int, deadline_s: float) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            s = socket.create_connection((host, port), timeout=0.5)
            s.close()
            return True
        except OSError:
            time.sleep(0.15)
    return False


def _wait_for_http_200(url: str, deadline_s: float) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.2)
    return False


def _http_get_json(url: str, timeout: float = 3.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _spawn(cmd, env, log_label):
    """Spawn a subprocess, return Popen. Output gets piped to capture
    for the assertion error message on failure."""
    return subprocess.Popen(
        cmd, env=env, cwd=str(V2),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


@pytest.mark.skipif(not _have_uvicorn(), reason="uvicorn not installed")
def test_two_strangers_auto_mesh_via_seed():
    seed_port = _free_port()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(V2) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PLUGINFER_LOG_LEVEL"] = "WARNING"
    # Hermetic: this test pins mesh FORMATION + routing, not inference.
    # Without the echo pin, nodes adopt whatever runtime the host has
    # (a real Ollama would serve real models — slow, machine-dependent).
    env["PLUGINFER_FORCE_ECHO"] = "1"
    # Same reason: a real mesh-llm node on the host must not join in.
    env["PLUGINFER_DISABLE_MESHLLM"] = "1"
    # Per-process passphrase keying so the two strangers can't
    # accidentally re-use the same wallet on disk.
    env_a = {**env, "PLUGINFER_NODE_ID": "stranger-A",
             "PLUGINFER_WALLET_PASSPHRASE": "test-A-passphrase"}
    env_b = {**env, "PLUGINFER_NODE_ID": "stranger-B",
             "PLUGINFER_WALLET_PASSPHRASE": "test-B-passphrase"}

    # Use a tmp dir for wallet files so prior CI state doesn't leak.
    import tempfile
    tmp_root = Path(tempfile.mkdtemp(prefix="pluginfer-automesh-"))

    procs: list = []
    try:
        # Seed boot with the same lost-bind-race retry as the nodes.
        for _ in range(NODE_BOOT_ATTEMPTS):
            seed_proc = _spawn(
                [sys.executable, "-m",
                 "infrastructure.seed_node.seed_server",
                 "--host", "127.0.0.1", "--port", str(seed_port)],
                env, "seed",
            )
            procs.append(seed_proc)
            if _wait_for_tcp("127.0.0.1", seed_port, TCP_DEADLINE_S):
                break
            alive = seed_proc.poll() is None
            seed_out = _tail_output(seed_proc)
            procs.remove(seed_proc)
            if alive:
                raise AssertionError(
                    f"seed alive but never listened:\n{seed_out}")
            seed_port = _free_port()
        else:
            raise AssertionError("seed never opened TCP listener")

        _, a_port = _boot_node(
            node_id="stranger-A", wallet_path=tmp_root / "A.pem",
            env=env_a, seed_port=seed_port, procs=procs,
        )
        _, b_port = _boot_node(
            node_id="stranger-B", wallet_path=tmp_root / "B.pem",
            env=env_b, seed_port=seed_port, procs=procs,
        )
        a_base = f"http://127.0.0.1:{a_port}"
        b_base = f"http://127.0.0.1:{b_port}"

        # Wait for cross-discovery (each side must see the other's
        # pubkey in its /peers).
        end = time.monotonic() + DISCOVERY_DEADLINE_S
        a_sees_b = False
        b_sees_a = False
        while time.monotonic() < end:
            try:
                ap = _http_get_json(f"{a_base}/peers")
                bp = _http_get_json(f"{b_base}/peers")
                a_sees_b = any(
                    p.get("pubkey_pem") for p in ap.get("discovered_peers", [])
                ) and len(ap.get("registered_cross_nodes", [])) >= 1
                b_sees_a = any(
                    p.get("pubkey_pem") for p in bp.get("discovered_peers", [])
                ) and len(bp.get("registered_cross_nodes", [])) >= 1
                if a_sees_b and b_sees_a:
                    break
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        assert a_sees_b, "stranger A never discovered stranger B"
        assert b_sees_a, "stranger B never discovered stranger A"

        # Both auctions should now have at least 2 providers (own
        # flagship + the cross-node wrapper for the other).
        ap = _http_get_json(f"{a_base}/peers")
        bp = _http_get_json(f"{b_base}/peers")
        assert ap["auction_size"] >= 2, f"A auction: {ap}"
        assert bp["auction_size"] >= 2, f"B auction: {bp}"

        # Submit a chat completion to A. The Pareto auction may pick
        # A's own flagship OR B via the CrossNodeProvider. Either is
        # a pass — both ends shared compute.
        req = urllib.request.Request(
            f"{a_base}/v1/chat/completions",
            data=json.dumps({
                "model": "pluginfer-alpha",
                "messages": [{"role": "user", "content": "auto-mesh hello"}],
                "max_tokens": 32,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30.0) as r:
            body = json.loads(r.read().decode("utf-8"))
            headers = dict(r.headers.items())

        # Receipt-signed flag (G7/G8 plumbing): A always emits it.
        rs = (headers.get("X-Pluginfer-Receipt-Signed")
              or headers.get("x-pluginfer-receipt-signed"))
        assert rs == "1", f"A receipt-signed flag={rs!r}, headers={headers}"
        # The response carries text — somebody served us.
        assert body["choices"][0]["message"]["content"]

    finally:
        for p in reversed(procs):
            p.terminate()
        for p in reversed(procs):
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)
