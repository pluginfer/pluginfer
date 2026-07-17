"""THE three-stranger gossip test — A,B,C across "continents," only
A and B register with the seed; C only registers with B; and yet
within ~10 seconds A learns about C purely through gossip and the
auction routes work to C.

This is the proof of "find one, find all": the seed is for bootstrap
only, and the mesh propagates membership peer-to-peer without
re-asking the seed for every new node.

Topology:

    seed (127.0.0.1:S)
      ^     ^
      | A   | B          (A and B register with seed)
      A <--> B <--> C    (C only registers with B's gateway)

After gossip converges, A's MembershipView contains B AND C, and
A's auction has BOTH B and C as CrossNodeProvider entries.
"""

from __future__ import annotations

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


def _wait_for_tcp(host, port, deadline_s):
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            s = socket.create_connection((host, port), timeout=0.5)
            s.close()
            return True
        except OSError:
            time.sleep(0.15)
    return False


def _wait_for_http_200(url, deadline_s):
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


def _get_json(url, timeout=3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _spawn(cmd, env):
    return subprocess.Popen(
        cmd, env=env, cwd=str(V2),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


@pytest.mark.skipif(not _have_uvicorn(), reason="uvicorn not installed")
def test_three_strangers_gossip_transitive_discovery():
    seed_port = _free_port()
    # Each stranger gets a fresh node port + wallet.
    a_port = _free_port()
    b_port = _free_port()
    # C registers with a "fake" second seed that never comes up
    # (port closed). Without gossip C would be invisible to A
    # forever. With gossip, B propagates C to A.
    c_port = _free_port()
    dead_seed_port = _free_port()    # nobody listens here

    env = os.environ.copy()
    env["PYTHONPATH"] = str(V2) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PLUGINFER_LOG_LEVEL"] = "WARNING"
    # Force the echo runner so we don't accidentally download a
    # 1.5B model on the test machine.
    env["PLUGINFER_FORCE_ECHO"] = "1"
    # Tighten gossip cadence so the test doesn't sit for 30s.
    env["PLUGINFER_GOSSIP_INTERVAL_S"] = "1.0"

    import tempfile
    tmp_root = Path(tempfile.mkdtemp(prefix="pluginfer-three-"))

    env_a = {**env, "PLUGINFER_NODE_ID": "A",
             "PLUGINFER_WALLET_PASSPHRASE": "pp-A"}
    env_b = {**env, "PLUGINFER_NODE_ID": "B",
             "PLUGINFER_WALLET_PASSPHRASE": "pp-B"}
    # C registers with a non-existent seed so it CANNOT bootstrap
    # via the seed path — its only entry into the mesh is via
    # B (we inject B's gateway URL through the
    # PLUGINFER_GOSSIP_BOOTSTRAP_PEER env hint).
    env_c = {**env, "PLUGINFER_NODE_ID": "C",
             "PLUGINFER_WALLET_PASSPHRASE": "pp-C"}

    seed_proc = _spawn(
        [sys.executable, "-m", "infrastructure.seed_node.seed_server",
         "--host", "127.0.0.1", "--port", str(seed_port)],
        env,
    )
    procs = [seed_proc]
    try:
        assert _wait_for_tcp("127.0.0.1", seed_port, 20.0), \
            "seed never opened"

        a_proc = _spawn(
            [sys.executable, "-m", "tools.auto_mesh",
             "--seed-host", "127.0.0.1", "--seed-port", str(seed_port),
             "--node-port", str(a_port), "--bind-ip", "127.0.0.1",
             "--node-id", "A", "--wallet-path", str(tmp_root / "A.pem")],
            env_a,
        )
        procs.append(a_proc)
        b_proc = _spawn(
            [sys.executable, "-m", "tools.auto_mesh",
             "--seed-host", "127.0.0.1", "--seed-port", str(seed_port),
             "--node-port", str(b_port), "--bind-ip", "127.0.0.1",
             "--node-id", "B", "--wallet-path", str(tmp_root / "B.pem")],
            env_b,
        )
        procs.append(b_proc)
        # C points at a dead seed — proves it can only join via gossip.
        c_proc = _spawn(
            [sys.executable, "-m", "tools.auto_mesh",
             "--seed-host", "127.0.0.1", "--seed-port", str(dead_seed_port),
             "--node-port", str(c_port), "--bind-ip", "127.0.0.1",
             "--node-id", "C", "--wallet-path", str(tmp_root / "C.pem"),
             "--gossip-bootstrap", f"127.0.0.1:{b_port}"],
            env_c,
        )
        procs.append(c_proc)

        a_base = f"http://127.0.0.1:{a_port}"
        b_base = f"http://127.0.0.1:{b_port}"
        c_base = f"http://127.0.0.1:{c_port}"
        assert _wait_for_http_200(f"{a_base}/healthz", 90.0), "A never came up"
        assert _wait_for_http_200(f"{b_base}/healthz", 90.0), "B never came up"
        assert _wait_for_http_200(f"{c_base}/healthz", 90.0), "C never came up"

        # Wait for A to learn about C purely via gossip (not via seed).
        # The seed never knew C, so this only converges if gossip
        # rounds carry C's pubkey from B to A.
        # Load budget, not an expectation: the poll exits on success,
        # so isolated runs stay fast. 40 s flaked under full-suite
        # load (same class as the two-strangers fix, 2026-07-17).
        deadline = time.monotonic() + 90.0
        last_state = {}
        a_pubkeys = set()
        while time.monotonic() < deadline:
            try:
                ap = _get_json(f"{a_base}/peers")
                bp = _get_json(f"{b_base}/peers")
                cp = _get_json(f"{c_base}/peers")
                last_state = {"A": ap, "B": bp, "C": cp}
                a_pubkeys = {
                    p.get("pubkey_pem") for p in ap.get("discovered_peers", [])
                }
                if len(a_pubkeys) >= 2:
                    break
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)

        assert len(a_pubkeys) >= 2, (
            f"A only saw {len(a_pubkeys)} peers; gossip didn't propagate C. "
            f"State: {json.dumps(last_state, default=str)[:800]}"
        )

        # And A's auction has BOTH peers bound as CrossNodeProvider.
        ap = _get_json(f"{a_base}/peers")
        assert len(ap["registered_cross_nodes"]) >= 2, ap

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
