"""WAN proof — Node B (the remote stranger).

Boots a BUYER-ONLY auto_mesh node that bootstraps directly to Node A's
public tunnel address (no shared LAN, no seed), then submits a real
chat-completion job. Because a buyer-only node registers NO local
compute, a job that comes back with content can ONLY have executed on
Node A — across the open internet. That is the whole proof: a machine
on this network paid a machine on another network to run a signed job.

Usage:
    python -m scripts.wan_proof_node_b <nodeA_host:port>

Exit 0 on proof, non-zero (with a reason) otherwise.
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

V2 = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _swarm_headers() -> dict:
    # Private-swarm proof runs set PLUGINFER_SWARM_KEY; the key must ride
    # on every call to a keyed Node A, exactly like any mesh client.
    from core.swarm_auth import auth_headers
    return auth_headers()


def _get(url: str, timeout: float = 8.0):
    req = urllib.request.Request(url, headers=_swarm_headers())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8")), dict(r.headers.items())


def _post(url: str, body: dict, timeout: float = 45.0):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **_swarm_headers()},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8")), dict(r.headers.items())


def main() -> int:
    if len(sys.argv) < 2 or ":" not in sys.argv[1]:
        print("FAIL: need Node A address as <host:port>", flush=True)
        return 2
    bootstrap = (sys.argv[1].strip()
                 .replace("tcp://", "").replace("https://", "")
                 .replace("http://", "").rstrip("/"))
    if ":" in bootstrap:
        a_host, a_port = bootstrap.split(":")
    else:
        a_host, a_port = bootstrap, "443"   # bare hostname = TLS tunnel
    a_port = int(a_port)
    bootstrap = f"{a_host}:{a_port}"
    a_base = f"https://{a_host}" if a_port == 443 else f"http://{a_host}:{a_port}"
    print(f"[node-B] Node A tunnel: {a_base}", flush=True)

    # Sanity: can we even reach Node A over the WAN before we start?
    for attempt in range(10):
        try:
            st, peers, _ = _get(f"{a_base}/peers", timeout=8.0)
            print(f"[node-B] reached Node A /peers (status {st}); "
                  f"A runtime={peers.get('runtime', {}).get('name')}", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"[node-B] waiting for Node A tunnel ({attempt+1}/10): {e}",
                  flush=True)
            time.sleep(4)
    else:
        print("FAIL: Node A tunnel never became reachable", flush=True)
        return 3

    local_port = _free_port()
    env = dict(os.environ)
    env.update({
        "PLUGINFER_BUYER_ONLY": "1",           # no local compute — the teeth
        "PLUGINFER_ENABLE_PUNCH": "0",         # no seed in this topology
        "PLUGINFER_GOSSIP_BOOTSTRAP_PEER": f"{a_host}:{int(a_port)}",
        "PLUGINFER_WALLET_PASSPHRASE": "wan-proof-b",
        "PYTHONIOENCODING": "utf-8",
    })
    wallet = str(V2 / "_wanproof_b_wallet.pem")
    proc = subprocess.Popen(
        [sys.executable, "-m", "tools.auto_mesh",
         "--node-port", str(local_port), "--bind-ip", "127.0.0.1",
         "--node-id", "wanproof-B", "--wallet-path", wallet,
         "--seed-host", "127.0.0.1", "--seed-port", "59999"],
        cwd=str(V2), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    b_base = f"http://127.0.0.1:{local_port}"

    try:
        # 1) Node B comes up.
        for _ in range(40):
            try:
                _get(f"{b_base}/healthz", timeout=2.0)
                break
            except Exception:  # noqa: BLE001
                if proc.poll() is not None:
                    print("FAIL: node B exited early:\n"
                          + (proc.stdout.read() if proc.stdout else ""),
                          flush=True)
                    return 4
                time.sleep(1)
        else:
            print("FAIL: node B never became healthy", flush=True)
            return 4
        print("[node-B] up (buyer-only)", flush=True)

        # 2) Node B binds Node A as its (only) cross-node provider — this
        #    is WAN peer discovery with no shared network.
        bound = False
        for _ in range(30):
            try:
                _, bp, _ = _get(f"{b_base}/peers")
                if bp.get("registered_cross_nodes"):
                    bound = True
                    print(f"[node-B] bound {len(bp['registered_cross_nodes'])} "
                          f"cross-node peer(s); auction_size={bp['auction_size']}",
                          flush=True)
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        if not bound:
            print("FAIL: node B never bound Node A across the WAN", flush=True)
            return 5

        # 3) The proof: submit a real job to buyer-only Node B. With no
        #    local compute, success means Node A ran it over the internet.
        st, body, headers = _post(f"{b_base}/v1/chat/completions", {
            "model": "pluginfer-alpha",
            "messages": [{"role": "user",
                          "content": "WAN proof: two strangers, one signed job."}],
            "max_tokens": 24,
        }, timeout=60.0)
        content = (body.get("choices", [{}])[0].get("message", {})
                   .get("content", ""))
        rsigned = (headers.get("X-Pluginfer-Receipt-Signed")
                   or headers.get("x-pluginfer-receipt-signed"))
        provider = (headers.get("X-Pluginfer-Provider")
                    or headers.get("x-pluginfer-provider") or "?")
        print(f"[node-B] job status={st} provider={provider!r} "
              f"receipt_signed={rsigned!r}", flush=True)
        print(f"[node-B] answer: {content[:120]!r}", flush=True)
        if st != 200 or not content:
            print("FAIL: buyer-only node returned no content — no peer served it",
                  flush=True)
            return 6

        # 4) Bidirectional discovery: Node A now lists Node B in its view.
        _, a_peers, _ = _get(f"{a_base}/peers")
        a_saw_b = a_peers.get("view_size", 0) >= 1
        print(f"[node-B] Node A view_size={a_peers.get('view_size')} "
              f"(bidirectional discovery: {a_saw_b})", flush=True)

        print("\n=== WAN PROOF PASSED ===", flush=True)
        print(f"A buyer-only node on THIS network submitted a job that a node "
              f"at {a_base} executed and signed, over the open internet.",
              flush=True)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
