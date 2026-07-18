"""pluginfer up — the zero-config onboarding command.

One command. No flags. Sixty seconds to a live node:

    python pluginfer.py up          (repo checkout)
    python -m tools.up              (equivalent)

What it does, in order, with honest output at every step:

  1. Detects hardware (GPU via nvidia-smi, else CPU).
  2. Detects a local model runtime (Ollama first). If found, binds the
     first installed model so the node serves REAL inference out of
     the box — no env vars. If not found, says so plainly and serves
     the honest echo until the user installs one.
  3. Loads or creates the encrypted wallet (W31 path, synthesized
     per-machine passphrase fallback — no prompt, no plaintext key).
  4. Resolves a mesh seed: PLUGINFER_SEED_HOST env wins; otherwise
     every record in data/seed_registry.json is probed over TCP and
     the first reachable seed wins; otherwise the node runs in SOLO
     mode — the local OpenAI-compatible gateway still works, receipts
     still sign, and the same command joins a mesh the moment a seed
     is reachable.
  5. Boots the auto_mesh node (gateway + auction + gossip + provider
     loops) and prints the three lines a new user actually needs:
     the gateway URL, the OPENAI_BASE_URL export, and a curl they
     can paste.

ASCII-only output by project rule (Windows cp1252 consoles).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

OLLAMA_URL = os.environ.get("PLUGINFER_OLLAMA_URL", "http://127.0.0.1:11434")
PREFERRED_PORT = int(os.environ.get("PLUGINFER_NODE_PORT", "8100"))
SEED_REGISTRY_PATH = V2 / "data" / "seed_registry.json"
PROBE_TIMEOUT_S = 2.0


def _say(msg: str) -> None:
    print(msg, flush=True)


def _step(n: int, total: int, msg: str) -> None:
    _say(f"[{n}/{total}] {msg}")


# ---------------------------------------------------------------------------
# 1. Hardware
# ---------------------------------------------------------------------------

def detect_gpu() -> str:
    """Cheap GPU detection. nvidia-smi if present; never imports torch
    (a 10s import is not acceptable inside an onboarding command)."""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            line = (out.stdout or "").strip().splitlines()
            if out.returncode == 0 and line:
                return line[0].strip()
        except Exception:
            pass
    return "CPU only (no NVIDIA GPU detected)"


# ---------------------------------------------------------------------------
# 2. Runtime (Ollama probe)
# ---------------------------------------------------------------------------

def detect_ollama_models() -> List[str]:
    """Names of locally installed Ollama models, or [] if Ollama is
    not running. Pure stdlib, 2s budget, never raises."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def pick_model(models: List[str]) -> str:
    """Smallest-first heuristic: onboarding should feel fast, and a
    3B answers in seconds where a 14B may take a minute on CPU."""
    def size_key(name: str) -> float:
        import re
        m = re.search(r"(\d+(?:\.\d+)?)b", name.lower())
        return float(m.group(1)) if m else 1e9
    return sorted(models, key=size_key)[0]


# ---------------------------------------------------------------------------
# 4. Seed resolution
# ---------------------------------------------------------------------------

def _tcp_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=PROBE_TIMEOUT_S):
            return True
    except Exception:
        return False


def resolve_seed() -> Tuple[Optional[str], int, str]:
    """Returns (seed_host, seed_port, how). how is one of
    'env', 'registry:<id>', 'solo'."""
    env_host = os.environ.get("PLUGINFER_SEED_HOST", "").strip()
    if env_host and env_host != "127.0.0.1":
        port = int(os.environ.get("PLUGINFER_SEED_PORT", "9000"))
        return env_host, port, "env"
    try:
        records = json.loads(SEED_REGISTRY_PATH.read_text("utf-8")).get("records", [])
    except Exception:
        records = []
    for rec in records:
        host, port = rec.get("host", ""), int(rec.get("port", 9000))
        # Placeholder fingerprints (pre-launch registry) are probed
        # anyway: DNS that does not resolve fails in <2s and the probe
        # is the single source of truth.
        if host and _tcp_reachable(host, port):
            return host, port, f"registry:{rec.get('id', host)}"
    return None, 9000, "solo"


# ---------------------------------------------------------------------------
# Port pick
# ---------------------------------------------------------------------------

def pick_port(preferred: int) -> int:
    for candidate in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", candidate))
                return s.getsockname()[1]
        except OSError:
            continue
    raise RuntimeError("no free TCP port available")


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        prog="pluginfer up",
        description="Zero-config: start a Pluginfer node and gateway.",
    )
    ap.add_argument("--seed-host", default="", help="Override mesh seed host.")
    ap.add_argument("--seed-port", type=int, default=0, help="Override mesh seed port.")
    ap.add_argument("--port", type=int, default=0,
                    help=f"Gateway port (default: {PREFERRED_PORT}, else first free).")
    ap.add_argument("--wallet-path",
                    default=str(Path.home() / ".pluginfer" / "auto_mesh_wallet.pem"))
    ap.add_argument("--share", action="store_true",
                    help="Make this node reachable by the whole mesh via a "
                         "free auto-tunnel, so others can send it jobs. "
                         "Without this, the node is local-only.")
    ap.add_argument("--swarm-key", default=None,
                    help="Form/join a PRIVATE mesh: only nodes and clients "
                         "presenting this shared key can join or send jobs. "
                         "Set the same key on every node (e.g. each of your "
                         "datacenters). Omit for the public mesh.")
    args = ap.parse_args(argv)
    # PLUGINFER_SHARE=1 is the env-equivalent (for installers / one-click).
    share = args.share or os.environ.get("PLUGINFER_SHARE", "0") == "1"
    if args.swarm_key:
        os.environ["PLUGINFER_SWARM_KEY"] = args.swarm_key
    if os.environ.get("PLUGINFER_SWARM_KEY", "").strip():
        _say("")
        _say("  PRIVATE SWARM: this node only talks to nodes/clients that "
             "present your swarm key.")

    # Windows consoles (and redirected stdout) default to cp1252, which
    # hard-crashes on the first non-ASCII char ANY imported module ever
    # prints — a failure class this project has hit repeatedly. Forcing
    # utf-8 with replacement closes the whole class for this command.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Keep the console clean for first-run UX; power users get detail
    # via PLUGINFER_LOG_LEVEL=INFO/DEBUG.
    logging.basicConfig(
        level=os.environ.get("PLUGINFER_LOG_LEVEL", "WARNING"),
        format="[pluginfer] %(levelname)s %(message)s",
    )

    total = 5
    _say("")
    _say("  Pluginfer -- decentralized AI compute, one command.")
    _say("")

    _step(1, total, f"Hardware: {detect_gpu()}")

    models = detect_ollama_models()
    if models:
        chosen = os.environ.get("PLUGINFER_ALPHA_MODEL_ID") or pick_model(models)
        os.environ.setdefault("PLUGINFER_ALPHA_MODEL_ID", chosen)
        _step(2, total, f"Runtime: Ollama detected -- serving '{chosen}' "
                        f"({len(models)} model(s) installed)")
    else:
        chosen = None
        _step(2, total,
              "Runtime: no local model runtime found. The node will serve an "
              "honest echo (clearly tagged, never billed as real inference).")
        _say("        To serve real models: install Ollama from "
             "https://ollama.com then run 'ollama pull qwen2.5:1.5b' "
             "and re-run this command.")

    wallet_path = Path(args.wallet_path).expanduser()
    existed = wallet_path.exists()
    _step(3, total, f"Wallet: {'loaded' if existed else 'creating'} "
                    f"encrypted wallet at {wallet_path}")

    if args.seed_host:
        seed_host: Optional[str] = args.seed_host
        seed_port = args.seed_port or 9000
        how = "flag"
    else:
        seed_host, seed_port, how = resolve_seed()
    if seed_host:
        _step(4, total, f"Mesh: joining via seed {seed_host}:{seed_port} ({how})")
    else:
        _step(4, total,
              "Mesh: no public seed reachable -- running SOLO. Your local "
              "gateway works fully; join a mesh anytime with --seed-host.")

    port = args.port or pick_port(PREFERRED_PORT)

    # --share: auto-tunnel so the whole mesh can reach this node with no
    # router config. The node advertises the public https host (port 443
    # -> https, handled by the mesh) instead of an unroutable LAN ip.
    tunnel_proc = None
    public_host = None
    if share:
        _say("")
        _say("  Sharing your node with the mesh (opening a public tunnel)...")
        from tools.tunnel import start_quick_tunnel
        public_host, tunnel_proc = start_quick_tunnel(port, _say)
        if public_host:
            os.environ["PLUGINFER_PUBLIC_IP"] = public_host
            os.environ["PLUGINFER_PUBLIC_PORT"] = "443"
            _say(f"  You are LIVE on the internet: https://{public_host}")
            _say("  Anyone on the mesh can now send jobs to your node.")

    _step(5, total, f"Starting node on port {port} ...")

    base = f"http://127.0.0.1:{port}"
    _say("")
    _say("  " + "=" * 62)
    _say("  YOUR NODE IS STARTING")
    _say("")
    _say(f"  OpenAI-compatible gateway:  {base}/v1")
    _say(f"  Point any OpenAI client:    OPENAI_BASE_URL={base}/v1")
    _say(f"  Peers / mesh status:        {base}/peers")
    _say("")
    _say("  Try it:")
    model_arg = chosen or "echo"
    _say(f'    curl -s {base}/v1/chat/completions -H "Content-Type: application/json" \\')
    _say(f'      -d "{{\\"model\\": \\"{model_arg}\\", \\"messages\\": '
         f'[{{\\"role\\": \\"user\\", \\"content\\": \\"hello\\"}}]}}"')
    _say("")
    _say(f"  Control panel (opens in your browser):  {base}/")
    _say("  Every response carries a signed receipt (X-Pluginfer-Receipt-ID).")
    _say("  Ctrl+C stops the node. Wallet + identity persist across runs.")
    _say("  " + "=" * 62)
    _say("")

    # Open the control panel like a normal app would. A background timer
    # gives the server a moment to bind first. Opt out with
    # PLUGINFER_NO_BROWSER=1 (headless servers, CI).
    if os.environ.get("PLUGINFER_NO_BROWSER", "0") != "1":
        import threading
        import webbrowser

        def _open():
            import time as _t
            _t.sleep(3.5)
            try:
                webbrowser.open(f"{base}/")
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    from tools import auto_mesh
    run_args = argparse.Namespace(
        seed_host=seed_host or "127.0.0.1",
        seed_port=seed_port,
        node_port=port,
        bind_ip="",
        node_id=os.environ.get("PLUGINFER_NODE_ID", ""),
        wallet_path=str(wallet_path),
        gossip_bootstrap=[],
    )

    # Supervision: the node NEVER stays down. Any crash restarts the
    # whole node loop with exponential backoff (2s -> 60s cap); a run
    # that stays healthy for 10 minutes resets the backoff. Only
    # Ctrl+C exits.
    import time as _time

    def _stop_tunnel():
        if tunnel_proc is not None:
            try:
                tunnel_proc.terminate()
            except Exception:
                pass

    backoff_s = 2.0
    try:
        while True:
            started = _time.monotonic()
            try:
                asyncio.run(auto_mesh._run(run_args))
                _say("  Node loop exited cleanly; restarting in 2s "
                     "(Ctrl+C to stop).")
                backoff_s = 2.0
            except KeyboardInterrupt:
                _say("\n  Node stopped. Run 'pluginfer up' anytime to come "
                     "back online.")
                return
            except Exception as e:
                healthy_s = _time.monotonic() - started
                if healthy_s > 600:
                    backoff_s = 2.0
                _say(f"  Node crashed after {healthy_s:.0f}s "
                     f"({type(e).__name__}: {e}) -- restarting in "
                     f"{backoff_s:.0f}s. It never stays down.")
            # If the tunnel died but the node lives, the public URL is
            # stale — honestly tell the operator rather than pretend.
            if tunnel_proc is not None and tunnel_proc.poll() is not None:
                _say("  Note: the public tunnel dropped; you are local-only "
                     "until you re-run with --share.")
            try:
                _time.sleep(backoff_s)
            except KeyboardInterrupt:
                _say("\n  Node stopped.")
                return
            backoff_s = min(backoff_s * 2, 60.0)
    finally:
        _stop_tunnel()


if __name__ == "__main__":
    main()
