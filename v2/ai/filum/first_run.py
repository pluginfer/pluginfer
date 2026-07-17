"""§H3 First-Run Orchestrator — one entrypoint, zero config, GUI opens.

The promise: the user double-clicks ONE thing. Everything else is
the application's job. This module is what that one thing calls.

Sequence (each step is idempotent — re-running just re-validates):

  1. Locate the Pluginfer state directory (platform-appropriate).
  2. If runtime_config.json is missing, run §H1 auto_setup —
     detect HW, generate identity, pick tier, write config.
  3. Probe for an accelerator backend (CUDA / ROCm / MPS / XPU /
     CPU). Print one line.
  4. Probe for the model federation (Ollama / Filum / remote).
     If none are bound, print a one-line hint — but never block.
  5. Hand off to the GUI launcher.

The function never raises on benign conditions (Tk missing, mesh
unreachable, etc.) — it falls back to headless service mode and
prints a stderr line so the user can debug if they care, but the
default user need-do-nothing experience is preserved.

Entry point: ``python -m ai.filum.first_run``.

Design note: a method of bringing
a decentralised AI compute node from cold install to active mesh
participation through a single user gesture, in which (a) the
hardware accelerator is auto-detected across multiple vendor
backends, (b) cryptographic identity is generated and persisted,
(c) the runtime configuration tier is picked from telemetry,
(d) the user-facing GUI launches automatically when present and
the node falls back to a headless service when not, all without
the user issuing any command, editing any file, or supplying any
credential.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _hr(line: str = "") -> None:
    """Print a separator. Cheap visual structure for first-run output."""
    print(line if line else "  " + "-" * 68)


def _probe_backend() -> str:
    """Return a one-line human-readable backend description."""
    try:
        from .hpa.backend import detect_backend
        b = detect_backend()
        if b.name == "cpu":
            return "CPU only (no GPU detected — light tier)"
        return (f"{b.name.upper()} ({b.accelerator_name}, "
                f"{int(b.total_memory_bytes / (1 << 20))} MiB)")
    except Exception as e:
        return f"backend probe failed: {e}"


def _probe_federation() -> str:
    """Report what's bound to the §J1 federation (Filum local / Ollama)."""
    try:
        from .hpa.model_federation import ModelFederation
        fed = ModelFederation()
        avail = fed.list_available()
        if not avail:
            return ("no LLM bound — install Ollama "
                    "(https://ollama.com) for local AI, optional")
        names = []
        for entry in avail:
            mlist = entry.get("models", [])
            head = mlist[0] if mlist else "(no models)"
            names.append(f"{entry['backend']}={head}")
        return "federation: " + ", ".join(names)
    except Exception as e:
        return f"federation probe failed: {e}"


def _form_mesh(cfg) -> dict:
    """§H4 — discover peers (LAN mDNS + DNS seeds + history) and persist
    them to peers.json so the running node can dial them. This is the
    'mesh forms automatically' step. Returns a dict with:
        {peer_count, lan_active, public_ip, my_node_id}
    Never raises on benign failures (no zeroconf, no network, no DNS).
    """
    try:
        from .mesh_discovery import MeshDiscovery, save_peers, load_peers
    except Exception as e:
        print(f"  mesh_discovery unavailable ({e}) — skipping auto-form")
        return {"peer_count": 0, "lan_active": False, "public_ip": None}
    try:
        d = MeshDiscovery(
            my_node_id=cfg.identity.pubkey_hex,
            my_port=5300,
            state_dir=cfg.state_dir,
            seeds=cfg.seed_addresses,
        )
        result = d.find_peers(lan_timeout_s=2.0)
        d.close()
    except Exception as e:
        print(f"  mesh discovery error: {e}")
        return {"peer_count": 0, "lan_active": False, "public_ip": None}

    # Merge new discoveries into peers.json.
    existing = load_peers(cfg.state_dir)
    have = {(p.get("ip"), int(p.get("port", 5300))) for p in existing}
    for p in result.peers:
        key = (p.addr, p.port)
        if key not in have:
            existing.append({
                "ip": p.addr, "port": int(p.port),
                "node_id": p.node_id, "source": p.source,
                "added_ts": p.last_seen,
            })
    if existing:
        save_peers(cfg.state_dir, existing)

    return {
        "peer_count": len(result.peers),
        "lan_active": result.lan_active,
        "public_ip": result.public_ip,
        "my_node_id": cfg.identity.pubkey_hex,
    }


def _ensure_setup() -> "AutoConfig":  # noqa: F821 - forward ref
    """Run §H1 auto_setup if not already done. Idempotent."""
    from .auto_setup import auto_setup, default_state_dir, load_runtime_config
    state = default_state_dir()
    existing = load_runtime_config(state)
    if existing:
        # Already configured — re-run auto_setup() so any new fields land,
        # but the identity/keys are preserved (see get_or_create_identity).
        return auto_setup(state_dir=state)
    print("  First-run setup — detecting hardware, generating identity...")
    return auto_setup(state_dir=state)


def _launch_gui_or_service() -> int:
    """Open the GUI if Tk is available; otherwise start headless."""
    # Try GUI first (the user-friendly path).
    try:
        import tkinter  # noqa: F401
        from .gui_launcher import main as gui_main
    except Exception as e:
        print(f"  GUI unavailable ({e}); starting headless service mode...")
        try:
            from .service_mode import main as svc_main
            return svc_main()
        except Exception as e2:
            print(f"  Headless service also failed: {e2}", file=sys.stderr)
            return 1
    return gui_main()


def main() -> int:
    print()
    _hr("=" * 72)
    print("                          PLUGINFER")
    print("                    First-run orchestrator")
    _hr("=" * 72)

    # 1+2. Setup (idempotent).
    try:
        cfg = _ensure_setup()
    except Exception as e:
        print(f"  [FATAL] auto-setup failed: {e}", file=sys.stderr)
        return 1
    print(f"  state dir   : {cfg.state_dir}")
    print(f"  node id     : {cfg.identity.pubkey_hex[:16]}...")
    print(f"  tier        : {cfg.tier}  (role: {cfg.role})")

    # 3. Backend.
    print(f"  accelerator : {_probe_backend()}")

    # 4. Federation (informational only — never blocks).
    print(f"  {_probe_federation()}")

    # 5. Auto-form the mesh: LAN mDNS + DNS seeds + persisted peers.
    mesh = _form_mesh(cfg)
    print(f"  mesh        : {mesh['peer_count']} peer(s) known"
          f"{'  (LAN-announced)' if mesh.get('lan_active') else ''}")
    if mesh.get('public_ip'):
        print(f"  public IP   : {mesh['public_ip']}  "
              f"(share `{cfg.identity.pubkey_hex[:16]}...@"
              f"{mesh['public_ip']}:5300` with a friend)")

    _hr("-" * 72)
    print("  Ready. Opening GUI...")
    _hr("=" * 72)
    print()

    # 6. GUI.
    return _launch_gui_or_service()


if __name__ == "__main__":
    sys.exit(main())
