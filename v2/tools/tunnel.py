"""Auto-tunnel: make a home node reachable from the whole internet with
zero networking knowledge.

A node behind a home router (NAT) can talk OUT to the mesh but nobody
can reach it to send jobs IN — so it can't actually contribute compute
without port-forwarding, which no normal person should have to do. This
module starts a Cloudflare "quick tunnel" (free, no account, no card)
that gives the node a public https URL, and reports the host so the
node can advertise it. If cloudflared isn't installed we say exactly
how to get it in one line and let the node run local-only.

ASCII-only output (Windows cp1252 consoles).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

_TRYCF = re.compile(r"https://([a-z0-9-]+\.trycloudflare\.com)")


def find_cloudflared() -> Optional[str]:
    """Locate cloudflared: PATH, common install dirs, or the spot our
    own installer drops it. Returns the executable path or None."""
    onpath = shutil.which("cloudflared")
    if onpath:
        return onpath
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        / "cloudflared" / "cloudflared.exe",
        Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        / "cloudflared" / "cloudflared.exe",
        Path.home() / ".pluginfer" / "bin" / "cloudflared",
        Path.home() / ".pluginfer" / "bin" / "cloudflared.exe",
        Path("/usr/local/bin/cloudflared"),
        Path("/usr/bin/cloudflared"),
        Path("/opt/homebrew/bin/cloudflared"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def install_hint() -> str:
    """The single command to install cloudflared on this OS."""
    if sys.platform.startswith("win"):
        return "winget install Cloudflare.cloudflared"
    if sys.platform == "darwin":
        return "brew install cloudflared"
    return ("curl -L https://github.com/cloudflare/cloudflared/releases/"
            "latest/download/cloudflared-linux-amd64 -o cloudflared "
            "&& chmod +x cloudflared && sudo mv cloudflared /usr/local/bin/")


def start_quick_tunnel(
    local_port: int,
    say: Callable[[str], None],
    *,
    wait_s: float = 40.0,
) -> Tuple[Optional[str], Optional[subprocess.Popen]]:
    """Start a Cloudflare quick tunnel to http://localhost:<local_port>.

    Returns (public_host, process) on success, or (None, None) if
    cloudflared is missing or no URL appears in time. The caller keeps
    the process handle to terminate it on shutdown.
    """
    exe = find_cloudflared()
    if not exe:
        say("  Sharing needs 'cloudflared' (free, no account). Install it:")
        say(f"      {install_hint()}")
        say("  Then re-run. Meanwhile the node runs local-only.")
        return None, None

    log_dir = Path.home() / ".pluginfer"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "tunnel.log"
    logf = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [exe, "tunnel", "--no-autoupdate",
         "--url", f"http://localhost:{local_port}"],
        stdout=logf, stderr=subprocess.STDOUT, text=True)

    deadline = time.monotonic() + wait_s
    host: Optional[str] = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        m = _TRYCF.search(text)
        if m:
            host = m.group(1)
            break
        time.sleep(1.0)

    if not host:
        try:
            proc.terminate()
        except Exception:
            pass
        say("  Could not establish a public tunnel in time; node runs "
            "local-only. Details: " + str(log_path))
        return None, None
    return host, proc
