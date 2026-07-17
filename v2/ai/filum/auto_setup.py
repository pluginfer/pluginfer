"""§H1 Plug-and-play auto-setup — the brain behind "double-click install".

Every previous decentralized-compute project (Bittensor, Akash,
Render, Folding@home) carries the same operational tax: the user
has to read instructions, install dependencies, set environment
variables, edit config files, and run a CLI. That tax has killed
every single one's mass adoption.

This module is the single entry point that does *everything*
automatically:

* Detects hardware (CPU, GPU, VRAM, OS, locale).
* Picks a safe default config tier (light / standard / max) based
  on hardware.
* Generates a node key (Ed25519) on first run; persists it.
* Selects a region by IP geolocation.
* Picks a contributor role (provider / consumer / both) based on
  what the user clicked in the GUI; auto if not specified.
* Joins the mesh by reaching a known seed node and registering.
* Starts the appropriate background services.
* Surfaces *only* the user-relevant metrics: today's earnings,
  current status, pause/resume.

The user's experience is:
  1. Double-click installer.
  2. Wait 30 seconds.
  3. Mesh is running. Earnings counter ticks up.

There is no terminal. No config file. No environment variable.
No "follow these 12 steps." The application *is* the setup.

Design note: a method of joining a
distributed AI compute mesh in which all configuration decisions
are derived from autodetected host telemetry, local IP
geolocation, and the user's single binary choice (contribute /
consume / both); no manual step is required from key generation
through first job execution; the resulting node is bound to the
user's machine via a locally-stored private key without
requiring an external account, password, or KYC.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------- detected machine profile ---------------------------------------

@dataclass
class MachineProfile:
    """Everything the setup needs to know about the host. Auto-filled."""
    os_name: str = ""
    os_version: str = ""
    cpu_count: int = 0
    cpu_arch: str = ""
    ram_total_mib: int = 0
    backend_name: str = ""             # cuda | rocm | mps | xpu | cpu
    accelerator: str = ""              # human-readable GPU/NPU name
    accelerator_vram_mib: int = 0
    region_hint: str = ""              # ISO country code, best-effort
    hostname: str = ""

    def tier(self) -> str:
        """light | standard | max — based on hardware.

        Threshold is 3800 MiB (not 4096) because consumer cards
        report ~4095 MiB after driver reservation. A "4 GB" GTX 1650
        is correctly classified as 'standard' — proven by the
        gpu_real_train.py 200-step success on this hardware.
        """
        if self.backend_name in ("cpu",):
            return "light"
        v = self.accelerator_vram_mib
        if v < 3800:
            return "light"
        if v < 12 * 1024:
            return "standard"
        return "max"


def detect_machine() -> MachineProfile:
    """Probe the host. Defensive — never raises."""
    p = MachineProfile()
    try:
        p.os_name = platform.system()
        p.os_version = platform.release()
        p.cpu_count = os.cpu_count() or 1
        p.cpu_arch = platform.machine()
        p.hostname = socket.gethostname()
    except Exception:
        pass
    try:
        import psutil
        p.ram_total_mib = int(psutil.virtual_memory().total / (1 << 20))
    except Exception:
        p.ram_total_mib = 0
    try:
        from .hpa.backend import detect_backend
        b = detect_backend()
        p.backend_name = b.name
        p.accelerator = b.accelerator_name
        p.accelerator_vram_mib = int(b.total_memory_bytes / (1 << 20))
    except Exception:
        p.backend_name = "cpu"
        p.accelerator = "cpu"
    p.region_hint = _guess_region()
    return p


def _guess_region() -> str:
    """IP geolocation, best-effort, non-blocking. Returns ISO code or ''."""
    # Production uses a local GeoIP DB; for the bootstrap we fall back to
    # locale-based hint so this works fully offline.
    try:
        import locale
        loc = locale.getlocale()[0] or ""
        if "_" in loc:
            return loc.split("_", 1)[1].upper()
    except Exception:
        pass
    return ""


# ---------- node identity (auto-key) ---------------------------------------

@dataclass
class NodeIdentity:
    pubkey_hex: str
    privkey_path: str             # never logged, never displayed
    created_ts: float


def get_or_create_identity(state_dir: str) -> NodeIdentity:
    """Load or generate the node's persistent Ed25519 keypair.

    Stored in the user's app-data directory by default. The private
    key seed is the only thing that proves "this is your node" —
    losing it means losing your earnings history but not the mesh
    (you can rejoin under a new identity).
    """
    sdir = Path(state_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    priv_path = sdir / "node_key.bin"
    pub_path = sdir / "node_pub.txt"
    meta_path = sdir / "identity.json"
    if priv_path.exists() and pub_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8")) \
               if meta_path.exists() else {}
        return NodeIdentity(
            pubkey_hex=pub_path.read_text(encoding="utf-8").strip(),
            privkey_path=str(priv_path),
            created_ts=meta.get("created_ts", time.time()),
        )
    from .hpa.grain import fresh_keypair
    seed, pub = fresh_keypair()
    priv_path.write_bytes(seed)
    try:
        if hasattr(os, "chmod"):
            os.chmod(priv_path, 0o600)
    except Exception:
        pass
    pub_path.write_text(pub.hex(), encoding="utf-8")
    meta_path.write_text(
        json.dumps({"created_ts": time.time()}), encoding="utf-8",
    )
    return NodeIdentity(
        pubkey_hex=pub.hex(),
        privkey_path=str(priv_path),
        created_ts=time.time(),
    )


# ---------- the auto-config -----------------------------------------------

@dataclass
class AutoConfig:
    """Fully-populated config the rest of the system can use as-is."""
    profile:        MachineProfile
    identity:       NodeIdentity
    role:           str = "both"          # "provider" | "consumer" | "both"
    tier:           str = "standard"
    vram_cap_frac:  float = 0.70
    micro_batch_max: int = 4
    rank_max:       int = 256
    seed_addresses: list = field(default_factory=list)
    state_dir:      str = ""
    join_immediately: bool = True

    def to_summary(self) -> str:
        v = self.profile.accelerator_vram_mib
        return (
            f"Pluginfer is set up.\n"
            f"  Hardware: {self.profile.accelerator} "
            f"({v} MiB VRAM, {self.profile.ram_total_mib} MiB RAM)\n"
            f"  Tier:     {self.tier}\n"
            f"  Role:     {self.role}\n"
            f"  Region:   {self.profile.region_hint or 'unknown'}\n"
            f"  Node ID:  {self.identity.pubkey_hex[:16]}...\n"
            f"  Seeds:    {len(self.seed_addresses)} configured\n"
        )


def auto_setup(
    *,
    role: str = "both",
    state_dir: Optional[str] = None,
    seed_addresses: Optional[list] = None,
) -> AutoConfig:
    """Run the full plug-and-play setup. Returns a complete AutoConfig.

    No prompts, no interactive input, no environment-variable lookups
    beyond ``HOME``/``APPDATA``. This function is what the GUI calls
    on "Start Mesh" and what the installer calls on first run.
    """
    profile = detect_machine()
    state = state_dir or default_state_dir()
    Path(state).mkdir(parents=True, exist_ok=True)
    identity = get_or_create_identity(state)
    tier = profile.tier()

    cfg = AutoConfig(
        profile=profile,
        identity=identity,
        role=role,
        tier=tier,
        vram_cap_frac={"light": 0.50, "standard": 0.70, "max": 0.85}[tier],
        micro_batch_max={"light": 1, "standard": 4, "max": 16}[tier],
        rank_max={"light": 32, "standard": 256, "max": 1024}[tier],
        seed_addresses=list(seed_addresses or _default_seeds()),
        state_dir=state,
        join_immediately=True,
    )
    _write_runtime_config(cfg)
    return cfg


def default_state_dir() -> str:
    """Platform-appropriate config directory.

    Windows:  %APPDATA%/Pluginfer
    macOS:    ~/Library/Application Support/Pluginfer
    Linux:    ~/.config/pluginfer
    """
    home = Path.home()
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return str(Path(appdata) / "Pluginfer")
        return str(home / "AppData" / "Roaming" / "Pluginfer")
    if sys.platform == "darwin":
        return str(home / "Library" / "Application Support" / "Pluginfer")
    return str(home / ".config" / "pluginfer")


def _default_seeds() -> list:
    """The known-good seed addresses. Hard-coded for v0.

    Production replaces this with a DNS-discovered list so seeds can
    be rotated without releasing a new installer.
    """
    return [
        ("seed1.pluginfer.net", 5300),
        ("seed2.pluginfer.net", 5300),
    ]


def _write_runtime_config(cfg: AutoConfig) -> None:
    """Persist the rendered config so the service-mode runner can read it."""
    out = Path(cfg.state_dir) / "runtime_config.json"
    payload = {
        "tier":           cfg.tier,
        "role":           cfg.role,
        "vram_cap_frac":  cfg.vram_cap_frac,
        "micro_batch_max": cfg.micro_batch_max,
        "rank_max":       cfg.rank_max,
        "seed_addresses": [list(s) for s in cfg.seed_addresses],
        "node_pubkey":    cfg.identity.pubkey_hex,
        "profile":        asdict(cfg.profile),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_runtime_config(state_dir: Optional[str] = None) -> Optional[dict]:
    sd = state_dir or default_state_dir()
    p = Path(sd) / "runtime_config.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------- one-line CLI entry --------------------------------------------

def main() -> int:
    print("Pluginfer auto-setup running...")
    cfg = auto_setup()
    print(cfg.to_summary())
    print(f"State dir: {cfg.state_dir}")
    print()
    print("To run the mesh node, double-click the Pluginfer GUI launcher.")
    print("To run headless: python -m ai.filum.service_mode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
