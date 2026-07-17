"""Build a Pluginfer .deb package.

Produces a Debian package that:
  - Installs the Pluginfer CLI to /usr/lib/pluginfer/
  - Drops a launcher at /usr/bin/pluginfer
  - Registers a systemd unit at /etc/systemd/system/pluginfer.service
  - Writes default config to /etc/pluginfer/config.yml
  - Stores user wallet at ~/.config/pluginfer/wallet.pem (per-user; not
    in /etc, never overwritten on upgrade)

The script uses Python's stdlib + `dpkg-deb` (which ships with Debian /
Ubuntu and is also available via apt on most CI runners). We do NOT
require fpm or other heavy tooling.

The shipped tree is curated by `SOURCE_DIRS` + `SOURCE_FILES` allowlists
(NOT a denylist over v2/) — vendored numpy/torch/etc. live under v2/ for
dev convenience but must never end up in the .deb. python3-cryptography
+ pip-installed runtime deps cover the actual runtime.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
V2_DIR = REPO_ROOT / "v2"

# Allowlist of source dirs / files that go into the .deb. Anything not
# listed here stays out (vendored packages, snapshots, build artefacts,
# .docx whitepapers, .exe fixtures, etc.).
SOURCE_DIRS = ("core", "ai", "infrastructure", "plugins", "ui", "utils")
SOURCE_FILES = (
    "pluginfer_node.py",
    "requirements.txt",
    "requirements-prod.txt",
    "README.md",
    "LICENSE",
)


SYSTEMD_UNIT = """\
[Unit]
Description=Pluginfer Distributed Compute Node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pluginfer
Group=pluginfer
WorkingDirectory=/usr/lib/pluginfer
ExecStart=/usr/bin/pluginfer start --role provider
Restart=on-failure
RestartSec=10
LimitNOFILE=65536
ReadWritePaths=/var/lib/pluginfer

[Install]
WantedBy=multi-user.target
"""

POSTINST_SCRIPT = """\
#!/bin/sh
set -e
# Create system user if not present.
id pluginfer >/dev/null 2>&1 || useradd --system --home /var/lib/pluginfer \\
    --shell /usr/sbin/nologin pluginfer
mkdir -p /var/lib/pluginfer
chown -R pluginfer:pluginfer /var/lib/pluginfer
systemctl daemon-reload || true
echo "[pluginfer] install OK. Start with: sudo systemctl start pluginfer"
"""

PRERM_SCRIPT = """\
#!/bin/sh
set -e
systemctl stop pluginfer || true
systemctl disable pluginfer || true
echo "[pluginfer] stopped. Wallet at /var/lib/pluginfer/wallet.pem is "
echo "preserved -- delete manually if you want a clean uninstall."
"""

LAUNCHER_SCRIPT = """\
#!/bin/sh
exec python3 /usr/lib/pluginfer/pluginfer_node.py "$@"
"""


def build_deb(
    *,
    version: str,
    git_sha: str,
    out_dir: Path,
    source_dir: Optional[Path] = None,
) -> Path:
    """Assemble a .deb at `out_dir`. Returns the .deb path.

    `source_dir` defaults to V2_DIR for production builds. Tests pass a
    small synthetic dir to keep assembly fast and deterministic.
    """
    src_root = Path(source_dir) if source_dir else V2_DIR
    arch = "amd64"  # Pluginfer is a pure-python pkg; arch is informational.
    pkg_name = f"pluginfer_{version}_{arch}"
    work = Path(tempfile.mkdtemp(prefix="pluginfer-deb-"))
    pkg = work / pkg_name
    debian = pkg / "DEBIAN"
    usr_lib = pkg / "usr" / "lib" / "pluginfer"
    usr_bin = pkg / "usr" / "bin"
    etc_systemd = pkg / "etc" / "systemd" / "system"
    etc_pluginfer = pkg / "etc" / "pluginfer"

    for d in (debian, usr_lib, usr_bin, etc_systemd, etc_pluginfer):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Curated copy: only the SOURCE_DIRS + SOURCE_FILES allowlist plus
    # any additional dirs the caller stages under `source_dir`.
    # ------------------------------------------------------------------
    skip_dirs = {"__pycache__", ".pytest_cache", "_archive_v2", "node_modules"}
    skip_file_suffixes = (".bak", ".log", ".tmp", ".pyc")
    skip_files = {
        "identity.json", "peers.json", "ledger.json",
        "wallet.pem", "governance.json", "marketplace.json",
        "reputation.json", "contracts.json", "learning_history.json",
    }

    def _copy_tree(src: Path, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        for entry in src.iterdir():
            if entry.is_dir():
                if entry.name in skip_dirs:
                    continue
                _copy_tree(entry, dst / entry.name)
            else:
                if entry.name in skip_files:
                    continue
                if entry.suffix in skip_file_suffixes:
                    continue
                shutil.copy2(entry, dst / entry.name)

    for dirname in SOURCE_DIRS:
        d = src_root / dirname
        if d.is_dir():
            _copy_tree(d, usr_lib / dirname)
    for fname in SOURCE_FILES:
        f = src_root / fname
        if f.is_file():
            shutil.copy2(f, usr_lib / fname)

    # ------------------------------------------------------------------
    # Launcher + systemd + control + scripts
    # ------------------------------------------------------------------
    (usr_bin / "pluginfer").write_text(LAUNCHER_SCRIPT)
    (usr_bin / "pluginfer").chmod(0o755)

    (etc_systemd / "pluginfer.service").write_text(SYSTEMD_UNIT)

    (etc_pluginfer / "config.yml").write_text(dedent(f"""\
        # Pluginfer node config (system-wide defaults).
        # Per-user state lives in ~/.config/pluginfer/.
        version: {version}
        node:
          port: 8100
          role: provider
        bootstrap:
          # Add seed-node host:port pairs here. See
          # /usr/lib/pluginfer/infrastructure/seed_node/README.md.
          seeds: []
        """))

    control_text = dedent(f"""\
        Package: pluginfer
        Version: {version}
        Architecture: {arch}
        Maintainer: Pluginfer Team <ops@pluginfer.network>
        Depends: python3 (>= 3.10), python3-pip, python3-cryptography
        Description: Pluginfer Distributed Compute Node
         Pluginfer is a peer-to-peer GPU compute marketplace. This
         package installs the node binary, registers a systemd unit,
         and creates a system user to run it under.
        Homepage: https://pluginfer.network
        Section: net
        Priority: optional
        """)
    (debian / "control").write_text(control_text)

    (debian / "postinst").write_text(POSTINST_SCRIPT)
    (debian / "postinst").chmod(0o755)
    (debian / "prerm").write_text(PRERM_SCRIPT)
    (debian / "prerm").chmod(0o755)

    # Conffiles -> systemd unit + config.yml are conf files (not replaced
    # blindly on upgrade).
    (debian / "conffiles").write_text(
        "/etc/systemd/system/pluginfer.service\n"
        "/etc/pluginfer/config.yml\n"
    )

    # ------------------------------------------------------------------
    # Build the .deb. We prefer dpkg-deb when available; otherwise we
    # produce an `ar`-archive manually so this also works in CI runners
    # without dpkg installed.
    # ------------------------------------------------------------------
    deb_path = out_dir / f"{pkg_name}.deb"

    if shutil.which("dpkg-deb"):
        subprocess.check_call(
            ["dpkg-deb", "--build", "--root-owner-group", str(pkg), str(deb_path)],
        )
    else:
        _ar_pack_manual(pkg, deb_path)

    print(f"[build_deb] wrote {deb_path}")

    # Clean up scratch dir
    shutil.rmtree(work, ignore_errors=True)
    return deb_path


def _ar_pack_manual(pkg_dir: Path, out_path: Path) -> None:
    """Fallback .deb builder using stdlib only (no dpkg-deb available).

    Emits a valid .deb: ar archive with debian-binary, control.tar.gz,
    data.tar.gz members in that exact order.
    """
    import gzip
    import io
    import struct

    debian_binary = b"2.0\n"

    # control.tar.gz
    control_buf = io.BytesIO()
    with tarfile.open(fileobj=control_buf, mode="w") as tf:
        for entry in (pkg_dir / "DEBIAN").iterdir():
            tf.add(entry, arcname=entry.name)
    control_gz = gzip.compress(control_buf.getvalue())

    # data.tar.gz (everything except DEBIAN/)
    data_buf = io.BytesIO()
    with tarfile.open(fileobj=data_buf, mode="w") as tf:
        for entry in pkg_dir.iterdir():
            if entry.name == "DEBIAN":
                continue
            tf.add(entry, arcname="./" + entry.name)
    data_gz = gzip.compress(data_buf.getvalue())

    def _ar_header(name: str, size: int) -> bytes:
        # Standard ar member header: 16 chars name, 12 mtime, 6 uid, 6 gid,
        # 8 mode, 10 size, 2 magic.
        return (
            f"{name:<16}{int(0):<12}{int(0):<6}{int(0):<6}"
            f"{'644':<8}{size:<10}".encode("ascii") + b"\x60\n"
        )

    with open(out_path, "wb") as f:
        f.write(b"!<arch>\n")
        for name, body in (
            ("debian-binary", debian_binary),
            ("control.tar.gz", control_gz),
            ("data.tar.gz", data_gz),
        ):
            f.write(_ar_header(name, len(body)))
            f.write(body)
            if len(body) % 2:
                f.write(b"\n")
