"""Windows build pipeline: PyInstaller -> NSIS -> Authenticode signing.

Steps:
  1. Run PyInstaller to produce a single-folder bundle in v2/build/dist/
  2. Run makensis on installer.nsi (interpolates VERSION + GIT_SHA)
  3. Optionally sign the resulting .exe with signtool.exe if an EV
     cert is configured via PLUGINFER_AUTHENTICODE_PFX +
     PLUGINFER_AUTHENTICODE_PASS env vars.
  4. Print SHA-256 of the final .exe so the manifest can pin it.

Requires (Windows host):
  - Python with pyinstaller installed
  - NSIS (makensis on PATH)
  - signtool.exe (Windows SDK) only if signing is requested

This script does NOT execute on non-Windows hosts; build_all.py guards.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
V2_DIR = REPO_ROOT / "v2"
WIN_DIR = V2_DIR / "build" / "windows"
# The REAL product CLI ('pluginfer up' → control panel, auto-mesh,
# gateway). The previous entrypoint (pluginfer_node.py) was the legacy
# standalone node — installers were shipping a different product than
# the README describes.
ENTRYPOINT = V2_DIR / "pluginfer.py"


def build_windows(*, version: str, git_sha: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _pyinstaller(out_dir)
    setup_exe = _make_nsi(version=version, git_sha=git_sha, out_dir=out_dir)
    _maybe_sign(setup_exe)
    sha = hashlib.sha256(setup_exe.read_bytes()).hexdigest()
    print(f"[build_windows] {setup_exe}  sha256={sha}")
    return setup_exe


def _pyinstaller(out_dir: Path) -> None:
    work = out_dir / "pluginfer"
    work.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "pyinstaller", "--clean", "--noconfirm",
        "--onedir",
        "--name", "pluginfer",
        "--distpath", str(out_dir),
        "--workpath", str(out_dir / "_pyinstaller_work"),
        "--specpath", str(out_dir / "_pyinstaller_spec"),
        # The control panel is a DATA file next to tools/auto_mesh.py —
        # PyInstaller only bundles imports, so without this the installed
        # node serves the fallback page instead of the panel.
        "--add-data",
        f"{V2_DIR / 'tools' / 'control_panel.html'}{os.pathsep}tools",
        # uvicorn resolves its event loop / protocol classes dynamically;
        # the frozen build needs them named explicitly.
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.loops.asyncio",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.http.h11_impl",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan.on",
        str(ENTRYPOINT),
    ]
    subprocess.check_call(cmd, cwd=str(V2_DIR))


def _make_nsi(*, version: str, git_sha: str, out_dir: Path) -> Path:
    if not (cmd := _which("makensis")):
        raise RuntimeError(
            "makensis not on PATH; install NSIS from https://nsis.sourceforge.io/"
        )
    rel_app_dir = os.path.relpath(out_dir / "pluginfer", WIN_DIR)
    setup_exe = out_dir / f"Pluginfer-{version}-Setup.exe"
    # The .nsi takes the output path from us — previously it hardcoded
    # ../../dist (a directory nothing creates) while we looked in
    # out_dir, so CI failed with "Can't open output file".
    rel_out_exe = os.path.relpath(setup_exe, WIN_DIR)
    subprocess.check_call(
        [
            cmd,
            f"-DVERSION={version}",
            f"-DGIT_SHA={git_sha}",
            f"-DAPP_DIR_REL={rel_app_dir}",
            f"-DOUT_EXE_REL={rel_out_exe}",
            str(WIN_DIR / "installer.nsi"),
        ],
        cwd=str(WIN_DIR),
    )
    if not setup_exe.exists():
        raise RuntimeError(f"NSIS did not produce {setup_exe}")
    return setup_exe


def _maybe_sign(setup_exe: Path) -> None:
    pfx = os.environ.get("PLUGINFER_AUTHENTICODE_PFX")
    pwd = os.environ.get("PLUGINFER_AUTHENTICODE_PASS")
    if not pfx or not pwd:
        print("[build_windows] no Authenticode cert configured; skipping signing")
        return
    if not (signtool := _which("signtool.exe")):
        raise RuntimeError("signtool.exe not on PATH (Windows SDK required)")
    subprocess.check_call([
        signtool, "sign",
        "/f", pfx, "/p", pwd,
        "/fd", "sha256",
        "/tr", "http://timestamp.digicert.com",
        "/td", "sha256",
        str(setup_exe),
    ])
    subprocess.check_call([signtool, "verify", "/pa", "/v", str(setup_exe)])


def _which(name: str) -> str | None:
    import shutil
    return shutil.which(name)
