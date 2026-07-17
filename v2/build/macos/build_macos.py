"""macOS build pipeline: PyInstaller -> .app -> codesign + notarize -> .pkg.

Requires:
  - Python with pyinstaller
  - Apple Developer ID Application certificate in the keychain
  - APPLE_DEVELOPER_TEAM_ID (e.g. ABCD123456)
  - APPLE_NOTARY_PROFILE (from `xcrun notarytool store-credentials`)

Without those, the script still runs and produces an UNSIGNED .pkg
suitable for sideloading (Gatekeeper will warn). For App Store / wide
distribution, real notarization is required and is gated by the
env vars.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
V2_DIR = REPO_ROOT / "v2"
MAC_DIR = V2_DIR / "build" / "macos"
ENTRYPOINT = V2_DIR / "pluginfer_node.py"


def build_macos(*, version: str, git_sha: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    app_path = _pyinstaller(out_dir, version)
    _maybe_codesign(app_path)
    pkg_path = _make_pkg(app_path, version, out_dir)
    _maybe_notarize(pkg_path)
    sha = hashlib.sha256(pkg_path.read_bytes()).hexdigest()
    print(f"[build_macos] {pkg_path}  sha256={sha}")
    return pkg_path


def _pyinstaller(out_dir: Path, version: str) -> Path:
    work = out_dir / "_pyinstaller_work"
    cmd = [
        "pyinstaller", "--clean", "--noconfirm",
        "--windowed",
        "--name", "Pluginfer",
        "--distpath", str(out_dir),
        "--workpath", str(work),
        "--specpath", str(out_dir / "_pyinstaller_spec"),
        f"--osx-bundle-identifier=network.pluginfer.node",
        str(ENTRYPOINT),
    ]
    subprocess.check_call(cmd, cwd=str(V2_DIR))
    return out_dir / "Pluginfer.app"


def _maybe_codesign(app_path: Path) -> None:
    team = os.environ.get("APPLE_DEVELOPER_TEAM_ID")
    if not team:
        print("[build_macos] APPLE_DEVELOPER_TEAM_ID unset; UNSIGNED build")
        return
    identity = f"Developer ID Application: ({team})"
    subprocess.check_call([
        "codesign",
        "--force", "--verbose=2",
        "--deep", "--strict",
        "--options", "runtime",
        "--entitlements", str(MAC_DIR / "entitlements.plist"),
        "--sign", identity,
        str(app_path),
    ])
    subprocess.check_call(["codesign", "--verify", "--verbose=2", str(app_path)])
    subprocess.check_call(["spctl", "--assess", "--verbose=4", str(app_path)])


def _make_pkg(app_path: Path, version: str, out_dir: Path) -> Path:
    pkg_path = out_dir / f"Pluginfer-{version}.pkg"
    subprocess.check_call([
        "productbuild",
        "--component", str(app_path), "/Applications",
        "--version", version,
        "--identifier", "network.pluginfer.node",
        str(pkg_path),
    ])
    return pkg_path


def _maybe_notarize(pkg_path: Path) -> None:
    profile = os.environ.get("APPLE_NOTARY_PROFILE")
    if not profile:
        print("[build_macos] APPLE_NOTARY_PROFILE unset; skipping notarization")
        return
    subprocess.check_call([
        "xcrun", "notarytool", "submit",
        str(pkg_path), "--keychain-profile", profile, "--wait",
    ])
    subprocess.check_call([
        "xcrun", "stapler", "staple", str(pkg_path),
    ])
