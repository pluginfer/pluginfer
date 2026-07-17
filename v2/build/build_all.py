"""Master build orchestrator. Dispatches to a per-platform builder.

Usage:
    python -m build.build_all --platform linux        # .deb + .rpm + AppImage
    python -m build.build_all --platform windows      # NSIS .exe (Windows only)
    python -m build.build_all --platform macos        # signed + notarised .pkg
    python -m build.build_all --platform all          # whichever works on host

Environment variables read by per-platform builders:

    PLUGINFER_VERSION              defaults to git describe / 1.0.0
    PLUGINFER_RELEASE_PRIVKEY_PEM  PEM private key for manifest signing
                                   (NEVER commit; CI secret only)
    PLUGINFER_AUTHENTICODE_PFX     path to .pfx (Windows code-signing)
    PLUGINFER_AUTHENTICODE_PASS    password for .pfx
    APPLE_DEVELOPER_TEAM_ID        e.g. 'ABCD123456'
    APPLE_NOTARY_PROFILE           keychain profile from `xcrun notarytool
                                   store-credentials`

The build pipeline is structured so the SAME script runs in CI and on
a developer machine. Any platform-specific tooling that isn't installed
is reported clearly and the build fails fast.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_DIR = REPO_ROOT / "v2"
DIST_DIR = V2_DIR / "build" / "dist"


def detect_version() -> str:
    if "PLUGINFER_VERSION" in os.environ:
        return os.environ["PLUGINFER_VERSION"]
    try:
        v = subprocess.check_output(
            ["git", "describe", "--tags", "--always"],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
        return v.lstrip("v")
    except Exception:
        return "1.0.0"


def detect_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Pluginfer build pipeline")
    parser.add_argument(
        "--platform",
        choices=("linux", "windows", "macos", "all", "host"),
        default="host",
        help="which OS target to build (default: host)",
    )
    parser.add_argument("--out", default=str(DIST_DIR),
                        help="output directory")
    args = parser.parse_args()

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    version = detect_version()
    git_sha = detect_git_sha()
    print(f"[build] version={version} git_sha={git_sha} target={args.platform}")

    target = args.platform
    if target == "host":
        sysplat = platform.system().lower()
        if sysplat == "darwin":
            target = "macos"
        elif sysplat == "windows":
            target = "windows"
        else:
            target = "linux"

    if target in ("linux", "all"):
        from .linux.build_deb import build_deb
        build_deb(version=version, git_sha=git_sha, out_dir=out_dir)

    if target in ("windows", "all"):
        if platform.system().lower() != "windows":
            print("[build] windows target skipped: must run on Windows host")
        else:
            from .windows.build_windows import build_windows
            build_windows(version=version, git_sha=git_sha, out_dir=out_dir)

    if target in ("macos", "all"):
        if platform.system().lower() != "darwin":
            print("[build] macos target skipped: must run on macOS host")
        else:
            from .macos.build_macos import build_macos
            build_macos(version=version, git_sha=git_sha, out_dir=out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
