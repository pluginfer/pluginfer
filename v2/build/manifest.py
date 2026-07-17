"""Build-time manifest signer.

Produces the manifest.json consumed by `core.updater` at runtime. The
runtime side already does ECDSA-verify of the manifest body against
`PLUGINFER_RELEASE_PUBKEY_PEM` (see W31 / commit `94f25c2`); this
script is the matching "release" side that signs.

Manifest schema:

    {
      "version": "1.0.0",
      "git_sha": "abcdef1234",
      "released_at": <unix>,
      "artefacts": {
        "linux_deb":   {"url": "...", "sha256": "..."},
        "linux_rpm":   {"url": "...", "sha256": "..."},
        "linux_appimage": {"url": "...", "sha256": "..."},
        "windows_exe": {"url": "...", "sha256": "..."},
        "macos_pkg":   {"url": "...", "sha256": "..."}
      },
      "manifest_signature": "<base64 ECDSA over the body without this field>"
    }

Usage from CI:

    export PLUGINFER_RELEASE_PRIVKEY_PEM=$(cat /secrets/release.priv.pem)
    python -m build.manifest \\
        --version 1.0.0 --git-sha abcdef1234 \\
        --linux-deb dist/pluginfer_1.0.0_amd64.deb \\
        --windows-exe dist/Pluginfer-1.0.0-Setup.exe \\
        --macos-pkg dist/Pluginfer-1.0.0.pkg \\
        --output dist/manifest.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    *,
    version: str,
    git_sha: str,
    artefacts: dict[str, dict],
) -> dict:
    return {
        "version": version,
        "git_sha": git_sha,
        "released_at": time.time(),
        "artefacts": artefacts,
    }


def sign_manifest(manifest: dict, *, privkey_pem: str) -> dict:
    """Return a copy of `manifest` with `manifest_signature` set."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    body = {k: v for k, v in manifest.items() if k != "manifest_signature"}
    canonical = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    priv = serialization.load_pem_private_key(privkey_pem.encode(), password=None)
    if not isinstance(priv, ec.EllipticCurvePrivateKey):
        raise ValueError("PLUGINFER_RELEASE_PRIVKEY_PEM must be an EC private key")
    sig = priv.sign(canonical, ec.ECDSA(hashes.SHA256()))
    out = dict(manifest)
    out["manifest_signature"] = base64.b64encode(sig).decode("ascii")
    return out


def verify_manifest(manifest: dict, *, pubkey_pem: str) -> bool:
    """Mirror of the runtime check (kept here so tests verify both sides)."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    sig_b64 = manifest.get("manifest_signature")
    if not sig_b64:
        return False
    body = {k: v for k, v in manifest.items() if k != "manifest_signature"}
    canonical = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    try:
        pub = serialization.load_pem_public_key(pubkey_pem.encode())
        pub.verify(
            base64.b64decode(sig_b64), canonical, ec.ECDSA(hashes.SHA256()),
        )
        return True
    except InvalidSignature:
        return False
    except Exception:
        return False


def _artefact(path: Optional[str], url_pattern: str) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"artefact missing: {p}")
    return {
        "url": url_pattern.format(name=p.name),
        "sha256": sha256_of_file(p),
        "size": p.stat().st_size,
        "filename": p.name,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Pluginfer manifest builder")
    ap.add_argument("--version", required=True)
    ap.add_argument("--git-sha", required=True)
    ap.add_argument(
        "--url-pattern",
        default="https://github.com/pluginfer/pluginfer/releases/download/v{version}/{name}",
        help="URL template; {name} expands to the artefact filename, "
             "{version} to the release version.",
    )
    ap.add_argument("--linux-deb")
    ap.add_argument("--linux-rpm")
    ap.add_argument("--linux-appimage")
    ap.add_argument("--windows-exe")
    ap.add_argument("--macos-pkg")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pattern = args.url_pattern.replace("{version}", args.version)
    artefacts: dict[str, dict] = {}
    for key, path in (
        ("linux_deb", args.linux_deb),
        ("linux_rpm", args.linux_rpm),
        ("linux_appimage", args.linux_appimage),
        ("windows_exe", args.windows_exe),
        ("macos_pkg", args.macos_pkg),
    ):
        rec = _artefact(path, pattern)
        if rec:
            artefacts[key] = rec

    manifest = build_manifest(
        version=args.version, git_sha=args.git_sha, artefacts=artefacts,
    )
    privkey = os.environ.get("PLUGINFER_RELEASE_PRIVKEY_PEM")
    if not privkey:
        raise SystemExit(
            "PLUGINFER_RELEASE_PRIVKEY_PEM env var unset; refusing to write "
            "an unsigned manifest"
        )
    signed = sign_manifest(manifest, privkey_pem=privkey)
    Path(args.output).write_text(json.dumps(signed, indent=2), encoding="utf-8")
    print(f"[manifest] wrote {args.output} ({len(artefacts)} artefacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
