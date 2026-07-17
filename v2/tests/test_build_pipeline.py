"""CP-3 tests: build pipeline (without invoking the actual toolchain).

We test the parts that DO NOT require PyInstaller / NSIS / codesign:

  - Manifest builder schema correctness
  - Manifest signature round-trip (sign with priv, verify with pub)
  - Tampered manifest fails verification
  - Linux .deb assembler produces a parseable ar archive
  - SHA-256 of file utility
  - build_all.detect_version + detect_git_sha don't crash

The actual binary build (PyInstaller + NSIS + signing) is exercised by
the CI release pipeline; we don't run it from pytest.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tarfile
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest  # noqa: E402

from build import build_all  # noqa: E402
from build.linux.build_deb import build_deb  # noqa: E402
from build.manifest import (  # noqa: E402
    build_manifest,
    sha256_of_file,
    sign_manifest,
    verify_manifest,
)


# ---------------------------------------------------------------------------
# Manifest sign/verify round-trip
# ---------------------------------------------------------------------------

def _gen_keypair() -> tuple[str, str]:
    """Generate a fresh SECP256K1 keypair for testing."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    priv = ec.generate_private_key(ec.SECP256K1())
    pub = priv.public_key()
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    pub_pem = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv_pem, pub_pem


def test_manifest_sign_and_verify_round_trip() -> None:
    priv, pub = _gen_keypair()
    m = build_manifest(
        version="1.0.0",
        git_sha="abcdef",
        artefacts={
            "linux_deb": {"url": "https://example/p.deb",
                           "sha256": "00" * 32, "size": 1234, "filename": "p.deb"},
        },
    )
    signed = sign_manifest(m, privkey_pem=priv)
    assert "manifest_signature" in signed
    assert verify_manifest(signed, pubkey_pem=pub) is True


def test_manifest_tamper_fails_verify() -> None:
    priv, pub = _gen_keypair()
    m = build_manifest(version="1.0.0", git_sha="abc", artefacts={})
    signed = sign_manifest(m, privkey_pem=priv)
    signed["version"] = "1.0.1"  # tamper post-signing
    assert verify_manifest(signed, pubkey_pem=pub) is False


def test_manifest_wrong_pubkey_fails_verify() -> None:
    priv, _pub = _gen_keypair()
    _other_priv, other_pub = _gen_keypair()
    m = build_manifest(version="1.0.0", git_sha="abc", artefacts={})
    signed = sign_manifest(m, privkey_pem=priv)
    assert verify_manifest(signed, pubkey_pem=other_pub) is False


def test_manifest_empty_signature_fails_verify() -> None:
    _priv, pub = _gen_keypair()
    m = build_manifest(version="1.0.0", git_sha="abc", artefacts={})
    # No signature field at all
    assert verify_manifest(m, pubkey_pem=pub) is False


def test_sha256_of_file_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "blob.bin"
    p.write_bytes(b"hello pluginfer")
    import hashlib
    expected = hashlib.sha256(b"hello pluginfer").hexdigest()
    assert sha256_of_file(p) == expected


# ---------------------------------------------------------------------------
# Linux .deb assembly (manual ar fallback path)
# ---------------------------------------------------------------------------

def _make_fake_source_tree(root: Path) -> Path:
    """Stage a minimal `v2/`-shaped tree the deb can be built from.

    Production builds copy the real v2/ source via SOURCE_DIRS allowlist;
    tests use this synthetic dir so assembly is fast and deterministic.
    """
    src = root / "fake_v2"
    src.mkdir()
    (src / "core").mkdir()
    (src / "core" / "__init__.py").write_text("# core stub")
    (src / "ai").mkdir()
    (src / "ai" / "__init__.py").write_text("# ai stub")
    (src / "infrastructure").mkdir()
    (src / "infrastructure" / "__init__.py").write_text("# infra stub")
    (src / "pluginfer_node.py").write_text("print('hello')")
    (src / "README.md").write_text("# Pluginfer test fixture")
    return src


def test_build_deb_produces_valid_ar_archive(tmp_path: Path) -> None:
    """Run the .deb assembler and check the output starts with ar magic."""
    out_dir = tmp_path / "dist"
    out_dir.mkdir()
    src = _make_fake_source_tree(tmp_path)
    deb_path = build_deb(
        version="1.0.0", git_sha="testsha", out_dir=out_dir, source_dir=src,
    )
    assert deb_path.exists()
    head = deb_path.read_bytes()[:8]
    assert head == b"!<arch>\n", f"not an ar archive: {head!r}"


def test_build_deb_contains_required_members(tmp_path: Path) -> None:
    out_dir = tmp_path / "dist"
    out_dir.mkdir()
    src = _make_fake_source_tree(tmp_path)
    deb_path = build_deb(
        version="1.0.0", git_sha="testsha", out_dir=out_dir, source_dir=src,
    )
    members = _ar_members(deb_path.read_bytes())
    names = [m[0].rstrip() for m in members]
    assert "debian-binary" in names
    assert "control.tar.gz" in names
    assert "data.tar.gz" in names
    # Order: debian-binary first (deb spec).
    assert names[0] == "debian-binary"


def _ar_members(data: bytes) -> list[tuple[str, bytes]]:
    """Minimal ar parser used only for the test."""
    assert data[:8] == b"!<arch>\n"
    cursor = 8
    out = []
    while cursor < len(data):
        if cursor + 60 > len(data):
            break
        header = data[cursor:cursor + 60]
        name = header[:16].decode().strip()
        size = int(header[48:58].decode().strip())
        body = data[cursor + 60:cursor + 60 + size]
        out.append((name, body))
        cursor += 60 + size
        if size % 2:
            cursor += 1
    return out


def test_build_deb_control_metadata_lists_pluginfer(tmp_path: Path) -> None:
    out_dir = tmp_path / "dist"
    out_dir.mkdir()
    src = _make_fake_source_tree(tmp_path)
    deb_path = build_deb(
        version="1.0.0", git_sha="testsha", out_dir=out_dir, source_dir=src,
    )
    import gzip
    import io

    members = dict(_ar_members(deb_path.read_bytes()))
    control_gz = members["control.tar.gz"]
    control_tar = io.BytesIO(gzip.decompress(control_gz))
    with tarfile.open(fileobj=control_tar, mode="r") as tf:
        files = tf.getnames()
        assert "control" in files
        # Read control file
        with tf.extractfile("control") as f:
            control = f.read().decode()
    assert "Package: pluginfer" in control
    assert "Version: 1.0.0" in control
    assert "Architecture: amd64" in control


# ---------------------------------------------------------------------------
# build_all utilities
# ---------------------------------------------------------------------------

def test_detect_version_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.delenv("PLUGINFER_VERSION", raising=False)
    # We don't assert a specific value because the env may have git
    # describe output; just verify it returns a non-empty string.
    v = build_all.detect_version()
    assert isinstance(v, str) and v


def test_detect_version_honours_env_override(monkeypatch) -> None:
    monkeypatch.setenv("PLUGINFER_VERSION", "9.9.9-test")
    assert build_all.detect_version() == "9.9.9-test"


def test_detect_git_sha_returns_string() -> None:
    sha = build_all.detect_git_sha()
    assert isinstance(sha, str) and sha


# ---------------------------------------------------------------------------
# Manifest cli main()
# ---------------------------------------------------------------------------

def test_manifest_main_refuses_without_privkey(tmp_path, monkeypatch) -> None:
    """Build pipeline must REFUSE to emit an unsigned manifest."""
    from build import manifest as _m

    monkeypatch.delenv("PLUGINFER_RELEASE_PRIVKEY_PEM", raising=False)
    out = tmp_path / "manifest.json"
    monkeypatch.setattr(
        sys, "argv",
        ["manifest", "--version", "1.0.0", "--git-sha", "abc",
         "--output", str(out)],
    )
    with pytest.raises(SystemExit) as ei:
        _m.main()
    assert "PLUGINFER_RELEASE_PRIVKEY_PEM" in str(ei.value)
