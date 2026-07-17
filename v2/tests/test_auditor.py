"""
Auditor / signed-manifest tests (W30)
=====================================

Cases:
  1. No manifest, no pubkey -> falls back to runtime snapshot;
     attestation_mode == 'self-signed-snapshot'.
  2. Manifest present, no pubkey -> attestation_mode ==
     'manifest-unverified' (loaded but accept-without-verify).
  3. Manifest signed with a known release pubkey -> attestation_mode
     == 'release-manifest-verified'.
  4. Tampered manifest signature -> auditor refuses to trust the
     manifest, falls back to runtime snapshot.
  5. Manifest expects a file that no longer matches on disk ->
     audit reports MODIFIED.
  6. build_signed_manifest builds a valid manifest that the auditor
     accepts.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
for parent in [_HERE.parents[1], _HERE.parents[2]]:
    if (parent / "core").is_dir():
        sys.path.insert(0, str(parent))
        break

from core.compute_ledger import ComputeLedger              # noqa: E402
from core.auditor import (                                # noqa: E402
    SystemAuditor, build_signed_manifest, _verify_manifest_signature,
)


def _make_release_keypair():
    """Generate a test SECP256K1 keypair, return (private_pem, pub_pem)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    sk = ec.generate_private_key(ec.SECP256K1())
    sk_pem = sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    pk_pem = sk.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return sk_pem, pk_pem


def _make_fake_core(td: Path) -> str:
    """Create a tiny fake core/ tree with a couple of .py files."""
    core = td / "fakecore"
    core.mkdir(parents=True, exist_ok=True)
    (core / "__init__.py").write_text("VERSION = '0.0.1'\n")
    (core / "module_a.py").write_text("def f(): return 42\n")
    (core / "module_b.py").write_text("CONST = 'b'\n")
    return str(core)


def test_no_manifest_falls_back_to_snapshot():
    print("\n[1] NO MANIFEST -> SNAPSHOT FALLBACK")
    print("-" * 60)
    led = ComputeLedger("a1")
    with tempfile.TemporaryDirectory() as td:
        core = _make_fake_core(Path(td))
        # manifest_path points at non-existent file.
        a = SystemAuditor(led, core_path=core,
                           manifest_path=str(Path(td) / "nope.json"))
        assert a.attestation_mode == "self-signed-snapshot"
        rep = a.perform_audit()
        assert rep["status"] == "PASS"
        assert rep["attestation_mode"] == "self-signed-snapshot"
        print(f"  attestation: {a.attestation_mode}; "
              f"files={len(a.known_checksums)} OK")
    print("  PASS")


def test_manifest_no_pubkey_unverified():
    print("\n[2] MANIFEST PRESENT, NO PUBKEY -> UNVERIFIED ACCEPT")
    print("-" * 60)
    led = ComputeLedger("a2")
    with tempfile.TemporaryDirectory() as td:
        core = _make_fake_core(Path(td))
        sk_pem, _ = _make_release_keypair()
        manifest_path = Path(td) / "manifest.json"
        build_signed_manifest(core, sk_pem, str(manifest_path), "v1")

        # No PLUGINFER_RELEASE_PUBKEY_PEM env var -> unverified mode.
        os.environ.pop("PLUGINFER_RELEASE_PUBKEY_PEM", None)
        a = SystemAuditor(led, core_path=core,
                           manifest_path=str(manifest_path),
                           release_pubkey_pem=None)
        assert a.attestation_mode == "manifest-unverified"
        rep = a.perform_audit()
        assert rep["status"] == "PASS"
        print(f"  attestation: {a.attestation_mode}; status={rep['status']}")
    print("  PASS")


def test_manifest_with_pubkey_verified():
    print("\n[3] MANIFEST + PUBKEY -> VERIFIED")
    print("-" * 60)
    led = ComputeLedger("a3")
    with tempfile.TemporaryDirectory() as td:
        core = _make_fake_core(Path(td))
        sk_pem, pk_pem = _make_release_keypair()
        manifest_path = Path(td) / "manifest.json"
        build_signed_manifest(core, sk_pem, str(manifest_path), "v1")
        a = SystemAuditor(led, core_path=core,
                           manifest_path=str(manifest_path),
                           release_pubkey_pem=pk_pem)
        assert a.attestation_mode == "release-manifest-verified"
        rep = a.perform_audit()
        assert rep["status"] == "PASS"
        print(f"  attestation: {a.attestation_mode}; status={rep['status']}")
    print("  PASS")


def test_tampered_signature_falls_back():
    print("\n[4] TAMPERED MANIFEST SIGNATURE -> FALLBACK")
    print("-" * 60)
    led = ComputeLedger("a4")
    with tempfile.TemporaryDirectory() as td:
        core = _make_fake_core(Path(td))
        sk_pem, pk_pem = _make_release_keypair()
        manifest_path = Path(td) / "manifest.json"
        build_signed_manifest(core, sk_pem, str(manifest_path), "v1")
        # Corrupt the signature.
        m = json.loads(manifest_path.read_text())
        m["manifest_signature"] = (
            "AAAA" + m["manifest_signature"][4:]
        )
        manifest_path.write_text(json.dumps(m))
        a = SystemAuditor(led, core_path=core,
                           manifest_path=str(manifest_path),
                           release_pubkey_pem=pk_pem)
        assert a.attestation_mode == "self-signed-snapshot", \
            f"expected fallback, got {a.attestation_mode}"
        print(f"  tampered sig rejected; attestation={a.attestation_mode}")
    print("  PASS")


def test_modified_file_detected():
    print("\n[5] FILE MODIFIED AFTER MANIFEST -> AUDIT FAIL")
    print("-" * 60)
    led = ComputeLedger("a5")
    with tempfile.TemporaryDirectory() as td:
        core = _make_fake_core(Path(td))
        sk_pem, pk_pem = _make_release_keypair()
        manifest_path = Path(td) / "manifest.json"
        build_signed_manifest(core, sk_pem, str(manifest_path), "v1")
        a = SystemAuditor(led, core_path=core,
                           manifest_path=str(manifest_path),
                           release_pubkey_pem=pk_pem)
        # Tamper with module_a.py AFTER manifest is built.
        Path(core, "module_a.py").write_text("def f(): return 99\n")
        rep = a.perform_audit()
        assert rep["status"] == "FAIL"
        assert any("MODIFIED" in i for i in rep["issues"])
        print(f"  status=FAIL; issues={[i.split(':')[0] for i in rep['issues']]}")
    print("  PASS")


def test_build_signed_manifest_roundtrip():
    print("\n[6] build_signed_manifest -> verify roundtrip")
    print("-" * 60)
    with tempfile.TemporaryDirectory() as td:
        core = _make_fake_core(Path(td))
        sk_pem, pk_pem = _make_release_keypair()
        out = Path(td) / "manifest.json"
        m = build_signed_manifest(core, sk_pem, str(out), "v0.1")
        assert "manifest_signature" in m
        assert _verify_manifest_signature(m, pk_pem) is True
        # Re-signing with a different key should fail to verify with
        # the original pubkey.
        sk2_pem, pk2_pem = _make_release_keypair()
        assert _verify_manifest_signature(m, pk2_pem) is False
        print("  signed by sk_a verifies under pk_a but not pk_b OK")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("AUDITOR / SIGNED-MANIFEST TEST (W30)")
    print("=" * 60)
    t0 = time.time()
    test_no_manifest_falls_back_to_snapshot()
    test_manifest_no_pubkey_unverified()
    test_manifest_with_pubkey_verified()
    test_tampered_signature_falls_back()
    test_modified_file_detected()
    test_build_signed_manifest_roundtrip()
    print("\n" + "=" * 60)
    print(f"ALL AUDITOR TESTS PASSED in {time.time() - t0:.1f}s")
    print("=" * 60)
