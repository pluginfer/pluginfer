"""G3 — self-signed dev-cert generator for the installer pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from tools.dev_cert import generate_self_signed   # noqa: E402


def test_generates_a_codesigning_cert_with_correct_eku(tmp_path):
    cert_path, key_path, fp_path = generate_self_signed(
        common_name="Pluginfer Test", organisation="Pluginfer Test",
        out_dir=tmp_path, days_valid=30,
    )
    assert cert_path.exists() and key_path.exists() and fp_path.exists()
    # Re-load + assert the EKU includes Code Signing OID.
    from cryptography import x509
    from cryptography.x509.oid import ExtendedKeyUsageOID
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert ExtendedKeyUsageOID.CODE_SIGNING in eku


def test_fingerprint_file_matches_cert_fingerprint(tmp_path):
    cert_path, _, fp_path = generate_self_signed(
        common_name="Pluginfer Test", organisation="Pluginfer Test",
        out_dir=tmp_path, days_valid=30,
    )
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    assert fp_path.read_text().strip() == cert.fingerprint(hashes.SHA256()).hex()


def test_private_key_has_restricted_permissions(tmp_path):
    """On POSIX systems chmod 0o600 should land; on Windows os.chmod
    is a no-op but the test asserts the file exists either way."""
    _, key_path, _ = generate_self_signed(
        common_name="Pluginfer Test", organisation="Pluginfer Test",
        out_dir=tmp_path, days_valid=30,
    )
    assert key_path.exists()
    import os
    import stat
    if os.name == "posix":
        mode = stat.S_IMODE(key_path.stat().st_mode)
        # Group + other should not have read; owner read+write.
        assert mode & stat.S_IRWXG == 0
        assert mode & stat.S_IRWXO == 0


def test_re_running_overwrites_cleanly(tmp_path):
    cert_a, _, _ = generate_self_signed(
        common_name="A", organisation="A", out_dir=tmp_path, days_valid=30,
    )
    bytes_a = cert_a.read_bytes()
    cert_b, _, _ = generate_self_signed(
        common_name="B", organisation="B", out_dir=tmp_path, days_valid=30,
    )
    bytes_b = cert_b.read_bytes()
    # Each invocation generates fresh material.
    assert bytes_a != bytes_b
