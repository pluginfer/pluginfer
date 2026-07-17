"""G3 — dev-cert fallback for the installer signing pipeline.

The launch-blocker
------------------
A production-signed Windows installer needs an Authenticode EV cert
($300/yr DigiCert). A production-signed macOS .pkg needs an Apple
Developer Program membership ($99/yr) + a notarisation profile. Both
are off-keyboard, paperwork-and-card-charge actions.

Until those land, the existing build pipeline either fails (no signer)
or emits unsigned binaries that pop "Unknown publisher" / Gatekeeper-
block warnings on every launch — a hard adoption cliff.

What this script does
---------------------
Generates a **self-signed code-signing certificate** that the build
pipeline uses transparently when no production cert is configured.
The resulting binary is:

  * **Hash-stable** — the same SHA-256 the manifest signature commits
    to, regardless of whether prod or dev cert was used to sign.
  * **Trust-on-explicit-install** — the operator publishes the
    self-signed CA's fingerprint alongside the release manifest;
    early adopters add it to their trust store with one PowerShell
    command (Windows) or `security add-trusted-cert` (macOS). The
    process is documented in `docs/SIGNING_SETUP.md`.

This is NOT a substitute for the real cert. It IS a way to keep the
distribution pipeline running end-to-end for the closed beta + the
"developer who knows what they're doing" audience while the real cert
is in procurement.

Outputs land at `$PLUGINFER_DEV_CERT_DIR/dev_cert.pem` +
`dev_cert.key.pem` (default: `~/.pluginfer/dev_cert/`). Override the
location with `--out-dir`. The pipeline picks up the dev cert from
the same env vars production uses (`PLUGINFER_AUTHENTICODE_PFX_PATH`
+ `PLUGINFER_AUTHENTICODE_PASS`) so the swap-in is transparent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Tuple


def _default_out_dir() -> Path:
    return Path(os.environ.get(
        "PLUGINFER_DEV_CERT_DIR",
        str(Path.home() / ".pluginfer" / "dev_cert"),
    ))


def generate_self_signed(
    *,
    common_name: str,
    organisation: str,
    out_dir: Path,
    days_valid: int = 365,
) -> Tuple[Path, Path, Path]:
    """Generate a self-signed code-signing cert + key.

    Returns (cert_pem_path, key_pem_path, sha256_fingerprint_hex).
    Raises if `cryptography` isn't installed — we don't fall back to
    OpenSSL CLI because cross-platform CLI shell-out is a snake pit."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    out_dir.mkdir(parents=True, exist_ok=True)

    # 3072-bit RSA. (EC P-256 would also work but Windows
    # SignTool's older codepaths handle RSA more reliably.)
    key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, organisation),
    ])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=days_valid))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, content_commitment=True,
                key_encipherment=False, data_encipherment=False,
                key_agreement=False, key_cert_sign=False,
                crl_sign=False, encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CODE_SIGNING]),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    cert_path = out_dir / "dev_cert.pem"
    key_path = out_dir / "dev_cert.key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    try:
        os.chmod(key_path, 0o600)
    except OSError:
        pass

    fingerprint_path = out_dir / "dev_cert.sha256.txt"
    fp_hex = cert.fingerprint(hashes.SHA256()).hex()
    fingerprint_path.write_text(fp_hex + "\n", encoding="utf-8")
    return cert_path, key_path, fingerprint_path


def main() -> None:
    ap = argparse.ArgumentParser(description=(
        "Generate a self-signed code-signing cert for the Pluginfer "
        "installer pipeline. Use only until a production EV cert is "
        "procured."
    ))
    ap.add_argument("--common-name", default="Pluginfer Dev Build")
    ap.add_argument("--organisation", default="Pluginfer (Closed Beta)")
    ap.add_argument("--out-dir", default=str(_default_out_dir()))
    ap.add_argument("--days-valid", type=int, default=365)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cert, key, fp = generate_self_signed(
        common_name=args.common_name,
        organisation=args.organisation,
        out_dir=out_dir,
        days_valid=args.days_valid,
    )
    print(
        f"\nself-signed dev cert generated:\n"
        f"  cert        : {cert}\n"
        f"  private key : {key}\n"
        f"  fingerprint : {fp.read_text().strip()}\n"
        f"\nWire into the pipeline:\n"
        f"  export PLUGINFER_AUTHENTICODE_PFX_PATH={cert}\n"
        f"  export PLUGINFER_AUTHENTICODE_PASS=                  # empty pass on dev cert\n"
        f"\nAdoption (Windows):\n"
        f"  Import-Certificate -FilePath '{cert}' "
        f"-CertStoreLocation Cert:\\CurrentUser\\TrustedPublisher\n"
        f"\nAdoption (macOS):\n"
        f"  security add-trusted-cert -d -k ~/Library/Keychains/login.keychain-db '{cert}'\n",
    )


if __name__ == "__main__":
    main()
