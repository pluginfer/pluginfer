"""TLS for the gateway and node (HG22, transport leg).

Two pieces:

* :func:`tls_kwargs` — reads ``<PREFIX>_TLS_CERT`` / ``<PREFIX>_TLS_KEY``
  and returns the ``ssl_certfile``/``ssl_keyfile`` kwargs for uvicorn.
  Half-configured TLS (cert without key, or an unreadable file) FAILS
  STARTUP with a clear message rather than silently serving plaintext —
  config must mean what it says.
* ``python -m governance.tls gencert <dir>`` — mint a self-signed cert
  for private meshes and pilots, where the swarm key already
  authenticates peers and TLS's job is wire privacy (the swarm key
  travels as a header, so plaintext HTTP across the internet would
  expose it — README says so wherever the key is documented).

Honest scope: a self-signed cert gives ENCRYPTION but not third-party
IDENTITY — clients must pin it or disable verification knowingly.
Public-facing deployments should use a real certificate (Let's Encrypt
behind a reverse proxy, or a Cloudflare tunnel, both already
supported paths).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict


class TLSConfigError(RuntimeError):
    """TLS half-configured or files unreadable — refuse to start."""


def tls_kwargs(prefix: str = "PLUGINFER_GW") -> Dict[str, str]:
    """uvicorn kwargs for ``<prefix>_TLS_CERT``/``<prefix>_TLS_KEY``.
    Empty dict when TLS is not configured (plain HTTP, the default)."""
    cert = os.environ.get(f"{prefix}_TLS_CERT", "").strip()
    key = os.environ.get(f"{prefix}_TLS_KEY", "").strip()
    if not cert and not key:
        return {}
    if not (cert and key):
        raise TLSConfigError(
            f"half-configured TLS: set BOTH {prefix}_TLS_CERT and "
            f"{prefix}_TLS_KEY (or neither). Refusing to start rather "
            f"than silently serving plaintext.")
    for label, path in (("cert", cert), ("key", key)):
        if not Path(path).is_file():
            raise TLSConfigError(
                f"TLS {label} file not found: {path}")
    return {"ssl_certfile": cert, "ssl_keyfile": key}


def generate_self_signed(out_dir: os.PathLike, *,
                         common_name: str = "pluginfer-node",
                         days: int = 825) -> Dict[str, str]:
    """Mint a self-signed cert + key pair into ``out_dir``. Returns
    {"cert": path, "key": path}. Requires the ``cryptography``
    package (same dependency Ed25519 receipt signing uses)."""
    try:
        import datetime

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
    except ImportError as e:
        raise TLSConfigError(
            "generating a certificate needs the 'cryptography' "
            "package: pip install cryptography") from e
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,
                                         common_name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName(
            [x509.DNSName(common_name), x509.DNSName("localhost")]),
            critical=False)
        .sign(key, hashes.SHA256())
    )
    key_path = out / "node-key.pem"
    cert_path = out / "node-cert.pem"
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    try:
        os.chmod(key_path, 0o600)
    except OSError:
        pass
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return {"cert": str(cert_path), "key": str(key_path)}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "gencert":
        print("usage: python -m governance.tls gencert <out-dir> "
              "[common-name]")
        return 2
    out_dir = argv[1] if len(argv) > 1 else "."
    cn = argv[2] if len(argv) > 2 else "pluginfer-node"
    paths = generate_self_signed(out_dir, common_name=cn)
    print(f"cert: {paths['cert']}\nkey:  {paths['key']}")
    print("Self-signed: gives wire ENCRYPTION, not third-party "
          "identity — clients must pin it. Public deployments should "
          "use a real certificate or a TLS-terminating proxy/tunnel.")
    print(f"Enable: set PLUGINFER_GW_TLS_CERT={paths['cert']} and "
          f"PLUGINFER_GW_TLS_KEY={paths['key']} (gateway), or "
          f"PLUGINFER_NODE_TLS_CERT/_KEY (node).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
