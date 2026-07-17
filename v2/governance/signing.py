"""Receipt signing for the governance audit log.

The audit's honest threat model (see AUDIT.md): a hash chain alone stops
NAIVE edits but not a motivated operator, who can recompute every
downstream hash and re-verify a doctored chain. Signatures raise that
bar: rewriting history now requires the signing key, and — with
Ed25519 — anyone can verify each receipt against the PUBLIC key without
trusting the gateway at all. That is the difference between "integrity
you must take our word for" and "integrity a third party can check".

Two backends, chosen at runtime and ALWAYS labelled on the receipt so
nobody is misled about which guarantee they have:

  * ``ed25519`` (preferred, needs the ``cryptography`` package):
    public-key signatures. The gateway holds the private key; auditors
    hold only the public key and can verify independently. This is the
    real answer to "prove the operator didn't doctor the log".
  * ``hmac-sha256`` (stdlib fallback): a keyed MAC. It proves the log
    was not altered by anyone WITHOUT the secret — but the gateway
    holds that secret, so it is integrity, NOT independent
    verifiability. Honestly labelled ``algorithm: "hmac-sha256"`` so a
    reader knows the weaker guarantee.

Neither backend makes an insider-proof ledger on its own — for that the
chain HEAD must be anchored externally (append-only third-party store /
public timestamp). ``GatewaySigner`` exposes the head for exactly that;
external anchoring is the operator's deployment choice, documented, not
silently assumed.

The private key persists (0600) in the budget state dir so receipts
stay verifiable across restarts; a fresh dir mints a fresh key.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from pathlib import Path
from typing import Optional

logger = logging.getLogger("pluginfer.governance.signing")

try:  # real public-key signatures when available
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey, Ed25519PublicKey,
    )
    _HAVE_ED25519 = True
except Exception:  # pragma: no cover - environment without cryptography
    _HAVE_ED25519 = False


class GatewaySigner:
    """Signs receipt bodies. Prefer :meth:`create` — it persists/loads
    the key from a state dir. Construct directly only in tests."""

    def __init__(self, algorithm: str, *, ed_private=None,
                 hmac_secret: Optional[bytes] = None,
                 public_key_pem: str = ""):
        self.algorithm = algorithm
        self._ed_private = ed_private
        self._hmac_secret = hmac_secret
        self.public_key_pem = public_key_pem

    # ------------------------------------------------------------------
    @classmethod
    def create(cls, state_dir: Optional[str] = None,
               *, prefer: str = "ed25519") -> "GatewaySigner":
        if prefer == "ed25519" and _HAVE_ED25519:
            return cls._ed25519(state_dir)
        return cls._hmac(state_dir)

    @classmethod
    def _ed25519(cls, state_dir) -> "GatewaySigner":
        key = None
        path = Path(state_dir) / "gateway_ed25519.key" if state_dir else None
        if path and path.exists():
            try:
                key = serialization.load_pem_private_key(
                    path.read_bytes(), password=None)
            except Exception as e:
                logger.warning("gateway key unreadable (%s) — minting "
                               "a new one; old receipts stay verifiable "
                               "against the old public key only", e)
        if key is None:
            key = Ed25519PrivateKey.generate()
            if path:
                try:
                    path.write_bytes(key.private_bytes(
                        serialization.Encoding.PEM,
                        serialization.PrivateFormat.PKCS8,
                        serialization.NoEncryption()))
                    try:
                        os.chmod(path, 0o600)
                    except OSError:
                        pass
                except Exception as e:
                    logger.warning("could not persist gateway key: %s", e)
        pub_pem = key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("ascii")
        return cls("ed25519", ed_private=key, public_key_pem=pub_pem)

    @classmethod
    def _hmac(cls, state_dir) -> "GatewaySigner":
        secret = None
        path = Path(state_dir) / "gateway_hmac.key" if state_dir else None
        if path and path.exists():
            try:
                secret = bytes.fromhex(path.read_text().strip())
            except Exception:
                secret = None
        if secret is None:
            secret = secrets.token_bytes(32)
            if path:
                try:
                    path.write_text(secret.hex())
                    try:
                        os.chmod(path, 0o600)
                    except OSError:
                        pass
                except Exception as e:
                    logger.warning("could not persist hmac key: %s", e)
        # The "public" identifier for an HMAC signer is a non-secret
        # fingerprint of the key, so receipts can be grouped by signer
        # without exposing the secret.
        fp = hashlib.sha256(secret).hexdigest()[:16]
        return cls("hmac-sha256", hmac_secret=secret,
                   public_key_pem=f"hmac-key:{fp}")

    # ------------------------------------------------------------------
    def sign(self, message: str) -> str:
        data = message.encode("utf-8")
        if self.algorithm == "ed25519":
            return self._ed_private.sign(data).hex()
        return hmac.new(self._hmac_secret, data, hashlib.sha256).hexdigest()

    def verify(self, message: str, signature_hex: str) -> bool:
        data = message.encode("utf-8")
        try:
            if self.algorithm == "ed25519":
                self._ed_private.public_key().verify(
                    bytes.fromhex(signature_hex), data)
                return True
            expected = hmac.new(self._hmac_secret, data,
                                hashlib.sha256).hexdigest()
            return hmac.compare_digest(expected, signature_hex)
        except Exception:
            return False


def verify_with_public_pem(public_key_pem: str, message: str,
                           signature_hex: str) -> bool:
    """Independent verification against a PUBLIC key only — the whole
    point of ed25519 mode: an auditor who never trusts the gateway can
    still check every receipt. Returns False for hmac keys (no public
    verifiability) and when cryptography is unavailable."""
    if not public_key_pem.startswith("-----BEGIN PUBLIC KEY-----"):
        return False
    if not _HAVE_ED25519:
        return False
    try:
        pub = serialization.load_pem_public_key(
            public_key_pem.encode("ascii"))
        if not isinstance(pub, Ed25519PublicKey):
            return False
        pub.verify(bytes.fromhex(signature_hex),
                   message.encode("utf-8"))
        return True
    except Exception:
        return False
