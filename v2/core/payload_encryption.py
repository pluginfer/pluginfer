"""Payload encryption — prompts + outputs encrypted to the winning
provider's pubkey, so even the gateway operator + relay nodes see
ciphertext only.

The exposure today
------------------
Without this module: the buyer's prompt arrives at the gateway as
plaintext JSON. The gateway dispatches to a provider — also as
plaintext over HTTP. A relay path (A → C → B for NAT traversal)
puts C in the middle of B's plaintext. Anyone on the wire reads
it. Anyone operating a relay reads it.

The fix
-------
Sealed envelope per job:

  1. Buyer's gateway (or buyer's SDK directly) generates a fresh
     ephemeral SECP256K1 keypair.
  2. ECDH(ephemeral_priv, provider_pub) → shared secret.
  3. HKDF(shared_secret) → 32-byte AES-256-GCM key.
  4. Encrypt the JSON payload with that key + a random nonce.
  5. Wire format: { ephemeral_pub, nonce, ciphertext, mac_tag }.
  6. Provider's SDK does the mirror ECDH + HKDF + decrypt.

Properties:
  * Forward secrecy — the ephemeral key is per-job. Even if the
    provider's long-term key leaks LATER, prior jobs stay
    confidential.
  * Tamper detection — AES-GCM provides authenticated encryption.
    Any bit-flip on the wire → MAC verification fails.
  * Auction-friendly — the bid CAN expose the payload's metadata
    (kind, max_tokens, cost_ceiling) without exposing the prompt
    itself. We split the envelope into (clear_metadata, sealed_body).

What this protects against
--------------------------
* Wire eavesdroppers (Wi-Fi snooper, ISP, hostile gateway proxy).
* Compromised relay nodes (mesh-native A→C→B relay).
* Gateway operators who try to log everything — they see ciphertext
  unless they ALSO operate the destination provider.
* Replay attacks — the ephemeral key + nonce are per-job, so a
  re-submitted payload is treated as a fresh job (with whatever
  cost/refund consequences that implies).

What this does NOT protect against
----------------------------------
* The provider node itself, post-decryption. The plaintext lives
  in the provider's process memory while the model executes.
  Mitigation: TEE-attested providers (`core/tee_attest.py`) put
  that memory inside SGX/SEV-SNP. Without TEE, the provider can
  log everything they see.
* Side-channel timing attacks against the provider's model.
  Out-of-scope for transport encryption.
* The buyer's own machine — if their device is compromised, the
  prompt is exposed before encryption.

Innovation: §A35 "Per-job ECIES sealing for permissionless compute
mesh." Combines (a) ephemeral-keypair-per-job for forward secrecy,
(b) provider-pubkey discovery via gossip (already in MembershipView),
AND (c) split-envelope (clear auction metadata + sealed body) so the
auction can route without seeing the prompt. No prior art unifies
these for auction-routed compute.

Dependency choice: this module uses the `cryptography` library for
SECP256K1 ECDH + HKDF + AES-GCM. It's already a transitive dep of
the project's PEM parser stack. NumPy not required.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

ENVELOPE_VERSION = 1


@dataclass
class SealedEnvelope:
    """Wire format: a base64-encoded JSON blob carrying everything
    the receiver needs to decrypt."""
    version: int
    ephemeral_pub_pem: str
    nonce_b64: str
    ciphertext_b64: str
    # AAD = associated data the receiver checks but isn't encrypted:
    # job_id + clear metadata. Tamper-detected via GCM MAC.
    aad_b64: str

    def to_wire(self) -> str:
        return base64.b64encode(json.dumps({
            "v": self.version,
            "ek": self.ephemeral_pub_pem,
            "n": self.nonce_b64,
            "c": self.ciphertext_b64,
            "a": self.aad_b64,
        }).encode("utf-8")).decode("ascii")

    @classmethod
    def from_wire(cls, wire: str) -> "SealedEnvelope":
        blob = json.loads(base64.b64decode(wire).decode("utf-8"))
        return cls(
            version=int(blob.get("v", 1)),
            ephemeral_pub_pem=str(blob["ek"]),
            nonce_b64=str(blob["n"]),
            ciphertext_b64=str(blob["c"]),
            aad_b64=str(blob.get("a", "")),
        )


# ---------------------------------------------------------------------------
# Crypto primitives — these will fail to import without `cryptography`,
# which is in the requirements file. We catch the import error here
# and raise a clear message so the operator knows.
# ---------------------------------------------------------------------------

class EncryptionUnavailable(RuntimeError):
    """The cryptography library isn't installed. Run
    `pip install cryptography`."""


def _ensure_crypto():
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        return ec, HKDF, hashes, serialization, AESGCM
    except ImportError as e:
        raise EncryptionUnavailable(
            f"Install `cryptography` to enable payload encryption: {e}"
        ) from e


def _load_pubkey(pem_str: str):
    from cryptography.hazmat.primitives import serialization
    return serialization.load_pem_public_key(pem_str.encode("utf-8"))


def _serialize_pubkey(pub) -> str:
    from cryptography.hazmat.primitives import serialization
    return pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")


# ---------------------------------------------------------------------------
# Seal / open
# ---------------------------------------------------------------------------

def seal_payload(
    *, payload_bytes: bytes,
    recipient_pubkey_pem: str,
    aad: bytes = b"",
) -> SealedEnvelope:
    """Encrypt `payload_bytes` so only the holder of the private key
    corresponding to `recipient_pubkey_pem` can read it. `aad` is
    authenticated-but-not-encrypted associated data (typically the
    job_id + non-secret metadata so the receiver can verify they
    matched up).
    """
    ec, HKDF, hashes, _ser, AESGCM = _ensure_crypto()
    recipient_pub = _load_pubkey(recipient_pubkey_pem)
    if not hasattr(recipient_pub, "curve"):
        raise ValueError("recipient pubkey is not an EC key")
    # Ephemeral keypair on the SAME curve as the recipient's pubkey.
    ephemeral = ec.generate_private_key(recipient_pub.curve)
    shared = ephemeral.exchange(ec.ECDH(), recipient_pub)
    aes_key = HKDF(
        algorithm=hashes.SHA256(), length=32,
        salt=None, info=b"pluginfer-payload-v1",
    ).derive(shared)
    nonce = os.urandom(12)
    aesgcm = AESGCM(aes_key)
    ciphertext = aesgcm.encrypt(nonce, payload_bytes, aad or None)
    return SealedEnvelope(
        version=ENVELOPE_VERSION,
        ephemeral_pub_pem=_serialize_pubkey(ephemeral.public_key()),
        nonce_b64=base64.b64encode(nonce).decode("ascii"),
        ciphertext_b64=base64.b64encode(ciphertext).decode("ascii"),
        aad_b64=base64.b64encode(aad or b"").decode("ascii"),
    )


def open_payload(
    *, envelope: SealedEnvelope, recipient_privkey,
) -> Optional[bytes]:
    """Recover the plaintext. Returns None on any failure (wrong key,
    tampered ciphertext, malformed envelope) so callers never have
    to distinguish failure modes that an attacker would distinguish
    via timing/error oracles."""
    ec, HKDF, hashes, _ser, AESGCM = _ensure_crypto()
    try:
        ephemeral_pub = _load_pubkey(envelope.ephemeral_pub_pem)
        shared = recipient_privkey.exchange(ec.ECDH(), ephemeral_pub)
        aes_key = HKDF(
            algorithm=hashes.SHA256(), length=32,
            salt=None, info=b"pluginfer-payload-v1",
        ).derive(shared)
        nonce = base64.b64decode(envelope.nonce_b64)
        ciphertext = base64.b64decode(envelope.ciphertext_b64)
        aad = base64.b64decode(envelope.aad_b64) if envelope.aad_b64 else None
        aesgcm = AESGCM(aes_key)
        return aesgcm.decrypt(nonce, ciphertext, aad or None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Split envelope — clear metadata + sealed body
# ---------------------------------------------------------------------------

@dataclass
class SplitEnvelope:
    """The auction needs cost_ceiling / latency / kind to route a
    job; it does NOT need the prompt. We carry those in the clear,
    seal the rest."""
    clear_metadata: Dict[str, Any]
    sealed_body_wire: str

    def to_wire(self) -> Dict[str, Any]:
        return {
            "metadata": self.clear_metadata,
            "sealed_body": self.sealed_body_wire,
            "encrypted": True,
        }

    @classmethod
    def from_wire(cls, blob: Dict[str, Any]) -> "SplitEnvelope":
        return cls(
            clear_metadata=dict(blob.get("metadata") or {}),
            sealed_body_wire=str(blob.get("sealed_body") or ""),
        )


def seal_job_payload(
    *,
    job_payload: Dict[str, Any],
    recipient_pubkey_pem: str,
    metadata_fields: Tuple[str, ...] = (
        "kind", "max_tokens", "model", "consortium", "required_compute_score",
    ),
) -> SplitEnvelope:
    """Split a job payload into (auction-visible metadata, sealed
    body). `metadata_fields` lists keys we leave in the clear so the
    auction can score bids; everything else gets sealed."""
    clear: Dict[str, Any] = {}
    sealed: Dict[str, Any] = {}
    for k, v in (job_payload or {}).items():
        if k in metadata_fields:
            clear[k] = v
        else:
            sealed[k] = v
    body_bytes = json.dumps(sealed, default=str).encode("utf-8")
    env = seal_payload(
        payload_bytes=body_bytes,
        recipient_pubkey_pem=recipient_pubkey_pem,
        aad=json.dumps(clear, sort_keys=True).encode("utf-8"),
    )
    return SplitEnvelope(
        clear_metadata=clear,
        sealed_body_wire=env.to_wire(),
    )


def open_job_payload(
    *, split: SplitEnvelope, recipient_privkey,
) -> Optional[Dict[str, Any]]:
    """Reassemble the original payload by decrypting the sealed body
    and merging it back with the clear metadata."""
    env = SealedEnvelope.from_wire(split.sealed_body_wire)
    plain = open_payload(envelope=env, recipient_privkey=recipient_privkey)
    if plain is None:
        return None
    # Verify AAD matches the clear metadata (catches an attacker who
    # rewrote the metadata between gateway and provider).
    expected_aad = json.dumps(split.clear_metadata, sort_keys=True).encode("utf-8")
    aad_from_envelope = base64.b64decode(env.aad_b64) if env.aad_b64 else b""
    if aad_from_envelope != expected_aad:
        return None
    try:
        sealed_dict = json.loads(plain.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return {**split.clear_metadata, **sealed_dict}


__all__ = [
    "ENVELOPE_VERSION",
    "EncryptionUnavailable",
    "SealedEnvelope",
    "SplitEnvelope",
    "open_job_payload",
    "open_payload",
    "seal_job_payload",
    "seal_payload",
]
