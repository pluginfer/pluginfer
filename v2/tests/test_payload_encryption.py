"""End-to-end payload encryption: prompts + outputs encrypted to the
provider's pubkey. Gateway operators + relay nodes see ciphertext
only.

Invariants:
  * Round-trip: seal → open recovers the exact bytes.
  * Wrong key: open with a different private key returns None.
  * Tamper: any bit-flip in ciphertext OR AAD invalidates the MAC.
  * Split envelope: metadata stays in the clear, body sealed.
  * Forward secrecy: ephemeral key is per-job; two seals to the
    same recipient produce different ephemeral pubs + ciphertexts.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

try:
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import serialization
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CRYPTO_AVAILABLE, reason="cryptography lib required",
)

from core.payload_encryption import (
    SealedEnvelope,
    SplitEnvelope,
    open_job_payload,
    open_payload,
    seal_job_payload,
    seal_payload,
)


def _gen_keypair():
    sk = ec.generate_private_key(ec.SECP256K1())
    pk_pem = sk.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    return sk, pk_pem


# ---------------------------------------------------------------------------
# Round-trip + properties
# ---------------------------------------------------------------------------

def test_seal_open_round_trip_recovers_bytes():
    sk, pk = _gen_keypair()
    plaintext = b"secret prompt: what is the recipe for guavaberry rum"
    env = seal_payload(payload_bytes=plaintext, recipient_pubkey_pem=pk)
    out = open_payload(envelope=env, recipient_privkey=sk)
    assert out == plaintext


def test_wrong_key_returns_none():
    _, pk_a = _gen_keypair()
    sk_b, _ = _gen_keypair()
    env = seal_payload(payload_bytes=b"top secret", recipient_pubkey_pem=pk_a)
    # Decrypt with the wrong recipient's private key.
    assert open_payload(envelope=env, recipient_privkey=sk_b) is None


def test_tampered_ciphertext_returns_none():
    sk, pk = _gen_keypair()
    env = seal_payload(payload_bytes=b"hello", recipient_pubkey_pem=pk)
    # Flip a bit in the ciphertext.
    raw = base64.b64decode(env.ciphertext_b64)
    raw = bytes([raw[0] ^ 0x01]) + raw[1:]
    env.ciphertext_b64 = base64.b64encode(raw).decode("ascii")
    assert open_payload(envelope=env, recipient_privkey=sk) is None


def test_tampered_aad_returns_none():
    sk, pk = _gen_keypair()
    env = seal_payload(
        payload_bytes=b"hello", recipient_pubkey_pem=pk, aad=b"job-1",
    )
    env.aad_b64 = base64.b64encode(b"job-2").decode("ascii")
    assert open_payload(envelope=env, recipient_privkey=sk) is None


def test_forward_secrecy_ephemeral_keys_differ_across_seals():
    """Two seals to the same recipient produce DIFFERENT ephemeral
    pubs + ciphertexts. Per-job key compromise doesn't reveal prior
    or future jobs."""
    sk, pk = _gen_keypair()
    e1 = seal_payload(payload_bytes=b"x", recipient_pubkey_pem=pk)
    e2 = seal_payload(payload_bytes=b"x", recipient_pubkey_pem=pk)
    assert e1.ephemeral_pub_pem != e2.ephemeral_pub_pem
    assert e1.ciphertext_b64 != e2.ciphertext_b64


def test_envelope_wire_round_trip():
    sk, pk = _gen_keypair()
    env = seal_payload(payload_bytes=b"sealed message", recipient_pubkey_pem=pk)
    wire = env.to_wire()
    env2 = SealedEnvelope.from_wire(wire)
    assert open_payload(envelope=env2, recipient_privkey=sk) == b"sealed message"


# ---------------------------------------------------------------------------
# Split envelope: clear metadata + sealed body
# ---------------------------------------------------------------------------

def test_split_envelope_keeps_auction_metadata_clear():
    sk, pk = _gen_keypair()
    payload = {
        "kind": "llm.completion",
        "max_tokens": 200,
        "prompt": "the actual secret",
        "user_email": "private@example.com",
    }
    split = seal_job_payload(
        job_payload=payload, recipient_pubkey_pem=pk,
    )
    # Auction-visible.
    assert split.clear_metadata["kind"] == "llm.completion"
    assert split.clear_metadata["max_tokens"] == 200
    # Secret fields NOT in the metadata.
    assert "prompt" not in split.clear_metadata
    assert "user_email" not in split.clear_metadata
    # Recipient can reconstruct everything.
    recovered = open_job_payload(split=split, recipient_privkey=sk)
    assert recovered["prompt"] == "the actual secret"
    assert recovered["user_email"] == "private@example.com"
    assert recovered["kind"] == "llm.completion"


def test_split_envelope_detects_metadata_tampering():
    """An attacker who flips a field in clear_metadata between
    gateway and provider gets rejected — the AAD MAC catches it."""
    sk, pk = _gen_keypair()
    payload = {"kind": "llm.completion", "prompt": "secret"}
    split = seal_job_payload(
        job_payload=payload, recipient_pubkey_pem=pk,
    )
    # Attacker modifies the metadata to charge buyer more.
    split.clear_metadata["max_tokens"] = 999999
    assert open_job_payload(split=split, recipient_privkey=sk) is None
