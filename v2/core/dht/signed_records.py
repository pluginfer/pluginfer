"""Signed-value wrapper for DHT records.

Every value stored in the DHT is wrapped in a SignedRecord:

    {
        "value":      <any JSON-serialisable thing>,
        "publisher":  <PEM pubkey of whoever stored it>,
        "timestamp":  <unix seconds>,
        "signature":  <base64 ECDSA over canonical JSON of (value,
                                publisher, timestamp)>
    }

Receiving nodes verify the signature before accepting the STORE.
This prevents the classic Kademlia attack where an adversary squats
on a key by being the first STORE responder; here the legitimate
publisher's records always verify, the squatter's don't.

Why timestamp: lets receivers prefer the most recent record when
multiple competing values exist for the same key (last-writer-wins
under signature). Replay protection is the responsibility of the
caller (DHT records are typically content-addressed, so replay of
the SAME value is fine).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SignedRecordError(ValueError):
    pass


@dataclass
class SignedRecord:
    value: Any
    publisher_pem: str
    timestamp: float
    signature: str  # base64

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, body: dict) -> "SignedRecord":
        try:
            return cls(
                value=body["value"],
                publisher_pem=body["publisher_pem"],
                timestamp=float(body["timestamp"]),
                signature=body["signature"],
            )
        except (KeyError, TypeError, ValueError) as e:
            raise SignedRecordError(f"malformed SignedRecord: {e!r}") from e

    def signed_bytes(self) -> bytes:
        """Canonical bytes the signature covers."""
        return _canonical_signed_payload(
            self.value, self.publisher_pem, self.timestamp
        )

    def fingerprint(self) -> str:
        """SHA256 of (value, publisher, timestamp). Useful as a record id."""
        return hashlib.sha256(self.signed_bytes()).hexdigest()


def _canonical_signed_payload(
    value: Any, publisher_pem: str, timestamp: float,
) -> bytes:
    body = {
        "value": value,
        "publisher_pem": publisher_pem,
        "timestamp": float(timestamp),
    }
    return json.dumps(body, sort_keys=True, default=str).encode("utf-8")


def sign_record(value: Any, *, wallet) -> SignedRecord:
    """Wrap `value` in a SignedRecord signed by `wallet`.

    `wallet` is any object exposing `.public_key_pem` and
    `.sign(message: str) -> str`  (matches `core.tokenomics.Wallet`).
    """
    timestamp = time.time()
    payload = _canonical_signed_payload(value, wallet.public_key_pem, timestamp)
    sig = wallet.sign(payload.decode("utf-8"))
    return SignedRecord(
        value=value,
        publisher_pem=wallet.public_key_pem,
        timestamp=timestamp,
        signature=sig,
    )


def verify_record(rec: SignedRecord) -> bool:
    """Return True if `rec.signature` verifies under `rec.publisher_pem`."""
    try:
        from core.tokenomics import Wallet
    except Exception:  # pragma: no cover - fallback for slim deploys
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        try:
            pub = serialization.load_pem_public_key(rec.publisher_pem.encode())
            pub.verify(
                base64.b64decode(rec.signature),
                rec.signed_bytes(),
                ec.ECDSA(hashes.SHA256()),
            )
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            logger.warning("verify_record fallback failed: %s", e)
            return False
    return Wallet.verify(
        rec.publisher_pem,
        rec.signed_bytes().decode("utf-8"),
        rec.signature,
    )
