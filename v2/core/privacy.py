"""
Confidential-Amount Privacy via Pedersen Commitments + Schnorr PoK
==================================================================
Real implementation, backed by `core/pedersen.py`.

Replaces the previous honest-stub that raised `PrivacyNotImplementedError`
and the (much worse) prior version that returned `"ZK_PROOF_MOCK_DATA_XP25"`
literal strings while pretending to be ZK.

This module exposes a small, audit-friendly facade over the underlying
`pedersen` primitives so callers don't need to manipulate EC points
directly. The facade is what `payments.py`, `smart_contracts.py`, and
the broker should depend on. `pedersen.py` is the math; this is the API.

API
---
    ZKPrivacy.is_available() -> True (now that the math is real)

    create_commitment(value)               -> (commitment_hex, blinding)
        Pedersen-commit `value` (a non-negative int) under a fresh
        random blinding. Returns the public commitment as a hex string
        and the secret blinding as an int — caller must keep the
        blinding to open or prove later.

    verify_commitment(commitment_hex, value, blinding) -> bool
        Re-derive the commitment from (value, blinding) and check it
        equals the supplied commitment.

    generate_proof(value, blinding, commitment_hex) -> dict
        Schnorr proof of knowledge of (value, blinding) opening the
        commitment. Non-interactive (Fiat–Shamir). Returns a dict
        suitable for JSON transport.

    verify_proof(commitment_hex, proof) -> bool
        Verify a Schnorr PoK against a public commitment.

    add_commitments(c1_hex, c2_hex)        -> commitment_hex
        Homomorphic add: commit(a) + commit(b) == commit(a+b). Used
        for confidential-amount transaction balancing without
        revealing individual amounts.

Design notes
------------
* Hex-encoded EC points (uncompressed, 65 bytes / 130 hex chars) are
  the wire format. Easy to log, easy to compare, JSON-safe.
* Blinding is returned as a Python int, not hex. Callers stash this
  in their wallet; it never leaves the prover.
* For the v3-alpha network we expose **bit OR-proofs** (`prove_bit` /
  `verify_bit`) but not yet a full 64-bit aggregated range proof.
  Rationale: bit-OR is enough for "amount is 0 or 1" use cases
  (governance votes, on/off-flags); aggregated Bulletproofs is
  ~800 lines of additional EC math — slated for v3.1.

Performance
-----------
Pure-Python EC: ~3 ms / scalar mul. Commitment creation: ~3 ms.
Proof creation: ~6 ms. Proof verification: ~9 ms. Acceptable for
transaction-rate. For batch-verification on validator nodes, swap
the EC backend in `pedersen.py` to `coincurve` (libsecp256k1
binding, ~50× speedup).
"""

from __future__ import annotations

import logging
from typing import Tuple

from . import pedersen

logger = logging.getLogger(__name__)


class ZKPrivacy:
    """
    Confidential-amount commitments and zero-knowledge proofs.
    Backed by real Pedersen + Schnorr math (`core/pedersen.py`).
    """

    AVAILABLE = True

    def __init__(self):
        logger.debug("ZKPrivacy initialized (Pedersen on SECP256K1).")

    @classmethod
    def is_available(cls) -> bool:
        return cls.AVAILABLE

    # ---- commitments -----------------------------------------------------
    def create_commitment(self, value: int) -> Tuple[str, int]:
        """
        Commit to `value` under a fresh random blinding.

        Returns (commitment_hex, blinding). Caller keeps `blinding`
        secret (it's needed to open or prove later); `commitment_hex`
        is the public artifact safe to broadcast.
        """
        if not isinstance(value, int) or value < 0:
            raise ValueError("value must be a non-negative integer")
        c, blinding = pedersen.commit(value)
        return c.to_hex(), blinding

    def verify_commitment(self, commitment_hex: str,
                          value: int, blinding: int) -> bool:
        """Verify that (value, blinding) opens to `commitment_hex`."""
        try:
            c = pedersen.Commitment.from_hex(commitment_hex)
        except Exception as e:
            logger.warning("verify_commitment: bad commitment encoding: %s", e)
            return False
        return pedersen.verify_open(c, int(value), int(blinding))

    # ---- Schnorr proof of knowledge --------------------------------------
    def generate_proof(self, value: int, blinding: int,
                       commitment_hex: str) -> dict:
        """
        Non-interactive Schnorr PoK that prover knows (value, blinding)
        opening `commitment_hex`. Returns a JSON-safe dict.
        """
        c = pedersen.Commitment.from_hex(commitment_hex)
        proof = pedersen.prove_knowledge(int(value), int(blinding), c)
        return proof.to_dict()

    def verify_proof(self, commitment_hex: str, proof: dict) -> bool:
        """Verify a Schnorr PoK against a public commitment."""
        try:
            c = pedersen.Commitment.from_hex(commitment_hex)
            p = pedersen.SchnorrPoK.from_dict(proof)
        except Exception as e:
            logger.warning("verify_proof: malformed input: %s", e)
            return False
        return pedersen.verify_knowledge(c, p)

    # ---- homomorphic ops -------------------------------------------------
    def add_commitments(self, c1_hex: str, c2_hex: str) -> str:
        """
        Homomorphic addition: commit(a) + commit(b) = commit(a+b).
        Useful for confidential-amount tx balancing: sum of input
        commitments equals sum of output commitments without
        revealing any individual amount.
        """
        c1 = pedersen.Commitment.from_hex(c1_hex)
        c2 = pedersen.Commitment.from_hex(c2_hex)
        return pedersen.add(c1, c2).to_hex()

    # ---- bit OR-proof (atomic Bulletproofs building block) ---------------
    def generate_bit_proof(self, bit: int, blinding: int,
                           commitment_hex: str) -> dict:
        """
        OR-proof that the committed value is 0 or 1, without revealing
        which. Building block for full range proofs (v3.1).
        """
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1")
        c = pedersen.Commitment.from_hex(commitment_hex)
        proof = pedersen.prove_bit(int(bit), int(blinding), c)
        return proof.to_dict()

    def verify_bit_proof(self, commitment_hex: str, proof: dict) -> bool:
        try:
            c = pedersen.Commitment.from_hex(commitment_hex)
            # Reconstruct BitProof from dict (no from_dict on BitProof — do it inline)
            a0_b = bytes.fromhex(proof["a0"])
            a1_b = bytes.fromhex(proof["a1"])
            a0 = (int.from_bytes(a0_b[1:33], "big"), int.from_bytes(a0_b[33:65], "big"))
            a1 = (int.from_bytes(a1_b[1:33], "big"), int.from_bytes(a1_b[33:65], "big"))
            bp = pedersen.BitProof(
                a0=a0, a1=a1,
                e0=int(proof["e0"], 16), e1=int(proof["e1"], 16),
                z0=int(proof["z0"], 16), z1=int(proof["z1"], 16),
            )
        except Exception as e:
            logger.warning("verify_bit_proof: malformed input: %s", e)
            return False
        return pedersen.verify_bit(c, bp)


# Backwards-compat alias for the deprecated exception (in case any
# caller still catches it during the transition).
class PrivacyNotImplementedError(NotImplementedError):
    """Deprecated. Kept only so existing `except` clauses don't break."""
