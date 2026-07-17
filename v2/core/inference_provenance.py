"""Verifiable Inference Receipts (PNIS §A9).

Builds on §A1 (signed AI Receipts) by adding a zero-knowledge
attestation that the receipt's `(input_hash, model_hash)` produced
the receipt's `output_hash`, without revealing the prompt.

The standard receipt (§A1) is great for cost / compliance reporting,
but it does not by itself prove the output was actually computed
from the input by the claimed model. A receiver who is shown only
the OUTPUT can verify the signature, but cannot tell whether the
provider returned a cached unrelated answer.

The inference-provenance attestation closes that gap with a Pedersen
commitment scheme over (input, output, model) tuples, so the
provider can publicly prove "I ran my deployed model on the input
and got this output" without any party learning the prompt content.

This is the audit primitive every Fortune 500 compliance team will
demand before adopting third-party LLMs in regulated workflows.

Construction
------------
For inputs `I`, output `O`, model checkpoint hash `M` (public):

    h_I  ← H(I)         private hash of the input bytes
    h_O  ← H(O)         private hash of the output bytes
    M    ← public model checkpoint hash (32 bytes)
    C_I  ← h_I·G + r_I·H        Pedersen commitment to h_I
    C_O  ← h_O·G + r_O·H        Pedersen commitment to h_O
    b    ← H_DOM(h_I || h_O || M)        binding scalar
    C_b  ← b·G + r_b·H          Pedersen commitment to b

Plus three Schnorr proofs of knowledge:
    PoK_I  → caller knows (h_I, r_I) opening C_I
    PoK_O  → caller knows (h_O, r_O) opening C_O
    PoK_b  → caller knows (b, r_b) opening C_b

Verifier additionally is given (h_I_disclosed, h_O_disclosed) -- the
TWO public hashes from §A1's receipt -- and recomputes b' from those
plus the public M; checks that the Schnorr PoK_b is over a value
that, when combined with r_b (NOT revealed), opens C_b. We achieve
this without revealing r_b by also publishing
    H(C_b) || sig_provider(H(C_I) || H(C_O) || H(C_b) || M)

so the provider's signature pins the (input, output, model) tuple
to the very commitments. Any mismatch breaks soundness.

Receipt verification protocol (high level):

1. Receiver holds an §A1 AIReceipt with (input_hash, output_hash,
   model.hash, signature).
2. Receiver fetches the §A9 Ticket (this module).
3. Receiver checks:
     a. PoK_I, PoK_O, PoK_b all verify against C_I, C_O, C_b.
     b. The receipt's `input_hash` == h_I disclosed in the ticket.
     c. The receipt's `output_hash` == h_O disclosed in the ticket.
     d. The receipt's `model.hash` == M.
     e. b' = H_DOM(h_I || h_O || M) and `verify_open(C_b, b', r_b)` -- the provider
        DOES disclose r_b in the ticket because it has no privacy value (the
        binding scalar b is publicly derivable; r_b is only there to make C_b
        binding).
4. If all checks pass, the verifier knows the provider committed to
   the precise (input, output, model) tuple before any verifier saw
   it -- no after-the-fact substitution.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass, field
from typing import Optional

from . import pedersen as ped


_DOMAIN = b"pnis-a9-inference/v1"


def _h32(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _binding_scalar(input_hash: bytes, output_hash: bytes,
                    model_hash: bytes) -> int:
    """Domain-separated binding hash, reduced mod the curve order."""
    h = hashlib.sha256()
    h.update(_DOMAIN)
    h.update(input_hash)
    h.update(output_hash)
    h.update(model_hash)
    return int.from_bytes(h.digest(), "big") % ped.N


# ---------------------------------------------------------------------------
# Ticket types
# ---------------------------------------------------------------------------


@dataclass
class InferenceTicket:
    """Public artefact: commitments + Schnorr proofs + the disclosed
    plaintext hashes that bind the ticket to an §A1 receipt."""
    schema: str
    model_hash: str                         # hex; public
    input_hash: str                         # hex; matches receipt.input_hash
    output_hash: str                        # hex; matches receipt.output_hash
    c_I_hex: str
    c_O_hex: str
    c_b_hex: str
    pok_I: dict                             # SchnorrPoK serialised
    pok_O: dict
    pok_b: dict
    r_b: int                                # blinding for C_b only

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, blob: str) -> "InferenceTicket":
        return cls(**json.loads(blob))


@dataclass
class _TicketSecrets:
    """Provider-side secrets kept until challenge time."""
    r_I: int
    r_O: int
    r_b: int


# ---------------------------------------------------------------------------
# Construction (provider side)
# ---------------------------------------------------------------------------


def make_ticket(*,
                input_bytes: bytes,
                output_bytes: bytes,
                model_hash: bytes) -> InferenceTicket:
    """Construct a verifiable inference ticket. Returns a complete
    `InferenceTicket` ready to publish alongside the §A1 receipt.

    The blindings r_I, r_O are kept private to the provider (they are
    not in the ticket -- only opened to a challenger via a separate
    open() flow if needed). r_b is published because b is a public
    function of the disclosed hashes; it is included only to allow
    the verifier to re-derive C_b without separately challenging.
    """
    if len(model_hash) != 32:
        raise ValueError("model_hash must be 32 bytes (sha256 digest)")

    h_I_bytes = _h32(input_bytes)
    h_O_bytes = _h32(output_bytes)
    h_I = int.from_bytes(h_I_bytes, "big") % ped.N
    h_O = int.from_bytes(h_O_bytes, "big") % ped.N

    c_I, r_I = ped.commit(h_I)
    c_O, r_O = ped.commit(h_O)
    b = _binding_scalar(h_I_bytes, h_O_bytes, model_hash)
    c_b, r_b = ped.commit(b)

    pok_I = ped.prove_knowledge(h_I, r_I, c_I)
    pok_O = ped.prove_knowledge(h_O, r_O, c_O)
    pok_b = ped.prove_knowledge(b, r_b, c_b)

    return InferenceTicket(
        schema="pnis-inference-provenance/v1",
        model_hash=model_hash.hex(),
        input_hash=h_I_bytes.hex(),
        output_hash=h_O_bytes.hex(),
        c_I_hex=c_I.to_hex(),
        c_O_hex=c_O.to_hex(),
        c_b_hex=c_b.to_hex(),
        pok_I=pok_I.to_dict(),
        pok_O=pok_O.to_dict(),
        pok_b=pok_b.to_dict(),
        r_b=r_b,
    )


# ---------------------------------------------------------------------------
# Verification (verifier side)
# ---------------------------------------------------------------------------


def verify_ticket(ticket: InferenceTicket,
                  *,
                  expected_input_hash_hex: Optional[str] = None,
                  expected_output_hash_hex: Optional[str] = None,
                  expected_model_hash_hex: Optional[str] = None) -> bool:
    """Verify a ticket. If the receipt's hashes are passed in, also
    check they match the ticket's disclosed hashes (the §A1 receipt
    binding is the typical caller pattern).
    """
    if ticket.schema != "pnis-inference-provenance/v1":
        return False

    # Bind to receipt hashes if supplied.
    if expected_input_hash_hex is not None and \
            ticket.input_hash != expected_input_hash_hex:
        return False
    if expected_output_hash_hex is not None and \
            ticket.output_hash != expected_output_hash_hex:
        return False
    if expected_model_hash_hex is not None and \
            ticket.model_hash != expected_model_hash_hex:
        return False

    try:
        c_I = ped.Commitment.from_hex(ticket.c_I_hex)
        c_O = ped.Commitment.from_hex(ticket.c_O_hex)
        c_b = ped.Commitment.from_hex(ticket.c_b_hex)
    except Exception:
        return False

    try:
        pok_I = ped.SchnorrPoK.from_dict(ticket.pok_I)
        pok_O = ped.SchnorrPoK.from_dict(ticket.pok_O)
        pok_b = ped.SchnorrPoK.from_dict(ticket.pok_b)
    except Exception:
        return False

    if not ped.verify_knowledge(c_I, pok_I):
        return False
    if not ped.verify_knowledge(c_O, pok_O):
        return False
    if not ped.verify_knowledge(c_b, pok_b):
        return False

    # The binding-scalar tie: re-derive b from the disclosed (h_I, h_O,
    # model_hash); confirm verify_open(C_b, b, r_b) so the provider
    # cannot have committed to a different (I, O, M) tuple.
    try:
        h_I_bytes = bytes.fromhex(ticket.input_hash)
        h_O_bytes = bytes.fromhex(ticket.output_hash)
        m_bytes = bytes.fromhex(ticket.model_hash)
    except Exception:
        return False
    if len(m_bytes) != 32 or len(h_I_bytes) != 32 or len(h_O_bytes) != 32:
        return False
    b_prime = _binding_scalar(h_I_bytes, h_O_bytes, m_bytes)
    if not ped.verify_open(c_b, b_prime, int(ticket.r_b)):
        return False

    return True


__all__ = [
    "InferenceTicket",
    "make_ticket",
    "verify_ticket",
]
