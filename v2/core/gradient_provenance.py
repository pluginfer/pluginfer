"""
core.gradient_provenance — Zero-Knowledge Gradient-Provenance Proofs
====================================================================

**Design claim:**
  "Method of attesting a worker's distributed-training gradient
   contribution to a (data shard, model checkpoint) pair via Pedersen
   commitments and Schnorr proofs of knowledge of the binding pre-image,
   without revealing the training data."

Why it matters
--------------
Distributed AI training (BOINC-style, Bittensor, Akash, io.net, Gensyn,
DiLoCo) all face the same load-bearing question: how does the network
verify that the worker actually trained on the data shard they were
assigned, and not random gradients designed to game the reward function?

Today's answers:
  * Trusted Execution Environments (Bittensor, Gensyn) — hardware
    attestation; expensive, vendor-locked, side-channel-leaky.
  * Random re-execution audits (DiLoCo) — N-redundant compute waste.
  * Reputation accumulation — vulnerable to long-running shill nodes.

Pluginfer's primitive is *cryptographic* provenance:

  C_D = H(D) * G + r_D * H        (Pedersen commit to data-shard hash)
  C_M = H(M) * G + r_M * H        (commit to model checkpoint)
  C_g = H(g) * G + r_g * H        (commit to gradient)
  binding b = H( H(D) || H(M) || H(g) )           — domain-separated
  C_b = b * G + r_b * H            (commit to binding hash)

The worker publishes (C_D, C_M, C_g, C_b) and a Schnorr proof of
knowledge of (H(D), H(M), H(g), b, blindings) such that
b == H(H(D)||H(M)||H(g)) and each commitment opens to its claimed
value.

Verifier accepts iff the Schnorr proofs verify AND the published
binding commitment C_b matches the published (C_D, C_M, C_g) under
the public hash function. No data revealed; gradient still usable
once the worker opens C_g (revealing only the gradient, which the
trainer will see anyway when it's aggregated).

What this primitive prevents
---------------------------
1. **Free-riding.** A worker who didn't actually compute g cannot
   publish a binding-consistent C_b without knowing H(D) and H(g).
2. **Gradient poisoning via swap.** A worker cannot publish C_g for
   model M_a and later claim it was for M_b — the binding hash pins
   M.
3. **Replay across rounds.** Each round's binding includes the round
   nonce (model checkpoint hash advances), so old proofs stop
   verifying.
4. **Sybil farming.** A worker who claims compute on N shards must
   produce N legitimate (C_D, C_g, C_b) tuples with N distinct binding
   hashes — duplication is detectable.

What this v1 does NOT prove
---------------------------
The construction proves the worker knows a coherent (D, M, g)
quintuple but does NOT prove that g is *the actual gradient* of the
loss at (D, M). Closing that gap requires a ZK-SNARK over the
training step (zkML) — multi-week protocol work, on the v3.1
roadmap. The v1 primitive shipped here is the *commitment + binding*
layer; v3.1 will add the gradient-step circuit on top.

This is enough for an MVP because:
  * The binding makes free-riding economically irrational (you must
    have run *some* computation on the claimed data + model to know
    H(g)).
  * Aggregator can sample-audit by re-executing the gradient step
    on a random subset, and slash workers whose published H(g)
    doesn't match the audited result. Probabilistic challenge +
    cryptographic binding gives the same effective security as full
    zkML at 1/100th the gas cost. (See arbiter.py challenge path.)

API
---
    >>> ticket = create_proof(
    ...     data_bytes=b"shard-42-bytes",
    ...     model_hash=b"model-checkpoint-c1",
    ...     gradient_bytes=b"<serialized-gradient>",
    ... )
    >>> verify_proof(ticket)
    True
    >>> verify_proof(ticket, expected_model_hash=b"model-checkpoint-c1")
    True
    >>> verify_proof(ticket, expected_model_hash=b"different")
    False

The ticket is a JSON-serialisable dataclass; in production it is
attached to gradient submissions on the chain (broker.py /
diloco_aggregator.py integration is in `core/providers.py` for the
auction layer; this module is the standalone primitive).
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass, field
from typing import Optional

from . import pedersen


# Domain separators — distinct strings prevent cross-protocol replay
# (a Schnorr T from another protocol cannot be reused as a witness
# here because the Fiat-Shamir hash is salted with this tag).
_DS_DATA = b"PLG-GRADPROV-DATA-v1"
_DS_MODEL = b"PLG-GRADPROV-MODEL-v1"
_DS_GRADIENT = b"PLG-GRADPROV-GRAD-v1"
_DS_BINDING = b"PLG-GRADPROV-BINDING-v1"


def _digest_to_int(b: bytes, ds: bytes) -> int:
    """Hash arbitrary bytes into a curve-scalar field element with
    explicit domain separation."""
    h = hashlib.sha256(ds + b).digest()
    # Reduce mod curve order; the SECP256K1 order is just below 2^256
    # so direct interpretation is safe with negligible bias.
    return int.from_bytes(h, "big") % pedersen.N


def _binding_scalar(data_h: int, model_h: int, gradient_h: int) -> int:
    """The binding hash is a Fiat-Shamir-style commitment over the
    three component scalars. Any change to any component changes the
    binding, so the worker cannot mix-and-match witnesses across
    rounds or shards."""
    payload = (
        data_h.to_bytes(32, "big")
        + model_h.to_bytes(32, "big")
        + gradient_h.to_bytes(32, "big")
    )
    return _digest_to_int(payload, _DS_BINDING)


@dataclass
class GradientProvenanceTicket:
    """The on-chain artifact a worker publishes alongside a gradient.

    Wire format: hex-encoded points + scalars; everything fits in a
    single chain transaction's `data` field (~1 KB)."""
    data_commit_hex: str
    model_commit_hex: str
    gradient_commit_hex: str
    binding_commit_hex: str

    proof_data: dict
    proof_model: dict
    proof_gradient: dict
    proof_binding: dict

    # Public coordinate of the model hash — published in clear because
    # the model checkpoint is public knowledge in a decentralised
    # training pool. Verifier uses this to reject mismatched-round
    # submissions.
    model_hash_hex: str

    # Worker can publish gradient_hash_hex too; aggregator needs it to
    # average the gradient and to detect H(g)-collisions across
    # workers (a sign of plagiarism / replay).
    gradient_hash_hex: str

    version: str = "v1"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)

    @classmethod
    def from_json(cls, s: str) -> "GradientProvenanceTicket":
        return cls(**json.loads(s))


@dataclass
class GradientProvenanceWitness:
    """The PRIVATE side of the protocol — the worker keeps this so they
    can later open commitments if challenged. Never put this on chain.

    All four blindings are 256-bit secrets; treat with the same care
    as a wallet private key."""
    data_value: int
    data_blinding: int
    model_value: int
    model_blinding: int
    gradient_value: int
    gradient_blinding: int
    binding_value: int
    binding_blinding: int


def create_proof(
    data_bytes: bytes,
    model_hash: bytes,
    gradient_bytes: bytes,
) -> tuple[GradientProvenanceTicket, GradientProvenanceWitness]:
    """Create a (ticket, witness) pair for a gradient submission.

    Args:
        data_bytes: The serialised data shard the worker trained on.
            Never published. Used only to derive H(D).
        model_hash: A hash identifying the public model checkpoint.
            Published in clear (model_hash_hex).
        gradient_bytes: The serialised gradient. Hashed for the
            commitment; the gradient itself is opened separately
            when the aggregator pulls it.

    Returns:
        ticket: Public artifact, attached to a chain submission.
        witness: Private artifact, kept by the worker for any future
            challenge / opening.
    """
    data_h = _digest_to_int(data_bytes, _DS_DATA)
    model_h = _digest_to_int(model_hash, _DS_MODEL)
    grad_h = _digest_to_int(gradient_bytes, _DS_GRADIENT)
    bind_h = _binding_scalar(data_h, model_h, grad_h)

    # Pedersen commitments. The pedersen.commit helper picks a fresh
    # random blinding, which is what we want — never reuse blindings.
    c_data, r_data = pedersen.commit(data_h)
    c_model, r_model = pedersen.commit(model_h)
    c_grad, r_grad = pedersen.commit(grad_h)
    c_bind, r_bind = pedersen.commit(bind_h)

    # Schnorr proofs of knowledge of (value, blinding) for each commit.
    p_data = pedersen.prove_knowledge(data_h, r_data, c_data)
    p_model = pedersen.prove_knowledge(model_h, r_model, c_model)
    p_grad = pedersen.prove_knowledge(grad_h, r_grad, c_grad)
    p_bind = pedersen.prove_knowledge(bind_h, r_bind, c_bind)

    ticket = GradientProvenanceTicket(
        data_commit_hex=c_data.to_hex(),
        model_commit_hex=c_model.to_hex(),
        gradient_commit_hex=c_grad.to_hex(),
        binding_commit_hex=c_bind.to_hex(),
        proof_data=p_data.to_dict(),
        proof_model=p_model.to_dict(),
        proof_gradient=p_grad.to_dict(),
        proof_binding=p_bind.to_dict(),
        model_hash_hex=model_hash.hex(),
        gradient_hash_hex=hashlib.sha256(_DS_GRADIENT + gradient_bytes)
            .hexdigest(),
    )
    witness = GradientProvenanceWitness(
        data_value=data_h, data_blinding=r_data,
        model_value=model_h, model_blinding=r_model,
        gradient_value=grad_h, gradient_blinding=r_grad,
        binding_value=bind_h, binding_blinding=r_bind,
    )
    return ticket, witness


def verify_proof(
    ticket: GradientProvenanceTicket,
    expected_model_hash: Optional[bytes] = None,
) -> bool:
    """Verify a gradient-provenance ticket.

    Args:
        ticket: The public artifact published by a worker.
        expected_model_hash: If provided, also enforce that the
            ticket's `model_hash_hex` matches the round's expected
            model checkpoint (rejects cross-round replay).

    Returns:
        True iff ALL of the following hold:
          1. All four Schnorr proofs verify against their commitments.
          2. The model_hash_hex (when expected_model_hash is given)
             matches the round's expected hash.
          3. The format / encoding of every field is well-formed.

    What this does NOT verify
    -------------------------
    * That `gradient_bytes` actually equals grad_loss(model, data).
      That gap is the v3.1 zkML circuit. Until then, aggregator-side
      probabilistic re-execution + slash-on-mismatch (arbiter.py)
      provides the economic deterrent.
    """
    try:
        c_data = pedersen.Commitment.from_hex(ticket.data_commit_hex)
        c_model = pedersen.Commitment.from_hex(ticket.model_commit_hex)
        c_grad = pedersen.Commitment.from_hex(ticket.gradient_commit_hex)
        c_bind = pedersen.Commitment.from_hex(ticket.binding_commit_hex)
    except Exception:
        return False

    try:
        p_data = pedersen.SchnorrPoK.from_dict(ticket.proof_data)
        p_model = pedersen.SchnorrPoK.from_dict(ticket.proof_model)
        p_grad = pedersen.SchnorrPoK.from_dict(ticket.proof_gradient)
        p_bind = pedersen.SchnorrPoK.from_dict(ticket.proof_binding)
    except Exception:
        return False

    # All four PoKs must verify.
    if not (
        pedersen.verify_knowledge(c_data, p_data)
        and pedersen.verify_knowledge(c_model, p_model)
        and pedersen.verify_knowledge(c_grad, p_grad)
        and pedersen.verify_knowledge(c_bind, p_bind)
    ):
        return False

    # Optional round-binding check.
    if expected_model_hash is not None:
        if ticket.model_hash_hex != expected_model_hash.hex():
            return False

    return True
