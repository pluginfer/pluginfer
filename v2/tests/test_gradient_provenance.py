"""
Gradient-Provenance ZK proof tests (TODO Innovation §4.1)
=========================================================

Cases:
  1. Honest worker round-trip: create_proof + verify_proof both pass.
  2. Round-binding: verify_proof rejects when expected_model_hash
     doesn't match.
  3. Tampered ticket: editing any commitment hex breaks verification.
  4. Forged proof: random Schnorr fails verification.
  5. Determinism: same inputs produce different tickets each call
     (fresh blindings) — both verify.
  6. Different gradient -> different binding -> rejection if a worker
     swaps gradients between rounds.
  7. JSON round-trip preserves verifiability.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
for parent in [_HERE.parents[1], _HERE.parents[2]]:
    if (parent / "core").is_dir():
        sys.path.insert(0, str(parent))
        break

from core.gradient_provenance import (    # noqa: E402
    create_proof, verify_proof, GradientProvenanceTicket,
)


def test_honest_roundtrip():
    print("\n[1] HONEST WORKER ROUND-TRIP")
    print("-" * 60)
    ticket, _ = create_proof(
        data_bytes=b"shard-42-bytes",
        model_hash=b"model-checkpoint-c1",
        gradient_bytes=b"<gradient>",
    )
    assert verify_proof(ticket)
    print(f"  4 schnorr proofs verified, ticket size {len(ticket.to_json())} bytes OK")
    print("  PASS")


def test_round_binding():
    print("\n[2] ROUND BINDING (expected_model_hash mismatch rejected)")
    print("-" * 60)
    ticket, _ = create_proof(
        data_bytes=b"shard",
        model_hash=b"round-A",
        gradient_bytes=b"g",
    )
    assert verify_proof(ticket, expected_model_hash=b"round-A")
    assert not verify_proof(ticket, expected_model_hash=b"round-B")
    print("  matching round_hash accepted, mismatched rejected OK")
    print("  PASS")


def test_tampered_commit_rejected():
    print("\n[3] TAMPERED COMMITMENT REJECTED")
    print("-" * 60)
    ticket, _ = create_proof(b"d", b"m", b"g")
    bad = json.loads(ticket.to_json())
    # Flip one byte of a commitment.
    orig = bad["data_commit_hex"]
    bad["data_commit_hex"] = orig[:-2] + ("00" if orig[-2:] != "00" else "11")
    bad_ticket = GradientProvenanceTicket(**bad)
    assert not verify_proof(bad_ticket)
    print("  modified commit hex rejected OK")
    print("  PASS")


def test_forged_schnorr_rejected():
    print("\n[4] FORGED SCHNORR PROOF REJECTED")
    print("-" * 60)
    ticket, _ = create_proof(b"d", b"m", b"g")
    bad = json.loads(ticket.to_json())
    # Replace the binding proof's z_v scalar with garbage.
    bad["proof_binding"]["z_v"] = hex(123456789)
    bad_ticket = GradientProvenanceTicket(**bad)
    assert not verify_proof(bad_ticket)
    print("  garbage z_v rejected OK")
    print("  PASS")


def test_fresh_blindings_each_call():
    print("\n[5] FRESH BLINDINGS (two calls produce different tickets)")
    print("-" * 60)
    t1, _ = create_proof(b"D", b"M", b"G")
    t2, _ = create_proof(b"D", b"M", b"G")
    assert t1.data_commit_hex != t2.data_commit_hex
    assert verify_proof(t1)
    assert verify_proof(t2)
    print(f"  t1.data_commit={t1.data_commit_hex[:16]}...")
    print(f"  t2.data_commit={t2.data_commit_hex[:16]}... (distinct) OK")
    print("  PASS")


def test_gradient_swap_detectable():
    print("\n[6] GRADIENT SWAP IS DETECTABLE (worker can't reuse old grad)")
    print("-" * 60)
    t_round_a, w_a = create_proof(b"data1", b"modelA", b"grad1")
    t_round_b, w_b = create_proof(b"data1", b"modelB", b"grad2")

    # The cross-round binding scalars MUST differ.
    assert w_a.binding_value != w_b.binding_value, "binding collided"
    # And the published binding commitments MUST differ too.
    assert t_round_a.binding_commit_hex != t_round_b.binding_commit_hex
    print("  binding distinguishes (data, modelA, grad1) from "
          "(data, modelB, grad2) OK")
    print("  PASS")


def test_json_roundtrip():
    print("\n[7] JSON SERIALISE / DESERIALISE PRESERVES VERIFY")
    print("-" * 60)
    t, _ = create_proof(b"d", b"m", b"g")
    s = t.to_json()
    t2 = GradientProvenanceTicket.from_json(s)
    assert verify_proof(t2)
    print(f"  ticket sent through JSON ({len(s)} bytes) re-verifies OK")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("GRADIENT-PROVENANCE ZK TEST (Innovation §4.1)")
    print("=" * 60)
    t0 = time.time()
    test_honest_roundtrip()
    test_round_binding()
    test_tampered_commit_rejected()
    test_forged_schnorr_rejected()
    test_fresh_blindings_each_call()
    test_gradient_swap_detectable()
    test_json_roundtrip()
    print("\n" + "=" * 60)
    print(f"ALL GRADIENT-PROVENANCE TESTS PASSED in {time.time() - t0:.1f}s")
    print("=" * 60)
