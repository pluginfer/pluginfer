"""
ZKPrivacy -- Pedersen smoke test
================================
Proves the privacy facade is wired to real cryptography and the prior
"ZK_PROOF_MOCK_DATA_XP25" / NotImplementedError stubs are gone.

Cases:
  1. create_commitment + verify_commitment round-trip.
  2. verify_commitment with wrong value or wrong blinding fails.
  3. generate_proof + verify_proof round-trip (Schnorr PoK).
  4. verify_proof against a different commitment fails.
  5. Homomorphic add: commit(a) + commit(b) opens to a+b under
     blinding_a + blinding_b.
  6. Bit OR-proof: commits to 0 or 1 verify; proofs cross-check fails.
  7. ZKPrivacy.is_available() returns True (was False under the stub).
"""

from __future__ import annotations

import secrets
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
for parent in [_HERE.parents[1], _HERE.parents[2]]:
    if (parent / "core").is_dir():
        sys.path.insert(0, str(parent))
        break

from core.privacy import ZKPrivacy                                  # noqa: E402
from core import pedersen                                            # noqa: E402


def test_round_trip_commitment():
    print("\n[1] COMMITMENT ROUND-TRIP")
    print("-" * 60)
    zk = ZKPrivacy()
    value = 42
    c_hex, blinding = zk.create_commitment(value)
    assert len(c_hex) == 130, f"hex length is {len(c_hex)} not 130"
    assert zk.verify_commitment(c_hex, value, blinding), "open failed"
    print(f"  commitment len=130 OK  open ok OK")
    print("  PASS")


def test_commitment_rejects_wrong_open():
    print("\n[2] COMMITMENT REJECTS WRONG OPEN")
    print("-" * 60)
    zk = ZKPrivacy()
    c_hex, blinding = zk.create_commitment(100)
    assert not zk.verify_commitment(c_hex, 99, blinding), "wrong value passed!"
    assert not zk.verify_commitment(c_hex, 100, blinding ^ 1), "wrong blinding passed!"
    print("  wrong value rejected OK  wrong blinding rejected OK")
    print("  PASS")


def test_schnorr_pok():
    print("\n[3] SCHNORR PROOF-OF-KNOWLEDGE")
    print("-" * 60)
    zk = ZKPrivacy()
    value = 7
    c_hex, blinding = zk.create_commitment(value)
    proof = zk.generate_proof(value, blinding, c_hex)
    assert "T" in proof and "z_v" in proof and "z_r" in proof, f"missing fields: {proof.keys()}"
    assert zk.verify_proof(c_hex, proof), "valid proof rejected"
    print(f"  proof has T,z_v,z_r OK  verify ok OK")
    print("  PASS")


def test_proof_against_wrong_commitment_fails():
    print("\n[4] PROOF AGAINST WRONG COMMITMENT FAILS")
    print("-" * 60)
    zk = ZKPrivacy()
    c1_hex, b1 = zk.create_commitment(7)
    c2_hex, _b2 = zk.create_commitment(7)              # different blinding ⇒ different C
    proof = zk.generate_proof(7, b1, c1_hex)
    assert not zk.verify_proof(c2_hex, proof), "proof passed against wrong C"
    print("  proof for C1 rejected against C2 OK")
    print("  PASS")


def test_homomorphic_add():
    print("\n[5] HOMOMORPHIC ADD: commit(a)+commit(b)==commit(a+b)")
    print("-" * 60)
    zk = ZKPrivacy()
    a, b = 11, 31
    c_a_hex, r_a = zk.create_commitment(a)
    c_b_hex, r_b = zk.create_commitment(b)
    c_sum_hex = zk.add_commitments(c_a_hex, c_b_hex)
    # commit(a+b) under blinding (r_a + r_b) should match c_sum
    assert zk.verify_commitment(c_sum_hex, a + b, r_a + r_b), "homomorphism broken"
    print(f"  commit(11)+commit(31) opens to (42, r_a+r_b) OK")
    print("  PASS")


def test_bit_or_proof():
    print("\n[6] BIT OR-PROOF: v in {0,1}")
    print("-" * 60)
    zk = ZKPrivacy()
    for bit in (0, 1):
        c_hex, blinding = zk.create_commitment(bit)
        proof = zk.generate_bit_proof(bit, blinding, c_hex)
        assert zk.verify_bit_proof(c_hex, proof), f"valid bit-proof for {bit} rejected"
        print(f"  bit={bit}: prove OK  verify OK")
    # Cross-check: a proof for bit=0 against a commitment to 1 should fail
    c0_hex, r0 = zk.create_commitment(0)
    c1_hex, _r1 = zk.create_commitment(1)
    p_for_0 = zk.generate_bit_proof(0, r0, c0_hex)
    assert not zk.verify_bit_proof(c1_hex, p_for_0), "bit-proof for C0 passed against C1"
    print("  cross-check (proof for C0 against C1) rejected OK")
    print("  PASS")


def test_no_more_stub():
    print("\n[7] NO MORE STUB")
    print("-" * 60)
    assert ZKPrivacy.is_available() is True, "ZKPrivacy.AVAILABLE flag is still False"
    # The deprecated exception class still exists (for back-compat) but
    # ZKPrivacy methods must NOT raise it on the happy path.
    zk = ZKPrivacy()
    c_hex, _b = zk.create_commitment(0)
    assert isinstance(c_hex, str) and len(c_hex) == 130
    print("  is_available()==True OK  no NotImplementedError thrown OK")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("ZKPRIVACY -- PEDERSEN INTEGRATION TEST")
    print("=" * 60)
    t0 = time.time()
    test_round_trip_commitment()
    test_commitment_rejects_wrong_open()
    test_schnorr_pok()
    test_proof_against_wrong_commitment_fails()
    test_homomorphic_add()
    test_bit_or_proof()
    test_no_more_stub()
    print("\n" + "=" * 60)
    print(f"ALL PRIVACY TESTS PASSED in {time.time() - t0:.1f}s")
    print("=" * 60)
