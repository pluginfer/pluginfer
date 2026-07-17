"""Tests for the W32 BFT slash-evidence protocol.

Constructs equivocation evidence from real signed block headers,
verifies the chain-side check, applies the slash to a real
StakingContract + ComputeLedger, and asserts the post-state
invariants (stake zeroed, address blacklisted, unbonding lock set).
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.compute_ledger import ComputeLedger  # noqa: E402
from core.slash_evidence import (  # noqa: E402
    UNBONDING_PERIOD_BLOCKS,
    Attestation,
    SlashEvidenceError,
    _address_from_pubkey_pem,
    apply_slash,
    attest,
    construct_evidence,
    is_outbound_locked,
    verify_evidence,
)
from core.staking import StakingContract, STAKING_POOL_ADDRESS  # noqa: E402
from core.tokenomics import Wallet  # noqa: E402


def _signed_header(wallet: Wallet, height: int, parent: str, payload: str) -> dict:
    """Build a block header dict + sign it with the validator wallet."""
    import hashlib
    h = {
        "index": height,
        "previous_hash": parent,
        "timestamp": 0.0,
        "txs_root": hashlib.sha256(payload.encode()).hexdigest(),
        "hash": hashlib.sha256(
            f"{height}|{parent}|{payload}".encode()
        ).hexdigest(),
    }
    canon = json.dumps(h, sort_keys=True, default=str)
    sig = wallet.sign(canon)
    return h, sig


def test_construct_rejects_non_equivocation():
    """If the two headers are at different heights, it's not
    equivocation -- different validators legitimately produce
    different blocks at different heights."""
    w = Wallet()
    h_a, sig_a = _signed_header(w, height=5, parent="0", payload="A")
    h_b, sig_b = _signed_header(w, height=6, parent="x", payload="B")
    with pytest.raises(SlashEvidenceError):
        construct_evidence(
            offender_pubkey_pem=w.public_key_pem,
            block_a_header=h_a, block_a_sig_b64=sig_a,
            block_b_header=h_b, block_b_sig_b64=sig_b,
        )


def test_construct_rejects_identical_blocks():
    w = Wallet()
    h, sig = _signed_header(w, height=5, parent="0", payload="A")
    with pytest.raises(SlashEvidenceError):
        construct_evidence(
            offender_pubkey_pem=w.public_key_pem,
            block_a_header=h, block_a_sig_b64=sig,
            block_b_header=h, block_b_sig_b64=sig,
        )


def test_construct_rejects_unsigned_block():
    w_real = Wallet()
    w_other = Wallet()
    h_a, sig_a = _signed_header(w_real, 5, "0", "A")
    # block_b "signed" by a different key but offender_pubkey is w_real --
    # the sig won't verify under w_real.
    h_b, _ = _signed_header(w_other, 5, "0", "B")
    bad_sig = w_other.sign(json.dumps(h_b, sort_keys=True, default=str))
    with pytest.raises(SlashEvidenceError):
        construct_evidence(
            offender_pubkey_pem=w_real.public_key_pem,
            block_a_header=h_a, block_a_sig_b64=sig_a,
            block_b_header=h_b, block_b_sig_b64=bad_sig,
        )


def test_verify_passes_with_two_thirds_attestations():
    """Build genuine evidence, collect attestations from validators
    who together control > 2/3 of the stake. Verifier accepts."""
    offender = Wallet()
    h_a, sig_a = _signed_header(offender, 10, "0", "A")
    h_b, sig_b = _signed_header(offender, 10, "0", "B")
    ev = construct_evidence(
        offender_pubkey_pem=offender.public_key_pem,
        block_a_header=h_a, block_a_sig_b64=sig_a,
        block_b_header=h_b, block_b_sig_b64=sig_b,
    )
    # 3 validators, 100 stake each, total 300; need 200 stake to pass.
    v1, v2, v3 = Wallet(), Wallet(), Wallet()
    stakes = {
        v1.public_key_pem: Decimal("100"),
        v2.public_key_pem: Decimal("100"),
        v3.public_key_pem: Decimal("100"),
    }
    ev.attestations = [
        attest(ev, validator_pubkey_pem=v1.public_key_pem, sign_fn=v1.sign),
        attest(ev, validator_pubkey_pem=v2.public_key_pem, sign_fn=v2.sign),
    ]
    ok, reason = verify_evidence(ev, validator_stakes=stakes,
                                 require_attestation_quorum=True)
    assert ok, reason


def test_verify_fails_below_quorum():
    offender = Wallet()
    h_a, sig_a = _signed_header(offender, 10, "0", "A")
    h_b, sig_b = _signed_header(offender, 10, "0", "B")
    ev = construct_evidence(
        offender_pubkey_pem=offender.public_key_pem,
        block_a_header=h_a, block_a_sig_b64=sig_a,
        block_b_header=h_b, block_b_sig_b64=sig_b,
    )
    v1, v2, v3 = Wallet(), Wallet(), Wallet()
    stakes = {
        v1.public_key_pem: Decimal("100"),
        v2.public_key_pem: Decimal("100"),
        v3.public_key_pem: Decimal("100"),
    }
    # Only one attestation -- 100/300 < 2/3.
    ev.attestations = [
        attest(ev, validator_pubkey_pem=v1.public_key_pem, sign_fn=v1.sign),
    ]
    ok, reason = verify_evidence(ev, validator_stakes=stakes,
                                 require_attestation_quorum=True)
    assert ok is False
    assert "below_quorum" in (reason or "")


def test_verify_rejects_forged_attestation_signature():
    """An attestation whose signature doesn't verify is silently
    dropped (not counted toward quorum). With only honest attestations
    below 2/3, verify must fail."""
    offender = Wallet()
    h_a, sig_a = _signed_header(offender, 10, "0", "A")
    h_b, sig_b = _signed_header(offender, 10, "0", "B")
    ev = construct_evidence(
        offender_pubkey_pem=offender.public_key_pem,
        block_a_header=h_a, block_a_sig_b64=sig_a,
        block_b_header=h_b, block_b_sig_b64=sig_b,
    )
    v1, v2 = Wallet(), Wallet()
    impostor = Wallet()
    stakes = {
        v1.public_key_pem: Decimal("100"),
        v2.public_key_pem: Decimal("100"),
        impostor.public_key_pem: Decimal("100"),
    }
    ev.attestations = [
        attest(ev, validator_pubkey_pem=v1.public_key_pem, sign_fn=v1.sign),
        # Forged attestation: claims to be v2 but signed by impostor.
        Attestation(
            validator_pubkey_pem=v2.public_key_pem,
            signature_b64=impostor.sign(ev.canonical_body()),
        ),
    ]
    ok, reason = verify_evidence(ev, validator_stakes=stakes)
    assert ok is False, reason


def test_apply_slash_zeros_stake_and_blacklists():
    """End-to-end: stake the offender, prove the equivocation, apply
    the slash, confirm the stake is zero + the address is in the
    blacklist + the unbonding lock is set."""
    offender = Wallet()
    ledger = ComputeLedger("slash-1")
    staking = StakingContract(ledger)
    offender_addr = _address_from_pubkey_pem(offender.public_key_pem)
    # Inject a stake directly so we have something to destroy.
    staking.stakes[offender_addr] = {
        "amount": "1000", "timestamp": 0, "last_block": 0,
    }

    h_a, sig_a = _signed_header(offender, 5, "0", "A")
    h_b, sig_b = _signed_header(offender, 5, "0", "B")
    ev = construct_evidence(
        offender_pubkey_pem=offender.public_key_pem,
        block_a_header=h_a, block_a_sig_b64=sig_a,
        block_b_header=h_b, block_b_sig_b64=sig_b,
    )

    payload = apply_slash(
        evidence=ev, staking_contract=staking, ledger=ledger,
    )
    assert payload["offender_addr"] == offender_addr
    assert Decimal(payload["slashed_stake"]) == Decimal("1000")
    assert staking.stakes[offender_addr]["amount"] == "0"
    assert staking.stakes[offender_addr]["slashed"] is True
    assert offender_addr in getattr(staking, "_slashed", set())
    # Blacklisted at the ledger level too.
    if hasattr(ledger, "blacklist"):
        assert offender_addr in getattr(ledger, "blacklist")


def test_unbonding_lock_blocks_outbound_until_period_elapses():
    offender = Wallet()
    ledger = ComputeLedger("slash-2")
    staking = StakingContract(ledger)
    offender_addr = _address_from_pubkey_pem(offender.public_key_pem)
    staking.stakes[offender_addr] = {"amount": "100", "timestamp": 0,
                                     "last_block": 0}

    h_a, sig_a = _signed_header(offender, 5, "0", "A")
    h_b, sig_b = _signed_header(offender, 5, "0", "B")
    ev = construct_evidence(
        offender_pubkey_pem=offender.public_key_pem,
        block_a_header=h_a, block_a_sig_b64=sig_a,
        block_b_header=h_b, block_b_sig_b64=sig_b,
    )
    apply_slash(evidence=ev, staking_contract=staking, ledger=ledger)
    assert is_outbound_locked(
        sender_address=offender_addr, staking_contract=staking, ledger=ledger,
    ) is True

    # Simulate UNBONDING_PERIOD_BLOCKS+1 fresh blocks elapsing. Each
    # mine_block requires a pending tx; we feed system coinbases so
    # the chain actually grows.
    from core.tokenomics import TokenMinter
    miner = Wallet()
    minter = TokenMinter(ledger=ledger)
    for _ in range(UNBONDING_PERIOD_BLOCKS + 1):
        coinbase = minter.mint_coinbase(miner.address, block_height=0,
                                        difficulty_factor=1.0)
        assert ledger.add_transaction(coinbase, _internal=True)
        ledger.mine_block(miner.address, difficulty=1)
    assert is_outbound_locked(
        sender_address=offender_addr, staking_contract=staking, ledger=ledger,
    ) is False


def test_evidence_fingerprint_is_deterministic():
    offender = Wallet()
    h_a, sig_a = _signed_header(offender, 7, "0", "A")
    h_b, sig_b = _signed_header(offender, 7, "0", "B")
    ev1 = construct_evidence(
        offender_pubkey_pem=offender.public_key_pem,
        block_a_header=h_a, block_a_sig_b64=sig_a,
        block_b_header=h_b, block_b_sig_b64=sig_b,
    )
    ev2 = construct_evidence(
        offender_pubkey_pem=offender.public_key_pem,
        block_a_header=h_a, block_a_sig_b64=sig_a,
        block_b_header=h_b, block_b_sig_b64=sig_b,
    )
    assert ev1.fingerprint() == ev2.fingerprint()
    assert len(ev1.fingerprint()) == 64           # sha256 hex
