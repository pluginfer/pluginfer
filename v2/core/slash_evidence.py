"""W32 BFT slash-evidence protocol.

What we close
-------------
Pre-this-module, `arbiter.slash_node` would log the intent of a
chain-wide slash but never produce a tx the chain would actually
honor. `compute_ledger.receive_remote_block` rejected ANY slash tx
with a warning ("W32 path not wired yet"). Validators could
equivocate (sign two conflicting blocks at the same height) without
losing a coin of their stake.

This module ships the actual protocol:

  1. **Detect** -- An arbiter (or any honest validator) observes two
     blocks at the same height, both signed by the same validator
     pubkey, with different block hashes. That's the equivocation
     primitive Tendermint / Casper FFG / GRANDPA all slash on.

  2. **Construct** -- Build a `SlashEvidence` carrying both signed
     block headers + the offender's pubkey. Optionally collect ≥2/3
     stake-weighted attestations from the validator set so the
     evidence is itself BFT-attested (defense against a malicious
     accuser fabricating "evidence" against an honest validator).

  3. **Verify** -- The chain layer (and any peer receiving a slash
     block) re-verifies:
       * Both block headers carry valid ECDSA signatures from the
         offender's pubkey over the canonical header.
       * The two block hashes differ.
       * The two block heights are equal.
       * If attestations are required, ≥2/3 of the validator set
         (by stake weight) signed the evidence body.
     Any failure rejects the SlashTx outright.

  4. **Apply** -- The chain executes a SlashTx that:
       * Zeros the offender's stake in `StakingContract`.
       * Adds the offender to a permanent `_slashed` set.
       * Locks the offender's address from new transfers for an
         `UNBONDING_PERIOD_BLOCKS` window so the offender can't
         immediately drain their balance to a fresh address.

What's deliberately NOT here
----------------------------
The actual evidence-broadcast layer (gossip the SlashEvidence to all
validators, collect attestations, mine the SlashTx into the next
block). That's a coordination protocol between validators that
needs the live BFT consensus surface to be running. The unit tests
in this module construct the evidence directly + drive the verifier
through to the apply step end-to-end; the broadcast wire-up plugs in
once a real validator quorum is online.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# How long a slashed address is locked from outbound transfers /
# unstakes. Blocks (not seconds) so the protocol is independent of
# wall-clock drift. 100 blocks ~= 5-10 minutes at our target block time.
UNBONDING_PERIOD_BLOCKS: int = 100

# Quorum threshold for attestations. ≥2/3 of validator stake.
ATTEST_QUORUM_NUMERATOR: int = 2
ATTEST_QUORUM_DENOMINATOR: int = 3


# ---------------------------------------------------------------------------
# Evidence dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BlockHeaderProof:
    """A block header + the validator's signature over it.

    The header dict MUST contain enough fields to fully identify the
    block (height, hash, parent_hash, timestamp). Sign over the
    canonical JSON of the header, NOT the raw block (we don't want
    the signature to depend on transaction bytes -- equivocation is
    about commitment to height+hash, not about which txs were inside).
    """
    header: Dict[str, Any]
    validator_pubkey_pem: str
    signature_b64: str

    def canonical(self) -> str:
        return json.dumps(self.header, sort_keys=True, default=str)

    def verify(self) -> bool:
        try:
            from .tokenomics import Wallet
            return Wallet.verify(
                self.validator_pubkey_pem, self.canonical(), self.signature_b64,
            )
        except Exception:
            return False


@dataclass
class Attestation:
    """One validator's signature confirming a SlashEvidence body is
    legitimate. Collected to prove ≥2/3 stake weight backs the slash."""
    validator_pubkey_pem: str
    signature_b64: str

    def verify(self, evidence_canonical: str) -> bool:
        try:
            from .tokenomics import Wallet
            return Wallet.verify(
                self.validator_pubkey_pem, evidence_canonical,
                self.signature_b64,
            )
        except Exception:
            return False


@dataclass
class SlashEvidence:
    """Equivocation evidence: same validator signed two distinct
    blocks at the same height."""
    offender_pubkey_pem: str
    block_a: BlockHeaderProof
    block_b: BlockHeaderProof
    height: int
    attestations: List[Attestation] = field(default_factory=list)

    def canonical_body(self) -> str:
        """The bytes attestors sign. Excludes attestations themselves
        (otherwise the body would mutate as attestations are added)."""
        body = {
            "offender_pubkey_pem": self.offender_pubkey_pem,
            "height": self.height,
            "block_a_header": self.block_a.header,
            "block_a_sig": self.block_a.signature_b64,
            "block_b_header": self.block_b.header,
            "block_b_sig": self.block_b.signature_b64,
        }
        return json.dumps(body, sort_keys=True, default=str)

    def fingerprint(self) -> str:
        """Stable id for the offence. Used as the SlashTx tx_id seed
        and as a dedup key so the same equivocation can't be slashed
        twice."""
        return hashlib.sha256(
            f"slash|{self.offender_pubkey_pem}|{self.height}|"
            f"{self.block_a.header.get('hash')}|"
            f"{self.block_b.header.get('hash')}".encode()
        ).hexdigest()


# ---------------------------------------------------------------------------
# Construction + verification
# ---------------------------------------------------------------------------


class SlashEvidenceError(ValueError):
    pass


def construct_evidence(
    *,
    offender_pubkey_pem: str,
    block_a_header: Dict[str, Any], block_a_sig_b64: str,
    block_b_header: Dict[str, Any], block_b_sig_b64: str,
) -> SlashEvidence:
    """Build a `SlashEvidence` and validate the basic invariants
    (heights match, hashes differ, both signatures verify)."""
    if int(block_a_header.get("index", -1)) != int(block_b_header.get("index", -2)):
        raise SlashEvidenceError(
            "block heights differ -- not an equivocation"
        )
    if block_a_header.get("hash") == block_b_header.get("hash"):
        raise SlashEvidenceError(
            "block hashes are identical -- not an equivocation"
        )
    height = int(block_a_header["index"])
    a = BlockHeaderProof(
        header=block_a_header,
        validator_pubkey_pem=offender_pubkey_pem,
        signature_b64=block_a_sig_b64,
    )
    b = BlockHeaderProof(
        header=block_b_header,
        validator_pubkey_pem=offender_pubkey_pem,
        signature_b64=block_b_sig_b64,
    )
    if not a.verify():
        raise SlashEvidenceError("block_a signature does not verify")
    if not b.verify():
        raise SlashEvidenceError("block_b signature does not verify")
    return SlashEvidence(
        offender_pubkey_pem=offender_pubkey_pem,
        block_a=a, block_b=b, height=height,
    )


def attest(
    evidence: SlashEvidence,
    *,
    validator_pubkey_pem: str,
    sign_fn,
) -> Attestation:
    """A validator signs the evidence body to attest it observed the
    equivocation. `sign_fn` takes a string and returns base64 sig."""
    sig = sign_fn(evidence.canonical_body())
    return Attestation(
        validator_pubkey_pem=validator_pubkey_pem,
        signature_b64=sig,
    )


def verify_evidence(
    evidence: SlashEvidence,
    *,
    validator_stakes: Optional[Dict[str, Decimal]] = None,
    require_attestation_quorum: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Re-run every check the chain will run before honoring a slash.

    `validator_stakes` -- {pubkey_pem: Decimal stake}. Required if
    `require_attestation_quorum=True`. The function checks ≥2/3 of
    TOTAL stake is represented by valid attestations.

    Returns (ok, reason). On False, `reason` is a short string the
    chain logs as the rejection cause.
    """
    if int(evidence.block_a.header.get("index", -1)) != \
            int(evidence.block_b.header.get("index", -2)):
        return False, "height_mismatch"
    if evidence.block_a.header.get("hash") == \
            evidence.block_b.header.get("hash"):
        return False, "blocks_identical"
    if not evidence.block_a.verify():
        return False, "block_a_sig_bad"
    if not evidence.block_b.verify():
        return False, "block_b_sig_bad"
    if (evidence.block_a.validator_pubkey_pem
            != evidence.offender_pubkey_pem
            or evidence.block_b.validator_pubkey_pem
            != evidence.offender_pubkey_pem):
        return False, "block_pubkey_mismatch"

    if not require_attestation_quorum:
        return True, None

    if not validator_stakes:
        return False, "validator_stakes_required"

    canon = evidence.canonical_body()
    seen_attestors: set[str] = set()
    attest_stake = Decimal("0")
    for a in evidence.attestations:
        if a.validator_pubkey_pem in seen_attestors:
            continue
        if a.validator_pubkey_pem not in validator_stakes:
            continue
        if not a.verify(canon):
            continue
        seen_attestors.add(a.validator_pubkey_pem)
        attest_stake += validator_stakes[a.validator_pubkey_pem]

    total_stake = sum(validator_stakes.values(), Decimal("0"))
    if total_stake <= 0:
        return False, "no_validator_stake"
    threshold = (total_stake * Decimal(ATTEST_QUORUM_NUMERATOR)
                 / Decimal(ATTEST_QUORUM_DENOMINATOR))
    if attest_stake < threshold:
        return False, (
            f"attestation_below_quorum "
            f"({attest_stake}/{total_stake} < {ATTEST_QUORUM_NUMERATOR}/"
            f"{ATTEST_QUORUM_DENOMINATOR})"
        )
    return True, None


# ---------------------------------------------------------------------------
# Apply the slash to live state
# ---------------------------------------------------------------------------


def apply_slash(
    *,
    evidence: SlashEvidence,
    staking_contract,
    ledger,
) -> Dict[str, Any]:
    """Zero the offender's stake; add to ledger blacklist; record the
    unbonding lock. Returns a dict the caller can include as the
    payload of a SlashTx for chain inclusion."""
    # Resolve the offender's address from their pubkey. Same scheme
    # the rest of the chain uses (Wallet.generate_address).
    from .tokenomics import Wallet
    offender_addr = _address_from_pubkey_pem(evidence.offender_pubkey_pem)

    slashed_amount = Decimal("0")
    if hasattr(staking_contract, "stakes"):
        info = staking_contract.stakes.get(offender_addr)
        if info is not None:
            slashed_amount = Decimal(str(info.get("amount", 0)))
            # Zero the entry; never delete it (we want a record that
            # this address WAS staked + WAS slashed).
            info["amount"] = "0"
            info["slashed"] = True
            info["slashed_at_block"] = ledger_height(ledger)
    if hasattr(staking_contract, "_slashed"):
        staking_contract._slashed.add(offender_addr)
    else:
        try:
            staking_contract._slashed = {offender_addr}
        except Exception:
            pass
    if hasattr(staking_contract, "_unbonding_locks"):
        staking_contract._unbonding_locks[offender_addr] = (
            ledger_height(ledger) + UNBONDING_PERIOD_BLOCKS
        )
    else:
        try:
            staking_contract._unbonding_locks = {
                offender_addr: ledger_height(ledger) + UNBONDING_PERIOD_BLOCKS
            }
        except Exception:
            pass
    if hasattr(ledger, "add_to_blacklist"):
        ledger.add_to_blacklist(offender_addr)

    return {
        "fingerprint": evidence.fingerprint(),
        "offender_addr": offender_addr,
        "offender_pubkey_pem": evidence.offender_pubkey_pem,
        "height_of_offence": evidence.height,
        "slashed_stake": str(slashed_amount),
        "unbonding_lock_until_block": (
            ledger_height(ledger) + UNBONDING_PERIOD_BLOCKS
        ),
    }


def _address_from_pubkey_pem(pubkey_pem: str) -> str:
    from cryptography.hazmat.primitives import serialization
    pub = serialization.load_pem_public_key(pubkey_pem.encode())
    der = pub.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    addr = "PLG" + hashlib.sha256(der).hexdigest()[:40]
    return addr


def ledger_height(ledger) -> int:
    if hasattr(ledger, "get_height"):
        try:
            return int(ledger.get_height())
        except Exception:
            pass
    chain = getattr(ledger, "chain", None)
    if chain:
        return max(0, len(chain) - 1)
    return 0


# ---------------------------------------------------------------------------
# Helpers for transfer-block enforcement
# ---------------------------------------------------------------------------


def is_outbound_locked(
    *, sender_address: str, staking_contract, ledger,
) -> bool:
    """True if `sender_address` is in an unbonding window and cannot
    transfer outbound. The mempool / receive_remote_block path can
    call this to refuse transfers from slashed addresses."""
    locks = getattr(staking_contract, "_unbonding_locks", None)
    if not locks:
        return False
    locked_until = locks.get(sender_address)
    if locked_until is None:
        return False
    return ledger_height(ledger) < int(locked_until)
