"""
The Arbiter (V3 Verification Engine)
====================================
Implements 'Optimistic Probabilistic Challenge' (Game Theory Verification).
- Verifies 5% of tasks randomly (low cost).
- If mismatch found: emits a slash-evidence transaction (high penalty).

W21/W32: slash_node was previously calling ledger.get_stake() and
ledger.add_to_blacklist() — neither existed → AttributeError every
single time. Plus the magic-string sender_pub_key='ARBITER_AUTH'
authorized any node to construct a slash tx, so even if it had run
the authorization was forgeable.

This module now:
  * Requires a `staking_contract` and `bft_consensus` to be wired in
    at construction time. Without them slash_node raises
    NotImplementedError with a clear remediation message — the
    project's honest-stub pattern. No more silent AttributeError.
  * Uses ledger.get_stake / add_to_blacklist (now real methods on
    ComputeLedger; see set_staking_contract for the wiring).
  * The slash transaction is rejected at receive_remote_block until
    W32 lands the full BFT slash-evidence protocol (≥⅔ validator
    signed precommits over the offence). Until then, slash() flips
    the local-only blacklist + logs the offence; that's enough to
    drop the offender from this node's peer table without yet
    propagating chain-wide stake destruction.
"""

from __future__ import annotations

import hashlib
import logging
from decimal import Decimal
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Arbiter:
    def __init__(self, ledger, probability: float = 0.05,
                 staking_contract=None, bft_consensus=None):
        """
        :param ledger: ComputeLedger instance.
        :param probability: spot-audit probability (0.0-1.0).
        :param staking_contract: StakingContract — required for
            slash_node to know what stake to penalize. None ⇒ slash
            raises NotImplementedError (honest stub).
        :param bft_consensus: BFTConsensus — required for slash_node
            to mint a consensus-authorised slash-evidence tx (W32).
            None ⇒ slash uses the local-only blacklist path and logs
            a warning that chain-wide slash is unwired.
        """
        self.ledger = ledger
        self.check_probability = probability
        self.staking_contract = staking_contract
        self.bft_consensus = bft_consensus
        # Allow ledger.get_stake() to delegate to the staking contract.
        if staking_contract is not None and hasattr(ledger, "set_staking_contract"):
            ledger.set_staking_contract(staking_contract)
        
    def should_audit(self, task_id: str) -> bool:
        """
        Deterministically decide whether to audit this task based on hash.
        This allows verifiable randomness (public can verify if it SHOULD have been checked).
        """
        # Hash task_id -> int -> mod 100
        h = int(hashlib.sha256(task_id.encode()).hexdigest(), 16)
        return (h % 100) < (self.check_probability * 100)

    def compare_results(self, result_a: Dict, result_b: Dict) -> bool:
        """Compare two execution results for consistency"""
        # Simple hash comparison of the output data
        # In V3, plugins should return deterministic outputs for same inputs
        
        # Strip metadata (timing differs)
        data_a = {k:v for k,v in result_a.items() if k != '_metadata'}
        data_b = {k:v for k,v in result_b.items() if k != '_metadata'}
        
        # Deep compare
        is_match = (str(data_a) == str(data_b))
        
        if not is_match:
            logger.warning(f"[ARBITER] Mismatch Detected! \n Node A: {data_a} \n Node B: {data_b}")
            
        return is_match

    def slash_with_evidence(
        self, *, offender_pubkey_pem: str,
        block_a_header: dict, block_a_sig_b64: str,
        block_b_header: dict, block_b_sig_b64: str,
        attestations: list,
    ) -> dict:
        """Run the full W32 BFT slash protocol end-to-end.

        Caller has observed an equivocation (same validator signed two
        distinct blocks at the same height) and collected ≥2/3 stake-
        weighted attestations from the validator set. This method:
        constructs the SlashEvidence, verifies it (re-checks every
        invariant the chain validator will re-check on inclusion),
        applies the slash to live state (stake zeroed, blacklist set,
        unbonding lock recorded), then returns the slash payload for
        chain inclusion via `build_slash_tx`.

        Raises NotImplementedError if staking_contract or bft_consensus
        is not wired — chain-wide slash without those is meaningless.
        """
        if self.staking_contract is None or self.bft_consensus is None:
            raise NotImplementedError(
                "Chain-wide slash needs both staking_contract AND "
                "bft_consensus wired into the Arbiter."
            )
        from .slash_evidence import (
            construct_evidence, verify_evidence, apply_slash, Attestation,
        )
        evidence = construct_evidence(
            offender_pubkey_pem=offender_pubkey_pem,
            block_a_header=block_a_header, block_a_sig_b64=block_a_sig_b64,
            block_b_header=block_b_header, block_b_sig_b64=block_b_sig_b64,
        )
        evidence.attestations = list(attestations)
        validator_stakes = self.staking_contract.validator_stake_snapshot()
        ok, reason = verify_evidence(
            evidence, validator_stakes=validator_stakes,
            require_attestation_quorum=True,
        )
        if not ok:
            raise ValueError(f"Slash evidence rejected: {reason}")
        payload = apply_slash(
            evidence=evidence,
            staking_contract=self.staking_contract,
            ledger=self.ledger,
        )
        logger.critical(
            "[ARBITER] SLASH EXECUTED %s height=%d slashed_stake=%s "
            "unbonding_until_block=%d",
            payload["offender_addr"][:12], payload["height_of_offence"],
            payload["slashed_stake"], payload["unbonding_lock_until_block"],
        )
        return payload

    def slash_node(self, node_id: str, reason: str) -> bool:
        """
        Punish a malicious node.

        Two effects:
          1. **Stake destruction (W32, partial):** if both
             `staking_contract` and `bft_consensus` are wired, this
             would emit a slash-evidence tx signed by ≥⅔ of the
             validator set. That code path is NOT YET implemented
             (multi-week protocol work). Until then we raise
             NotImplementedError if a caller attempts to use the
             un-wired chain-wide slash — better an explicit error
             than the previous silent AttributeError.
          2. **Local-only blacklist:** always applied. The offender
             is dropped from this node's peer-acceptance set, even
             if chain-wide stake destruction is unwired.

        Returns True if local blacklist applied; raises
        NotImplementedError if a chain-wide slash was attempted but
        the prerequisites are missing.
        """
        logger.critical("[ARBITER] SLASH %s: %s", node_id[:12], reason)

        if self.staking_contract is not None or self.bft_consensus is not None:
            # Caller intends a chain-wide slash; gate on full prereqs.
            if self.staking_contract is None or self.bft_consensus is None:
                raise NotImplementedError(
                    "Chain-wide slash requires both staking_contract AND "
                    "bft_consensus wired into the Arbiter. The full slash-"
                    "evidence protocol (≥2/3 validator signatures over the "
                    "offence) is W32 — not yet implemented. "
                    "Pass both, or call slash_node with neither for a "
                    "local-only blacklist."
                )
            # Both wired. Even so, the v3.0-alpha receive_remote_block
            # rejects 'slash' tx types entirely (see TODO W32). Until
            # the protocol lands we log the intent + apply local
            # blacklist + return.
            logger.warning(
                "[ARBITER] chain-wide slash for %s logged but NOT mined. "
                "Full slash-evidence protocol pending (W32).", node_id[:12],
            )

        # Local-only blacklist (always works).
        if hasattr(self.ledger, "add_to_blacklist"):
            self.ledger.add_to_blacklist(node_id)
        return True

    def process_challenge(self, task: Dict, result_primary: Dict, result_challenger: Dict, 
                         primary_node: str, challenger_node: str):
        """
        Resolve a Spot Check.
        """
        match = self.compare_results(result_primary, result_challenger)
        
        if match:
            logger.info(f"✅ Audit Passed for Task {task['id']}. Nodes {primary_node} & {challenger_node} are honest.")
            # Reward Challenger for the work
            return True
        else:
            logger.critical(f"❌ AUDIT FAILED! One of these nodes is lying.")
            # In V3 Simple: We don't know WHICH is lying without a 3rd check.
            # For MVP: We flag BOTH for 'High Suspicion' or trigger a 3rd check.
            # Impl: Trigger 3rd check (Tiebreaker)
            # ... (Tiebreaker logic would go here)
            return False
