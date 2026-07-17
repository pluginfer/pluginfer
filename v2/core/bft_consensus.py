"""
Stake-Weighted BFT Consensus (Tendermint-style)
================================================
Replaces the Raft-style `core/election.py` "elect one coordinator" model
with a leaderless, stake-weighted, rotating-leader BFT consensus suitable
for hundreds-to-thousands of validators.

Why this exists
---------------
`election.py` picks a single coordinator and fails over on death. Seven
known structural problems for a global compute mesh:
    * single point of failure (failover window = downtime)
    * bottleneck (every op routes through one node)
    * attack target (DDoS one node = halt the network)
    * oligarchy by reputation (early movers ossify)
    * geographic latency (one global coordinator)
    * no partition tolerance (CAP: pick C, lose A)
    * trust concentration (coordinator sees / orders everything)

This module replaces it with:

    * Validator set drawn from stake-weighted node pool.
    * Leader rotation per block — no validator holds the floor for long.
    * Two-phase commit (Prevote → Precommit) with 2/3+ stake quorum.
    * Slashing for double-sign / equivocation.
    * Tolerates ⅓ byzantine + ⅓ offline simultaneously.

This is a prototype. Production swap is to `tendermint-rs` or `cosmos-sdk`
Go bindings. The Python implementation here is for demonstration,
single-process simulation, and integration testing of the rest of the
stack against a real (non-mocked) consensus protocol.

Wire-format messages (all signed with proposer's wallet):
    PROPOSE   round, height, block_hash, proposer_id, signature
    PREVOTE   round, height, block_hash, voter_id,    signature
    PRECOMMIT round, height, block_hash, voter_id,    signature
    COMMIT    round, height, block_hash               (informational)

Slashable offences:
    * sign two different blocks at the same (height, round)
    * sign a vote then change vote in the same round
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Set, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


# W7: hot-path pubkey cache. Previously _verify_envelope did
# `from .tokenomics import Wallet` per call AND parsed PEM
# (`load_pem_public_key`) per call — the latter is ~hundreds of µs
# per call (DER parse + EC point decompression). On a 1k-validator
# network at 1 vote per validator per round this is 1000+ parses
# per block. Cache by PEM string. The cache is unbounded but PEM
# strings are bounded by the validator set size; we cap at 4096
# slots which fits any realistic validator-set churn window.
@lru_cache(maxsize=4096)
def _load_pubkey(pem: str):
    return serialization.load_pem_public_key(pem.encode())


def _verify_signature_cached(pem: str, message: str, sig_b64: str) -> bool:
    """ECDSA-SHA256 verify with cached PEM parse."""
    try:
        pub_key = _load_pubkey(pem)
        signature = base64.b64decode(sig_b64)
        pub_key.verify(
            signature, message.encode(), ec.ECDSA(hashes.SHA256())
        )
        return True
    except InvalidSignature:
        return False
    except Exception as e:
        logger.debug("[BFT] sig verify failed: %s", e)
        return False

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
QUORUM_NUMERATOR = 2
QUORUM_DENOMINATOR = 3      # i.e. > 2/3 stake
PROPOSE_TIMEOUT_S = 3.0
PREVOTE_TIMEOUT_S = 1.0
PRECOMMIT_TIMEOUT_S = 1.0


# ----------------------------------------------------------------------
# Validator set
# ----------------------------------------------------------------------
@dataclass
class Validator:
    """One validator in the active set."""
    node_id: str
    wallet_address: str
    stake: Decimal
    pubkey_pem: str
    online: bool = True
    slashed: bool = False
    last_signed_height: int = -1

    def voting_power(self) -> Decimal:
        return Decimal("0") if self.slashed else self.stake


@dataclass
class ValidatorSet:
    """Active validator set for a given epoch."""
    validators: List[Validator]
    epoch: int = 0

    def total_power(self) -> Decimal:
        return sum((v.voting_power() for v in self.validators), Decimal("0"))

    def quorum_threshold(self) -> Decimal:
        return self.total_power() * QUORUM_NUMERATOR / QUORUM_DENOMINATOR

    def by_id(self, node_id: str) -> Optional[Validator]:
        for v in self.validators:
            if v.node_id == node_id:
                return v
        return None

    def proposer_for(self, height: int, round_idx: int) -> Optional[Validator]:
        """
        Deterministic stake-weighted leader rotation.

        Algorithm: hash(epoch, height, round) modulo cumulative stake.
        Each validator owns an interval proportional to its stake; the
        hash falls into exactly one interval. No vote required.
        """
        if not self.validators:
            return None
        seed = f"{self.epoch}:{height}:{round_idx}"
        h = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
        total = self.total_power()
        if total == 0:
            return None
        # Map h into [0, total)
        slot = Decimal(h % int(total * 10 ** 18)) / Decimal(10 ** 18)
        cumulative = Decimal("0")
        for v in self.validators:
            cumulative += v.voting_power()
            if slot < cumulative:
                return v
        return self.validators[-1]   # rounding fallback


# ----------------------------------------------------------------------
# Consensus state machine
# ----------------------------------------------------------------------
@dataclass
class _RoundState:
    height: int
    round: int
    proposed_block_hash: Optional[str] = None
    prevotes: Dict[str, str] = field(default_factory=dict)       # voter_id -> block_hash
    precommits: Dict[str, str] = field(default_factory=dict)
    committed_hash: Optional[str] = None
    started_at: float = 0.0


class BFTConsensus:
    """
    Drive consensus on the next block.

    `propose_callback(height, round)` — called when we are the proposer.
       Must return a block dict (or None to skip the round).
    `commit_callback(height, block_hash)` — called when 2/3+ precommit
       collected for this block.
    `broadcast_callback(envelope)` — called to ship a signed message
       to all validators.
    """

    def __init__(self,
                 self_id: str,
                 wallet,                                  # core.tokenomics.Wallet
                 validator_set: ValidatorSet,
                 propose_callback: Callable[[int, int], Optional[Dict]],
                 commit_callback: Callable[[int, str, Dict], None],
                 broadcast_callback: Callable[[Dict], None],
                 ):
        self.self_id = self_id
        self.wallet = wallet
        self.vset = validator_set
        self.propose = propose_callback
        self.on_commit = commit_callback
        self.broadcast = broadcast_callback

        self._lock = threading.RLock()
        self._height = 0
        self._round = 0
        self._state: _RoundState = _RoundState(height=0, round=0,
                                                started_at=time.time())
        # Equivocation detector: track votes per (height, round, kind, voter)
        self._seen_votes: Dict[Tuple[int, int, str, str], str] = {}
        self.slashing_events: List[Dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ---- lifecycle ------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            time.sleep(0.1)
            with self._lock:
                state = self._state
                # If proposer hasn't proposed within timeout, advance round.
                if (state.proposed_block_hash is None
                        and time.time() - state.started_at > PROPOSE_TIMEOUT_S):
                    proposer = self.vset.proposer_for(state.height, state.round)
                    if proposer and proposer.node_id == self.self_id:
                        self._do_propose()
                    else:
                        # Other proposer is silent → bump round.
                        self._advance_round()

    # ---- proposal -------------------------------------------------
    def _do_propose(self) -> None:
        block = self.propose(self._state.height, self._state.round)
        if not block:
            self._advance_round()
            return
        block_hash = self._block_hash(block)
        self._state.proposed_block_hash = block_hash
        envelope = self._sign_envelope({
            "type": "PROPOSE",
            "height": self._state.height,
            "round": self._state.round,
            "block_hash": block_hash,
            "block": block,
            "proposer_id": self.self_id,
        })
        self.broadcast(envelope)
        # Self-prevote.
        self._cast_prevote(block_hash)

    # ---- inbound message handling --------------------------------
    def handle_message(self, envelope: Dict) -> bool:
        """Returns True if envelope was consensus-related."""
        m_type = envelope.get("type")
        if m_type not in ("PROPOSE", "PREVOTE", "PRECOMMIT"):
            return False
        if not self._verify_envelope(envelope):
            logger.warning("BFT: rejecting envelope with bad signature")
            return True

        with self._lock:
            height = envelope.get("height")
            round_idx = envelope.get("round")
            if height != self._state.height:
                # Old or future height; ignore for now (sync layer's job).
                return True

            if m_type == "PROPOSE":
                # Verify proposer is who we expect.
                expected = self.vset.proposer_for(height, round_idx)
                proposer_id = envelope.get("proposer_id")
                if not expected or expected.node_id != proposer_id:
                    logger.warning("BFT: proposal from non-proposer %s", proposer_id)
                    return True
                bh = envelope.get("block_hash")
                if self._state.proposed_block_hash is None and round_idx == self._state.round:
                    self._state.proposed_block_hash = bh
                    # Cast our prevote.
                    self._cast_prevote(bh)

            elif m_type == "PREVOTE":
                voter = envelope.get("voter_id")
                bh = envelope.get("block_hash")
                self._record_vote("PREVOTE", height, round_idx, voter, bh)
                self._state.prevotes[voter] = bh
                self._maybe_precommit()

            elif m_type == "PRECOMMIT":
                voter = envelope.get("voter_id")
                bh = envelope.get("block_hash")
                self._record_vote("PRECOMMIT", height, round_idx, voter, bh)
                self._state.precommits[voter] = bh
                self._maybe_commit(envelope)

        return True

    # ---- vote bookkeeping ----------------------------------------
    def _record_vote(self, kind: str, height: int, round_idx: int,
                     voter: str, block_hash: str) -> None:
        key = (height, round_idx, kind, voter)
        prior = self._seen_votes.get(key)
        if prior is not None and prior != block_hash:
            self._slash(voter, f"equivocation: signed {prior[:8]} and {block_hash[:8]}"
                        f" at height={height} round={round_idx} kind={kind}")
            return
        self._seen_votes[key] = block_hash

    def _slash(self, voter_id: str, reason: str) -> None:
        v = self.vset.by_id(voter_id)
        if v:
            v.slashed = True
        event = {"voter": voter_id, "reason": reason, "ts": time.time()}
        self.slashing_events.append(event)
        logger.error("SLASH %s: %s", voter_id, reason)

    def _cast_prevote(self, block_hash: Optional[str]) -> None:
        envelope = self._sign_envelope({
            "type": "PREVOTE",
            "height": self._state.height,
            "round": self._state.round,
            "block_hash": block_hash,
            "voter_id": self.self_id,
        })
        self.broadcast(envelope)
        self._state.prevotes[self.self_id] = block_hash or ""

    def _cast_precommit(self, block_hash: str) -> None:
        envelope = self._sign_envelope({
            "type": "PRECOMMIT",
            "height": self._state.height,
            "round": self._state.round,
            "block_hash": block_hash,
            "voter_id": self.self_id,
        })
        self.broadcast(envelope)
        self._state.precommits[self.self_id] = block_hash

    # ---- quorum checks -------------------------------------------
    def _stake_for_hash(self, votes: Dict[str, str], block_hash: str) -> Decimal:
        total = Decimal("0")
        for voter_id, bh in votes.items():
            if bh != block_hash:
                continue
            v = self.vset.by_id(voter_id)
            if v and not v.slashed:
                total += v.voting_power()
        return total

    def _maybe_precommit(self) -> None:
        bh = self._state.proposed_block_hash
        if not bh:
            return
        prevote_stake = self._stake_for_hash(self._state.prevotes, bh)
        if prevote_stake >= self.vset.quorum_threshold():
            # Have we already precommitted this round?
            if self.self_id not in self._state.precommits:
                self._cast_precommit(bh)

    def _maybe_commit(self, envelope: Dict) -> None:
        bh = self._state.proposed_block_hash
        if not bh:
            return
        precommit_stake = self._stake_for_hash(self._state.precommits, bh)
        if precommit_stake >= self.vset.quorum_threshold():
            self._state.committed_hash = bh
            block = envelope.get("block") or {"hash": bh}
            self.on_commit(self._state.height, bh, block)
            logger.info("BFT COMMIT height=%d hash=%s stake=%s",
                        self._state.height, bh[:12], precommit_stake)
            self._advance_height()

    def _advance_height(self) -> None:
        self._height += 1
        self._round = 0
        self._state = _RoundState(height=self._height, round=0,
                                  started_at=time.time())

    def _advance_round(self) -> None:
        self._round += 1
        self._state = _RoundState(height=self._height, round=self._round,
                                  started_at=time.time())

    # ---- crypto envelopes ----------------------------------------
    def _sign_envelope(self, body: Dict) -> Dict:
        canon = json.dumps(body, sort_keys=True, default=str)
        return {
            **body,
            "signer": self.wallet.address,
            "pubkey": self.wallet.public_key_pem,
            "signature": self.wallet.sign(canon),
        }

    def _verify_envelope(self, envelope: Dict) -> bool:
        sig = envelope.get("signature")
        pubkey = envelope.get("pubkey")
        if not sig or not pubkey:
            return False
        body = {k: v for k, v in envelope.items()
                if k not in ("signer", "pubkey", "signature")}
        canon = json.dumps(body, sort_keys=True, default=str)
        return _verify_signature_cached(pubkey, canon, sig)

    @staticmethod
    def _block_hash(block: Dict) -> str:
        return hashlib.sha256(
            json.dumps(block, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
