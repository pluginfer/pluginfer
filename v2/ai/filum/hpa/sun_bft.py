"""Sun-BFT bridge — wires §C2 sun_election to core.bft_consensus.

The Sun-of-Suns ring is small (typically <= 100 nodes) — exactly the
regime classical BFT works well in. This module is the adapter
between two modules that don't otherwise know about each other:

* ``hpa.sun_election`` — gives us the Sun set elected by hardware
  pressure stability
* ``core.bft_consensus.BFTConsensus`` — runs Tendermint-style
  rounds with stake-weighted voting

The bridge makes the Sun set act as the BFT validator set: each
elected Sun becomes a validator with weight proportional to its
``stability_score * advertised_capacity_tflops``. When a Sun is
demoted (S < S_cut), its weight drops to zero — it's still in the
set but cannot propose or vote until re-promoted.

Re-elections are handled gracefully: a fresh ``ElectionResult`` is
applied as a soft membership change, propagated via the next BFT
block. The chain commits a *checkpoint* every K rounds that locks
in the current Sun set; in between, soft membership changes are
proposed by the current proposer and committed via the same 2/3
quorum. No hard fork on Sun churn.

This is the missing piece between "we have a topology" and "the
topology agrees on something." Once this lands, the §C protocol
can record ledger commitments via the existing chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from .sun_election import (
    ElectionResult,
    NodeMembership,
    SunElection,
    SunOfSunsRing,
)

logger = logging.getLogger(__name__)


@dataclass
class SunValidator:
    """A Sun acting as a BFT validator. Mirrors core.bft_consensus.Validator
    fields the SunBFTBridge cares about, decoupled so this module doesn't
    hard-import the core module at the top level."""
    node_id: str
    public_key_pem: str
    weight: int                  # vote weight in BFT
    stability_score: float
    capacity_tflops: float


def sun_to_validator(sun: NodeMembership, *, weight_unit: int = 1000) -> SunValidator:
    """Project a NodeMembership to BFT-validator weight.

    Weight = round(stability * capacity * weight_unit).
    Demoted Suns (stability == 0) get weight 0 — they're members but
    cannot influence consensus until they recover.
    """
    weight = int(round(
        max(0.0, sun.stability_score)
        * max(0.0, sun.advertised_capacity_tflops)
        * weight_unit
    ))
    pubkey_pem = ""
    if sun.public_key:
        try:
            pubkey_pem = sun.public_key.decode("utf-8", errors="replace")
        except Exception:
            pubkey_pem = ""
    return SunValidator(
        node_id=sun.node_id,
        public_key_pem=pubkey_pem,
        weight=max(0, weight),
        stability_score=float(sun.stability_score),
        capacity_tflops=float(sun.advertised_capacity_tflops),
    )


class SunBFTBridge:
    """Glue between SunElection and core.bft_consensus.

    Two responsibilities:
    1. Propagate election results into the BFT validator set as soft
       membership changes.
    2. Expose a propose-callback that emits §C training-state
       blocks (NBGGA shard versions, sealed receipt-log Merkle roots,
       sun-set checkpoints) for BFT to commit.

    Construction is deferred so this module imports cleanly even if
    `core.bft_consensus` is unavailable in the current environment
    (e.g. unit tests that don't need the full chain stack).
    """

    def __init__(
        self,
        self_id: str,
        ring: SunOfSunsRing,
        *,
        weight_unit: int = 1000,
        bft_consensus=None,
    ):
        self.self_id = self_id
        self.ring = ring
        self.weight_unit = weight_unit
        self.bft = bft_consensus
        self._last_validator_set: list[SunValidator] = []
        self._on_commit_callbacks: list[Callable[[int, str, dict], None]] = []

    # --- election integration --------------------------------------------

    def apply_election(self, result: ElectionResult) -> list[SunValidator]:
        """Project an ElectionResult into a fresh BFT validator set.

        Returns the list of SunValidators the BFT layer should use.
        Caller passes this through to ``ValidatorSet.update(...)``
        whenever the chain spec allows soft membership change.
        """
        validators: list[SunValidator] = []
        for sun in result.suns:
            self.ring.update(sun)
            validators.append(sun_to_validator(sun, weight_unit=self.weight_unit))
        # Demote anyone in the previous set who isn't in the new set
        # by giving them weight 0 — they remain in the validator
        # registry for one transition cycle so existing votes can be
        # finalised, then they're evicted by the next checkpoint.
        prev_ids = {v.node_id for v in self._last_validator_set}
        new_ids = {v.node_id for v in validators}
        for nid in prev_ids - new_ids:
            validators.append(SunValidator(
                node_id=nid,
                public_key_pem="",
                weight=0,
                stability_score=0.0,
                capacity_tflops=0.0,
            ))
        self._last_validator_set = validators
        return validators

    # --- block production ------------------------------------------------

    @dataclass
    class TrainingStateBlock:
        height: int
        nbgga_shard_versions: dict           # shard_id -> version_v
        receipt_anchor_root: str             # hex sha256 of latest sealed root
        sun_set_hash: str                    # commitment to current sun set
        ts: float

    def propose_training_state(
        self,
        height: int,
        round_n: int,
        *,
        nbgga_shard_versions: dict,
        receipt_anchor_root: str = "",
    ) -> Optional[dict]:
        """The BFT propose-callback. Builds a TrainingStateBlock dict.

        Returns ``None`` to skip (no state changed since last commit).
        """
        import hashlib, json, time
        if not nbgga_shard_versions and not receipt_anchor_root:
            return None
        sun_set = sorted(v.node_id for v in self._last_validator_set
                          if v.weight > 0)
        sun_set_hash = hashlib.sha256(
            json.dumps(sun_set, sort_keys=True).encode()
        ).hexdigest()
        block = {
            "kind": "training_state",
            "height": int(height),
            "round": int(round_n),
            "nbgga": dict(nbgga_shard_versions),
            "receipt_anchor_root": receipt_anchor_root,
            "sun_set_hash": sun_set_hash,
            "ts": time.time(),
        }
        return block

    # --- commit callback -------------------------------------------------

    def on_commit(self, fn: Callable[[int, str, dict], None]) -> None:
        """Register a callback fired when a TrainingStateBlock commits."""
        self._on_commit_callbacks.append(fn)

    def handle_commit(self, height: int, block_hash: str, block: dict) -> None:
        """Forward a committed block to all listeners.

        BFT calls this; we fan out to (a) NBGGA so it can advance its
        durability checkpoint, (b) the receipt log so it can mark
        sealed roots as anchored, (c) any user listeners.
        """
        for fn in self._on_commit_callbacks:
            try:
                fn(height, block_hash, block)
            except Exception as e:
                logger.exception("commit listener raised: %s", e)
