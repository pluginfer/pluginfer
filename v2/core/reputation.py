"""
Reputation Manager — chain-derived (W28)
========================================

The previous version persisted reputation to ``reputation.json`` —
an unprotected, operator-editable JSON file. A node operator could
grant themselves infinite reputation by editing the file. There was
no decay, no negative path (slashing, audit failures, dropped
tasks), and the score was fundamentally unfalsifiable.

W28 rewrite: reputation is **derived** from chain-observable events:

  * blocks_mined: count of blocks signed by this address as miner
                  (chain.coinbase[recipient] tally).
  * tasks_completed: count of `task_receipt` system transactions
                     attesting completed work for this address.
  * slashes: count of `slash` system transactions naming this
             address (penalty path).

Score = blocks*W_BLOCK + tasks*W_TASK - slashes*W_SLASH + hw_baseline.
Score is capped at MAX_SCORE so a single wealthy validator can't
dominate election by mining 100k blocks.

Caching
-------
For O(1) reads after the first call, we cache (chain_height,
score_components) per address. Cache invalidates when chain extends
or reorgs (the latter detected by chain[height-1].hash divergence).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# Per-event weights. Tunable via governance proposal in W27.
W_BLOCK = 20.0
W_TASK = 5.0
W_SLASH = 100.0
W_HARDWARE_BASELINE = 100.0
MAX_SCORE = 1_000_000.0       # absolute cap


@dataclass
class ReputationComponents:
    """Per-address reputation derived from chain state."""
    blocks_mined: int = 0
    tasks_completed: int = 0
    slashes: int = 0
    hw_score: float = 1.0          # 1.0 = CPU-only baseline
    last_height: int = -1
    last_tip_hash: Optional[str] = None

    def score(self) -> float:
        s = (
            self.blocks_mined * W_BLOCK
            + self.tasks_completed * W_TASK
            - self.slashes * W_SLASH
            + self.hw_score * W_HARDWARE_BASELINE
        )
        return round(max(0.0, min(s, MAX_SCORE)), 2)


class ReputationManager:
    """Chain-backed reputation accumulator.

    Caller-provided ledger is mandatory (the security claim is that
    reputation derives from chain state, not local files). HW score
    is a SOFT input — used as a one-time baseline, not persisted to
    disk.
    """
    def __init__(self, ledger, address: Optional[str] = None,
                 hw_score: Optional[float] = None):
        if ledger is None:
            raise ValueError(
                "ReputationManager requires a ledger — chain-derived "
                "reputation is the entire point. Pass the active "
                "ComputeLedger instance."
            )
        self.ledger = ledger
        self.address = address
        self.cache: Dict[str, ReputationComponents] = {}
        self._fallback_hw_score = hw_score
        if address:
            # Pre-populate so get_score() with no args is cheap.
            self.recompute(address)

    # ------------------------------------------------------------------
    # Derivation from chain
    # ------------------------------------------------------------------
    def recompute(self, address: str) -> ReputationComponents:
        """Walk the chain and recount events for `address`. Returns the
        fresh ReputationComponents (also cached)."""
        comp = ReputationComponents()
        comp.hw_score = self._detect_hw_score()

        for block in self.ledger.chain:
            for tx in block.transactions or []:
                tx_type = tx.get("type")
                if tx_type in ("coinbase", "mint"):
                    if tx.get("recipient") == address:
                        comp.blocks_mined += 1
                elif tx_type == "task_receipt":
                    if tx.get("recipient") == address:
                        comp.tasks_completed += 1
                elif tx_type == "slash":
                    if tx.get("recipient") == address:
                        comp.slashes += 1
        comp.last_height = self.ledger.get_height()
        try:
            comp.last_tip_hash = self.ledger.chain[-1].hash
        except Exception:
            comp.last_tip_hash = None
        self.cache[address] = comp
        return comp

    def get_components(self, address: Optional[str] = None) -> ReputationComponents:
        """Cached if chain hasn't moved; otherwise re-derive."""
        addr = address or self.address
        if addr is None:
            raise ValueError(
                "get_components needs an address (constructor or arg)"
            )
        cached = self.cache.get(addr)
        cur_height = self.ledger.get_height()
        try:
            cur_tip = self.ledger.chain[-1].hash
        except Exception:
            cur_tip = None
        if cached and cached.last_height == cur_height \
                and cached.last_tip_hash == cur_tip:
            return cached
        return self.recompute(addr)

    def get_score(self, address: Optional[str] = None) -> float:
        return self.get_components(address).score()

    # ------------------------------------------------------------------
    # HW baseline (one input, not persisted)
    # ------------------------------------------------------------------
    def _detect_hw_score(self) -> float:
        if self._fallback_hw_score is not None:
            return float(self._fallback_hw_score)
        try:
            from .hardware_detector import HardwareDetector
            hw = HardwareDetector()
            return float(hw.get_performance_score())
        except Exception:
            return 1.0
