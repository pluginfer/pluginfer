"""
Governance DAO (W27 hardened)
=============================
On-chain governance for protocol parameters, treasury spend, and
contract upgrades.

Previous version had three CRITICAL bugs:
  1. Vote weight was `ledger.get_balance(voter)` at the *current*
     block, not a snapshot. **Vote-pump attack:** vote with 100 PLG,
     transfer to a new address, vote again from the new address,
     repeat. The `voters` set guarded same-address, not same-human.
  2. No quorum requirement — a single 1-yes-0-no proposal "PASSED".
  3. No execution path — `get_result()` returned "PASSED" but nothing
     ever applied the change. Decorative DAO.

This version:
  * **Snapshot block**: each proposal locks `snapshot_height` at
    creation. Vote weight = `ledger.get_balance_at(voter, height)`.
    Transferring funds during the voting window doesn't move the
    needle.
  * **Quorum**: require `participation >= QUORUM_FRACTION * supply`
    (default 5% of supply at snapshot). Below quorum -> NO_QUORUM.
  * **Execution payload**: each proposal carries a structured
    `action` dict. After PASS + end_time, the registered
    `execute_callback(action)` fires. Caller passes a handler when
    constructing the DAO.
  * **Persistence**: state is journalled to `governance.json` so
    restart doesn't lose proposals.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class Proposal:
    """An on-chain governance proposal."""

    def __init__(self, id: str, title: str, creator: str,
                 end_time: float, snapshot_height: int,
                 action: Optional[Dict[str, Any]] = None,
                 description: str = ""):
        self.id = id
        self.title = title
        self.description = description
        self.creator = creator
        self.end_time = end_time
        self.snapshot_height = snapshot_height
        # `action` describes what executes on PASS. Conventional shape:
        #   {"kind": "set_param", "key": "MIN_TX_FEE", "value": "0.002"}
        #   {"kind": "treasury_spend", "to": "PLG...", "amount": "1000"}
        #   {"kind": "upgrade", "binary_sha256": "...", "from_version": "..."}
        # The DAO doesn't know how to execute these — that's the
        # `execute_callback` registered by the host. The DAO just
        # certifies that "this action was approved by stake-weighted
        # majority with quorum at this snapshot height".
        self.action: Dict[str, Any] = action or {}
        self.votes_for = Decimal("0.0")
        self.votes_against = Decimal("0.0")
        self.voters: Dict[str, str] = {}     # voter_addr -> "yes"/"no"
        self.status = "ACTIVE"               # ACTIVE | PASSED | REJECTED | NO_QUORUM | EXECUTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "description": self.description,
            "creator": self.creator, "end_time": self.end_time,
            "snapshot_height": self.snapshot_height, "action": self.action,
            "votes_for": str(self.votes_for),
            "votes_against": str(self.votes_against),
            "voters": dict(self.voters),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Proposal":
        p = cls(d["id"], d["title"], d["creator"], d["end_time"],
                int(d["snapshot_height"]),
                action=d.get("action") or {},
                description=d.get("description", ""))
        p.votes_for = Decimal(d.get("votes_for", "0"))
        p.votes_against = Decimal(d.get("votes_against", "0"))
        p.voters = dict(d.get("voters") or {})
        p.status = d.get("status", "ACTIVE")
        return p


class GovernanceDAO:
    """Stake-weighted DAO with snapshot voting + quorum + execution."""

    # Quorum: fraction of `total_supply_at(snapshot_height)` that must
    # have voted (yes OR no) for the result to count. Below this
    # threshold the proposal is NO_QUORUM regardless of vote ratio.
    QUORUM_FRACTION = Decimal("0.05")     # 5%

    def __init__(self, ledger,
                 storage_path: str = "governance.json",
                 execute_callback: Optional[Callable[[Dict[str, Any]], bool]] = None):
        self.ledger = ledger
        self.storage_path = storage_path
        self.execute_callback = execute_callback
        self.proposals: Dict[str, Proposal] = {}
        self._load()

    # ---- proposal lifecycle ---------------------------------------------
    def create_proposal(self, creator: str, title: str,
                        action: Optional[Dict[str, Any]] = None,
                        description: str = "",
                        duration_hours: int = 24) -> str:
        """Create a new proposal. Snapshots voter balances at this height."""
        pid = uuid.uuid4().hex[:8]
        end_time = time.time() + duration_hours * 3600
        snapshot_height = self.ledger.get_height() if self.ledger else 0
        prop = Proposal(pid, title, creator, end_time, snapshot_height,
                        action=action, description=description)
        self.proposals[pid] = prop
        self._save()
        logger.info("[DAO] Proposal %s created at height=%d: %s",
                    pid, snapshot_height, title)
        return pid

    def vote(self, proposal_id: str, voter_addr: str, choice: bool) -> bool:
        """
        Cast a vote. Weight = voter's balance at proposal snapshot
        height (NOT current). Returns False if proposal expired,
        already voted by this address, or balance == 0.
        """
        prop = self.proposals.get(proposal_id)
        if not prop:
            return False
        if time.time() > prop.end_time:
            self._finalize(prop)
            return False
        if voter_addr in prop.voters:
            return False           # one vote per address (per-human is W28)

        weight = Decimal(str(
            self.ledger.get_balance_at(voter_addr, prop.snapshot_height)
        )) if self.ledger else Decimal("0")
        if weight <= 0:
            return False

        if choice:
            prop.votes_for += weight
        else:
            prop.votes_against += weight
        prop.voters[voter_addr] = "yes" if choice else "no"
        self._save()
        logger.info("[DAO] %s voted '%s' on %s with weight %s "
                    "(snapshot height=%d)",
                    voter_addr[:8], "yes" if choice else "no",
                    proposal_id, weight, prop.snapshot_height)
        return True

    def get_result(self, proposal_id: str) -> str:
        """Returns one of ACTIVE | PASSED | REJECTED | NO_QUORUM | EXECUTED."""
        prop = self.proposals.get(proposal_id)
        if not prop:
            return "NOT_FOUND"
        if prop.status != "ACTIVE":
            return prop.status
        # Active proposals only finalize after end_time elapses.
        if time.time() <= prop.end_time:
            return "ACTIVE"
        return self._finalize(prop)

    def execute(self, proposal_id: str) -> bool:
        """Execute a PASSED proposal via the registered callback."""
        prop = self.proposals.get(proposal_id)
        if not prop or prop.status != "PASSED":
            return False
        if self.execute_callback is None:
            logger.warning("[DAO] %s passed but no execute_callback "
                           "registered; cannot apply action.", proposal_id)
            return False
        try:
            ok = bool(self.execute_callback(prop.action))
        except Exception as e:
            logger.error("[DAO] execute_callback raised on %s: %s",
                         proposal_id, e)
            ok = False
        if ok:
            prop.status = "EXECUTED"
            self._save()
            logger.info("[DAO] %s EXECUTED", proposal_id)
        return ok

    # ---- internal --------------------------------------------------------
    def _finalize(self, prop: Proposal) -> str:
        """Apply quorum + majority and persist final status."""
        if not self.ledger:
            prop.status = "REJECTED"
            self._save()
            return prop.status
        supply = Decimal(str(self.ledger.total_supply_at(prop.snapshot_height)))
        participation = prop.votes_for + prop.votes_against
        if supply <= 0:
            quorum_ok = participation > 0
        else:
            quorum_ok = participation >= supply * self.QUORUM_FRACTION

        if not quorum_ok:
            prop.status = "NO_QUORUM"
            logger.info("[DAO] %s NO_QUORUM (participation %s / supply %s, "
                        "needed %s%%)", prop.id, participation, supply,
                        self.QUORUM_FRACTION * 100)
        elif prop.votes_for > prop.votes_against:
            prop.status = "PASSED"
            logger.info("[DAO] %s PASSED (%s / %s for/against)",
                        prop.id, prop.votes_for, prop.votes_against)
        else:
            prop.status = "REJECTED"
            logger.info("[DAO] %s REJECTED (%s / %s for/against)",
                        prop.id, prop.votes_for, prop.votes_against)

        self._save()
        return prop.status

    def _save(self) -> None:
        try:
            data = {pid: p.to_dict() for pid, p in self.proposals.items()}
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error("[DAO] save failed: %s", e)

    def _load(self) -> None:
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r") as f:
                raw = json.load(f)
            for pid, d in raw.items():
                self.proposals[pid] = Proposal.from_dict(d)
        except Exception as e:
            logger.error("[DAO] load failed: %s", e)

    def list_proposals(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.proposals.values()]
