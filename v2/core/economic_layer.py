"""Economic layer over quorum verification (HG21).

Quorum (``core/quorum_verify.py``) makes cheating DETECTABLE: a wrong
result is outvoted and unpaid. This layer makes it UNPROFITABLE:

  * **Stake** — a provider bonds real ledger money to serve. No bond,
    no auction. The bond is the thing a cheater loses.
  * **Slashing** — when an accepted quorum majority proves a dissenter
    wrong, the dissenter's stake is slashed (job price × multiplier,
    capped at the bond) and the proceeds are split among the honest
    majority — verification pays, cheating costs. Slash proceeds NEVER
    touch the treasury (which earns commission only; a treasury that
    profits from slashing has an incentive to slash — the ledger's
    verify_balances enforces the ban structurally).
  * **Unbonding delay** — unstaking matures after a delay, and slashing
    draws from the FULL stake including amounts awaiting maturity, so
    "cheat then instantly withdraw the bond" is impossible.
  * **Reputation + quarantine** — dissents accumulate in a sliding
    window; past a threshold the provider is quarantined (auction-
    ineligible) for a cooling period, and its required stake rises
    with recent dissents. Repeat offenders pay more to play and
    eventually can't play at all.

Honest scope, stated plainly:

  * Slashing happens ONLY against an ACCEPTED majority. A dispute
    (no quorum reached) proves nobody wrong — nobody is slashed for
    one. Silent punishment on ambiguity is how honest nodes get robbed.
  * This defeats economically-rational cheating. A colluding majority
    that OUTVOTES the quorum still wins the round — quorum's documented
    limit — but now each attempt risks the whole bond if the collusion
    ever falls short, which changes the expected value of trying.
  * Testnet first: stakes are testnet-USD accruals like every other
    balance, and every surface says so.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CENT = Decimal("0.00000001")        # ledger money quantum (8 dp)


def _env_decimal(name: str, default: str) -> Decimal:
    try:
        return Decimal(os.environ.get(name, default))
    except ArithmeticError:
        return Decimal(default)


@dataclass
class UnbondingEntry:
    wallet_id: str
    amount_usd: Decimal
    mature_unix: float


@dataclass
class ProviderRecord:
    """Sliding-window participation history for one provider."""
    outcomes: deque = field(default_factory=lambda: deque(maxlen=50))
    slashes: int = 0
    slashed_total_usd: Decimal = Decimal("0")
    quarantined_until: float = 0.0

    def participations(self) -> int:
        return len(self.outcomes)

    def dissents(self) -> int:
        return sum(1 for _, o in self.outcomes if o == "dissent")


class EconomicLayer:
    """Stake + slashing + reputation over a BuyerLedger, consuming
    QuorumOutcome dissent signals. All knobs env-tunable:

      PLUGINFER_BASE_STAKE_USD        (default 1.00)
      PLUGINFER_SLASH_MULTIPLIER      (default 1.0 × job price)
      PLUGINFER_UNBONDING_S           (default 86400 — 24 h)
      PLUGINFER_QUARANTINE_S          (default 86400)
      PLUGINFER_QUARANTINE_MIN_DISSENTS (default 3)
      PLUGINFER_QUARANTINE_RATE       (default 0.34)

    ``now_fn`` is injectable for deterministic tests.
    """

    def __init__(self, ledger, state_dir: Optional[os.PathLike] = None,
                 *, now_fn: Callable[[], float] = time.time):
        self.ledger = ledger
        self.now = now_fn
        self.base_stake_usd = _env_decimal(
            "PLUGINFER_BASE_STAKE_USD", "1.00")
        self.slash_multiplier = _env_decimal(
            "PLUGINFER_SLASH_MULTIPLIER", "1.0")
        self.unbonding_s = float(os.environ.get(
            "PLUGINFER_UNBONDING_S", "86400"))
        self.quarantine_s = float(os.environ.get(
            "PLUGINFER_QUARANTINE_S", "86400"))
        self.quarantine_min_dissents = int(os.environ.get(
            "PLUGINFER_QUARANTINE_MIN_DISSENTS", "3"))
        self.quarantine_rate = float(os.environ.get(
            "PLUGINFER_QUARANTINE_RATE", "0.34"))
        self._lock = threading.Lock()
        self._providers: Dict[str, ProviderRecord] = {}
        self._unbonding: List[UnbondingEntry] = []
        self._slash_journal: List[Dict[str, Any]] = []
        self._state_path: Optional[Path] = None
        if state_dir is not None:
            self._state_path = Path(state_dir) / "economic_layer.json"
            self._load()

    # ------------------------------------------------------------------
    # Persistence (same atomic style as the money ledger)
    # ------------------------------------------------------------------

    def _save(self) -> None:
        if self._state_path is None:
            return
        try:
            data = {
                "providers": {
                    pid: {
                        "outcomes": [[ts, o] for ts, o in r.outcomes],
                        "slashes": r.slashes,
                        "slashed_total_usd": str(r.slashed_total_usd),
                        "quarantined_until": r.quarantined_until,
                    } for pid, r in self._providers.items()
                },
                "unbonding": [
                    {"wallet_id": u.wallet_id,
                     "amount_usd": str(u.amount_usd),
                     "mature_unix": u.mature_unix}
                    for u in self._unbonding
                ],
                "slash_journal": self._slash_journal[-500:],
            }
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError as e:
            logger.error("economic layer snapshot failed: %s", e)

    def _load(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for pid, rd in data.get("providers", {}).items():
            rec = ProviderRecord()
            for ts, o in rd.get("outcomes", []):
                rec.outcomes.append((float(ts), str(o)))
            rec.slashes = int(rd.get("slashes", 0))
            rec.slashed_total_usd = Decimal(
                rd.get("slashed_total_usd", "0"))
            rec.quarantined_until = float(rd.get("quarantined_until", 0))
            self._providers[pid] = rec
        self._unbonding = [
            UnbondingEntry(u["wallet_id"], Decimal(u["amount_usd"]),
                           float(u["mature_unix"]))
            for u in data.get("unbonding", [])
        ]
        self._slash_journal = list(data.get("slash_journal", []))

    # ------------------------------------------------------------------
    # Stake lifecycle
    # ------------------------------------------------------------------

    def stake(self, wallet_id: str, amount_usd: Decimal):
        """Bond funds. Raises InsufficientFunds via the ledger when the
        wallet can't cover it."""
        w = self.ledger.stake(wallet_id, amount_usd)
        with self._lock:
            self._save()
        return w

    def staked_of(self, wallet_id: str) -> Decimal:
        w = self.ledger.get_wallet(wallet_id)
        return w.staked_usd if w is not None else Decimal("0")

    def request_unstake(self, wallet_id: str,
                        amount_usd: Decimal) -> UnbondingEntry:
        """Start the unbonding clock. The money STAYS staked (and
        slashable) until maturity — that is the whole point."""
        if amount_usd <= Decimal("0"):
            raise ValueError("unstake amount must be positive")
        with self._lock:
            pending = sum((u.amount_usd for u in self._unbonding
                           if u.wallet_id == wallet_id), Decimal("0"))
            if self.staked_of(wallet_id) - pending < amount_usd:
                raise ValueError(
                    f"wallet {wallet_id} has "
                    f"{self.staked_of(wallet_id) - pending} USD staked "
                    f"and not already unbonding — cannot unbond "
                    f"{amount_usd}")
            entry = UnbondingEntry(
                wallet_id=wallet_id, amount_usd=amount_usd,
                mature_unix=self.now() + self.unbonding_s)
            self._unbonding.append(entry)
            self._save()
            return entry

    def claim_unstaked(self, wallet_id: str) -> Decimal:
        """Release every MATURED unbonding entry back to available.
        If slashing consumed part of the stake in the meantime, the
        claim shrinks to what actually remains — a slash is never
        undone by a queued withdrawal."""
        claimed = Decimal("0")
        with self._lock:
            now = self.now()
            keep: List[UnbondingEntry] = []
            for u in self._unbonding:
                if u.wallet_id != wallet_id or u.mature_unix > now:
                    keep.append(u)
                    continue
                remaining = self.staked_of(wallet_id)
                take = min(u.amount_usd, remaining)
                if take > Decimal("0"):
                    self.ledger.unstake_release(
                        wallet_id, take,
                        note="unbonded after delay")
                    claimed += take
                # A fully-slashed entry just evaporates — the money is
                # gone to the slash, honestly recorded there.
            self._unbonding = keep
            self._save()
        return claimed

    # ------------------------------------------------------------------
    # Eligibility (the auction's gate)
    # ------------------------------------------------------------------

    def required_stake(self, wallet_id: str) -> Decimal:
        """Base bond, scaled up by recent dissents: a provider with a
        history pays more to keep playing."""
        with self._lock:
            rec = self._providers.get(wallet_id)
            dissents = rec.dissents() if rec else 0
        return self.base_stake_usd * (1 + dissents)

    def is_eligible(self, wallet_id: str) -> Tuple[bool, str]:
        with self._lock:
            rec = self._providers.get(wallet_id)
            if rec and rec.quarantined_until > self.now():
                return False, (
                    f"quarantined until "
                    f"{rec.quarantined_until:.0f} (dissent history)")
        need = self.required_stake(wallet_id)
        have = self.staked_of(wallet_id)
        if have < need:
            return False, f"stake {have} USD < required {need} USD"
        return True, "ok"

    # ------------------------------------------------------------------
    # The quorum consumer — where detection becomes economics
    # ------------------------------------------------------------------

    def record_quorum_outcome(self, outcome, *, job_id: str,
                              job_price_usd: Decimal) -> Dict[str, Any]:
        """Consume a QuorumOutcome. Majority members build reputation;
        dissenters (proved wrong by an ACCEPTED majority) are slashed
        and the proceeds split equally among the majority. On a dispute
        nothing punitive happens — no majority means nobody was proved
        wrong."""
        summary: Dict[str, Any] = {"job_id": job_id, "slashed": {},
                                   "awards": {}, "quarantined": []}
        if not outcome.accepted:
            summary["action"] = ("none — dispute/no majority: nobody "
                                 "provably wrong, nobody slashed")
            return summary
        majority = outcome.paid_providers()
        dissenters = outcome.dissenting_providers()
        now = self.now()
        with self._lock:
            for pid in majority:
                self._providers.setdefault(
                    pid, ProviderRecord()).outcomes.append(
                        (now, "majority"))
            pot = Decimal("0")
            for pid in dissenters:
                rec = self._providers.setdefault(pid, ProviderRecord())
                rec.outcomes.append((now, "dissent"))
                want = (job_price_usd
                        * self.slash_multiplier).quantize(_CENT)
                taken = self.ledger.slash_stake(
                    pid, want, job_id=job_id,
                    note=f"quorum dissent (majority "
                         f"{outcome.agreement_count}/"
                         f"{outcome.responded})")
                if taken > Decimal("0"):
                    rec.slashes += 1
                    rec.slashed_total_usd += taken
                    pot += taken
                    summary["slashed"][pid] = str(taken)
                # Quarantine check — on the sliding window.
                n, d = rec.participations(), rec.dissents()
                if (d >= self.quarantine_min_dissents
                        and n > 0 and d / n > self.quarantine_rate):
                    rec.quarantined_until = now + self.quarantine_s
                    summary["quarantined"].append(pid)
            # Split the pot equally among the honest majority; the
            # indivisible remainder goes to the first member so every
            # cent is accounted for.
            if pot > Decimal("0") and majority:
                share = (pot / len(majority)).quantize(_CENT)
                paid = Decimal("0")
                for i, pid in enumerate(majority):
                    amt = (pot - paid if i == len(majority) - 1
                           else share)
                    if amt > Decimal("0"):
                        self.ledger.slash_award(
                            pid, amt, job_id=job_id,
                            note="honest-majority share of slashed "
                                 "bond")
                        summary["awards"][pid] = str(amt)
                        paid += amt
            self._slash_journal.append({
                "ts": now, "job_id": job_id,
                "price_usd": str(job_price_usd),
                "majority": majority, "dissenters": dissenters,
                "slashed": dict(summary["slashed"]),
                "awards": dict(summary["awards"]),
                "quarantined": list(summary["quarantined"]),
            })
            self._save()
        summary["action"] = "settled"
        return summary

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def reputation(self, wallet_id: str) -> Dict[str, Any]:
        with self._lock:
            rec = self._providers.get(wallet_id) or ProviderRecord()
            n, d = rec.participations(), rec.dissents()
            pending_unbond = sum(
                (u.amount_usd for u in self._unbonding
                 if u.wallet_id == wallet_id), Decimal("0"))
            quarantined = rec.quarantined_until > self.now()
        ok, why = self.is_eligible(wallet_id)
        return {
            "wallet_id": wallet_id,
            "participations": n,
            "majorities": n - d,
            "dissents": d,
            "dissent_rate": round(d / n, 4) if n else None,
            "slashes": rec.slashes,
            "slashed_total_usd": str(rec.slashed_total_usd),
            "staked_usd": str(self.staked_of(wallet_id)),
            "unbonding_usd": str(pending_unbond),
            "required_stake_usd": str(self.required_stake(wallet_id)),
            "quarantined": quarantined,
            "quarantined_until": (rec.quarantined_until
                                  if quarantined else None),
            "eligible": ok,
            "eligibility_reason": why,
        }

    def slash_journal(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._slash_journal[-max(1, min(limit, 500)):])


__all__ = ["EconomicLayer", "ProviderRecord", "UnbondingEntry"]
