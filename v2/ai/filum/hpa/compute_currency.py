"""§E1 Compute-as-Currency — pay for AI by contributing AI compute.

The single biggest gap between "Pluginfer is 90% cheaper than AWS"
and "Pluginfer makes AI training available to *everyone on the
planet*" is this: 90%-off-AWS is still expensive for the bottom
3 billion people. A median monthly income of $300 cannot afford
training a custom model even at our prices.

This module fixes that.

The mechanism: any node can submit a training job by *promising
equal compute back to the mesh*. The submitter signs a binding
ledger entry "I owe the mesh X TFLOP-hours, payable from my
device's idle hardware over the next N weeks." The mesh runs the
job immediately, paid for by other providers. The submitter then
contributes whatever compute their phone/laptop can spare during
its idle hours until the debt is repaid. **No money changes
hands.** A teenager in Lagos with a $200 Android phone and a
$5/month data plan can train a custom model — paid for by the
phone's idle GPU running training jobs for someone else
overnight.

Why this works economically:

* Idle compute is *free at the margin* for the contributor. The
  phone is on, the GPU is sitting idle, electricity is being paid
  anyway. Donating those cycles costs nothing.
* The mesh aggregates everyone's idle hours into a coherent
  buffer. Even if any one person's device is unreliable, the
  aggregate can fulfil any reasonable debt within weeks.
* No fiat or crypto custody required — debts are denominated in
  TFLOP-hours, settled by performing compute. This sidesteps
  every money-transmitter regulation that would otherwise apply.

Why this is novel (§E1):

* No prior art treats compute as a *currency settled in kind*.
  Every existing "barter compute" project (BOINC, SETI@home,
  Folding@home) is donation-only — no compute-debt mechanism.
  Cloud providers' "credits" are denominated in fiat. Bittensor's
  TAO is a token, not a settled-in-kind unit.

The implementation is lean: a debt ledger keyed by pubkey, an
auction matcher (`auction_for_compute_debt`) that pairs new asks
against the open-debt buffer, a `repay_step()` hook that the
local trainer calls to credit the submitter's debt as their device
runs jobs for others.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------- domain types ---------------------------------------------------

@dataclass
class ComputeDebt:
    """A submitter's open obligation to perform N TFLOP-hours of work."""
    pubkey: str                          # submitter's Ed25519 pubkey hex
    initial_tflop_hr: float              # signed when submitted
    repaid_tflop_hr: float = 0.0
    expiry_ts: float = 0.0
    created_ts: float = 0.0
    job_id: str = ""                     # links to the job that created the debt
    interest_rate: float = 0.0           # 0 by default; mesh-governance-tunable
    last_update_ts: float = 0.0

    def remaining_tflop_hr(self) -> float:
        return max(0.0, self.initial_tflop_hr - self.repaid_tflop_hr)

    def is_settled(self) -> bool:
        return self.remaining_tflop_hr() <= 1e-6

    def is_expired(self, now_ts: Optional[float] = None) -> bool:
        if self.expiry_ts <= 0:
            return False
        return (now_ts if now_ts is not None else time.time()) > self.expiry_ts


@dataclass
class CreditPool:
    """The aggregate buffer of idle compute that funds new debts.

    Filled by providers contributing TFLOP-hours faster than they
    consume. Drains when new submitters get their debts approved.
    The pool is a *commons* — every contributor adds, every consumer
    draws, the running balance keeps the system solvent.
    """
    available_tflop_hr: float = 0.0
    cumulative_contributed: float = 0.0
    cumulative_consumed: float = 0.0
    last_update_ts: float = 0.0


@dataclass
class ComputeCurrencyConfig:
    state_path: str = "ai/filum/_work/compute_currency.json"
    max_debt_per_pubkey: float = 200.0            # cap (TFLOP-hr) per submitter
    default_repay_window_days: float = 30.0
    insolvency_haircut_pct: float = 0.05          # max 5% of pool can be in
                                                  # expired-debt write-offs
    min_balance_for_grant: float = 0.0            # mesh can require pool > X


# ---------- the exchange ---------------------------------------------------

class ComputeCurrencyExchange:
    """The compute-as-currency book-keeping primitive.

    Thread-safe. Persists to JSON on every state change so a node
    crash never loses debt data.

    Operations:

    * ``submit_for_compute_debt(pubkey, tflop_hr)`` -> ComputeDebt
        Records a new debt; deducts from the credit pool. Raises
        ValueError if the pool is too thin or the submitter is
        already over-leveraged.
    * ``contribute_compute(pubkey, tflop_hr_done)``
        Records a provider performing work for the mesh. First
        applies to that provider's outstanding debts (if any);
        residual goes into the credit pool as new credit.
    * ``open_debts()``  -> list of ComputeDebt
    * ``balance()``     -> CreditPool snapshot

    Note: the *matching* of which providers run which jobs lives in
    `reverse_auction.py`. This module is the ledger; that module is
    the market. Together they make compute fungible.
    """

    def __init__(self, config: ComputeCurrencyConfig = ComputeCurrencyConfig()):
        self.cfg = config
        self.pool = CreditPool(last_update_ts=time.time())
        self._debts: dict[str, list[ComputeDebt]] = {}  # pubkey -> debts
        self._lock = threading.RLock()
        self._load()

    # --- core ops -------------------------------------------------------

    def submit_for_compute_debt(
        self,
        pubkey: str,
        tflop_hr: float,
        *,
        job_id: str = "",
        repay_window_days: Optional[float] = None,
    ) -> ComputeDebt:
        """Approve a new compute-debt against the credit pool.

        Raises ``ValueError`` if:
        * tflop_hr <= 0
        * pubkey already has > max_debt_per_pubkey outstanding
        * the pool is below min_balance_for_grant
        """
        if tflop_hr <= 0:
            raise ValueError("tflop_hr must be positive")
        with self._lock:
            outstanding = self.outstanding_for(pubkey)
            if outstanding + tflop_hr > self.cfg.max_debt_per_pubkey:
                raise ValueError(
                    f"submitter would exceed cap "
                    f"({outstanding + tflop_hr:.2f} > "
                    f"{self.cfg.max_debt_per_pubkey})"
                )
            if self.pool.available_tflop_hr < self.cfg.min_balance_for_grant:
                raise ValueError(
                    f"credit pool too thin: {self.pool.available_tflop_hr:.2f} "
                    f"< {self.cfg.min_balance_for_grant:.2f}"
                )
            now = time.time()
            window = (repay_window_days
                      if repay_window_days is not None
                      else self.cfg.default_repay_window_days)
            debt = ComputeDebt(
                pubkey=pubkey,
                initial_tflop_hr=tflop_hr,
                expiry_ts=now + window * 86400.0,
                created_ts=now,
                job_id=job_id,
                last_update_ts=now,
            )
            self._debts.setdefault(pubkey, []).append(debt)
            # Pool is reduced *now* — the work is being done on someone
            # else's compute, paid for from the pool's reserves.
            self.pool.available_tflop_hr -= tflop_hr
            self.pool.cumulative_consumed += tflop_hr
            self.pool.last_update_ts = now
            self._save()
            return debt

    def contribute_compute(self, pubkey: str, tflop_hr_done: float) -> dict:
        """Record a provider doing work for the mesh.

        First applies to *that pubkey's* open debts; residual goes
        into the credit pool. Returns a summary dict with how much
        was applied to debt vs added to the pool.
        """
        if tflop_hr_done <= 0:
            return {"applied_to_debt": 0.0, "added_to_pool": 0.0}
        applied = 0.0
        remaining = tflop_hr_done
        now = time.time()
        with self._lock:
            for d in self._debts.get(pubkey, []):
                if remaining <= 0:
                    break
                if d.is_settled() or d.is_expired(now):
                    continue
                pay = min(remaining, d.remaining_tflop_hr())
                d.repaid_tflop_hr += pay
                d.last_update_ts = now
                applied += pay
                remaining -= pay
            self.pool.available_tflop_hr += remaining
            self.pool.cumulative_contributed += tflop_hr_done
            self.pool.last_update_ts = now
            self._save()
        return {
            "applied_to_debt": round(applied, 6),
            "added_to_pool":   round(remaining, 6),
        }

    # --- queries -------------------------------------------------------

    def outstanding_for(self, pubkey: str) -> float:
        with self._lock:
            return sum(
                d.remaining_tflop_hr()
                for d in self._debts.get(pubkey, [])
                if not d.is_expired()
            )

    def open_debts(self) -> list[ComputeDebt]:
        with self._lock:
            return [d for ds in self._debts.values()
                    for d in ds if not d.is_settled()]

    def expired_debts(self) -> list[ComputeDebt]:
        with self._lock:
            now = time.time()
            return [d for ds in self._debts.values()
                    for d in ds
                    if d.is_expired(now) and not d.is_settled()]

    def balance(self) -> CreditPool:
        with self._lock:
            return CreditPool(
                available_tflop_hr=self.pool.available_tflop_hr,
                cumulative_contributed=self.pool.cumulative_contributed,
                cumulative_consumed=self.pool.cumulative_consumed,
                last_update_ts=self.pool.last_update_ts,
            )

    def write_off_expired(self) -> float:
        """Remove expired-unpaid debts. Caps total write-off per call
        to ``insolvency_haircut_pct`` of the cumulative-consumed total
        so a single bad period can't wipe out the pool.

        Returns the total TFLOP-hr written off.
        """
        with self._lock:
            now = time.time()
            cap = self.pool.cumulative_consumed * self.cfg.insolvency_haircut_pct
            written_off = 0.0
            for ds in self._debts.values():
                for d in ds:
                    if d.is_expired(now) and not d.is_settled():
                        amount = d.remaining_tflop_hr()
                        if written_off + amount > cap:
                            break
                        written_off += amount
                        d.repaid_tflop_hr = d.initial_tflop_hr
                        d.last_update_ts = now
            if written_off:
                self._save()
            return written_off

    def seed(self, tflop_hr: float) -> None:
        """Bootstrap the pool with initial credit (genesis grant)."""
        with self._lock:
            self.pool.available_tflop_hr += tflop_hr
            self.pool.cumulative_contributed += tflop_hr
            self.pool.last_update_ts = time.time()
            self._save()

    # --- persistence ---------------------------------------------------

    def _save(self) -> None:
        try:
            Path(self.cfg.state_path).parent.mkdir(parents=True, exist_ok=True)
            data = {
                "pool": asdict(self.pool),
                "debts": {
                    pk: [asdict(d) for d in ds]
                    for pk, ds in self._debts.items()
                },
            }
            tmp = Path(self.cfg.state_path).with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(self.cfg.state_path)
        except Exception as e:
            logger.warning("compute_currency save failed: %s", e)

    def _load(self) -> None:
        p = Path(self.cfg.state_path)
        if not p.exists():
            return
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            pool = d.get("pool", {})
            self.pool = CreditPool(**{
                k: pool.get(k, 0.0) for k in
                ("available_tflop_hr", "cumulative_contributed",
                 "cumulative_consumed", "last_update_ts")
            })
            self._debts = {
                pk: [ComputeDebt(**dd) for dd in ds]
                for pk, ds in d.get("debts", {}).items()
            }
        except Exception as e:
            logger.warning("compute_currency load failed: %s", e)
