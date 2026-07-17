"""Deposit + withdrawal flows — the cash rails on top of BuyerLedger
(§HG16).

The ledger (core/buyer_ledger.py) is correct internal accounting:
escrow, payouts, commission. This module is the boundary where REAL
money enters and leaves it, built on the honesty rules the project
already enforces:

  * **No mocks in the money path.** Deposits require a configured
    `PaymentGateway` (core/payments.py — real Stripe creds, no fake
    flows). No gateway → the deposit is refused with a clear message,
    never simulated. A wallet is credited ONLY after the gateway
    reports a successful charge.
  * **Idempotent by construction.** A deposit retried with the same
    idempotency key (network blip, crash, double-click) credits the
    wallet exactly once — enforced here by transaction-id dedup, and
    again at the gateway layer (belt + suspenders, matching Stripe's
    own idempotency contract).
  * **Withdrawals are two-phase and can never overdraw.** Phase 1
    (`request_withdrawal`) debits available funds immediately — locked
    (escrowed) money is untouchable — and records a `pending`
    withdrawal. Phase 2 is either `complete_withdrawal` with a real
    payout reference (a Stripe payout id, bank/UPI transaction id —
    automatic when the gateway supports `payout()`, operator-driven
    otherwise) or `cancel_withdrawal`, which returns the funds. Funds
    in a pending withdrawal exist in exactly one place: the withdrawal
    record. Nothing is ever silently in two places or zero places.
  * **Everything persists.** Deposits and withdrawals snapshot
    atomically alongside the ledger, so a restart can neither
    double-credit a deposit nor lose a pending withdrawal.

Operator payouts are a REAL flow, not a shortcut: marketplaces
routinely run manual payout queues before automating. What makes it
honest is the accounting — funds leave the wallet at request time,
the record demands a payout reference to close, and the journal shows
every state transition.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.buyer_ledger import BuyerLedger, InsufficientFunds

__all__ = [
    "DepositRecord",
    "InsufficientFunds",
    "PaymentFlows",
    "PaymentsNotConfigured",
    "UnknownWithdrawal",
    "WithdrawalRecord",
]


class PaymentsNotConfigured(RuntimeError):
    """Deposit attempted with no real payment gateway configured."""


class UnknownWithdrawal(KeyError):
    pass


@dataclass
class DepositRecord:
    deposit_id: str
    wallet_id: str
    amount_usd: Decimal
    transaction_id: str
    idempotency_key: Optional[str]
    created_unix: float = field(default_factory=time.time)
    idempotent_replay: bool = False

    def to_public(self) -> Dict[str, Any]:
        return {
            "deposit_id": self.deposit_id,
            "wallet_id": self.wallet_id,
            "amount_usd": str(self.amount_usd),
            "transaction_id": self.transaction_id,
            "created_unix": self.created_unix,
            "idempotent_replay": self.idempotent_replay,
        }


@dataclass
class WithdrawalRecord:
    withdrawal_id: str
    wallet_id: str
    amount_usd: Decimal
    destination: str               # bank/UPI/Stripe-account descriptor
    state: str = "pending"         # "pending" | "paid" | "cancelled"
    created_unix: float = field(default_factory=time.time)
    payout_reference: Optional[str] = None
    terminal_unix: Optional[float] = None

    def to_public(self) -> Dict[str, Any]:
        return {
            "withdrawal_id": self.withdrawal_id,
            "wallet_id": self.wallet_id,
            "amount_usd": str(self.amount_usd),
            "destination": self.destination,
            "state": self.state,
            "created_unix": self.created_unix,
            "payout_reference": self.payout_reference,
            "terminal_unix": self.terminal_unix,
        }


class PaymentFlows:
    def __init__(self, ledger: BuyerLedger,
                 gateway: Optional[Any] = None,
                 state_dir: Optional[str] = None) -> None:
        self.ledger = ledger
        self.gateway = gateway
        self._lock = threading.RLock()
        self._deposits: Dict[str, DepositRecord] = {}      # by deposit_id
        self._seen_tx: Dict[str, str] = {}                 # tx_id -> deposit_id
        self._seen_keys: Dict[str, str] = {}               # idem key -> deposit_id
        self._withdrawals: Dict[str, WithdrawalRecord] = {}
        self._state_path: Optional[Path] = None
        if state_dir:
            d = Path(state_dir)
            try:
                d.mkdir(parents=True, exist_ok=True)
                self._state_path = d / "payment_flows.json"
            except OSError:
                self._state_path = None
        self._load()

    # ------------------------------------------------------------------
    # Deposits — money IN
    # ------------------------------------------------------------------
    def deposit(self, *, wallet_id: str, amount_usd: Decimal,
                customer_id: str, idempotency_key: Optional[str] = None,
                ) -> DepositRecord:
        """Charge the buyer via the real gateway, then credit their
        wallet. Refuses honestly when no gateway is configured. The
        wallet is credited exactly once per gateway transaction, no
        matter how many times this is retried."""
        if amount_usd <= Decimal("0"):
            raise ValueError("deposit amount must be positive")
        if self.gateway is None:
            raise PaymentsNotConfigured(
                "no payment gateway configured — set "
                "PLUGINFER_STRIPE_SECRET_KEY (pip install stripe) or "
                "inject a PaymentGateway. Deposits are never simulated."
            )
        with self._lock:
            # Replay guard 1: same idempotency key → same deposit.
            if idempotency_key and idempotency_key in self._seen_keys:
                prior = self._deposits[self._seen_keys[idempotency_key]]
                return DepositRecord(**{**prior.__dict__,
                                        "idempotent_replay": True})
        result = self.gateway.charge(
            float(amount_usd), "usd", customer_id,
            description=f"Pluginfer wallet deposit → {wallet_id}",
            idempotency_key=idempotency_key,
        )
        if not result.success or not result.transaction_id:
            raise RuntimeError(
                f"payment failed: {result.error or 'gateway declined'}")
        with self._lock:
            # Replay guard 2: the gateway deduplicated and returned the
            # SAME transaction we already credited — do not credit again.
            if result.transaction_id in self._seen_tx:
                prior = self._deposits[self._seen_tx[result.transaction_id]]
                return DepositRecord(**{**prior.__dict__,
                                        "idempotent_replay": True})
            rec = DepositRecord(
                deposit_id="dep-" + secrets.token_urlsafe(10),
                wallet_id=wallet_id, amount_usd=amount_usd,
                transaction_id=result.transaction_id,
                idempotency_key=idempotency_key,
            )
            self.ledger.credit(
                wallet_id, amount_usd,
                note=f"deposit {rec.deposit_id} "
                     f"(tx {result.transaction_id})")
            self._deposits[rec.deposit_id] = rec
            self._seen_tx[result.transaction_id] = rec.deposit_id
            if idempotency_key:
                self._seen_keys[idempotency_key] = rec.deposit_id
            self._save()
            return rec

    # ------------------------------------------------------------------
    # Withdrawals — money OUT (two-phase)
    # ------------------------------------------------------------------
    def request_withdrawal(self, *, wallet_id: str, amount_usd: Decimal,
                           destination: str) -> WithdrawalRecord:
        """Phase 1: debit the wallet NOW (overdraw and escrowed funds
        are refused by the ledger) and open a pending withdrawal. If
        the gateway supports automatic payouts, one is attempted
        immediately; failure leaves the record pending for the
        operator — funds stay safely held either way."""
        if not destination.strip():
            raise ValueError("destination is required — a withdrawal "
                             "must say where the money goes")
        with self._lock:
            wid = "wd-" + secrets.token_urlsafe(10)
            # Debit first: after this the funds live ONLY in this record.
            self.ledger.debit(
                wallet_id, amount_usd,
                note=f"withdrawal {wid} → {destination}")
            rec = WithdrawalRecord(
                withdrawal_id=wid, wallet_id=wallet_id,
                amount_usd=amount_usd, destination=destination,
            )
            self._withdrawals[wid] = rec
            self._save()
        # Automatic payout when the gateway offers it (optional
        # capability — the base PaymentGateway has charge/refund only).
        payout = getattr(self.gateway, "payout", None)
        if callable(payout):
            try:
                res = payout(float(amount_usd), "usd", destination,
                             idempotency_key=wid)
                if getattr(res, "success", False) and \
                        getattr(res, "transaction_id", None):
                    return self.complete_withdrawal(
                        wid, payout_reference=res.transaction_id)
            except Exception:
                pass        # stays pending; operator pays out manually
        return rec

    def complete_withdrawal(self, withdrawal_id: str, *,
                            payout_reference: str) -> WithdrawalRecord:
        """Phase 2a: close a pending withdrawal against a REAL payout
        reference (Stripe payout id / bank / UPI transaction id).
        Refuses an empty reference — 'trust me' is not a reference."""
        if not payout_reference.strip():
            raise ValueError("payout_reference is required")
        with self._lock:
            rec = self._withdrawals.get(withdrawal_id)
            if rec is None:
                raise UnknownWithdrawal(withdrawal_id)
            if rec.state == "paid":
                return rec                    # idempotent
            if rec.state == "cancelled":
                raise RuntimeError(
                    f"withdrawal {withdrawal_id} was cancelled — funds "
                    f"already returned; cannot mark paid")
            rec.state = "paid"
            rec.payout_reference = payout_reference
            rec.terminal_unix = time.time()
            self._save()
            return rec

    def cancel_withdrawal(self, withdrawal_id: str) -> WithdrawalRecord:
        """Phase 2b: return the held funds to the wallet. Idempotent;
        refuses to cancel an already-paid withdrawal."""
        with self._lock:
            rec = self._withdrawals.get(withdrawal_id)
            if rec is None:
                raise UnknownWithdrawal(withdrawal_id)
            if rec.state == "cancelled":
                return rec                    # idempotent
            if rec.state == "paid":
                raise RuntimeError(
                    f"withdrawal {withdrawal_id} already paid out — "
                    f"cannot cancel")
            self.ledger.credit(
                rec.wallet_id, rec.amount_usd,
                note=f"withdrawal {withdrawal_id} cancelled — funds "
                     f"returned")
            rec.state = "cancelled"
            rec.terminal_unix = time.time()
            self._save()
            return rec

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------
    def withdrawals(self, *, wallet_id: Optional[str] = None
                    ) -> List[Dict[str, Any]]:
        with self._lock:
            return [w.to_public() for w in self._withdrawals.values()
                    if wallet_id is None or w.wallet_id == wallet_id]

    def deposits(self, *, wallet_id: Optional[str] = None
                 ) -> List[Dict[str, Any]]:
        with self._lock:
            return [d.to_public() for d in self._deposits.values()
                    if wallet_id is None or d.wallet_id == wallet_id]

    def get_withdrawal(self, withdrawal_id: str
                       ) -> Optional[WithdrawalRecord]:
        with self._lock:
            return self._withdrawals.get(withdrawal_id)

    # ------------------------------------------------------------------
    # Persistence — same atomic-snapshot discipline as the ledger
    # ------------------------------------------------------------------
    def _save(self) -> None:
        if self._state_path is None:
            return
        try:
            data = {
                "deposits": {
                    did: {**d.to_public(),
                          "idempotency_key": d.idempotency_key}
                    for did, d in self._deposits.items()
                },
                "withdrawals": {wid: w.to_public()
                                for wid, w in self._withdrawals.items()},
            }
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError as exc:
            import logging
            logging.getLogger(__name__).error(
                "payment flows snapshot FAILED: %s", exc)

    def _load(self) -> None:
        if self._state_path is None:
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        with self._lock:
            for did, dd in data.get("deposits", {}).items():
                rec = DepositRecord(
                    deposit_id=did, wallet_id=dd["wallet_id"],
                    amount_usd=Decimal(dd["amount_usd"]),
                    transaction_id=dd["transaction_id"],
                    idempotency_key=dd.get("idempotency_key"),
                    created_unix=dd.get("created_unix", 0.0),
                )
                self._deposits[did] = rec
                self._seen_tx[rec.transaction_id] = did
                if rec.idempotency_key:
                    self._seen_keys[rec.idempotency_key] = did
            for wid, wd in data.get("withdrawals", {}).items():
                self._withdrawals[wid] = WithdrawalRecord(
                    withdrawal_id=wid, wallet_id=wd["wallet_id"],
                    amount_usd=Decimal(wd["amount_usd"]),
                    destination=wd["destination"],
                    state=wd.get("state", "pending"),
                    created_unix=wd.get("created_unix", 0.0),
                    payout_reference=wd.get("payout_reference"),
                    terminal_unix=wd.get("terminal_unix"),
                )
