"""Buyer wallet + escrow ledger — the money flow that makes the
mesh economics work without Pluginfer ever subsidising a job.

Economic invariants (these are the bar — every test pins one):

  1. Pre-execution: the buyer's balance is REDUCED by the auction's
     locked price. The locked amount lives in escrow until a
     terminal state.
  2. On success: the locked amount splits into
       (1 - commission_rate) × locked → provider wallet
       commission_rate × locked        → Pluginfer treasury
     We always take a cut; we never pay out from treasury.
  3. On failure: the full locked amount returns to the buyer. We
     made zero on this job — but we also lost zero. Worst case is
     break-even, never negative.
  4. Idempotency: a `lock` / `release` / `refund` keyed by the same
     job_id is a no-op on the second call. Crash-safe.
  5. Negative-balance impossible: lock rejects if buyer can't
     afford the price. The auction caller surfaces this as a
     402 Payment Required to the buyer's request path.

The commission rate is env-tunable via `PLUGINFER_COMMISSION_RATE`
(default 0.10 = 10%). Forward-compatible with a per-tier or
per-region rate schedule via a future BuyerWallet.rate_override.

This module ships with an in-memory ledger. The on-chain version
lives in `core/compute_ledger.py`; this is the off-chain accounting
that the gateway runs synchronously inside the request path. Both
write to the same conceptual ledger; the on-chain version is the
durable receipt.

Innovation worth filing: §A25 "Auction-driven escrow with built-in
operator margin." The commission is enforced at release time, not
at price-discovery time, so the auction can clear at the buyer's
true willingness to pay; Pluginfer's economics are a function of
clearing prices, never a price floor that distorts them.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

COMMISSION_RATE = Decimal(os.environ.get("PLUGINFER_COMMISSION_RATE", "0.10"))
TREASURY_WALLET_ID = "pluginfer-treasury"
MIN_BALANCE = Decimal("0")


class FaucetAlreadyGranted(ValueError):
    """Second faucet request for the same wallet."""


class InsufficientFunds(RuntimeError):
    """Buyer doesn't have enough credit to lock the auction price."""


class UnknownEscrow(RuntimeError):
    """release/refund called for a job_id we don't have escrow for.
    Idempotent paths treat this as success (already settled)."""


@dataclass
class WalletEntry:
    """A single accounting line in the ledger.

    `kind` is one of: `credit`, `debit`, `lock`, `release`, `refund`,
    `commission`. The sum of (credits - debits - locks + releases +
    refunds + commissions_in - commissions_out) is the wallet's
    instantaneous balance."""
    kind: str
    amount_usd: Decimal
    timestamp_unix: float
    counterparty_id: Optional[str] = None
    job_id: Optional[str] = None
    note: str = ""


@dataclass
class BuyerWallet:
    """One wallet — buyer, provider, or treasury. The same struct
    serves all three roles to keep the ledger uniform.

    `available_usd` is the balance minus everything currently
    locked in escrow. `locked_usd` is the sum of un-released
    escrow holds. `balance_usd` returns available_usd for the
    common "how much can I spend" question."""
    wallet_id: str
    role: str = "buyer"            # "buyer" | "provider" | "treasury"
    available_usd: Decimal = Decimal("0")
    locked_usd: Decimal = Decimal("0")
    entries: List[WalletEntry] = field(default_factory=list, repr=False)

    @property
    def balance_usd(self) -> Decimal:
        return self.available_usd

    def to_public(self) -> Dict[str, Any]:
        return {
            "wallet_id": self.wallet_id,
            "role": self.role,
            "available_usd": str(self.available_usd),
            "locked_usd": str(self.locked_usd),
            "balance_usd": str(self.balance_usd),
            "entries_n": len(self.entries),
        }


@dataclass
class EscrowRecord:
    """One job's escrow hold. Survives until release/refund. Idempotent
    via `state in {locked, released, refunded}` — second call is a
    no-op (with a counter-resync if the wallets drifted)."""
    job_id: str
    buyer_wallet_id: str
    locked_usd: Decimal
    state: str = "locked"          # "locked" | "released" | "refunded"
    created_unix: float = field(default_factory=time.time)
    # On `release`, fill out who got paid + how much.
    provider_wallet_id: Optional[str] = None
    provider_amount: Optional[Decimal] = None
    commission_amount: Optional[Decimal] = None
    terminal_unix: Optional[float] = None
    # For consortium jobs, the per-member payout split so dispute()
    # can claw back from each one. None for single-winner jobs.
    consortium_members: Optional[List[Tuple[str, Decimal]]] = None


class BuyerLedger:
    """The off-chain wallet ledger. Synchronous, thread-safe via a
    single RLock — the gateway dispatches escrow ops on the auction
    thread, so any cross-thread access is wallet-id-keyed already.

    This class is the SINGLE point that mutates wallet balances.
    Every state transition writes a `WalletEntry` so audit can
    reconstruct any wallet's history.
    """

    def __init__(self, state_dir: Optional[str] = None) -> None:
        self._wallets: Dict[str, BuyerWallet] = {}
        self._escrows: Dict[str, EscrowRecord] = {}
        self._lock = threading.RLock()
        # Money records MUST survive restarts — an in-memory-only ledger
        # would wipe every balance and every commission on reboot, which
        # is unacceptable for anything called "money". With a state_dir,
        # every mutation snapshots atomically; without one (unit tests),
        # the ledger is explicitly ephemeral.
        self._state_path = None
        if state_dir:
            from pathlib import Path
            d = Path(state_dir)
            try:
                d.mkdir(parents=True, exist_ok=True)
                self._state_path = d / "money_ledger.json"
            except OSError:
                self._state_path = None
        # Treasury wallet always exists — every commission lands here.
        self._wallets[TREASURY_WALLET_ID] = BuyerWallet(
            wallet_id=TREASURY_WALLET_ID, role="treasury",
        )
        # Non-empty = the persisted state failed verification. Money
        # NEVER leaves (withdrawals refuse) while this is non-empty.
        self.integrity_alerts: List[str] = []
        self._load()
        bal = self.verify_balances()
        if not bal["ok"]:
            import logging as _lg
            _lg.getLogger(__name__).critical(
                "MONEY LEDGER INTEGRITY FAILURE: %d wallet(s) do not "
                "match their entry history: %s",
                len(bal["mismatches"]), bal["mismatches"])
            self.integrity_alerts.append(
                f"balance/history mismatch: {bal['mismatches']}")

    # ------------------------------------------------------------------
    # Persistence — atomic snapshot on every mutation
    # ------------------------------------------------------------------
    def _save(self) -> None:
        """Snapshot wallets + escrows. Called under self._lock by every
        mutator. Atomic (tmp + replace) so a crash mid-write can never
        leave a half-written money file."""
        if self._state_path is None:
            return
        import json
        try:
            data = {
                "wallets": {
                    wid: {
                        "role": w.role,
                        "available_usd": str(w.available_usd),
                        "locked_usd": str(w.locked_usd),
                        "entries": [
                            {"kind": e.kind, "amount_usd": str(e.amount_usd),
                             "timestamp_unix": e.timestamp_unix,
                             "counterparty_id": e.counterparty_id,
                             "job_id": e.job_id, "note": e.note}
                            for e in w.entries
                        ],
                    } for wid, w in self._wallets.items()
                },
                "escrows": {
                    jid: {
                        "buyer_wallet_id": e.buyer_wallet_id,
                        "locked_usd": str(e.locked_usd),
                        "state": e.state,
                        "created_unix": e.created_unix,
                        "provider_wallet_id": e.provider_wallet_id,
                        "provider_amount": (str(e.provider_amount)
                                            if e.provider_amount is not None
                                            else None),
                        "commission_amount": (str(e.commission_amount)
                                              if e.commission_amount
                                              is not None else None),
                        "terminal_unix": e.terminal_unix,
                        "consortium_members": (
                            [[pid, str(s)] for pid, s in
                             e.consortium_members]
                            if e.consortium_members else None),
                    } for jid, e in self._escrows.items()
                },
            }
            # Tamper-evidence seal: sha256 over the canonical body.
            # Honest scope — this catches corruption and casual edits;
            # a host who edits AND re-hashes with the code in hand is
            # not detectable locally. That is why authoritative
            # settlement (real payouts) is never trusted to a remote
            # host's file, and why withdrawals refuse while integrity
            # is in doubt (see payment_flows).
            import hashlib as _hl
            data["integrity_sha256"] = _hl.sha256(
                json.dumps(data, sort_keys=True).encode("utf-8")
            ).hexdigest()
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self._state_path)
            marker = self._state_path.with_suffix(".exists")
            if not marker.exists():
                marker.write_text("money ledger created; if the .json "
                                  "is ever missing while this file "
                                  "remains, the state was deleted\n",
                                  encoding="utf-8")
        except OSError as exc:
            import logging
            logging.getLogger(__name__).error(
                "money ledger snapshot FAILED: %s", exc)

    def _load(self) -> None:
        if self._state_path is None:
            return
        # Deletion-evidence: the marker outlives the ledger file. A
        # missing ledger with a present marker is a wipe, not a fresh
        # install — flagged, and withdrawals stay blocked.
        marker = self._state_path.with_suffix(".exists")
        if not self._state_path.exists():
            if marker.exists():
                import logging as _lg
                _lg.getLogger(__name__).critical(
                    "MONEY LEDGER INTEGRITY FAILURE: ledger file is "
                    "missing but its marker exists — state was deleted. "
                    "Withdrawals are BLOCKED until resolved.")
                self.integrity_alerts.append(
                    "ledger file missing but marker present — deleted?")
            return
        import json
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        claimed = data.pop("integrity_sha256", None)
        if claimed is not None:
            import hashlib as _hl
            actual = _hl.sha256(
                json.dumps(data, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if actual != claimed:
                import logging as _lg
                _lg.getLogger(__name__).critical(
                    "MONEY LEDGER INTEGRITY FAILURE: snapshot hash "
                    "mismatch — file was edited or corrupted. "
                    "Withdrawals are BLOCKED until resolved.")
                self.integrity_alerts.append(
                    "snapshot sha256 mismatch on load")
        with self._lock:
            for wid, wd in data.get("wallets", {}).items():
                w = BuyerWallet(
                    wallet_id=wid, role=wd.get("role", "buyer"),
                    available_usd=Decimal(wd["available_usd"]),
                    locked_usd=Decimal(wd["locked_usd"]),
                )
                w.entries = [
                    WalletEntry(
                        kind=ed["kind"],
                        amount_usd=Decimal(ed["amount_usd"]),
                        timestamp_unix=ed["timestamp_unix"],
                        counterparty_id=ed.get("counterparty_id"),
                        job_id=ed.get("job_id"), note=ed.get("note", ""),
                    ) for ed in wd.get("entries", [])
                ]
                self._wallets[wid] = w
            for jid, ed in data.get("escrows", {}).items():
                esc = EscrowRecord(
                    job_id=jid,
                    buyer_wallet_id=ed["buyer_wallet_id"],
                    locked_usd=Decimal(ed["locked_usd"]),
                    state=ed.get("state", "locked"),
                    created_unix=ed.get("created_unix", 0.0),
                    provider_wallet_id=ed.get("provider_wallet_id"),
                    provider_amount=(Decimal(ed["provider_amount"])
                                     if ed.get("provider_amount") else None),
                    commission_amount=(Decimal(ed["commission_amount"])
                                       if ed.get("commission_amount")
                                       else None),
                    terminal_unix=ed.get("terminal_unix"),
                    consortium_members=(
                        [(pid, Decimal(s)) for pid, s in
                         ed["consortium_members"]]
                        if ed.get("consortium_members") else None),
                )
                self._escrows[jid] = esc

    # ------------------------------------------------------------------
    # Wallet lifecycle
    # ------------------------------------------------------------------
    def get_or_create_wallet(
        self, wallet_id: str, *, role: str = "buyer",
    ) -> BuyerWallet:
        with self._lock:
            w = self._wallets.get(wallet_id)
            if w is None:
                w = BuyerWallet(wallet_id=wallet_id, role=role)
                self._wallets[wallet_id] = w
            return w

    def get_wallet(self, wallet_id: str) -> Optional[BuyerWallet]:
        with self._lock:
            return self._wallets.get(wallet_id)

    def credit(
        self, wallet_id: str, amount_usd: Decimal,
        *, note: str = "",
    ) -> BuyerWallet:
        """Add funds — buyer top-up, provider payout, treasury deposit.
        amount_usd must be > 0; we refuse negative credits."""
        if amount_usd <= Decimal("0"):
            raise ValueError("credit amount must be positive")
        with self._lock:
            w = self.get_or_create_wallet(wallet_id)
            w.available_usd += amount_usd
            w.entries.append(WalletEntry(
                kind="credit", amount_usd=amount_usd,
                timestamp_unix=time.time(), note=note,
            ))
            self._save()
            return w

    FAUCET_NOTE = "testnet-faucet"

    def faucet_grant(
        self, wallet_id: str, amount_usd: Decimal,
    ) -> BuyerWallet:
        """One-time starter credit so anyone can try the mesh as a buyer.

        TESTNET ONLY — the caller (the node endpoint) must refuse this
        in mainnet mode: an operator-minted mainnet balance would be a
        treasury subsidy, which this project never does. Idempotent per
        wallet: a second grant raises, so joining twice can't farm it.
        """
        if amount_usd <= Decimal("0"):
            raise ValueError("faucet amount must be positive")
        with self._lock:
            w = self.get_or_create_wallet(wallet_id)
            if any(e.kind == "credit" and e.note == self.FAUCET_NOTE
                   for e in w.entries):
                raise FaucetAlreadyGranted(
                    f"wallet {wallet_id} already received the testnet "
                    f"faucet grant")
            return self.credit(wallet_id, amount_usd,
                               note=self.FAUCET_NOTE)

    def debit(
        self, wallet_id: str, amount_usd: Decimal,
        *, note: str = "",
    ) -> BuyerWallet:
        """Remove available funds — the withdrawal leg. Refuses to
        overdraw and refuses to touch locked (escrowed) funds; a
        withdrawal can never claw money out of an in-flight job."""
        if amount_usd <= Decimal("0"):
            raise ValueError("debit amount must be positive")
        with self._lock:
            w = self._wallets.get(wallet_id)
            if w is None or w.available_usd < amount_usd:
                have = w.available_usd if w else Decimal("0")
                raise InsufficientFunds(
                    f"wallet {wallet_id} has {have} USD available, "
                    f"needs {amount_usd}"
                )
            w.available_usd -= amount_usd
            w.entries.append(WalletEntry(
                kind="debit", amount_usd=amount_usd,
                timestamp_unix=time.time(), note=note,
            ))
            self._save()
            return w

    # ------------------------------------------------------------------
    # Escrow — the auction lock
    # ------------------------------------------------------------------
    def lock_for_job(
        self, *, buyer_wallet_id: str, job_id: str,
        amount_usd: Decimal,
    ) -> EscrowRecord:
        """Pre-execution: move `amount_usd` from the buyer's
        available balance into escrow. Refuses if balance < amount.
        Idempotent on (job_id) — a second call returns the existing
        record without double-locking."""
        with self._lock:
            existing = self._escrows.get(job_id)
            if existing is not None:
                return existing
            w = self.get_or_create_wallet(buyer_wallet_id)
            if w.available_usd < amount_usd:
                raise InsufficientFunds(
                    f"wallet {buyer_wallet_id} has {w.available_usd} USD "
                    f"available, needs {amount_usd}"
                )
            w.available_usd -= amount_usd
            w.locked_usd += amount_usd
            w.entries.append(WalletEntry(
                kind="lock", amount_usd=amount_usd,
                timestamp_unix=time.time(), job_id=job_id,
                note="auction escrow lock",
            ))
            esc = EscrowRecord(
                job_id=job_id, buyer_wallet_id=buyer_wallet_id,
                locked_usd=amount_usd,
            )
            self._escrows[job_id] = esc
            self._save()
            return esc

    def release_to_provider(
        self, *, job_id: str, provider_wallet_id: str,
        commission_rate: Optional[Decimal] = None,
    ) -> EscrowRecord:
        """Job succeeded: split the locked amount → provider +
        treasury. Buyer's locked_usd decreases by the full amount;
        provider's available_usd grows by (1-c) × locked; treasury
        grows by c × locked.

        Idempotent on job_id. A release of a job that doesn't exist
        raises UnknownEscrow so callers can decide whether to
        surface or swallow."""
        if commission_rate is None:
            commission_rate = COMMISSION_RATE
        with self._lock:
            esc = self._escrows.get(job_id)
            if esc is None:
                raise UnknownEscrow(f"no escrow for job {job_id}")
            if esc.state == "released":
                return esc      # idempotent — return prior record
            if esc.state == "refunded":
                raise RuntimeError(
                    f"job {job_id} already refunded — cannot release"
                )
            commission = (esc.locked_usd * commission_rate).quantize(
                Decimal("0.00000001")
            )
            to_provider = esc.locked_usd - commission
            buyer = self.get_or_create_wallet(esc.buyer_wallet_id)
            buyer.locked_usd -= esc.locked_usd
            buyer.entries.append(WalletEntry(
                kind="debit", amount_usd=esc.locked_usd,
                timestamp_unix=time.time(),
                counterparty_id=provider_wallet_id,
                job_id=job_id, note="escrow released",
            ))
            prov = self.get_or_create_wallet(
                provider_wallet_id, role="provider",
            )
            prov.available_usd += to_provider
            prov.entries.append(WalletEntry(
                kind="release", amount_usd=to_provider,
                timestamp_unix=time.time(),
                counterparty_id=esc.buyer_wallet_id, job_id=job_id,
                note=f"earnings for job (commission {commission_rate * 100}%)",
            ))
            treas = self._wallets[TREASURY_WALLET_ID]
            treas.available_usd += commission
            treas.entries.append(WalletEntry(
                kind="commission", amount_usd=commission,
                timestamp_unix=time.time(),
                counterparty_id=esc.buyer_wallet_id, job_id=job_id,
                note=f"{commission_rate * 100}% of {esc.locked_usd}",
            ))
            esc.state = "released"
            esc.provider_wallet_id = provider_wallet_id
            esc.provider_amount = to_provider
            esc.commission_amount = commission
            esc.terminal_unix = time.time()
            self._save()
            return esc

    def refund_to_buyer(self, *, job_id: str) -> EscrowRecord:
        """Job failed: full locked amount returns to buyer's available
        balance. Pluginfer takes nothing on a failed job — but also
        loses nothing. Idempotent on job_id."""
        with self._lock:
            esc = self._escrows.get(job_id)
            if esc is None:
                raise UnknownEscrow(f"no escrow for job {job_id}")
            if esc.state == "refunded":
                return esc      # idempotent
            if esc.state == "released":
                raise RuntimeError(
                    f"job {job_id} already released — cannot refund"
                )
            buyer = self.get_or_create_wallet(esc.buyer_wallet_id)
            buyer.locked_usd -= esc.locked_usd
            buyer.available_usd += esc.locked_usd
            buyer.entries.append(WalletEntry(
                kind="refund", amount_usd=esc.locked_usd,
                timestamp_unix=time.time(), job_id=job_id,
                note="escrow refunded — job failed",
            ))
            esc.state = "refunded"
            esc.terminal_unix = time.time()
            self._save()
            return esc

    def split_release_to_consortium(
        self, *, job_id: str, members: List[Tuple[str, Decimal]],
        commission_rate: Optional[Decimal] = None,
    ) -> EscrowRecord:
        """For a consortium job: every successful member gets their
        share (proportional to their bid.price_usd within the
        consortium). Each share is post-commission. The buyer's
        locked amount drops by the released total; any remainder
        (failed members' shares) is automatically refunded to the
        buyer's available balance.

        `members` is [(provider_wallet_id, member_price_usd), ...].
        sum(member_prices) MAY be ≤ locked_usd; the difference is
        the partial refund."""
        if commission_rate is None:
            commission_rate = COMMISSION_RATE
        with self._lock:
            esc = self._escrows.get(job_id)
            if esc is None:
                raise UnknownEscrow(f"no escrow for job {job_id}")
            if esc.state == "released":
                return esc
            if esc.state == "refunded":
                raise RuntimeError(
                    f"job {job_id} already refunded — cannot release"
                )
            total = sum((m[1] for m in members), Decimal("0"))
            assert total <= esc.locked_usd, (
                f"consortium split exceeds locked amount: "
                f"sum={total}, locked={esc.locked_usd}"
            )
            refund_amount = esc.locked_usd - total
            buyer = self.get_or_create_wallet(esc.buyer_wallet_id)
            buyer.locked_usd -= esc.locked_usd
            buyer.entries.append(WalletEntry(
                kind="debit", amount_usd=total,
                timestamp_unix=time.time(),
                job_id=job_id,
                note=f"escrow released to consortium of {len(members)}",
            ))
            if refund_amount > Decimal("0"):
                # Partial refund — failed members' shares back to buyer.
                buyer.available_usd += refund_amount
                buyer.entries.append(WalletEntry(
                    kind="refund", amount_usd=refund_amount,
                    timestamp_unix=time.time(), job_id=job_id,
                    note="partial refund for failed consortium members",
                ))
            # Record per-member payout details on the escrow so a
            # later dispute() can claw back from each member.
            esc.consortium_members = [
                (str(pid), Decimal(str(share))) for pid, share in members
            ]
            treas = self._wallets[TREASURY_WALLET_ID]
            total_commission = Decimal("0")
            for prov_id, share in members:
                commission = (share * commission_rate).quantize(
                    Decimal("0.00000001")
                )
                to_provider = share - commission
                prov = self.get_or_create_wallet(prov_id, role="provider")
                prov.available_usd += to_provider
                prov.entries.append(WalletEntry(
                    kind="release", amount_usd=to_provider,
                    timestamp_unix=time.time(),
                    counterparty_id=esc.buyer_wallet_id, job_id=job_id,
                    note="consortium share",
                ))
                total_commission += commission
            treas.available_usd += total_commission
            treas.entries.append(WalletEntry(
                kind="commission", amount_usd=total_commission,
                timestamp_unix=time.time(),
                counterparty_id=esc.buyer_wallet_id, job_id=job_id,
                note=f"consortium commission ({commission_rate * 100}%)",
            ))
            esc.state = "released"
            esc.commission_amount = total_commission
            esc.terminal_unix = time.time()
            self._save()
            return esc

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------
    def escrow_for(self, job_id: str) -> Optional[EscrowRecord]:
        with self._lock:
            return self._escrows.get(job_id)

    def treasury_balance(self) -> Decimal:
        with self._lock:
            return self._wallets[TREASURY_WALLET_ID].available_usd

    def verify_balances(self) -> Dict[str, Any]:
        """Recompute every wallet's balances from its full entry history
        and compare with the stored figures. An edited balance without a
        consistently forged history shows up here immediately.

        Entry semantics (single source of truth for auditors):
          credit            → available += amount
          debit (job_id)    → locked    -= amount   (escrow settlement)
          debit (no job_id) → available -= amount   (withdrawal)
          lock              → available -= amount, locked += amount
          refund            → available += amount, locked -= amount
          release           → available += amount   (provider earnings)
          commission        → available += amount   (treasury only)
        """
        mismatches = []
        with self._lock:
            for wid, w in self._wallets.items():
                av = Decimal("0")
                lk = Decimal("0")
                for e in w.entries:
                    a = e.amount_usd
                    if e.kind == "credit":
                        av += a
                    elif e.kind == "debit":
                        if e.job_id:
                            lk -= a
                        else:
                            av -= a
                    elif e.kind == "lock":
                        av -= a
                        lk += a
                    elif e.kind == "refund":
                        av += a
                        lk -= a
                    elif e.kind in ("release", "commission"):
                        av += a
                    else:
                        mismatches.append(
                            f"{wid}: unknown entry kind {e.kind!r}")
                if av != w.available_usd or lk != w.locked_usd:
                    mismatches.append(
                        f"{wid}: stored available={w.available_usd} "
                        f"locked={w.locked_usd}, history says "
                        f"available={av} locked={lk}")
            treas = self._wallets.get(TREASURY_WALLET_ID)
            if treas is not None:
                comm = sum((e.amount_usd for e in treas.entries
                            if e.kind == "commission"), Decimal("0"))
                non_comm = [e for e in treas.entries
                            if e.kind != "commission"]
                if non_comm:
                    mismatches.append(
                        f"treasury has {len(non_comm)} non-commission "
                        f"entries — treasury only ever earns commission")
                if comm != treas.available_usd:
                    mismatches.append(
                        f"treasury balance {treas.available_usd} != "
                        f"sum of commissions {comm}")
        return {"ok": not mismatches and not self.integrity_alerts,
                "mismatches": mismatches,
                "integrity_alerts": list(self.integrity_alerts)}

    def treasury_report(self, *, limit: int = 100) -> Dict[str, Any]:
        """The commission book, made visible: total earned, entry count,
        and the most recent commission entries with the job + buyer each
        one came from. This is the operator's revenue view."""
        with self._lock:
            treas = self._wallets[TREASURY_WALLET_ID]
            entries = [e for e in treas.entries if e.kind == "commission"]
            return {
                "treasury_balance_usd": str(treas.available_usd),
                "commission_rate": str(COMMISSION_RATE),
                "commission_entries_n": len(entries),
                "recent_commissions": [
                    {"amount_usd": str(e.amount_usd),
                     "job_id": e.job_id,
                     "buyer_wallet_id": e.counterparty_id,
                     "ts": e.timestamp_unix, "note": e.note}
                    for e in entries[-max(1, min(limit, 1000)):]
                ],
            }


__all__ = [
    "BuyerLedger",
    "FaucetAlreadyGranted",
    "BuyerWallet",
    "COMMISSION_RATE",
    "EscrowRecord",
    "InsufficientFunds",
    "TREASURY_WALLET_ID",
    "UnknownEscrow",
    "WalletEntry",
]
