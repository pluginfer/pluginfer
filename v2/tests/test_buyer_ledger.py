"""Buyer wallet + escrow + commission ledger — the money invariants.

Every test pins ONE of the five non-negotiable economic invariants:
  (a) pre-execution lock reduces buyer balance,
  (b) success releases (1-c)×locked to provider, c×locked to treasury,
  (c) failure refunds 100% to buyer,
  (d) lock/release/refund are idempotent on job_id,
  (e) negative balance is impossible — InsufficientFunds raised.

Plus consortium split + partial-failure refund.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from core.buyer_ledger import (
    BuyerLedger,
    COMMISSION_RATE,
    InsufficientFunds,
    TREASURY_WALLET_ID,
    UnknownEscrow,
)


def _fund(ledger, wallet_id, amount):
    ledger.credit(wallet_id, Decimal(str(amount)))


# ---------------------------------------------------------------------------
# Invariant (e): negative balance impossible
# ---------------------------------------------------------------------------

def test_lock_refuses_to_overdraft():
    led = BuyerLedger()
    _fund(led, "alice", "10.0")
    with pytest.raises(InsufficientFunds):
        led.lock_for_job(
            buyer_wallet_id="alice", job_id="j1",
            amount_usd=Decimal("20.0"),
        )
    # Alice's balance is unchanged after the failed lock.
    assert led.get_wallet("alice").available_usd == Decimal("10.0")


def test_lock_at_exactly_balance_works():
    led = BuyerLedger()
    _fund(led, "alice", "10.0")
    esc = led.lock_for_job(
        buyer_wallet_id="alice", job_id="j1",
        amount_usd=Decimal("10.0"),
    )
    assert esc.locked_usd == Decimal("10.0")
    assert led.get_wallet("alice").available_usd == Decimal("0")
    assert led.get_wallet("alice").locked_usd == Decimal("10.0")


# ---------------------------------------------------------------------------
# Invariant (a): lock reduces buyer available + parks in locked
# ---------------------------------------------------------------------------

def test_lock_moves_balance_to_locked():
    led = BuyerLedger()
    _fund(led, "alice", "100.0")
    led.lock_for_job(
        buyer_wallet_id="alice", job_id="j1",
        amount_usd=Decimal("25.0"),
    )
    w = led.get_wallet("alice")
    assert w.available_usd == Decimal("75.0")
    assert w.locked_usd == Decimal("25.0")


# ---------------------------------------------------------------------------
# Invariant (b): success splits (1-c) → provider, c → treasury
# ---------------------------------------------------------------------------

def test_release_pays_provider_minus_commission_treasury_gets_rest():
    led = BuyerLedger()
    _fund(led, "alice", "100.0")
    led.lock_for_job(
        buyer_wallet_id="alice", job_id="j1",
        amount_usd=Decimal("10.0"),
    )
    led.release_to_provider(job_id="j1", provider_wallet_id="bob")
    bob = led.get_wallet("bob")
    treas = led.get_wallet(TREASURY_WALLET_ID)
    expected_commission = Decimal("10.0") * COMMISSION_RATE
    expected_provider = Decimal("10.0") - expected_commission
    assert bob.available_usd == expected_provider
    assert treas.available_usd == expected_commission
    # Buyer's locked_usd is zero again — escrow drained.
    assert led.get_wallet("alice").locked_usd == Decimal("0")
    # Sum invariant: alice paid 10; bob got (1-c)*10; treasury got c*10.
    # All accounted for.
    assert bob.available_usd + treas.available_usd == Decimal("10.0")


def test_release_with_custom_commission_overrides_env_rate():
    led = BuyerLedger()
    _fund(led, "alice", "100.0")
    led.lock_for_job(
        buyer_wallet_id="alice", job_id="j1",
        amount_usd=Decimal("10.0"),
    )
    led.release_to_provider(
        job_id="j1", provider_wallet_id="bob",
        commission_rate=Decimal("0.20"),     # bumped commission
    )
    treas = led.get_wallet(TREASURY_WALLET_ID)
    assert treas.available_usd == Decimal("2.0")
    assert led.get_wallet("bob").available_usd == Decimal("8.0")


# ---------------------------------------------------------------------------
# Invariant (c): failure refunds 100% to buyer
# ---------------------------------------------------------------------------

def test_refund_restores_full_lock_to_buyer():
    led = BuyerLedger()
    _fund(led, "alice", "50.0")
    led.lock_for_job(
        buyer_wallet_id="alice", job_id="j1",
        amount_usd=Decimal("12.5"),
    )
    led.refund_to_buyer(job_id="j1")
    w = led.get_wallet("alice")
    assert w.available_usd == Decimal("50.0")
    assert w.locked_usd == Decimal("0")
    # Treasury didn't get anything — failed job earns nothing.
    assert led.get_wallet(TREASURY_WALLET_ID).available_usd == Decimal("0")


# ---------------------------------------------------------------------------
# Invariant (d): idempotency
# ---------------------------------------------------------------------------

def test_lock_idempotent_returns_same_record():
    led = BuyerLedger()
    _fund(led, "alice", "100.0")
    e1 = led.lock_for_job(
        buyer_wallet_id="alice", job_id="j1",
        amount_usd=Decimal("10.0"),
    )
    e2 = led.lock_for_job(
        buyer_wallet_id="alice", job_id="j1",
        amount_usd=Decimal("10.0"),
    )
    assert e1 is e2
    # Balance unchanged on the duplicate call.
    assert led.get_wallet("alice").available_usd == Decimal("90.0")


def test_release_idempotent_no_double_pay():
    led = BuyerLedger()
    _fund(led, "alice", "100.0")
    led.lock_for_job(
        buyer_wallet_id="alice", job_id="j1",
        amount_usd=Decimal("10.0"),
    )
    led.release_to_provider(job_id="j1", provider_wallet_id="bob")
    led.release_to_provider(job_id="j1", provider_wallet_id="bob")  # dup
    bob = led.get_wallet("bob")
    # Bob got paid exactly once.
    expected = Decimal("10.0") - (Decimal("10.0") * COMMISSION_RATE)
    assert bob.available_usd == expected


def test_release_then_refund_is_rejected():
    led = BuyerLedger()
    _fund(led, "alice", "100.0")
    led.lock_for_job(
        buyer_wallet_id="alice", job_id="j1",
        amount_usd=Decimal("10.0"),
    )
    led.release_to_provider(job_id="j1", provider_wallet_id="bob")
    with pytest.raises(RuntimeError):
        led.refund_to_buyer(job_id="j1")


# ---------------------------------------------------------------------------
# Consortium split + partial refund
# ---------------------------------------------------------------------------

def test_consortium_split_pays_each_member_proportionally():
    led = BuyerLedger()
    _fund(led, "alice", "100.0")
    led.lock_for_job(
        buyer_wallet_id="alice", job_id="j1",
        amount_usd=Decimal("12.0"),
    )
    led.split_release_to_consortium(
        job_id="j1",
        members=[
            ("p1", Decimal("4.0")),
            ("p2", Decimal("6.0")),
            ("p3", Decimal("2.0")),
        ],
    )
    c = COMMISSION_RATE
    assert led.get_wallet("p1").available_usd == Decimal("4.0") - Decimal("4.0") * c
    assert led.get_wallet("p2").available_usd == Decimal("6.0") - Decimal("6.0") * c
    assert led.get_wallet("p3").available_usd == Decimal("2.0") - Decimal("2.0") * c
    treas = led.get_wallet(TREASURY_WALLET_ID)
    assert treas.available_usd == Decimal("12.0") * c


def test_consortium_partial_failure_refunds_failed_shares_to_buyer():
    led = BuyerLedger()
    _fund(led, "alice", "100.0")
    led.lock_for_job(
        buyer_wallet_id="alice", job_id="j1",
        amount_usd=Decimal("12.0"),
    )
    # Only 2 of 3 members succeeded; pass just the survivors.
    led.split_release_to_consortium(
        job_id="j1",
        members=[("p1", Decimal("4.0")), ("p2", Decimal("6.0"))],
    )
    # Alice's share of the failed member ($2) returned to her.
    alice = led.get_wallet("alice")
    assert alice.available_usd == Decimal("100.0") - Decimal("12.0") + Decimal("2.0")
    assert alice.locked_usd == Decimal("0")


# ---------------------------------------------------------------------------
# Treasury invariants — Pluginfer never loses
# ---------------------------------------------------------------------------

def test_treasury_is_monotonic_across_many_jobs():
    """Across a workload of 100 jobs — some succeed, some fail — the
    treasury balance never decreases. Pluginfer can refund a job
    without paying out commission; we can never pay out from treasury."""
    led = BuyerLedger()
    _fund(led, "alice", "1000.0")
    last_treas = Decimal("0")
    for i in range(100):
        jid = f"j-{i}"
        led.lock_for_job(
            buyer_wallet_id="alice", job_id=jid,
            amount_usd=Decimal("5.0"),
        )
        if i % 5 == 0:
            led.refund_to_buyer(job_id=jid)
        else:
            led.release_to_provider(job_id=jid, provider_wallet_id=f"p-{i}")
        cur = led.treasury_balance()
        assert cur >= last_treas, (i, cur, last_treas)
        last_treas = cur
    # 80 successful jobs × $5 × commission = treasury balance.
    assert led.treasury_balance() == Decimal("400.0") * COMMISSION_RATE


# ---------------------------------------------------------------------------
# Persistence — money records must survive restarts (audit gap, 2026-07-18)
# ---------------------------------------------------------------------------

def test_money_ledger_survives_restart(tmp_path):
    from decimal import Decimal
    from core.buyer_ledger import BuyerLedger, TREASURY_WALLET_ID

    led1 = BuyerLedger(str(tmp_path))
    led1.credit("buyer-1", Decimal("10.00"), note="top-up")
    led1.lock_for_job(buyer_wallet_id="buyer-1", job_id="job-A",
                      amount_usd=Decimal("2.00"))
    led1.release_to_provider(job_id="job-A",
                             provider_wallet_id="prov-1",
                             commission_rate=Decimal("0.10"))
    assert led1.treasury_balance() == Decimal("0.20")

    # "Restart": a fresh ledger on the same dir restores every balance,
    # escrow state, and commission entry.
    led2 = BuyerLedger(str(tmp_path))
    assert led2.treasury_balance() == Decimal("0.20")
    assert led2.get_wallet("buyer-1").available_usd == Decimal("8.00")
    assert led2.get_wallet("prov-1").available_usd == Decimal("1.80")
    esc = led2.escrow_for("job-A")
    assert esc.state == "released"
    assert esc.commission_amount == Decimal("0.20")
    # Idempotency survives the restart too — re-releasing is a no-op.
    led2.release_to_provider(job_id="job-A", provider_wallet_id="prov-1")
    assert led2.treasury_balance() == Decimal("0.20")


def test_treasury_report_shows_commission_book(tmp_path):
    from decimal import Decimal
    from core.buyer_ledger import BuyerLedger

    led = BuyerLedger(str(tmp_path))
    led.credit("buyer-1", Decimal("5.00"))
    for i in range(3):
        jid = f"job-{i}"
        led.lock_for_job(buyer_wallet_id="buyer-1", job_id=jid,
                         amount_usd=Decimal("1.00"))
        led.release_to_provider(job_id=jid, provider_wallet_id="prov-1",
                                commission_rate=Decimal("0.10"))
    rep = led.treasury_report()
    assert Decimal(rep["treasury_balance_usd"]) == Decimal("0.30")
    assert rep["commission_entries_n"] == 3
    assert len(rep["recent_commissions"]) == 3
    entry = rep["recent_commissions"][0]
    assert entry["job_id"] == "job-0"
    assert entry["buyer_wallet_id"] == "buyer-1"
    assert entry["amount_usd"] == "0.10000000"


# ---------------------------------------------------------------------------
# Testnet faucet — self-serve starter credit, once per wallet
# ---------------------------------------------------------------------------

def test_faucet_grants_once_and_survives_restart(tmp_path):
    from decimal import Decimal
    from core.buyer_ledger import BuyerLedger, FaucetAlreadyGranted
    import pytest

    led = BuyerLedger(str(tmp_path))
    w = led.faucet_grant("newbie-1", Decimal("25"))
    assert w.available_usd == Decimal("25")
    # Second request: refused, balance unchanged.
    with pytest.raises(FaucetAlreadyGranted):
        led.faucet_grant("newbie-1", Decimal("25"))
    assert led.get_wallet("newbie-1").available_usd == Decimal("25")

    # Restart: the grant marker persists, so it still can't be farmed.
    led2 = BuyerLedger(str(tmp_path))
    with pytest.raises(FaucetAlreadyGranted):
        led2.faucet_grant("newbie-1", Decimal("25"))
    assert led2.get_wallet("newbie-1").available_usd == Decimal("25")


def test_faucet_credit_spendable_in_escrow(tmp_path):
    from decimal import Decimal
    from core.buyer_ledger import BuyerLedger

    led = BuyerLedger(str(tmp_path))
    led.faucet_grant("newbie-2", Decimal("25"))
    led.lock_for_job(buyer_wallet_id="newbie-2", job_id="job-F",
                     amount_usd=Decimal("1.00"))
    led.release_to_provider(job_id="job-F", provider_wallet_id="prov-9",
                            commission_rate=Decimal("0.10"))
    assert led.get_wallet("newbie-2").available_usd == Decimal("24.00")
    assert led.get_wallet("prov-9").available_usd == Decimal("0.90")
    assert led.treasury_balance() == Decimal("0.10")
