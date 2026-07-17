"""§HG16 — deposit + withdrawal flows.

Pins the money-safety contract:
  * no gateway → deposits REFUSE (never simulate),
  * a deposit credits exactly once no matter how it is retried,
  * withdrawals debit up-front, can't overdraw, can't touch escrow,
  * two-phase close: real payout reference or cancel-with-refund,
  * everything survives a restart.

The gateway here is an injected test double implementing the real
PaymentGateway shape — the PRODUCT path still refuses when no gateway
is configured, which is itself pinned below.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from core.buyer_ledger import BuyerLedger, InsufficientFunds
from core.payment_flows import PaymentFlows, PaymentsNotConfigured
from core.payments import PaymentResult


class FakeGateway:
    """Deterministic gateway double: succeeds, honors idempotency by
    returning the same transaction id for the same key."""

    def __init__(self, fail=False):
        self.fail = fail
        self.charges = []
        self._by_key = {}

    def charge(self, amount_usd, currency, customer_id,
               description="", idempotency_key=None):
        if self.fail:
            return PaymentResult(False, None, "card_declined")
        if idempotency_key and idempotency_key in self._by_key:
            return PaymentResult(True, self._by_key[idempotency_key],
                                 idempotent_replay=True)
        tx = f"tx-{len(self.charges) + 1}"
        self.charges.append((amount_usd, customer_id, idempotency_key))
        if idempotency_key:
            self._by_key[idempotency_key] = tx
        return PaymentResult(True, tx)


class PayoutGateway(FakeGateway):
    def payout(self, amount_usd, currency, destination,
               idempotency_key=None):
        return PaymentResult(True, f"po-{destination}")


def test_deposit_refused_without_gateway(tmp_path):
    flows = PaymentFlows(BuyerLedger(str(tmp_path)), gateway=None)
    with pytest.raises(PaymentsNotConfigured):
        flows.deposit(wallet_id="b1", amount_usd=Decimal("5"),
                      customer_id="cus_1")


def test_deposit_credits_exactly_once_on_retry(tmp_path):
    led = BuyerLedger(str(tmp_path))
    flows = PaymentFlows(led, gateway=FakeGateway(), state_dir=str(tmp_path))
    r1 = flows.deposit(wallet_id="b1", amount_usd=Decimal("5"),
                       customer_id="cus_1", idempotency_key="k1")
    assert r1.idempotent_replay is False
    # Retry — same key. Wallet must NOT be credited again.
    r2 = flows.deposit(wallet_id="b1", amount_usd=Decimal("5"),
                       customer_id="cus_1", idempotency_key="k1")
    assert r2.idempotent_replay is True
    assert r2.transaction_id == r1.transaction_id
    assert led.get_wallet("b1").available_usd == Decimal("5")


def test_failed_charge_credits_nothing(tmp_path):
    led = BuyerLedger(str(tmp_path))
    flows = PaymentFlows(led, gateway=FakeGateway(fail=True))
    with pytest.raises(RuntimeError, match="card_declined"):
        flows.deposit(wallet_id="b1", amount_usd=Decimal("5"),
                      customer_id="cus_1")
    assert led.get_wallet("b1") is None


def test_withdrawal_two_phase_and_overdraw_refused(tmp_path):
    led = BuyerLedger(str(tmp_path))
    flows = PaymentFlows(led, gateway=None, state_dir=str(tmp_path))
    led.credit("prov-1", Decimal("10"))

    # Escrowed funds are untouchable by a withdrawal.
    led.lock_for_job(buyer_wallet_id="prov-1", job_id="j1",
                     amount_usd=Decimal("4"))
    with pytest.raises(InsufficientFunds):
        flows.request_withdrawal(wallet_id="prov-1",
                                 amount_usd=Decimal("7"),
                                 destination="upi:x@bank")

    rec = flows.request_withdrawal(wallet_id="prov-1",
                                   amount_usd=Decimal("6"),
                                   destination="upi:x@bank")
    assert rec.state == "pending"
    assert led.get_wallet("prov-1").available_usd == Decimal("0")

    # Empty payout reference refused; real one closes it, idempotently.
    with pytest.raises(ValueError):
        flows.complete_withdrawal(rec.withdrawal_id, payout_reference=" ")
    done = flows.complete_withdrawal(rec.withdrawal_id,
                                     payout_reference="UTR123")
    assert done.state == "paid"
    again = flows.complete_withdrawal(rec.withdrawal_id,
                                      payout_reference="UTR999")
    assert again.payout_reference == "UTR123"      # idempotent, first wins
    # A paid withdrawal cannot be cancelled (no clawback into wallets).
    with pytest.raises(RuntimeError):
        flows.cancel_withdrawal(rec.withdrawal_id)


def test_cancel_returns_funds_exactly_once(tmp_path):
    led = BuyerLedger(str(tmp_path))
    flows = PaymentFlows(led, gateway=None, state_dir=str(tmp_path))
    led.credit("prov-1", Decimal("3"))
    rec = flows.request_withdrawal(wallet_id="prov-1",
                                   amount_usd=Decimal("3"),
                                   destination="upi:x@bank")
    flows.cancel_withdrawal(rec.withdrawal_id)
    flows.cancel_withdrawal(rec.withdrawal_id)     # idempotent
    assert led.get_wallet("prov-1").available_usd == Decimal("3")
    # A cancelled withdrawal cannot later be marked paid.
    with pytest.raises(RuntimeError):
        flows.complete_withdrawal(rec.withdrawal_id,
                                  payout_reference="UTR1")


def test_auto_payout_when_gateway_supports_it(tmp_path):
    led = BuyerLedger(str(tmp_path))
    flows = PaymentFlows(led, gateway=PayoutGateway(),
                         state_dir=str(tmp_path))
    led.credit("prov-1", Decimal("2"))
    rec = flows.request_withdrawal(wallet_id="prov-1",
                                   amount_usd=Decimal("2"),
                                   destination="acct_1")
    assert rec.state == "paid"
    assert rec.payout_reference == "po-acct_1"


def test_flows_survive_restart(tmp_path):
    led = BuyerLedger(str(tmp_path))
    flows = PaymentFlows(led, gateway=FakeGateway(),
                         state_dir=str(tmp_path))
    flows.deposit(wallet_id="b1", amount_usd=Decimal("5"),
                  customer_id="cus_1", idempotency_key="k1")
    wd = flows.request_withdrawal(wallet_id="b1",
                                  amount_usd=Decimal("2"),
                                  destination="upi:x@bank")

    # "Restart" — same dirs, fresh objects. The pending withdrawal is
    # still pending; a replayed deposit still credits nothing new.
    led2 = BuyerLedger(str(tmp_path))
    flows2 = PaymentFlows(led2, gateway=FakeGateway(),
                          state_dir=str(tmp_path))
    pend = flows2.get_withdrawal(wd.withdrawal_id)
    assert pend is not None and pend.state == "pending"
    r = flows2.deposit(wallet_id="b1", amount_usd=Decimal("5"),
                       customer_id="cus_1", idempotency_key="k1")
    assert r.idempotent_replay is True
    assert led2.get_wallet("b1").available_usd == Decimal("3")  # 5 - 2
    # Close the withdrawal post-restart with a real reference.
    assert flows2.complete_withdrawal(
        wd.withdrawal_id, payout_reference="UTR42").state == "paid"
