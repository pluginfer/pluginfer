"""Cross-gateway settlement forwarder — every signed-credit notice
eventually lands at the provider's home gateway, even when the
target is briefly unreachable.

Invariants:
  * forward() POSTs a signed payload; on 200 the notice is marked
    delivered.
  * On failure the notice stays in the outbox; retry_pending()
    re-sends after backoff.
  * verify_credit_notice on the receiving side rejects bad
    signatures, unknown source gateways, non-positive amounts.
  * Idempotent at the notice_id level — receiver caller can use
    notice_id as a dedup key.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from core.settlement_forwarder import (
    CreditNotice,
    SettlementForwarder,
    verify_credit_notice,
)


class _FakeNetwork:
    """Records every POST + lets the test toggle reachability."""

    def __init__(self):
        self.calls: List[tuple] = []
        self.reachable = True
        self.fail_n_times = 0

    def __call__(self, url: str, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.calls.append((url, body))
        if self.fail_n_times > 0:
            self.fail_n_times -= 1
            return None
        if not self.reachable:
            return None
        return {"status": "credited", "notice_id": body["notice_id"]}


def _fw(net):
    return SettlementForwarder(
        source_gateway_pubkey="GW-A-pub",
        sign_fn=lambda m: "sig-of-" + m[:8],
        poster=net,
    )


def test_immediate_send_marks_delivered():
    net = _FakeNetwork()
    fw = _fw(net)
    n = fw.forward(
        target_gateway_url="http://gw-b.example",
        provider_wallet_id="bob",
        amount_usd=Decimal("0.45"),
        job_id="job-1",
    )
    assert n.delivered is True
    assert fw.pending_count() == 0
    assert len(net.calls) == 1
    url, body = net.calls[0]
    assert url.endswith("/v1/wallet/credit_notice")
    assert body["amount_usd"] == "0.45"
    assert body["signature"].startswith("sig-of-")


def test_failed_send_keeps_notice_in_outbox():
    net = _FakeNetwork()
    net.reachable = False
    fw = _fw(net)
    n = fw.forward(
        target_gateway_url="http://gw-down.example",
        provider_wallet_id="bob", amount_usd=Decimal("1.0"), job_id="j",
    )
    assert n.delivered is False
    assert fw.pending_count() == 1
    assert n.last_error


def test_retry_after_backoff_eventually_delivers(monkeypatch):
    import core.settlement_forwarder as mod
    # Speed up the backoff schedule for the test.
    monkeypatch.setattr(mod, "DEFAULT_RETRY_DELAYS_S", (0, 0, 0, 0, 0))
    net = _FakeNetwork()
    net.reachable = False
    fw = _fw(net)
    n = fw.forward(
        target_gateway_url="http://gw.example",
        provider_wallet_id="bob", amount_usd=Decimal("1.0"), job_id="j",
    )
    assert n.delivered is False
    # Bring the target back online; retry_pending eventually drains.
    net.reachable = True
    summary = fw.retry_pending()
    assert summary["sent"] == 1
    assert summary["still_pending"] == 0
    assert fw.pending_count() == 0


def test_multiple_notices_drain_independently(monkeypatch):
    import core.settlement_forwarder as mod
    monkeypatch.setattr(mod, "DEFAULT_RETRY_DELAYS_S", (0,))
    net = _FakeNetwork()
    fw = _fw(net)
    net.reachable = False
    for i in range(3):
        fw.forward(
            target_gateway_url=f"http://gw-{i}.example",
            provider_wallet_id=f"prov-{i}",
            amount_usd=Decimal("0.10"), job_id=f"j-{i}",
        )
    assert fw.pending_count() == 3
    net.reachable = True
    summary = fw.retry_pending()
    assert summary["sent"] == 3
    assert summary["still_pending"] == 0


# ---------------------------------------------------------------------------
# Receiver-side verification
# ---------------------------------------------------------------------------

def _good_body():
    return {
        "notice_id": "nid",
        "source_gateway_pubkey": "GW-A-pub",
        "provider_wallet_id": "bob",
        "amount_usd": "0.5",
        "job_id": "j",
        "issued_at_unix": 1.0,
        "signature": "sigsig",
    }


def test_verify_accepts_well_formed_notice_from_known_gateway():
    out = verify_credit_notice(
        _good_body(),
        known_gateway_pubkeys={"GW-A-pub": "GW-A-pub"},
        verify_signature=None,    # signature check skipped for unit
    )
    assert out is not None
    assert out["amount_usd"] == Decimal("0.5")
    assert out["provider_wallet_id"] == "bob"


def test_verify_rejects_unknown_source_gateway():
    out = verify_credit_notice(
        _good_body(),
        known_gateway_pubkeys={"GW-X-pub": "GW-X-pub"},
        verify_signature=None,
    )
    assert out is None


def test_verify_rejects_non_positive_amount():
    body = _good_body()
    body["amount_usd"] = "0"
    out = verify_credit_notice(
        body, known_gateway_pubkeys={"GW-A-pub": "GW-A-pub"},
        verify_signature=None,
    )
    assert out is None


def test_verify_rejects_bad_signature_when_callback_returns_false():
    out = verify_credit_notice(
        _good_body(),
        known_gateway_pubkeys={"GW-A-pub": "GW-A-pub"},
        verify_signature=lambda msg, sig, pub: False,
    )
    assert out is None


def test_verify_accepts_when_callback_returns_true():
    out = verify_credit_notice(
        _good_body(),
        known_gateway_pubkeys={"GW-A-pub": "GW-A-pub"},
        verify_signature=lambda msg, sig, pub: True,
    )
    assert out is not None


def test_verify_rejects_missing_fields():
    body = _good_body()
    del body["signature"]
    out = verify_credit_notice(
        body, known_gateway_pubkeys={"GW-A-pub": "GW-A-pub"},
        verify_signature=None,
    )
    assert out is None
