"""Hosted gateway — the end-to-end "startup ditching AWS" surface.

Pins:
  * API key issue + revoke + Bearer auth.
  * /v1/chat/completions returns OpenAI-shaped response, with
    Pluginfer headers (receipt, price, provider, wallet balance).
  * Underfunded wallet → HTTP 402 with structured remediation
    (deficit + topup_url + job_id for resume).
  * Topup → auto-resume of paused jobs.
  * Buyer wallet debited, provider wallet credited (minus
    commission), treasury credited — economics consistent with
    direct JobsService calls.
"""

from __future__ import annotations

import base64
import hashlib
import sys
from decimal import Decimal
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest
from fastapi.testclient import TestClient

from api.hosted_gateway import ApiKeyStore, build_hosted_gateway
from api.jobs_service import JobsService
from core.buyer_ledger import (
    BuyerLedger,
    COMMISSION_RATE,
    TREASURY_WALLET_ID,
)
from core.providers import (
    Auction,
    Bid,
    PRIVACY_PUBLIC,
    Provider,
)


class _W(Provider):
    def __init__(self, *, pid: str, price: float, output: bytes):
        self.provider_id = pid
        self.privacy_grade = PRIVACY_PUBLIC
        self._price = price
        self._output = output

    def bid(self, job):
        return Bid(
            provider_id=self.provider_id, price_usd=self._price, eta_ms=100,
            expected_quality=0.9, privacy_grade=PRIVACY_PUBLIC, evidence={},
        )

    def execute(self, job, bid):
        return {
            "status": "executed", "job_id": job.job_id,
            "result_bytes": base64.b64encode(self._output).decode("ascii"),
            "result_hash": hashlib.sha256(self._output).hexdigest(),
            "execution_ms": 100.0, "provider_sig": "AAAA",
            "provider_pubkey_pem": "fake",
        }


def _build_stack(price=0.05, output=b"hello-from-mesh"):
    ledger = BuyerLedger()
    keys = ApiKeyStore()
    auction = Auction()
    auction.register(_W(pid="mesh-provider-1", price=price, output=output))
    svc = JobsService(auction=auction, ledger=ledger)
    app = build_hosted_gateway(jobs_service=svc, ledger=ledger, api_keys=keys)
    return app, svc, ledger, keys


def _issue_key(client, wallet_id):
    r = client.post("/v1/keys", json={"wallet_id": wallet_id})
    assert r.status_code == 200, r.text
    return r.json()["api_key"]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_chat_completions_requires_bearer_token():
    app, *_ = _build_stack()
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 401


def test_revoked_key_rejected():
    app, _, _, keys = _build_stack()
    with TestClient(app) as client:
        raw = _issue_key(client, "alice")
        assert keys.revoke(raw) is True
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw}"},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# Happy path — wallet → mesh → result
# ---------------------------------------------------------------------------

def test_funded_buyer_gets_openai_shaped_response():
    app, _, ledger, _ = _build_stack(price=0.05, output=b"mesh-says-hi")
    with TestClient(app) as client:
        raw = _issue_key(client, "alice")
        ledger.credit("alice", Decimal("10.00"))
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw}"},
            json={
                "model": "pluginfer-alpha",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
                "pluginfer_cost_ceiling_usd": 1.0,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # OpenAI-shaped envelope.
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "mesh-says-hi"
        # Pluginfer headers.
        assert r.headers["X-Pluginfer-Receipt-ID"]
        assert r.headers["X-Pluginfer-Price-USD"] == "0.05"
        # Wallet debited, provider + treasury credited.
        c = COMMISSION_RATE
        alice = ledger.get_wallet("alice")
        prov = ledger.get_wallet("mesh-provider-1")
        treas = ledger.get_wallet(TREASURY_WALLET_ID)
        assert alice.available_usd == Decimal("10.00") - Decimal("0.05")
        assert prov.available_usd == Decimal("0.05") - Decimal("0.05") * c
        assert treas.available_usd == Decimal("0.05") * c


# ---------------------------------------------------------------------------
# Pause path — HTTP 402 with remediation, then topup + resume
# ---------------------------------------------------------------------------

def test_underfunded_buyer_gets_402_with_topup_url():
    app, _, ledger, _ = _build_stack(price=0.50)
    with TestClient(app) as client:
        raw = _issue_key(client, "alice")
        ledger.credit("alice", Decimal("0.10"))      # less than $0.50
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw}"},
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
                "pluginfer_cost_ceiling_usd": 1.0,
            },
        )
        assert r.status_code == 402, r.text
        body = r.json()
        assert body["error"] == "wallet_underfunded"
        assert body["wallet_id"] == "alice"
        assert body["topup_url"].endswith("/v1/wallets/alice/topup")
        # job_id is included so the buyer can resume after topup
        # instead of re-running the auction.
        assert body["job_id"]


def test_topup_auto_resumes_paused_job(monkeypatch):
    """Wallet credit triggers a resume_funding for every paused
    job owned by the wallet. This is the Claude-Code-style 'press
    continue and your work picks up' UX, but mechanical (the
    topup IS the continue)."""
    app, svc, ledger, _ = _build_stack(price=0.50, output=b"resumed")
    with TestClient(app) as client:
        raw = _issue_key(client, "alice")
        ledger.credit("alice", Decimal("0.10"))
        # Submit underfunded → paused.
        r1 = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw}"},
            json={"messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 16,
                  "pluginfer_cost_ceiling_usd": 1.0},
        )
        assert r1.status_code == 402
        job_id = r1.json()["job_id"]
        # Top up by enough.
        r2 = client.post(
            f"/v1/wallets/alice/topup",
            headers={"Authorization": f"Bearer {raw}"},
            json={"amount_usd": "1.00"},
        )
        assert r2.status_code == 200
        # Poll the job — should reach completed after auto-resume.
        import time
        for _ in range(50):
            rec = svc.get(job_id)
            if rec.state in ("completed", "completed_partial"):
                break
            time.sleep(0.05)
        rec = svc.get(job_id)
        assert rec.state == "completed", (rec.state, rec.detail)
        # And the wallet was debited exactly once across the
        # pause + resume cycle.
        assert ledger.get_wallet("alice").available_usd == (
            Decimal("0.10") + Decimal("1.00") - Decimal("0.50")
        )


# ---------------------------------------------------------------------------
# Wallet GET endpoint
# ---------------------------------------------------------------------------

def test_get_wallet_returns_balance():
    app, _, ledger, _ = _build_stack()
    with TestClient(app) as client:
        raw = _issue_key(client, "alice")
        ledger.credit("alice", Decimal("12.50"))
        r = client.get(
            "/v1/wallets/alice",
            headers={"Authorization": f"Bearer {raw}"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["wallet_id"] == "alice"
        assert body["available_usd"] == "12.50"


def test_get_wallet_other_wallet_forbidden():
    app, _, ledger, _ = _build_stack()
    with TestClient(app) as client:
        raw_alice = _issue_key(client, "alice")
        _issue_key(client, "bob")     # bob's wallet exists too
        r = client.get(
            "/v1/wallets/bob",
            headers={"Authorization": f"Bearer {raw_alice}"},
        )
        assert r.status_code == 403
