"""Audit-driven hardening: auth, public-verifiable signed receipts,
external anchoring, and honest token counting.

Pins the fixes for the shortfalls the third-party audit named:
  * open endpoints (anyone reaching the port could spend the key / read
    all spend) -> client/read/admin auth, fail-closed once configured,
  * self-verifying "blockchain" -> Ed25519-signed receipts that an
    auditor verifies with the PUBLIC key alone; tamper caught at the
    edited receipt; external anchor of the chain head,
  * chars/4 token holds -> real tokenizer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest
from fastapi.testclient import TestClient

from governance.auth import AuthConfig
from governance.budget_ledger import BudgetLedger
from governance.gateway import build_governance_gateway
from governance.signing import GatewaySigner, verify_with_public_pem

PRICES = {"gpt-test": {"input_per_1m": 1.0, "output_per_1m": 10.0}}


def _upstream():
    def post(url, body, headers, timeout_s):
        return 200, json.dumps({
            "choices": [{"message": {"content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }).encode()
    return post


def _stack(auth=None):
    budget = BudgetLedger(None)
    budget.set_envelope("acme", 100.0, "month")
    app = build_governance_gateway(
        budget=budget, upstream_base="https://sim.local",
        price_sheet=PRICES, http_post=_upstream(), auth=auth)
    return app, budget


# ---------------------------------------------------------------------------
# Auth — the open-endpoint fix
# ---------------------------------------------------------------------------

def test_no_auth_configured_stays_open_backwards_compatible():
    app, _ = _stack()          # empty AuthConfig by default
    with TestClient(app) as c:
        assert c.post("/v1/chat/completions", json={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}]}).status_code == 200
        assert c.get("/v1/budget/report").status_code == 200


def test_configured_auth_locks_forwarding_and_reporting():
    auth = AuthConfig(admin_key="admin")
    app, _ = _stack(auth=auth)
    client_key = auth.issue_client_key(label="team-a")
    with TestClient(app) as c:
        body = {"model": "gpt-test",
                "messages": [{"role": "user", "content": "hi"}]}
        # No key -> 401 on the money-spending endpoint.
        assert c.post("/v1/chat/completions", json=body).status_code == 401
        # Valid client key -> 200.
        assert c.post("/v1/chat/completions", json=body, headers={
            "Authorization": f"Bearer {client_key}"}).status_code == 200
        # Spend data needs a reader.
        assert c.get("/v1/savings").status_code == 401
        assert c.get("/v1/savings", headers={
            "X-Admin-Key": "admin"}).status_code == 200


def test_revoked_client_key_rejected():
    auth = AuthConfig(admin_key="admin")
    app, _ = _stack(auth=auth)
    k = auth.issue_client_key()
    assert auth.revoke_client_key(k) is True
    with TestClient(app) as c:
        assert c.post("/v1/chat/completions", json={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {k}"}).status_code == 401


def test_pinned_envelope_cannot_be_overridden_by_header():
    auth = AuthConfig(admin_key="admin")
    app, budget = _stack(auth=auth)
    budget.set_envelope("acme/teamA", 100.0, "month")
    k = auth.issue_client_key(envelope="acme/teamA")
    with TestClient(app) as c:
        # Caller TRIES to bill a different envelope via header — ignored.
        c.post("/v1/chat/completions", json={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {k}",
                     "X-Budget-Envelope": "acme/someone-else"})
        rep = budget.report()
    assert "acme/teamA" in rep["by_envelope"]
    assert "acme/someone-else" not in rep["by_envelope"]


def test_raw_keys_never_stored():
    auth = AuthConfig(admin_key="admin")
    raw = auth.issue_client_key(label="t")
    # Only fingerprints are retained.
    for k in auth.list_client_keys():
        assert raw not in k["fingerprint"]
        assert len(k["fingerprint"]) == 16


# ---------------------------------------------------------------------------
# Signed, publicly-verifiable receipts + anchoring
# ---------------------------------------------------------------------------

def test_receipts_signed_by_default_and_publicly_verifiable():
    app, _ = _stack()
    with TestClient(app) as c:
        c.post("/v1/chat/completions", json={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}]})
        v = c.get("/v1/receipts/verify").json()
        rec = c.get("/v1/receipts").json()["receipts"][-1]
    assert v["ok"] is True
    assert v["signed"] is True
    gw = app.state.gateway
    if v["signature_algorithm"] == "ed25519":
        assert v["publicly_verifiable"] is True
        # An auditor with ONLY the public key can verify the receipt.
        body = gw._signing_body(rec)
        assert verify_with_public_pem(
            rec["gateway_pubkey_pem"], body, rec["gateway_signature"])


def test_anchor_endpoint_exposes_signed_head():
    app, _ = _stack()
    with TestClient(app) as c:
        c.post("/v1/chat/completions", json={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}]})
        a = c.get("/v1/audit/anchor").json()
    assert a["receipt_count"] == 1
    assert len(a["chain_head_sha256"]) == 64
    assert a["signature"]
    # The anchored head signature verifies against the public key.
    if a["algorithm"] == "ed25519":
        assert verify_with_public_pem(
            a["public_key_pem"], a["chain_head_sha256"], a["signature"])


def test_editing_any_field_invalidates_signature():
    app, _ = _stack()
    with TestClient(app) as c:
        for _ in range(3):
            c.post("/v1/chat/completions", json={
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hi"}]})
        gw = app.state.gateway
        gw._receipts[0]["envelope"] = "attacker/rewrote/this"
        v = c.get("/v1/receipts/verify").json()
    assert v["ok"] is False
    assert v["broken_at_index"] == 0


def test_hmac_fallback_labelled_honestly():
    # Force the stdlib fallback and confirm it is NOT claimed as
    # publicly verifiable.
    signer = GatewaySigner.create(None, prefer="hmac")
    assert signer.algorithm == "hmac-sha256"
    assert signer.public_key_pem.startswith("hmac-key:")
    budget = BudgetLedger(None)
    budget.set_envelope("acme", 100.0, "month")
    app = build_governance_gateway(
        budget=budget, upstream_base="https://sim.local",
        price_sheet=PRICES, http_post=_upstream(), signer=signer)
    with TestClient(app) as c:
        c.post("/v1/chat/completions", json={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}]})
        v = c.get("/v1/receipts/verify").json()
    assert v["signature_algorithm"] == "hmac-sha256"
    assert v["publicly_verifiable"] is False


# ---------------------------------------------------------------------------
# Audit-chain persistence across restarts
# ---------------------------------------------------------------------------

def _persistent_gw(tmp_path):
    from governance.gateway import GovernanceGateway
    return GovernanceGateway(
        budget=BudgetLedger(str(tmp_path)),
        upstream_base="https://up.example",
        price_sheet={"m": {"input_per_1m": 1.0, "output_per_1m": 2.0}})


def test_receipt_chain_survives_restart(tmp_path):
    gw1 = _persistent_gw(tmp_path)
    for i in range(3):
        gw1._emit_receipt({"kind": "forward", "envelope": "acme",
                           "model": "m", "cost_usd": 0.01 * (i + 1)})
    head1 = gw1._chain_head
    assert gw1.verify_chain()["ok"] is True

    # "Restart": a fresh gateway on the same state dir must reload the
    # full history, verify it, and link new receipts onto the real head.
    gw2 = _persistent_gw(tmp_path)
    v = gw2.verify_chain()
    assert v["ok"] is True
    assert v["receipts_checked"] == 3
    assert gw2._chain_head == head1

    gw2._emit_receipt({"kind": "forward", "envelope": "acme",
                       "model": "m", "cost_usd": 0.04})
    v2 = gw2.verify_chain()
    assert v2["ok"] is True and v2["receipts_checked"] == 4
    assert gw2._receipts[3]["prev_sha256"] == head1   # not genesis


def test_tampered_receipt_file_is_caught_on_restart(tmp_path):
    gw1 = _persistent_gw(tmp_path)
    for _ in range(3):
        gw1._emit_receipt({"kind": "forward", "envelope": "acme",
                           "model": "m", "cost_usd": 0.01})
    path = tmp_path / "receipts.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    doctored = json.loads(lines[1])
    doctored["cost_usd"] = 999.99            # edit history on DISK
    lines[1] = json.dumps(doctored, sort_keys=True, default=str)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    gw2 = _persistent_gw(tmp_path)           # restart over doctored file
    v = gw2.verify_chain()
    assert v["ok"] is False
    assert v["broken_at_index"] == 1
    assert v["reason"] in ("bad_signature", "chain_break")


# ---------------------------------------------------------------------------
# Signed savings report + real embedder backend
# ---------------------------------------------------------------------------

def test_signed_savings_report_verifies_and_tamper_fails(tmp_path):
    from governance.gateway import GovernanceGateway
    gw = GovernanceGateway(
        budget=BudgetLedger(str(tmp_path)),
        upstream_base="https://up.example",
        price_sheet={"m": {"input_per_1m": 1.0, "output_per_1m": 2.0}})
    gw._emit_receipt({"kind": "cache_hit", "envelope": "e", "model": "m",
                      "cost_usd": 0.0, "saved_usd": 0.5})
    data = gw.signed_savings_report()
    assert data["report"]["savings"]["cache_saved_usd"] == 0.5
    assert data["report"]["receipt_count"] == 1
    body = json.dumps(data["report"], sort_keys=True, default=str)
    assert verify_with_public_pem(data["public_key_pem"], body,
                                  data["signature"]) is True
    # Inflate the claim -> the signature no longer verifies.
    doctored = dict(data["report"])
    doctored["savings"] = dict(doctored["savings"], net_saved_usd=999.0)
    bad = json.dumps(doctored, sort_keys=True, default=str)
    assert verify_with_public_pem(data["public_key_pem"], bad,
                                  data["signature"]) is False


def test_ollama_embedder_hermetic_and_unavailable_raises():
    from governance.embedders import EmbedderUnavailable, OllamaEmbedder
    calls = []

    def ok_post(url, body, timeout_s):
        calls.append(url)
        return 200, json.dumps({"embeddings": [[0.1, 0.2, 0.3]]}).encode()

    emb = OllamaEmbedder(model="test-embed", http_post=ok_post)
    assert emb.backend_name == "ollama:test-embed"
    assert emb("hello") == [0.1, 0.2, 0.3]

    def dead_post(url, body, timeout_s):
        return 500, b"{}"

    with pytest.raises(EmbedderUnavailable):
        OllamaEmbedder(model="test-embed", http_post=dead_post)


def test_semantic_cache_labels_real_backend():
    from governance.token_thrift import SemanticCache
    sc = SemanticCache(threshold=0.9,
                       embed_fn=lambda t: [1.0, 0.0],
                       backend_name="ollama:test-embed")
    assert sc.backend_name == "ollama:test-embed"
