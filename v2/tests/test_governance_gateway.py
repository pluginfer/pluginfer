"""§HG13b — Governance Gateway: budget control for TRADITIONAL setups.

Hermetic: `http_post` is injected, so the "upstream" is a plain
function — no sockets, no keys, no real provider. Pins the product
contract:

  * governed forward: reserve at the estimate → forward → settle at
    the upstream's OWN usage numbers → receipt + cost headers,
  * exhausted envelope → HTTP 402 with the honest reason BEFORE any
    upstream call is made,
  * unpriced model → HTTP 400 (we refuse to guess money numbers),
  * upstream errors release the hold (no phantom spend),
  * streaming refused until governed (HG13c),
  * chargeback report + envelope admin endpoints work end-to-end,
  * Anthropic-shaped usage (input_tokens/output_tokens) parses too.
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

from governance.gateway import build_governance_gateway
from governance.budget_ledger import BudgetLedger

PRICES = {
    "gpt-test": {"input_per_1m": 1.0, "output_per_1m": 10.0},
    "claude-test": {"input_per_1m": 3.0, "output_per_1m": 15.0},
}


def _openai_upstream(calls):
    def post(url, body, headers, timeout_s):
        calls.append({"url": url, "body": json.loads(body),
                      "headers": headers})
        resp = {
            "id": "cmpl-1", "object": "chat.completion",
            "choices": [{"message": {"role": "assistant",
                                     "content": "governed hello"}}],
            "usage": {"prompt_tokens": 1000,
                      "completion_tokens": 500},
        }
        return 200, json.dumps(resp).encode("utf-8")
    return post


def _stack(cap_usd=10.0, *, http_post=None, calls=None,
           **gw_kw):
    calls = calls if calls is not None else []
    budget = BudgetLedger(None)
    budget.set_envelope("acme", cap_usd, "month")
    app = build_governance_gateway(
        budget=budget,
        upstream_base="https://upstream.example",
        price_sheet=PRICES,
        http_post=http_post or _openai_upstream(calls),
        **gw_kw,
    )
    return app, budget, calls


def _chat(client, envelope="acme/support/bot", model="gpt-test",
          **body_extra):
    body = {"model": model,
            "messages": [{"role": "user", "content": "x" * 400}],
            "max_tokens": 100, **body_extra}
    return client.post("/v1/chat/completions", json=body,
                       headers={"X-Budget-Envelope": envelope})


# ---------------------------------------------------------------------------
# The governed forward
# ---------------------------------------------------------------------------

def test_forward_settles_at_upstream_usage_and_stamps_headers():
    app, budget, calls = _stack()
    with TestClient(app) as c:
        r = _chat(c)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "governed hello"
    assert len(calls) == 1
    assert calls[0]["url"].endswith("/v1/chat/completions")
    # Settled at the upstream's REAL usage (1000 in / 500 out):
    # 1000/1e6*1.0 + 500/1e6*10.0 = 0.006
    expect = 0.006
    assert float(r.headers["X-Pluginfer-Cost-USD"]) == pytest.approx(expect)
    assert r.headers["X-Pluginfer-Envelope"] == "acme/support/bot"
    assert r.headers["X-Pluginfer-Receipt-Id"]
    rep = budget.report()
    assert rep["total_spend_usd"] == pytest.approx(expect)
    assert rep["by_envelope"]["acme/support/bot"]["jobs"] == 1


def test_exhausted_envelope_402_before_any_upstream_call():
    app, _, calls = _stack(cap_usd=0.000001)
    with TestClient(app) as c:
        r = _chat(c)
    assert r.status_code == 402
    assert "budget_ledger:" in r.json()["error"]
    assert calls == []                     # upstream never contacted


def test_unpriced_model_refused_not_guessed():
    app, _, calls = _stack()
    with TestClient(app) as c:
        r = _chat(c, model="mystery-model-9000")
    assert r.status_code == 400
    assert "price sheet" in r.json()["error"]
    assert calls == []


def test_upstream_error_releases_hold_and_relays_body():
    def failing(url, body, headers, timeout_s):
        return 429, json.dumps(
            {"error": {"message": "rate limited"}}).encode()
    app, budget, _ = _stack(cap_usd=0.01, http_post=failing)
    with TestClient(app) as c:
        r = _chat(c)
        assert r.status_code == 429
        assert r.json()["error"]["message"] == "rate limited"
        # Hold released: the same tiny cap accepts the next call.
        r2 = _chat(c)
        assert r2.status_code == 429       # again upstream, not 402
    assert budget.report()["total_spend_usd"] == 0.0


def _sse_upstream(chunks):
    """Streaming upstream stub: returns the given SSE byte chunks."""
    def stream(url, body, headers, timeout_s):
        return 200, iter(chunks)
    return stream


def test_streaming_governed_settles_at_final_usage():
    final = {"choices": [], "usage": {"prompt_tokens": 1000,
                                      "completion_tokens": 500}}
    chunks = [
        b'data: {"choices": [{"delta": {"content": "hel"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n',
        ("data: " + json.dumps(final) + "\n\n").encode(),
        b"data: [DONE]\n\n",
    ]
    app, budget, _ = _stack(http_post=None,
                            http_stream=_sse_upstream(chunks))
    with TestClient(app) as c:
        r = _chat(c, stream=True)
        assert r.status_code == 200
        assert r.headers["X-Pluginfer-Stream"] == "governed"
        assert '"content": "hel"' in r.text     # chunks relayed
        assert "[DONE]" in r.text
        rec = c.get("/v1/receipts").json()["receipts"][-1]
    assert rec["kind"] == "stream"
    assert rec["estimated"] is False
    assert rec["cutoff"] is False
    # Settled at the upstream's final usage: 1000/1e6*1 + 500/1e6*10.
    assert budget.report()["total_spend_usd"] == pytest.approx(0.006)


def test_streaming_cutoff_protects_the_envelope():
    # Upstream streams far more output than the hold covers and never
    # sends usage — the gateway must hard-cut and settle at the hold.
    big = b'data: {"choices": [{"delta": {"content": "' \
          + b"x" * 3000 + b'"}}]}\n\n'
    chunks = [big] * 400                       # unbounded-ish stream
    app, budget, _ = _stack(http_post=None,
                            http_stream=_sse_upstream(chunks))
    with TestClient(app) as c:
        r = _chat(c, stream=True, max_tokens=10)   # tiny hold
        assert r.status_code == 200
        assert "pluginfer_budget_cutoff" in r.text
        rec = c.get("/v1/receipts").json()["receipts"][-1]
    assert rec["cutoff"] is True
    # Settled at exactly the held amount, never more.
    hold = 100 / 1e6 * 1.0 + 10 / 1e6 * 10.0   # 100 in-est, 10 out
    assert budget.report()["total_spend_usd"] <= hold + 1e-9


# ---------------------------------------------------------------------------
# Thrift: cache + cascade (measured savings only)
# ---------------------------------------------------------------------------

def test_cache_hit_costs_zero_and_records_measured_saving():
    from governance.gateway import ResponseCache
    calls = []
    app, budget, _ = _stack(
        http_post=_openai_upstream(calls), calls=calls,
        cache=ResponseCache(ttl_s=300, cache_all=True))
    with TestClient(app) as c:
        r1 = _chat(c)
        assert r1.status_code == 200
        assert len(calls) == 1
        r2 = _chat(c)                          # byte-identical repeat
        assert r2.status_code == 200
        assert len(calls) == 1                 # upstream NOT called
        assert r2.headers["X-Pluginfer-Cache"] == "hit"
        assert r2.headers["X-Pluginfer-Cost-USD"] == "0.00000000"
        # Saved = what the upstream billed for the identical request.
        assert float(r2.headers["X-Pluginfer-Saved-USD"]) == \
            pytest.approx(0.006)
        sav = c.get("/v1/savings").json()
    assert sav["cache_saved_usd"] == pytest.approx(0.006)
    assert sav["cache_hits"] == 1
    # Spend unchanged by the hit.
    assert budget.report()["total_spend_usd"] == pytest.approx(0.006)


def test_cache_skips_sampling_requests_by_default():
    from governance.gateway import ResponseCache
    calls = []
    app, _, _ = _stack(http_post=_openai_upstream(calls), calls=calls,
                       cache=ResponseCache(ttl_s=300))  # cache_all off
    with TestClient(app) as c:
        _chat(c, temperature=0.7)
        _chat(c, temperature=0.7)
    assert len(calls) == 2                     # never served from cache


def test_cascade_accept_settles_cheap_and_saves_difference():
    calls = []

    def upstream(url, body, headers, timeout_s):
        calls.append(json.loads(body))
        resp = {"choices": [{"message": {"role": "assistant",
                                         "content": "fine answer"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000,
                          "completion_tokens": 1000}}
        return 200, json.dumps(resp).encode()

    app, budget, _ = _stack(http_post=upstream,
                            cascades={"gpt-test": "claude-test"})
    # Note inverted prices: for THIS test make the cascade target the
    # cheap one — gpt-test out=10/1M, claude-test out=15/1M means
    # claude is pricier; swap direction: cascade gpt->claude would
    # LOSE money but the mechanics are identical; assert exact math.
    with TestClient(app) as c:
        r = _chat(c)                            # asks for gpt-test
    assert r.status_code == 200
    assert calls[0]["model"] == "claude-test"   # cheap try went first
    assert len(calls) == 1                      # accepted, no escalate
    assert r.headers["X-Pluginfer-Cascade"] == "accepted:claude-test"
    # Settled at the TRIED model's price: (1000*3 + 1000*15)/1e6.
    assert budget.report()["total_spend_usd"] == pytest.approx(0.018)
    # Saving vs requested model at same usage: (1+10-3-15)/1e3 < 0 →
    # clamped at 0 (never claim negative-as-positive).
    assert float(r.headers["X-Pluginfer-Saved-USD"]) == 0.0


def test_cascade_escalates_on_empty_and_charges_honestly():
    calls = []

    def upstream(url, body, headers, timeout_s):
        b = json.loads(body)
        calls.append(b)
        if b["model"] == "claude-test":         # cheap try: empty
            resp = {"choices": [{"message": {"content": ""},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100,
                              "completion_tokens": 1}}
        else:                                    # target: real answer
            resp = {"choices": [{"message": {"content": "real"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1000,
                              "completion_tokens": 500}}
        return 200, json.dumps(resp).encode()

    app, budget, _ = _stack(http_post=upstream,
                            cascades={"gpt-test": "claude-test"})
    with TestClient(app) as c:
        r = _chat(c)
        assert r.status_code == 200
        assert [b["model"] for b in calls] == ["claude-test",
                                               "gpt-test"]
        rec = c.get("/v1/receipts").json()["receipts"][-1]
        sav = c.get("/v1/savings").json()
    # Total = cheap try (100*3+1*15)/1e6 + target 0.006 — both real.
    cheap = (100 * 3 + 1 * 15) / 1e6
    assert budget.report()["total_spend_usd"] == \
        pytest.approx(0.006 + cheap)
    assert rec["kind"] == "cascade_escalate"
    assert rec["saved_usd"] == pytest.approx(-cheap)  # negative, shown
    assert sav["cascade_escalation_cost_usd"] == pytest.approx(cheap)
    assert sav["net_saved_usd"] == pytest.approx(-cheap)


# ---------------------------------------------------------------------------
# Audit chain + per-key attribution
# ---------------------------------------------------------------------------

def test_receipt_chain_verifies_and_detects_tampering():
    app, _, _ = _stack()
    with TestClient(app) as c:
        for _ in range(3):
            _chat(c)
        v = c.get("/v1/receipts/verify").json()
        assert v["ok"] is True
        assert v["receipts_checked"] == 3
        # Receipts are signed by default now — verify surfaces the algo.
        assert v["signed"] is True
        assert v["signature_algorithm"] in ("ed25519", "hmac-sha256")
        # Tamper with a historical receipt → detected AT the edited
        # receipt via its signature (stronger than the old chain-only
        # detection, which only caught the next link).
        gw = app.state.gateway
        gw._receipts[1]["cost_usd"] = 0.0
        v2 = c.get("/v1/receipts/verify").json()
    assert v2["ok"] is False
    assert v2["broken_at_index"] == 1          # the edited receipt itself
    assert v2["reason"] in ("bad_signature", "chain_break")


def test_spend_attributed_to_key_fingerprints():
    app, _, _ = _stack()
    with TestClient(app) as c:
        c.post("/v1/chat/completions", json={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer key-of-team-a",
                     "X-Budget-Envelope": "acme/a"})
        c.post("/v1/chat/completions", json={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer key-of-team-b",
                     "X-Budget-Envelope": "acme/b"})
        by_key = c.get("/v1/spend/by_key").json()["by_key"]
    assert len(by_key) == 2
    for fp, row in by_key.items():
        assert len(fp) == 16                   # fingerprint, never raw
        assert "key-of-team" not in fp
        assert row["calls"] == 1


def test_dashboard_served():
    app, _, _ = _stack()
    with TestClient(app) as c:
        r = c.get("/dashboard")
    assert r.status_code == 200
    assert "AI Spend Governance" in r.text


# ---------------------------------------------------------------------------
# HG13f semantic cache + HG13h compression wired into the gateway
# ---------------------------------------------------------------------------

def test_semantic_cache_serves_near_duplicate_at_zero_cost():
    from governance.token_thrift import SemanticCache
    calls = []
    sc = SemanticCache(threshold=0.9, ttl_s=300, cache_all=True)
    app, budget, _ = _stack(http_post=_openai_upstream(calls),
                            calls=calls, semantic_cache=sc)
    with TestClient(app) as c:
        # First call populates both caches.
        r1 = c.post("/v1/chat/completions", json={
            "model": "gpt-test", "temperature": 0,
            "messages": [{"role": "user",
                          "content": "What is the capital of France?"}]},
            headers={"X-Budget-Envelope": "acme/x"})
        assert r1.status_code == 200
        assert len(calls) == 1
        # Near-duplicate (extra whitespace) — exact cache MISSES, the
        # semantic tier catches it.
        r2 = c.post("/v1/chat/completions", json={
            "model": "gpt-test", "temperature": 0,
            "messages": [{"role": "user",
                          "content": "What is the capital of France?   "}]},
            headers={"X-Budget-Envelope": "acme/x"})
        assert r2.status_code == 200
        assert len(calls) == 1                  # upstream NOT called again
        assert r2.headers["X-Pluginfer-Cache"].startswith("semantic:")
        assert r2.headers["X-Pluginfer-Cost-USD"] == "0.00000000"
        sav = c.get("/v1/savings").json()
    assert sav["semantic_saved_usd"] > 0
    assert sav["semantic_cache"]["backend"] == "lexical-3gram"


def test_compression_reduces_input_and_reports_estimated_saving():
    from governance.token_thrift import PromptCompressor
    calls = []
    pc = PromptCompressor(dedup_exact=True)
    app, budget, _ = _stack(http_post=_openai_upstream(calls),
                            calls=calls, compressor=pc)
    with TestClient(app) as c:
        # Three identical tool-output messages (agent retry pathology).
        r = c.post("/v1/chat/completions", json={
            "model": "gpt-test", "max_tokens": 100,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "duplicated blob " * 20},
                {"role": "user", "content": "duplicated blob " * 20},
                {"role": "user", "content": "duplicated blob " * 20},
            ]},
            headers={"X-Budget-Envelope": "acme/x"})
        assert r.status_code == 200
        # The upstream saw the DEDUPED body (2 messages, not 4).
        assert len(calls[0]["body"]["messages"]) == 2
        rec = c.get("/v1/receipts").json()["receipts"][-1]
        sav = c.get("/v1/savings").json()
    assert rec["compression"]["applied"]
    assert sav["compression_saved_est_usd"] > 0
    # Compression estimate is kept OUT of the measured net.
    assert sav["net_saved_usd"] == 0.0


def test_compression_and_semantic_savings_are_separate_buckets():
    from governance.token_thrift import PromptCompressor, SemanticCache
    calls = []
    app, _, _ = _stack(
        http_post=_openai_upstream(calls), calls=calls,
        compressor=PromptCompressor(collapse_whitespace=True),
        semantic_cache=SemanticCache(threshold=0.9, cache_all=True))
    with TestClient(app) as c:
        sav = c.get("/v1/savings").json()
    # The two live in distinct fields; measured net never absorbs the
    # compression estimate.
    assert "compression_saved_est_usd" in sav
    assert "net_saved_usd" in sav
    assert sav["semantic_cache"] is not None


def test_missing_usage_settles_at_estimate_and_says_so():
    def no_usage(url, body, headers, timeout_s):
        return 200, json.dumps({"choices": []}).encode()
    app, budget, _ = _stack(http_post=no_usage)
    with TestClient(app) as c:
        r = _chat(c)
        assert r.status_code == 200
        rec = c.get("/v1/receipts").json()["receipts"][-1]
    assert rec["estimated"] is True
    assert budget.report()["total_spend_usd"] > 0.0


# ---------------------------------------------------------------------------
# Anthropic shape
# ---------------------------------------------------------------------------

def test_anthropic_messages_usage_keys_parse():
    def anthropic(url, body, headers, timeout_s):
        resp = {"id": "msg-1", "content": [{"type": "text",
                                            "text": "hello"}],
                "usage": {"input_tokens": 2000, "output_tokens": 100}}
        return 200, json.dumps(resp).encode()
    app, budget, _ = _stack(http_post=anthropic)
    with TestClient(app) as c:
        r = c.post("/v1/messages", json={
            "model": "claude-test", "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Budget-Envelope": "acme/eng"})
    assert r.status_code == 200
    # 2000/1e6*3.0 + 100/1e6*15.0 = 0.0075
    assert budget.report()["total_spend_usd"] == pytest.approx(0.0075)


# ---------------------------------------------------------------------------
# Admin + reporting surface
# ---------------------------------------------------------------------------

def test_quote_endpoint_estimates_without_forwarding():
    app, _, calls = _stack()
    with TestClient(app) as c:
        r = c.post("/v1/quote", json={
            "model": "gpt-test", "max_tokens": 1000,
            "messages": [{"role": "user", "content": "x" * 4000}]})
    assert r.status_code == 200
    body = r.json()
    # Token count now comes from a real tokenizer (tiktoken when
    # present), so we assert structure + a real positive price rather
    # than a chars/4 constant. Settlement still uses upstream usage.
    assert body["estimated_input_tokens"] >= 1
    assert body["estimated_max_cost_usd"] > 0
    assert "tokenizer" in body
    assert calls == []


def test_envelope_admin_roundtrip_and_report():
    # With an admin key set, the WHOLE surface is now enforced (the
    # security fix): admin for admin ops, a client key to forward, and
    # a read credential to view spend.
    app, _, _ = _stack(admin_key="s3cret")
    admin = {"X-Admin-Key": "s3cret"}
    with TestClient(app) as c:
        r = c.post("/v1/budget/envelopes",
                   json={"path": "acme/newteam", "cap_usd": 5.0,
                         "period": "day"})
        assert r.status_code == 401            # admin key enforced
        r = c.post("/v1/budget/envelopes",
                   json={"path": "acme/newteam", "cap_usd": 5.0,
                         "period": "day"}, headers=admin)
        assert r.status_code == 200
        assert r.json()["cap_usd"] == 5.0
        # Mint a client key pinned to the envelope, then forward with it.
        raw = c.post("/v1/keys",
                     json={"envelope": "acme/newteam"},
                     headers=admin).json()["api_key"]
        rc = c.post("/v1/chat/completions", json={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {raw}"})
        assert rc.status_code == 200
        # Read endpoints need a read/admin credential now.
        assert c.get("/v1/budget/report").status_code == 401
        rep = c.get("/v1/budget/report", headers=admin).json()
        assert rep["by_envelope"]["acme/newteam"]["jobs"] == 1
        listed = c.get("/v1/budget/envelopes",
                       headers=admin).json()["envelopes"]
        assert any(e["path"] == "acme/newteam" for e in listed)


def test_gateway_key_replaces_client_auth():
    calls = []
    app, _, _ = _stack(calls=calls, upstream_api_key="real-key")
    with TestClient(app) as c:
        c.post("/v1/chat/completions", json={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer app-held-gateway-key"})
    sent = calls[0]["headers"]
    assert sent["Authorization"] == "Bearer real-key"


# ---------------------------------------------------------------------------
# Multi-upstream: X-Pluginfer-Upstream + allowlist (SSRF-gated)
# ---------------------------------------------------------------------------

def test_upstream_override_allowlisted_and_ssrf_refused():
    seen = []

    def upstream(url, body, headers, timeout_s):
        seen.append(url)
        resp = {"choices": [{"message": {"content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10}}
        return 200, json.dumps(resp).encode()

    app, _, _ = _stack(http_post=upstream,
                       upstream_allowlist=["https://api.groq.com/openai"])
    with TestClient(app) as c:
        # Allowlisted override → forwarded to the requested base.
        r = c.post("/v1/chat/completions", json={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Pluginfer-Upstream":
                     "https://api.groq.com/openai"})
        assert r.status_code == 200
        assert seen[-1].startswith("https://api.groq.com/openai/")
        rec = c.get("/v1/receipts").json()["receipts"][-1]
        assert rec["upstream"].startswith("https://api.groq.com/openai")
        # Non-allowlisted → 403, upstream NEVER called.
        n = len(seen)
        r2 = c.post("/v1/chat/completions", json={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Pluginfer-Upstream": "https://evil.example"})
        assert r2.status_code == 403
        assert len(seen) == n
        # No header → default upstream, unchanged behavior.
        r3 = c.post("/v1/chat/completions", json={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}]})
        assert r3.status_code == 200
        assert seen[-1].startswith("https://upstream.example")
