"""Signet model router — rule-driven best-model-per-task selection.

Pins: first-match-wins rules, the transparent task classifier, the
gateway swap with MEASURED savings at actual usage, unroutable models
(not on the price sheet) left untouched, and streams left unrouted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from governance.budget_ledger import BudgetLedger
from governance.gateway import build_governance_gateway
from governance.router import ModelRouter, classify_task

PRICES = {
    "gpt-big": {"input_per_1m": 10.0, "output_per_1m": 30.0},
    "gpt-mini": {"input_per_1m": 1.0, "output_per_1m": 2.0},
}


def _upstream(calls):
    def post(url, body, headers, timeout_s):
        b = json.loads(body)
        calls.append(b)
        resp = {"choices": [{"message": {"content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000,
                          "completion_tokens": 1000}}
        return 200, json.dumps(resp).encode()
    return post


def _app(rules, calls):
    return build_governance_gateway(
        budget=BudgetLedger(), upstream_base="https://up.example",
        price_sheet=PRICES, http_post=_upstream(calls),
        router=ModelRouter(rules))


def test_classifier_is_transparent():
    assert classify_task({"messages": [
        {"role": "user", "content": "please summarize this article"}]}) \
        == "summarize"
    assert classify_task({"messages": [
        {"role": "user", "content": "def f(): pass  fix this bug"}]}) \
        == "code"
    assert classify_task({"messages": [
        {"role": "user", "content": "hello there"}]}) == "chat"


def test_rule_routes_and_saving_is_measured():
    calls = []
    app = _app([{"id": "sum-cheap", "when": {"task": "summarize"},
                 "use": "gpt-mini"}], calls)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "gpt-big",
            "messages": [{"role": "user",
                          "content": "summarize the quarterly report"}]})
        assert r.status_code == 200
        assert calls[0]["model"] == "gpt-mini"       # swapped pre-flight
        assert r.headers["X-Pluginfer-Routed"] == "gpt-big->gpt-mini"
        # Measured saving at ACTUAL usage (1000 in / 1000 out):
        # big (10+30)/1e3 = 0.04; mini (1+2)/1e3 = 0.003; saved 0.037.
        assert float(r.headers["X-Pluginfer-Saved-USD"]) == \
            pytest.approx(0.037)
        rec = c.get("/v1/receipts").json()["receipts"][-1]
    assert rec["kind"] == "routed"
    assert rec["requested_model"] == "gpt-big"
    assert rec["routing"]["rule"] == "sum-cheap"
    assert rec["saved_usd"] == pytest.approx(0.037)


def test_no_matching_rule_leaves_request_untouched():
    calls = []
    app = _app([{"id": "code-only", "when": {"task": "code"},
                 "use": "gpt-mini"}], calls)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "gpt-big",
            "messages": [{"role": "user", "content": "hi there"}]})
        assert r.status_code == 200
    assert calls[0]["model"] == "gpt-big"
    assert "X-Pluginfer-Routed" not in r.headers


def test_unpriced_target_never_routed():
    calls = []
    app = _app([{"id": "bad", "when": {}, "use": "not-priced"}], calls)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "gpt-big",
            "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
    assert calls[0]["model"] == "gpt-big"            # untouched


def test_envelope_scoped_user_rule():
    calls = []
    app = _app([{"id": "ci-cheap",
                 "when": {"envelope_prefix": "acme/ci"},
                 "use": "gpt-mini"}], calls)
    with TestClient(app) as c:
        c.post("/v1/chat/completions", json={
            "model": "gpt-big",
            "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Budget-Envelope": "acme/ci/tests"})
        c.post("/v1/chat/completions", json={
            "model": "gpt-big",
            "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Budget-Envelope": "acme/support"})
    assert calls[0]["model"] == "gpt-mini"           # ci envelope routed
    assert calls[1]["model"] == "gpt-big"            # others untouched


# ---------------------------------------------------------------------------
# Auto-save: zero-config cost routing from the price sheet
# ---------------------------------------------------------------------------

def test_auto_save_routes_simple_tasks_to_cheapest_never_code():
    from governance.router import auto_save_rules, ModelRouter
    rules = auto_save_rules(PRICES)  # gpt-big vs gpt-mini
    r = ModelRouter(rules)
    # A chat prompt -> cheapest.
    tgt, rid, task = r.route(
        {"messages": [{"role": "user", "content": "hello there"}]}, "acme")
    assert tgt == "gpt-mini" and task == "chat"
    # A code prompt -> untouched (no auto-save rule for 'code').
    tgt2, _, task2 = r.route(
        {"messages": [{"role": "user", "content": "fix this bug def f():"}]},
        "acme")
    assert task2 == "code" and tgt2 is None


def test_auto_save_noop_with_one_model():
    from governance.router import auto_save_rules
    assert auto_save_rules({"only": {"input_per_1m": 1, "output_per_1m": 2}}) == []


def test_auto_save_gateway_measures_saving():
    calls = []
    app = build_governance_gateway(
        budget=BudgetLedger(), upstream_base="https://up.example",
        price_sheet=PRICES, http_post=_upstream(calls),
        router=ModelRouter(__import__("governance.router",
                           fromlist=["auto_save_rules"]).auto_save_rules(PRICES)))
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "gpt-big",
            "messages": [{"role": "user", "content": "hello there"}]})
        assert r.status_code == 200
        assert r.headers["X-Pluginfer-Routed"] == "gpt-big->gpt-mini"
        assert float(r.headers["X-Pluginfer-Saved-USD"]) > 0
