"""HG13g — judge-model cascade scorer.

Hermetic: every "model" (cheap, target, judge) is one injected
http_post that switches on body["model"]. Pins the contract:

  * the judge NARROWS acceptance — it runs only after hard signals
    pass, and its rejection escalates to the target model,
  * the judge's own call is REAL spend: metered into the settle on
    acceptance (saving reduced, signed — may go negative) and into the
    escalation overhead on rejection,
  * judge failure follows the explicit on_error policy,
  * a judge model outside the price sheet refuses construction,
  * evaluate_golden reports agreement / false-accept / false-escalate
    against operator labels, admin-gated over HTTP.
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

from governance.budget_ledger import BudgetLedger
from governance.gateway import build_governance_gateway
from governance.judge import CascadeJudge, JudgeVerdict

PRICES = {
    "gpt-test": {"input_per_1m": 1.0, "output_per_1m": 10.0},
    "cheap-test": {"input_per_1m": 0.1, "output_per_1m": 1.0},
    "judge-test": {"input_per_1m": 0.05, "output_per_1m": 0.5},
}


def _chat_resp(text, usage=(1000, 500), finish="stop"):
    return {
        "choices": [{"message": {"role": "assistant", "content": text},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
    }


def _upstream(judge_score=9, calls=None):
    """One fake provider serving all three models."""
    def post(url, body, headers, timeout_s):
        b = json.loads(body)
        if calls is not None:
            calls.append(b["model"])
        if b["model"] == "judge-test":
            resp = _chat_resp(json.dumps({"score": judge_score,
                                          "reason": "checked"}),
                              usage=(200, 20))
        elif b["model"] == "cheap-test":
            resp = _chat_resp("cheap answer")
        else:
            resp = _chat_resp("expensive answer")
        return 200, json.dumps(resp).encode("utf-8")
    return post


def _app(tmp_path, *, judge_score=9, judge=None, calls=None, **kw):
    budget = BudgetLedger(str(tmp_path / "budget"))
    budget.set_envelope("acme", 10.0, "month")
    app = build_governance_gateway(
        budget=budget, upstream_base="https://upstream.example",
        price_sheet=PRICES,
        http_post=_upstream(judge_score=judge_score, calls=calls),
        cascades={"gpt-test": "cheap-test"},
        cascade_judge=judge if judge is not None
        else CascadeJudge("judge-test", threshold=7.0),
        **kw)
    return app


def _chat(client, **extra):
    return client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test",
              "messages": [{"role": "user", "content": "question"}],
              "max_tokens": 100, **extra},
        headers={"X-Budget-Envelope": "acme/team"})


# ---------------------------------------------------------------------------
# parse_score robustness
# ---------------------------------------------------------------------------

def test_parse_score_variants():
    p = CascadeJudge.parse_score
    ok = p(_chat_resp('{"score": 8, "reason": "good"}'))
    assert ok == (8.0, "good")
    # JSON wrapped in prose / code fences still parses.
    assert p(_chat_resp('Sure!\n```json\n{"score": 3.5, "reason": "meh"}'
                        '\n```'))[0] == 3.5
    # Garbage, out-of-range, boolean, missing → None.
    assert p(_chat_resp("ACCEPT")) is None
    assert p(_chat_resp('{"score": 11}')) is None
    assert p(_chat_resp('{"score": true}')) is None
    assert p(_chat_resp('{"reason": "no score"}')) is None
    assert p({"choices": []}) is None


def test_judge_config_validation():
    with pytest.raises(ValueError):
        CascadeJudge("j", on_error="explode")
    with pytest.raises(ValueError):
        CascadeJudge("j", threshold=11)


def test_judge_on_error_policies():
    boom = CascadeJudge("j", on_error="escalate")
    v = boom.judge({"messages": []}, "x",
                   lambda b: (_ for _ in ()).throw(OSError("down")))
    assert v.accept is False and "judge_unreachable" in v.error

    lenient = CascadeJudge("j", on_error="accept")
    v2 = lenient.judge({"messages": []}, "x", lambda b: (500, None))
    assert v2.accept is True and v2.error == "judge_http_500"

    v3 = lenient.judge({"messages": []}, "x",
                       lambda b: (200, _chat_resp("not json")))
    assert v3.accept is True and v3.error == "judge_unparseable"


# ---------------------------------------------------------------------------
# Gateway flow — accept / reject / spend accounting
# ---------------------------------------------------------------------------

def test_judge_accept_settles_cheap_plus_judge(tmp_path):
    calls = []
    app = _app(tmp_path, judge_score=9, calls=calls)
    with TestClient(app) as c:
        r = _chat(c)
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == \
            "cheap answer"
        assert r.headers["X-Pluginfer-Cascade"] == \
            "accepted+judged:cheap-test"
        # cheap 0.0006 + judge 0.00002; saved = 0.006 − total (signed).
        assert float(r.headers["X-Pluginfer-Cost-USD"]) == \
            pytest.approx(0.00062, abs=1e-9)
        assert float(r.headers["X-Pluginfer-Saved-USD"]) == \
            pytest.approx(0.006 - 0.00062, abs=1e-9)
        # The target model was never called.
        assert calls == ["cheap-test", "judge-test"]
        rec = c.get("/v1/receipts").json()["receipts"][-1]
        assert rec["kind"] == "cascade_accept"
        assert rec["judge"]["model"] == "judge-test"
        assert rec["judge"]["score"] == 9.0
        assert rec["judge"]["cost_usd"] == pytest.approx(0.00002,
                                                         abs=1e-9)


def test_judge_reject_escalates_and_meters_judge_cost(tmp_path):
    calls = []
    app = _app(tmp_path, judge_score=2, calls=calls)
    with TestClient(app) as c:
        r = _chat(c)
        assert r.status_code == 200
        # Escalated: the caller gets the TARGET model's answer.
        assert r.json()["choices"][0]["message"]["content"] == \
            "expensive answer"
        assert calls == ["cheap-test", "judge-test", "gpt-test"]
        recs = c.get("/v1/receipts").json()["receipts"]
        esc = [x for x in recs if x["kind"] == "cascade_escalate"][-1]
        # The escalation receipt names the judge verdict and carries
        # the judge's own metering.
        assert "judge_rejected(score=2.0)" in \
            esc["cascade"]["escalated_because"]
        assert esc["cascade"]["judge"]["score"] == 2.0
        # Overhead = cheap try + judge call, surfaced as negative saving.
        sav = c.get("/v1/savings").json()
        assert sav["cascade_escalation_cost_usd"] == \
            pytest.approx(0.0006 + 0.00002, abs=1e-9)


def test_judge_error_default_escalates_via_gateway(tmp_path):
    def post(url, body, headers, timeout_s):
        b = json.loads(body)
        if b["model"] == "judge-test":
            return 503, b"overloaded"
        resp = _chat_resp("cheap answer" if b["model"] == "cheap-test"
                          else "expensive answer")
        return 200, json.dumps(resp).encode("utf-8")

    budget = BudgetLedger(str(tmp_path / "budget"))
    budget.set_envelope("acme", 10.0, "month")
    app = build_governance_gateway(
        budget=budget, upstream_base="https://upstream.example",
        price_sheet=PRICES, http_post=post,
        cascades={"gpt-test": "cheap-test"},
        cascade_judge=CascadeJudge("judge-test"))
    with TestClient(app) as c:
        r = _chat(c)
        # Judge down + on_error=escalate → target model answers.
        assert r.json()["choices"][0]["message"]["content"] == \
            "expensive answer"


def test_negative_saving_recorded_when_judge_eats_margin(tmp_path):
    prices = dict(PRICES,
                  **{"judge-test": {"input_per_1m": 10000.0,
                                    "output_per_1m": 0.5}})
    budget = BudgetLedger(str(tmp_path / "budget"))
    budget.set_envelope("acme", 10.0, "month")
    app = build_governance_gateway(
        budget=budget, upstream_base="https://upstream.example",
        price_sheet=prices, http_post=_upstream(judge_score=9),
        cascades={"gpt-test": "cheap-test"},
        cascade_judge=CascadeJudge("judge-test"))
    with TestClient(app) as c:
        r = _chat(c)
        assert r.status_code == 200
        # Judge cost (2.0) dwarfs the 0.0054 margin — the SIGNED saving
        # goes negative on the receipt instead of being clamped away.
        assert float(r.headers["X-Pluginfer-Saved-USD"]) < 0


def test_judge_outside_price_sheet_refuses_construction(tmp_path):
    budget = BudgetLedger(None)
    with pytest.raises(ValueError, match="price sheet"):
        build_governance_gateway(
            budget=budget, upstream_base="https://u.example",
            price_sheet={"gpt-test": PRICES["gpt-test"]},
            cascades={"gpt-test": "gpt-test"},
            cascade_judge=CascadeJudge("unpriced-judge"))


def test_no_judge_keeps_previous_behavior(tmp_path):
    budget = BudgetLedger(str(tmp_path / "budget"))
    budget.set_envelope("acme", 10.0, "month")
    app = build_governance_gateway(
        budget=budget, upstream_base="https://upstream.example",
        price_sheet=PRICES, http_post=_upstream(),
        cascades={"gpt-test": "cheap-test"})
    with TestClient(app) as c:
        r = _chat(c)
        assert r.headers["X-Pluginfer-Cascade"] == "accepted:cheap-test"
        assert float(r.headers["X-Pluginfer-Cost-USD"]) == \
            pytest.approx(0.0006, abs=1e-9)


# ---------------------------------------------------------------------------
# Golden-set evaluation
# ---------------------------------------------------------------------------

GOLDEN = [
    {"prompt": "2+2?", "answer": "4", "label": "accept"},
    {"prompt": "2+2?", "answer": "5", "label": "escalate"},
    {"prompt": "capital of France?", "answer": "Paris",
     "label": "accept"},
]


def test_evaluate_golden_metrics():
    judge = CascadeJudge("judge-test", threshold=7.0)
    scores = {"4": 9, "5": 1, "Paris": 3}   # judge wrongly dislikes Paris

    def post_fn(jbody):
        answer = jbody["messages"][1]["content"].rsplit(
            "CANDIDATE ANSWER:\n", 1)[1]
        return 200, _chat_resp(json.dumps(
            {"score": scores[answer], "reason": "r"}), usage=(10, 5))

    rep = judge.evaluate_golden(GOLDEN, post_fn)
    assert rep["items"] == 3 and rep["judged"] == 3
    assert rep["agreement"] == 2
    assert rep["false_accepts"] == 0
    assert rep["false_escalates"] == 1     # Paris was good; judge balked
    assert rep["agreement_rate"] == pytest.approx(2 / 3, abs=1e-4)

    with pytest.raises(ValueError, match="label"):
        judge.evaluate_golden([{"prompt": "x", "answer": "y",
                                "label": "maybe"}], post_fn)


def test_golden_endpoint_admin_gated_and_working(tmp_path):
    from governance.auth import AuthConfig
    app = _app(tmp_path, judge_score=8,
               auth=AuthConfig(admin_key="s3cret"))
    with TestClient(app) as c:
        assert c.post("/v1/cascade/judge/golden",
                      json={"items": GOLDEN}).status_code == 401
        r = c.post("/v1/cascade/judge/golden", json={"items": GOLDEN},
                   headers={"X-Admin-Key": "s3cret"})
        assert r.status_code == 200
        body = r.json()
        assert body["judge_model"] == "judge-test"
        assert body["items"] == 3
        # Fake judge scores everything 8 → both "accept" labels agree,
        # the "escalate" label becomes the dangerous false accept.
        assert body["false_accepts"] == 1
        # Guard rails.
        assert c.post("/v1/cascade/judge/golden", json={},
                      headers={"X-Admin-Key": "s3cret"}
                      ).status_code == 400
        too_many = {"items": [GOLDEN[0]] * 201}
        assert c.post("/v1/cascade/judge/golden", json=too_many,
                      headers={"X-Admin-Key": "s3cret"}
                      ).status_code == 400


def test_golden_endpoint_without_judge_is_honest(tmp_path):
    budget = BudgetLedger(str(tmp_path / "budget"))
    app = build_governance_gateway(
        budget=budget, upstream_base="https://upstream.example",
        price_sheet=PRICES, http_post=_upstream())
    with TestClient(app) as c:
        r = c.post("/v1/cascade/judge/golden", json={"items": GOLDEN})
        assert r.status_code == 400
        assert "no cascade judge configured" in r.json()["error"]
