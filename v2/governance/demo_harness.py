"""Live local demo of the Governance Gateway — no real API, no mesh,
no risk. Stands up the REAL gateway (real budget ledger, real thrift
stack, real receipt chain) in-process against a SIMULATED upstream that
returns OpenAI-shaped responses with realistic token usage, then drives
a realistic traffic mix so you can watch every mechanism fire:

    python -m governance.demo_harness            # print a report
    python -m governance.demo_harness --serve     # + serve the live
                                                  #   dashboard on :8799

The simulated upstream lets this run anywhere with zero credentials.
Swap `upstream_base` + a real API key and the identical code governs a
real provider — that path needs a PAID API key (subscription OAuth,
like Claude Code's, cannot be proxied; see WORKLOG 2026-07-08 pxpipe).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from governance.budget_ledger import BudgetLedger
from governance.gateway import ResponseCache, build_governance_gateway
from governance.token_thrift import PromptCompressor, SemanticCache

# A price sheet mirroring real July-2026 tiers (research/00_scoreboard).
PRICES = {
    "gpt-4o": {"input_per_1m": 2.5, "output_per_1m": 10.0},
    "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.60},
    "claude-sonnet": {"input_per_1m": 3.0, "output_per_1m": 15.0},
}
CASCADES = {"gpt-4o": "gpt-4o-mini"}   # try the mini first for gpt-4o


def _simulated_upstream():
    """Deterministic OpenAI-shaped upstream. Output size scales with
    the model so cascade + cache math is realistic. Counts calls."""
    state = {"calls": 0}

    def post(url, body, headers, timeout_s):
        state["calls"] += 1
        req = json.loads(body)
        model = req.get("model", "")
        prompt = req["messages"][-1].get("content", "")
        # gpt-4o-mini gives a short but VALID answer → cascade accepts.
        answer = f"[{model}] answer to: {prompt[:40]}"
        in_tok = max(1, sum(len(m.get("content", ""))
                            for m in req["messages"]) // 4)
        out_tok = 40 if "mini" in model else 300
        resp = {"id": "sim", "object": "chat.completion",
                "choices": [{"message": {"role": "assistant",
                                         "content": answer},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": in_tok,
                          "completion_tokens": out_tok}}
        return 200, json.dumps(resp).encode()

    return post, state


def build_demo(state_dir=None):
    budget = BudgetLedger(state_dir)
    # Envelope tree: org cap, and a tight team cap we will breach.
    budget.set_envelope("acme", 100.0, "month")
    budget.set_envelope("acme/support", 100.0, "month")
    budget.set_envelope("acme/support/chatbot", 0.02, "month")  # tight
    budget.set_envelope("acme/research", 100.0, "month")
    post, state = _simulated_upstream()
    app = build_governance_gateway(
        budget=budget, upstream_base="https://sim.local",
        price_sheet=PRICES, http_post=post,
        cache=ResponseCache(ttl_s=3600, cache_all=True),
        semantic_cache=SemanticCache(threshold=0.92, ttl_s=3600,
                                     cache_all=True),
        compressor=PromptCompressor(dedup_exact=True,
                                    collapse_whitespace=True),
        cascades=CASCADES,
    )
    return app, budget, state


def drive_traffic(client):
    """A realistic mix: fresh calls, exact repeats, near-duplicates,
    a cascade, a compressible agent loop, and a budget breach."""
    H = {"X-Budget-Envelope": "acme/research"}
    log = []

    def call(env, model, content, **extra):
        r = client.post("/v1/chat/completions", json={
            "model": model, "temperature": 0,
            "messages": [{"role": "user", "content": content}], **extra},
            headers={"X-Budget-Envelope": env,
                     "Authorization": "Bearer sk-team-" + env.split("/")[0]})
        log.append((env, model, content[:30], r.status_code,
                    r.headers.get("X-Pluginfer-Cache", "-"),
                    r.headers.get("X-Pluginfer-Cost-USD", "-")))
        return r

    # 5 distinct research questions on the premium model (cascade tries
    # mini first and accepts → big saving vs gpt-4o).
    for q in ("Summarize the Q3 board deck",
              "Draft a customer apology email",
              "Explain our refund policy",
              "Translate the onboarding guide",
              "List risks in the vendor contract"):
        call("acme/research", "gpt-4o", q)

    # 20 exact repeats of the most common question (cache → $0 each).
    for _ in range(20):
        call("acme/research", "gpt-4o", "Explain our refund policy")

    # 10 near-duplicates (whitespace/punctuation drift) → semantic cache.
    for i in range(10):
        call("acme/research", "gpt-4o",
             "Summarize the Q3 board deck" + "  " * i + ".")

    # Agent retry loop: same tool output pasted 4× → compression dedups.
    client.post("/v1/chat/completions", json={
        "model": "claude-sonnet", "temperature": 0, "messages": [
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": "TOOL OUTPUT: " + "data " * 50},
            {"role": "user", "content": "TOOL OUTPUT: " + "data " * 50},
            {"role": "user", "content": "TOOL OUTPUT: " + "data " * 50},
            {"role": "user", "content": "TOOL OUTPUT: " + "data " * 50},
        ]}, headers={"X-Budget-Envelope": "acme/research",
                     "Authorization": "Bearer sk-team-acme"})

    # Budget breach: the chatbot envelope caps at $0.02 — a handful of
    # GENUINELY DISTINCT questions (so neither cache tier serves them)
    # exhaust it, and the rest are refused with HTTP 402, no upstream
    # hit. This is the "AmEx can't cap spend" problem, solved.
    distinct = [
        "How do I reset my password?",
        "What are your business hours on holidays?",
        "Can I get a refund for a damaged item?",
        "Where is my order shipment right now?",
        "Do you offer student discounts?",
        "How do I cancel my subscription?",
        "Is international shipping available to Japan?",
        "What is your privacy policy on data retention?",
    ]
    for q in distinct:
        call("acme/support/chatbot", "claude-sonnet", q)

    return log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true",
                    help="serve the live dashboard after driving traffic")
    ap.add_argument("--port", type=int, default=8799)
    args = ap.parse_args()

    from fastapi.testclient import TestClient
    app, budget, state = build_demo()
    with TestClient(app) as client:
        t0 = time.time()
        log = drive_traffic(client)
        dt = time.time() - t0
        report = budget.report()
        savings = client.get("/v1/savings").json()
        by_key = client.get("/v1/spend/by_key").json()
        chain = client.get("/v1/receipts/verify").json()

    n = len(log)
    refused = sum(1 for r in log if r[3] == 402)
    cache_hits = sum(1 for r in log if r[4] != "-" and r[4] != "miss")
    print("=" * 68)
    print(" PLUGINFER SIGNET - AI SPEND GATEWAY - LIVE LOCAL DEMO")
    print("=" * 68)
    print(f" {n} governed calls in {dt*1000:.0f} ms "
          f"({n/dt:.0f}/s) · upstream hit {state['calls']} times")
    print(f" cache/thrift short-circuits: {n - state['calls']} calls "
          f"never reached the upstream")
    print(f" budget breaches refused (HTTP 402): {refused}")
    print("-" * 68)
    print(f" TOTAL SPEND (measured):        ${report['total_spend_usd']:.4f}")
    print(f" NET SAVED (measured):          ${savings['net_saved_usd']:.4f}")
    print(f"   exact-cache saved:           ${savings['cache_saved_usd']:.4f}")
    print(f"   semantic-cache saved:        ${savings['semantic_saved_usd']:.4f}")
    print(f"   cascade saved:               ${savings['cascade_saved_usd']:.4f}")
    print(f"   (cascade escalation cost:    ${savings['cascade_escalation_cost_usd']:.4f})")
    print(f" COMPRESSION saved (ESTIMATE):  ${savings['compression_saved_est_usd']:.4f}")
    spent = report['total_spend_usd']
    saved = savings['net_saved_usd']
    if spent + saved > 0:
        pct = saved / (spent + saved) * 100
        print(f"  => measured spend WITHOUT thrift would have been "
              f"${spent+saved:.4f}; thrift cut it by {pct:.0f}%")
    print("-" * 68)
    print(" spend by API key (fingerprints, never raw keys):")
    for fp, row in by_key["by_key"].items():
        print(f"   {fp}  ${row['spend_usd']:.4f}  "
              f"{row['calls']} calls  {row['envelopes']}")
    print("-" * 68)
    print(f" audit chain: {'INTACT' if chain['ok'] else 'BROKEN'} "
          f"({chain['receipts_checked']} receipts hash-linked)")
    print("=" * 68)

    if args.serve:
        import uvicorn
        # Demo-ONLY endpoints (deliberately NOT in the product gateway):
        # let the operator watch tamper-detection flip live. GET so a
        # browser can trigger them too.
        _orig = {}

        @app.get("/demo/tamper")
        async def _demo_tamper():
            gw = app.state.gateway
            with gw._receipts_lock:
                if len(gw._receipts) < 3:
                    return {"tampered": False, "reason": "too few receipts"}
                idx = 1
                _orig[idx] = gw._receipts[idx].get("cost_usd")
                gw._receipts[idx]["cost_usd"] = 999.99   # edit history
            return {"tampered": True, "edited_receipt_index": idx,
                    "was_cost_usd": _orig[idx], "now_cost_usd": 999.99,
                    "note": "every hash link AFTER this receipt is now "
                            "broken — /v1/receipts/verify will show it"}

        @app.get("/demo/restore")
        async def _demo_restore():
            gw = app.state.gateway
            with gw._receipts_lock:
                for idx, val in _orig.items():
                    gw._receipts[idx]["cost_usd"] = val
                _orig.clear()
            return {"restored": True,
                    "note": "receipt returned to its original value; "
                            "the chain verifies again"}

        print(f"\n Serving live dashboard -> http://127.0.0.1:{args.port}"
              f"/dashboard  (Ctrl-C to stop)")
        print(f" Demo controls: GET /demo/tamper  and  /demo/restore\n")
        uvicorn.run(app, host="127.0.0.1", port=args.port,
                    log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
