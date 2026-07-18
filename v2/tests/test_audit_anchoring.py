"""HG13j — external anchoring of the receipt-chain head.

Hermetic: calendar submissions are injected (or served by a loopback
stdlib HTTP server) — no public network in CI. Pins the contract:

  * the detached .ots proof is byte-exact OpenTimestamps format
    (magic + version + sha256 tag + digest + calendar ops),
  * anchoring is FAIL-OPEN (calendar failures are journaled, never
    raised into the spend path),
  * anchor_if_new skips genesis and unchanged heads,
  * the journal survives a manager restart,
  * gateway endpoints: /now is admin-gated, records + proof download
    are public, disabled mode answers honestly,
  * the scheduler anchors a moving head on its own thread.
"""

from __future__ import annotations

import http.server
import json
import sys
import threading
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest
from fastapi.testclient import TestClient

from governance.anchoring import (
    AnchorManager, AnchorScheduler, AnchorSubmitError, OTS_HEADER_MAGIC,
    build_detached_ots, submit_to_calendar,
)
from governance.budget_ledger import BudgetLedger
from governance.gateway import build_governance_gateway

HEAD = "ab" * 32          # a valid 64-hex sha256-shaped head
FAKE_OPS = b"\xf0\x10" + b"\x11" * 16 + b"\x00fake-calendar-ops"


def _fake_submit(url, digest):
    return FAKE_OPS


# ---------------------------------------------------------------------------
# Proof-file format
# ---------------------------------------------------------------------------

def test_detached_ots_is_exact_format():
    digest = bytes.fromhex(HEAD)
    proof = build_detached_ots(digest, FAKE_OPS)
    assert proof.startswith(OTS_HEADER_MAGIC)
    rest = proof[len(OTS_HEADER_MAGIC):]
    assert rest[0:1] == b"\x01"          # version 1
    assert rest[1:2] == b"\x08"          # sha256 file-hash op tag
    assert rest[2:34] == digest          # the anchored head, verbatim
    assert rest[34:] == FAKE_OPS         # calendar ops appended untouched


def test_detached_ots_rejects_bad_inputs():
    with pytest.raises(ValueError):
        build_detached_ots(b"\x00" * 31, FAKE_OPS)      # not a sha256
    with pytest.raises(ValueError):
        build_detached_ots(b"\x00" * 32, b"")           # empty response


# ---------------------------------------------------------------------------
# Real HTTP submission path (loopback server — hermetic)
# ---------------------------------------------------------------------------

class _CalendarHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        assert self.path == "/digest"
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n)
        assert len(body) == 32           # the raw digest, nothing else
        self.send_response(200)
        self.end_headers()
        self.wfile.write(FAKE_OPS)

    def log_message(self, *a):           # keep test output clean
        pass


@pytest.fixture()
def local_calendar():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _CalendarHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_submit_to_calendar_over_real_http(local_calendar):
    ops = submit_to_calendar(local_calendar, bytes.fromhex(HEAD),
                             timeout=5.0)
    assert ops == FAKE_OPS


def test_submit_to_calendar_connection_error_raises():
    with pytest.raises(AnchorSubmitError):
        submit_to_calendar("http://127.0.0.1:9", b"\x00" * 32,
                           timeout=0.5)


# ---------------------------------------------------------------------------
# AnchorManager
# ---------------------------------------------------------------------------

def test_anchor_writes_proofs_and_journal(tmp_path):
    mgr = AnchorManager(tmp_path, calendars=["https://cal-a", "https://cal-b"],
                        submit_fn=_fake_submit)
    rec = mgr.anchor(HEAD, 7)
    assert rec["ok"] is True
    assert rec["status"] == "pending"    # honest: not Bitcoin-attested yet
    assert rec["chain_head_sha256"] == HEAD
    assert rec["receipt_count"] == 7
    assert len(rec["calendars"]) == 2
    for entry in rec["calendars"]:
        assert entry["ok"] is True
        proof = (tmp_path / "anchors" / entry["proof_file"]).read_bytes()
        assert proof.startswith(OTS_HEADER_MAGIC)
        assert bytes.fromhex(HEAD) in proof
    journal = (tmp_path / "anchors" / "anchors.jsonl").read_text()
    assert rec["anchor_id"] in journal


def test_anchor_record_is_signed_and_verifiable(tmp_path):
    from governance.signing import GatewaySigner, verify_with_public_pem
    signer = GatewaySigner.create(str(tmp_path))
    mgr = AnchorManager(tmp_path, signer=signer,
                        calendars=["https://cal-a"], submit_fn=_fake_submit)
    rec = mgr.anchor(HEAD, 1)
    assert rec["signature"] and rec["algorithm"]
    body = json.dumps(
        {k: v for k, v in rec.items()
         if k not in ("signature", "algorithm", "public_key_pem")},
        sort_keys=True, default=str)
    if rec["algorithm"] == "ed25519":    # publicly verifiable path
        assert verify_with_public_pem(rec["public_key_pem"], body,
                                      rec["signature"])
        assert not verify_with_public_pem(rec["public_key_pem"],
                                          body + "tampered",
                                          rec["signature"])
    else:                                # hmac fallback still labelled
        assert signer.verify(body, rec["signature"])


def test_anchor_fail_open_partial_and_total(tmp_path):
    def flaky(url, digest):
        if "bad" in url:
            raise AnchorSubmitError(f"{url}: refused")
        return FAKE_OPS

    mgr = AnchorManager(tmp_path, calendars=["https://good", "https://bad"],
                        submit_fn=flaky)
    rec = mgr.anchor(HEAD, 1)
    assert rec["ok"] is True             # one proof is enough
    oks = {e["url"]: e["ok"] for e in rec["calendars"]}
    assert oks == {"https://good": True, "https://bad": False}

    dead = AnchorManager(tmp_path, calendars=["https://bad"],
                         submit_fn=flaky)
    rec2 = dead.anchor(HEAD, 1)          # total failure: recorded, no raise
    assert rec2["ok"] is False
    assert "no calendar accepted" in rec2["error"]


def test_anchor_if_new_skips_genesis_and_unchanged(tmp_path):
    mgr = AnchorManager(tmp_path, calendars=["https://cal"],
                        submit_fn=_fake_submit)
    assert mgr.anchor_if_new("0" * 64, 0) is None       # genesis
    first = mgr.anchor_if_new(HEAD, 3)
    assert first is not None and first["ok"]
    assert mgr.anchor_if_new(HEAD, 3) is None           # unchanged head
    moved = mgr.anchor_if_new("cd" * 32, 4)             # head moved
    assert moved is not None and moved["ok"]


def test_failed_anchor_is_retried_on_next_pass(tmp_path):
    calls = []

    def failing(url, digest):
        calls.append(url)
        raise AnchorSubmitError("down")

    mgr = AnchorManager(tmp_path, calendars=["https://cal"],
                        submit_fn=failing)
    assert mgr.anchor_if_new(HEAD, 1)["ok"] is False
    # No SUCCESS recorded yet, so the same head is retried, not skipped.
    assert mgr.anchor_if_new(HEAD, 1)["ok"] is False
    assert len(calls) == 2


def test_journal_survives_manager_restart(tmp_path):
    mgr = AnchorManager(tmp_path, calendars=["https://cal"],
                        submit_fn=_fake_submit)
    rec = mgr.anchor(HEAD, 2)
    reloaded = AnchorManager(tmp_path, calendars=["https://cal"],
                             submit_fn=_fake_submit)
    assert [r["anchor_id"] for r in reloaded.records()] == [rec["anchor_id"]]
    # Dedup works off the RELOADED journal too.
    assert reloaded.anchor_if_new(HEAD, 2) is None
    # And the proof file is findable through the reloaded journal.
    assert reloaded.find_proof(rec["anchor_id"], 0) is not None


# ---------------------------------------------------------------------------
# Gateway endpoints
# ---------------------------------------------------------------------------

PRICES = {"gpt-test": {"input_per_1m": 1.0, "output_per_1m": 10.0}}


def _upstream(url, body, headers, timeout_s):
    resp = {"choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    return 200, json.dumps(resp).encode("utf-8")


def _app(tmp_path, *, anchoring="on", **kw):
    budget = BudgetLedger(str(tmp_path / "budget"))
    budget.set_envelope("acme", 10.0, "month")
    mgr = None
    if anchoring == "on":
        mgr = AnchorManager(tmp_path / "budget",
                            calendars=["https://cal"],
                            submit_fn=_fake_submit)
    app = build_governance_gateway(
        budget=budget, upstream_base="https://upstream.example",
        price_sheet=PRICES, http_post=_upstream,
        anchoring=mgr, anchor_interval_s=0,   # manual-only in tests
        **kw)
    return app, mgr


def test_endpoints_anchor_now_records_and_proof(tmp_path):
    app, mgr = _app(tmp_path)
    with TestClient(app) as c:
        # Put a real receipt on the chain so the head is non-genesis.
        r = c.post("/v1/chat/completions",
                   json={"model": "gpt-test",
                         "messages": [{"role": "user", "content": "x"}],
                         "max_tokens": 10},
                   headers={"X-Budget-Envelope": "acme/team"})
        assert r.status_code == 200
        head = c.get("/v1/receipts/verify").json()["chain_head_sha256"]

        rec = c.post("/v1/audit/anchor/now").json()
        assert rec["ok"] is True
        assert rec["chain_head_sha256"] == head   # anchored the REAL head

        status = c.get("/v1/audit/anchor").json()["external_anchoring"]
        assert status["enabled"] is True
        assert status["last_success"]["anchor_id"] == rec["anchor_id"]

        listing = c.get("/v1/audit/anchors").json()
        assert listing["enabled"] is True
        assert listing["anchors"][-1]["anchor_id"] == rec["anchor_id"]

        proof = c.get(f"/v1/audit/anchors/{rec['anchor_id']}/proof/0")
        assert proof.status_code == 200
        assert proof.content.startswith(OTS_HEADER_MAGIC)
        assert bytes.fromhex(head) in proof.content

        assert c.get("/v1/audit/anchors/anc-nope/proof/0").status_code == 404


def test_endpoints_disabled_mode_is_honest(tmp_path):
    app, _ = _app(tmp_path, anchoring="off")
    with TestClient(app) as c:
        assert (c.get("/v1/audit/anchor").json()
                ["external_anchoring"]["enabled"] is False)
        assert c.get("/v1/audit/anchors").json() == {
            "enabled": False, "anchors": []}
        r = c.post("/v1/audit/anchor/now")
        assert r.status_code == 400
        assert "not enabled" in r.json()["error"]


def test_anchor_now_is_admin_gated(tmp_path):
    from governance.auth import AuthConfig
    app, _ = _app(tmp_path,
                  auth=AuthConfig(admin_key="s3cret",
                                  read_keys=["reader"]))
    with TestClient(app) as c:
        assert c.post("/v1/audit/anchor/now").status_code == 401
        r = c.post("/v1/audit/anchor/now",
                   headers={"X-Admin-Key": "s3cret"})
        assert r.status_code == 200
        # Reading anchors stays public — integrity data only.
        assert c.get("/v1/audit/anchors").status_code == 200


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def test_scheduler_anchors_moving_head(tmp_path):
    mgr = AnchorManager(tmp_path, calendars=["https://cal"],
                        submit_fn=_fake_submit)
    heads = iter([HEAD, HEAD, "cd" * 32])
    current = {"head": HEAD}

    def head_fn():
        try:
            current["head"] = next(heads)
        except StopIteration:
            pass
        return current["head"], 1

    sched = AnchorScheduler(mgr, head_fn, interval_s=1.0, tick_s=0.05)
    sched.interval_s = 0.05              # fast for the test
    sched.start()
    try:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            heads_seen = {r["chain_head_sha256"] for r in mgr.records()}
            if {HEAD, "cd" * 32} <= heads_seen:
                break
            time.sleep(0.05)
        heads_seen = {r["chain_head_sha256"] for r in mgr.records()}
        assert {HEAD, "cd" * 32} <= heads_seen
        # Dedup held: two records total, not one per tick.
        assert len(mgr.records()) == 2
    finally:
        sched.stop()
        sched.join(timeout=5.0)
        assert not sched.is_alive()
