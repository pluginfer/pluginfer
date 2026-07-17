"""MeshLLMProvider — bridge any OpenAI-compatible endpoint into the auction.

Hermetic: spins a fake mesh-llm node (stdlib http.server) in-process.
Pins the contract that matters for going public:
  * a live endpoint is discovered, bids, executes, and returns the
    signed-receipt result shape every other provider returns;
  * a dead endpoint ABSTAINS (never wins an auction it cannot serve);
  * OpenAI-style explicit nulls ("temperature": null) survive;
  * public mesh never serves privacy-routed jobs (privacy grade).
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import sys
from pathlib import Path
V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.meshllm_provider import MeshLLMProvider, autodetect_meshllm
from core.providers import JobSpec


class _FakeMeshLLM(BaseHTTPRequestHandler):
    served_models = ["GLM-4.7-Flash-Q4_K_M", "Qwen3-8B-Q4_K_M"]
    last_body: dict = {}

    def do_GET(self):
        if self.path.endswith("/models"):
            body = json.dumps({
                "data": [{"id": m} for m in self.served_models]
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        _FakeMeshLLM.last_body = json.loads(self.rfile.read(length))
        body = json.dumps({
            "choices": [{"message": {
                "role": "assistant",
                "content": f"mesh answer via {_FakeMeshLLM.last_body.get('model')}",
            }}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence
        pass


@pytest.fixture()
def fake_mesh():
    srv = HTTPServer(("127.0.0.1", 0), _FakeMeshLLM)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def _job(model="GLM-4.7-Flash-Q4_K_M", **over):
    payload = {"model": model,
               "messages": [{"role": "user", "content": "hello"}],
               "max_tokens": 64, "temperature": None}
    payload.update(over.pop("payload", {}))
    kw = dict(job_id="j1", kind="inference", payload=payload,
              cost_ceiling_usd=0.05, latency_ceiling_ms=30_000)
    kw.update(over)
    return JobSpec(**kw)


def test_probe_discovers_models(fake_mesh):
    p = MeshLLMProvider(base_url=fake_mesh)
    assert p.probe() is True
    assert "Qwen3-8B-Q4_K_M" in p.models


def test_bid_and_execute_roundtrip_with_receipt_shape(fake_mesh):
    p = MeshLLMProvider(base_url=fake_mesh)
    job = _job()
    bid = p.bid(job)
    assert bid is not None and bid.price_usd > 0
    out = p.execute(job, bid)
    assert out["status"] == "executed"
    text = base64.b64decode(out["result_bytes"]).decode()
    assert "mesh answer via GLM-4.7-Flash-Q4_K_M" in text
    assert out["result_hash"] == hashlib.sha256(text.encode()).hexdigest()
    assert out["model_id"] == "meshllm:GLM-4.7-Flash-Q4_K_M"
    # explicit-null temperature must NOT be forwarded as null
    assert _FakeMeshLLM.last_body.get("temperature") is None or \
        "temperature" not in _FakeMeshLLM.last_body


def test_wallet_signature_attached(fake_mesh):
    from core.tokenomics import Wallet
    w = Wallet()
    p = MeshLLMProvider(base_url=fake_mesh, wallet=w)
    out = p.execute(_job(), p.bid(_job()))
    assert out["provider_sig"], "result must be wallet-signed"
    assert out["provider_pubkey_pem"] == w.public_key_pem


def test_dead_endpoint_abstains():
    p = MeshLLMProvider(base_url="http://127.0.0.1:9")   # nothing there
    assert p.bid(_job()) is None


def test_unserved_model_abstains(fake_mesh):
    p = MeshLLMProvider(base_url=fake_mesh)
    assert p.bid(_job(model="gpt-4o")) is None
    # 'mesh' MoA pseudo-model always routable when mesh is up
    assert p.bid(_job(model="mesh")) is not None


def test_over_ceiling_abstains(fake_mesh):
    p = MeshLLMProvider(base_url=fake_mesh, price_per_1k_tok_usd=100.0)
    assert p.bid(_job()) is None


def test_public_privacy_grade():
    from core.providers import PRIVACY_PUBLIC
    p = MeshLLMProvider()
    assert p.privacy_grade == PRIVACY_PUBLIC


def test_autodetect_returns_none_when_absent(monkeypatch):
    monkeypatch.setenv("PLUGINFER_MESHLLM_URL", "http://127.0.0.1:9/v1")
    assert autodetect_meshllm() is None


def test_autodetect_env_optout(fake_mesh, monkeypatch):
    monkeypatch.setenv("PLUGINFER_MESHLLM_URL", fake_mesh)
    monkeypatch.setenv("PLUGINFER_DISABLE_MESHLLM", "1")
    assert autodetect_meshllm() is None
    monkeypatch.delenv("PLUGINFER_DISABLE_MESHLLM")
    assert autodetect_meshllm() is not None


def test_upstream_error_is_refund_eligible(fake_mesh):
    p = MeshLLMProvider(base_url=fake_mesh)
    bid = p.bid(_job())
    p.base_url = "http://127.0.0.1:9/v1"   # endpoint dies after bidding
    out = p.execute(_job(), bid)
    assert out["status"] == "error"
    assert out["refund_eligible"] is True
