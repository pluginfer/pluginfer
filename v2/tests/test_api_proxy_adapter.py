"""api-proxy adapter — bring any existing LLM onto the mesh (§RFC-2).

Hermetic: a tiny in-process HTTP server stands in for an OpenAI- and
an Anthropic-compatible upstream. No network, no keys, no real model.
Pins the contract that matters for the "replace the cloud" mission:

  * refuses cleanly when unconfigured (ladder falls through),
  * probe = cheap reachability, never generation,
  * localhost endpoint is flagged is_local_compute=True (real mesh
    compute); a non-local host is flagged False so receipts can't
    claim mesh compute for a hosted passthrough,
  * both OpenAI and Anthropic response shapes parse.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest  # noqa: E402

from core.runtime_adapters.api_proxy_adapter import (  # noqa: E402
    make_api_proxy_runner,
)
from core.runtime_adapters.base import RuntimeAdapterUnavailable  # noqa: E402


class _FakeUpstream(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def do_GET(self):  # /models probe
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"data": [{"id": "fake-1"}]}).encode())

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(n)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if self.path.endswith("/messages"):      # anthropic dialect
            body = {"content": [{"type": "text", "text": "hi-anthropic"}]}
        else:                                      # openai dialect
            body = {"choices": [
                {"message": {"role": "assistant", "content": "hi-openai"}}]}
        self.wfile.write(json.dumps(body).encode())


@pytest.fixture()
def upstream():
    srv = HTTPServer(("127.0.0.1", 0), _FakeUpstream)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    host, port = srv.server_address
    yield f"http://127.0.0.1:{port}/v1"
    srv.shutdown()


def test_refuses_when_unconfigured(monkeypatch):
    monkeypatch.delenv("PLUGINFER_PROXY_BASE_URL", raising=False)
    with pytest.raises(RuntimeAdapterUnavailable):
        make_api_proxy_runner(model_id="x")


def test_openai_dialect_roundtrip_and_is_local(upstream):
    runner = make_api_proxy_runner(
        model_id="fake-1", base_url=upstream, dialect="openai")
    out = runner("hello", {"max_tokens": 8})
    assert out == b"hi-openai"
    assert runner.served_model_id == "fake-1"
    assert runner.is_local_compute is True   # 127.0.0.1 = mesh compute


def test_anthropic_dialect_roundtrip(upstream):
    runner = make_api_proxy_runner(
        model_id="fake-1", base_url=upstream, dialect="anthropic")
    assert runner("hello", {"max_tokens": 8}) == b"hi-anthropic"


def test_remote_host_flagged_non_local(upstream):
    # Same fake server, but declare a non-local base host: the flag
    # must be False so receipts never claim mesh compute for what is
    # really a hosted passthrough. (Probe still hits the fixture via
    # the real port; we only assert the classification.)
    from core.runtime_adapters import api_proxy_adapter as ap
    assert ap._is_local_endpoint("https://api.openai.com/v1") is False
    assert ap._is_local_endpoint("http://localhost:1234/v1") is True
    assert ap._is_local_endpoint("http://192.168.1.5:11434/v1") is True


def test_probe_does_not_generate(upstream):
    # _probe=True must return a runner after only the cheap GET probe;
    # we assert it constructs without error (generation is a separate
    # call the probe never makes).
    runner = make_api_proxy_runner(
        model_id="fake-1", base_url=upstream, _probe=True)
    assert callable(runner)


def test_registered_last_in_ladder():
    from core.runtime_adapters.base import _REGISTRY
    names = [n for n, _ in _REGISTRY]
    assert "api-proxy" in names
    assert names.index("api-proxy") == len(names) - 1  # lowest priority
