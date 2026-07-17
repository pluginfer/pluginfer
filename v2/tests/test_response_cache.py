"""Gateway response cache — zero-token repeats, honestly labelled.

Unit-pins the policy (deterministic-only by default, opt-in for all,
TTL expiry, LRU bound) and the end-to-end devserver behaviour: second
identical temperature-0 request is a labelled cache hit with price 0
and the ORIGINAL receipt id.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.response_cache import ResponseCache


def _payload(temp=0, prompt="hi"):
    return {
        "model": "m1",
        "max_tokens": 32,
        "temperature": temp,
        "top_p": None,
        "openai": {"messages": [{"role": "user", "content": prompt}],
                   "model": "m1"},
    }


def test_deterministic_only_by_default(monkeypatch):
    monkeypatch.delenv("PLUGINFER_CACHE_ALL", raising=False)
    monkeypatch.delenv("PLUGINFER_CACHE_DISABLE", raising=False)
    assert ResponseCache.cacheable(_payload(temp=0)) is True
    assert ResponseCache.cacheable(_payload(temp=0.7)) is False
    assert ResponseCache.cacheable(_payload(temp=None)) is False


def test_opt_in_all_and_kill_switch(monkeypatch):
    monkeypatch.setenv("PLUGINFER_CACHE_ALL", "1")
    assert ResponseCache.cacheable(_payload(temp=0.7)) is True
    monkeypatch.setenv("PLUGINFER_CACHE_DISABLE", "1")
    assert ResponseCache.cacheable(_payload(temp=0)) is False


def test_key_distinguishes_content_and_params():
    k1 = ResponseCache.key_for(_payload(prompt="hi"))
    k2 = ResponseCache.key_for(_payload(prompt="bye"))
    k3 = ResponseCache.key_for({**_payload(prompt="hi"), "max_tokens": 64})
    assert len({k1, k2, k3}) == 3
    assert ResponseCache.key_for(_payload(prompt="hi")) == k1


def test_hit_miss_ttl_and_lru():
    c = ResponseCache(max_entries=2, ttl_s=0.2)
    k = ResponseCache.key_for(_payload())
    assert c.get(k) is None
    c.put(k, {"response": {"ok": 1}, "headers": {}})
    entry, age = c.get(k)
    assert entry["response"] == {"ok": 1} and age >= 0
    time.sleep(0.25)
    assert c.get(k) is None, "TTL must expire entries"
    # LRU bound
    for i in range(3):
        c.put(f"k{i}", {"response": {"i": i}, "headers": {}})
    assert c.stats()["entries"] == 2


def test_devserver_second_identical_request_is_cache_hit():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from api.devserver import build_devserver_app
    from api.jobs_service import JobsService
    from core.providers import Auction
    from core.flagship import register_alpha_flagship, spec_for_runtime

    svc = JobsService(auction=Auction())
    register_alpha_flagship(
        jobs_service=svc,
        spec=spec_for_runtime("echo", "echo"),
        runner_fn=lambda prompt, payload: f"echo:{prompt}".encode(),
    )
    app = build_devserver_app(jobs_service=svc, title="cache-test")
    client = TestClient(app)

    body = {"model": "pluginfer/alpha-echo",
            "messages": [{"role": "user", "content": "cache me"}],
            "max_tokens": 16, "temperature": 0}
    r1 = client.post("/v1/chat/completions", json=body)
    assert r1.status_code == 200
    assert r1.headers["x-pluginfer-cache"] == "miss"
    receipt_1 = r1.headers.get("x-pluginfer-receipt-id", "")

    r2 = client.post("/v1/chat/completions", json=body)
    assert r2.status_code == 200
    assert r2.headers["x-pluginfer-cache"] == "hit"
    assert r2.headers["x-pluginfer-price-usd"] == "0"
    assert r2.headers.get("x-pluginfer-receipt-id", "") == receipt_1, \
        "hit must carry the ORIGINAL receipt id (audit trail intact)"
    assert r2.json()["choices"] == r1.json()["choices"]

    # sampled request (temperature omitted -> None) must BYPASS
    body_sampled = {**body}
    del body_sampled["temperature"]
    r3 = client.post("/v1/chat/completions", json=body_sampled)
    assert r3.headers["x-pluginfer-cache"] == "bypass"
