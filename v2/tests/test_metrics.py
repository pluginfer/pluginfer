"""Tests for `core.metrics` (Counter, Gauge, Histogram, Registry) +
the FastAPI `/metrics` endpoint."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import httpx
import pytest

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from api.main import build_app  # noqa: E402
from core.metrics import (  # noqa: E402
    Counter,
    Gauge,
    Histogram,
    Registry,
    jobs_total,
    REGISTRY,
)
from core.providers import Auction, Bid, JobSpec, PRIVACY_PUBLIC, Provider  # noqa: E402


# ---------------------------------------------------------------------------
# unit
# ---------------------------------------------------------------------------


def test_counter_render_matches_prometheus_format():
    c = Counter("pluginfer_test_counter", "test counter")
    c.inc()
    c.inc(2.5, labels={"k": "v"})
    c.inc(1.0, labels={"k": "v"})
    out = c.render()
    assert "# HELP pluginfer_test_counter test counter" in out
    assert "# TYPE pluginfer_test_counter counter" in out
    assert "pluginfer_test_counter 1.0" in out
    assert 'pluginfer_test_counter{k="v"} 3.5' in out


def test_counter_rejects_negative():
    c = Counter("c", "h")
    with pytest.raises(ValueError):
        c.inc(-1)


def test_gauge_set_and_dec():
    g = Gauge("pluginfer_test_gauge", "test gauge")
    g.set(10.0)
    g.dec(3.0)
    g.inc(0.5)
    out = g.render()
    assert "pluginfer_test_gauge 7.5" in out


def test_histogram_buckets_and_quantile_form():
    h = Histogram("pluginfer_test_histogram", "test hist", buckets=[0.1, 1.0, 10.0])
    for v in (0.05, 0.5, 5.0, 50.0):
        h.observe(v)
    out = h.render()
    assert "# TYPE pluginfer_test_histogram histogram" in out
    # Cumulative: <=0.1 -> 1, <=1.0 -> 2, <=10.0 -> 3, <=+Inf -> 4
    assert 'pluginfer_test_histogram_bucket{le="0.1"} 1' in out
    assert 'pluginfer_test_histogram_bucket{le="1.0"} 2' in out
    assert 'pluginfer_test_histogram_bucket{le="10.0"} 3' in out
    assert 'pluginfer_test_histogram_bucket{le="+Inf"} 4' in out
    assert "pluginfer_test_histogram_count 4" in out


def test_histogram_time_context_manager():
    h = Histogram("pluginfer_test_timer", "test", buckets=[0.001, 0.01, 0.1])
    with h.time():
        time.sleep(0.005)
    assert "_count 1" in h.render()


def test_registry_concatenates_all_metrics():
    r = Registry()
    r.register(Counter("a", "h"))
    r.register(Gauge("b", "h"))
    out = r.render()
    assert "# HELP a" in out and "# HELP b" in out


# ---------------------------------------------------------------------------
# /metrics endpoint integration
# ---------------------------------------------------------------------------


class _FakeProvider(Provider):
    provider_id = "fake-metrics"
    privacy_grade = PRIVACY_PUBLIC
    kind = "compute"

    def bid(self, job: JobSpec) -> Bid:
        return Bid(provider_id=self.provider_id, price_usd=0.001,
                   eta_ms=10, expected_quality=0.99,
                   privacy_grade=PRIVACY_PUBLIC)

    def execute(self, job, bid):
        return {"status": "executed",
                "result_bytes_b64": "aGV5", "result_hash": "00" * 32,
                "provider_sig": "c2ln"}


def test_metrics_endpoint_serves_prometheus_text():
    auction = Auction()
    auction.register(_FakeProvider())
    app = build_app(auction=auction)

    async def _run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as c:
            r = await c.get("/metrics")
            assert r.status_code == 200
            ctype = r.headers.get("content-type", "")
            assert ctype.startswith("text/plain")
            body = r.text
            for line in (
                "# HELP pluginfer_jobs_total",
                "# TYPE pluginfer_jobs_total counter",
                "# HELP pluginfer_peers_connected",
                "# HELP pluginfer_chain_height",
                "# HELP pluginfer_uptime_seconds",
            ):
                assert line in body, f"missing: {line!r}\nbody:\n{body[:400]}"
    asyncio.run(_run())


def test_jobs_total_counter_increments_after_completion():
    """End-to-end: submit a job, wait for completion, observe that
    pluginfer_jobs_total{status=completed} ticked up."""
    auction = Auction()
    auction.register(_FakeProvider())
    app = build_app(auction=auction)
    api_key = app.state.auth_backend.issue_api_key("metrics-user")

    # Snapshot the current value.
    before = jobs_total._values.get((("status", "completed"),), 0.0)

    async def _run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            headers={"Authorization": f"Bearer {api_key}"},
        ) as c:
            r = await c.post("/v1/jobs", json={"kind": "x"})
            jid = r.json()["job_id"]
            for _ in range(50):
                rr = await c.get(f"/v1/jobs/{jid}")
                if rr.json()["state"]["state"] in (
                    "completed", "failed", "timeout", "cancelled",
                ):
                    break
                await asyncio.sleep(0.05)
    asyncio.run(_run())
    after = jobs_total._values.get((("status", "completed"),), 0.0)
    assert after >= before + 1.0, f"counter did not advance: {before} -> {after}"
