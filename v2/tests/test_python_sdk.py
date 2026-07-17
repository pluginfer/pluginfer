"""Python SDK tests, driven through the live FastAPI app via httpx
ASGITransport. Exercises the wire contract end-to-end without a socket.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
import pytest

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))
SDK_PATH = V2 / "sdk" / "python"
if str(SDK_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PATH))

from api.main import build_app  # noqa: E402
from core.providers import Auction, Bid, JobSpec, PRIVACY_PUBLIC, Provider  # noqa: E402
from pluginfer import (  # noqa: E402
    AuthenticationError,
    JobNotFoundError,
    Pluginfer,
    PluginferError,
    RateLimitError,
)


class _FakeProvider(Provider):
    provider_id = "fake-sdk-provider"
    privacy_grade = PRIVACY_PUBLIC
    kind = "compute"

    def bid(self, job: JobSpec) -> Bid:
        return Bid(
            provider_id=self.provider_id, price_usd=0.001, eta_ms=10,
            expected_quality=0.95, privacy_grade=PRIVACY_PUBLIC,
        )

    def execute(self, job: JobSpec, bid: Bid) -> dict:
        import base64 as _b64, hashlib as _h
        out = b"sdk-fake-result"
        return {
            "status": "executed",
            "result_bytes_b64": _b64.b64encode(out).decode(),
            "result_hash": _h.sha256(out).hexdigest(),
            "provider_sig": _b64.b64encode(b"sdk-sig").decode(),
        }


def _make_sync_client(app, api_key=None):
    """Build a sync httpx.Client wrapping a FastAPI ASGI app.

    httpx.ASGITransport is async-only, so the SDK's sync httpx.Client
    cannot use it directly. Starlette ships an ASGI->sync bridge as
    `starlette.testclient.TestClient` — it speaks httpx.Client API and
    drives the ASGI app under an internal asyncio loop. We pass that
    instance straight through to the SDK via `http_client=`.
    """
    from starlette.testclient import TestClient
    tc = TestClient(app, base_url="http://testserver")
    if api_key:
        tc.headers["Authorization"] = f"Bearer {api_key}"
    return tc


@pytest.fixture
def client():
    auction = Auction()
    auction.register(_FakeProvider())
    app = build_app(auction=auction, rate_limit_capacity=1000.0,
                    rate_limit_refill_per_sec=1000.0)
    api_key = app.state.auth_backend.issue_api_key("sdk-test-user")
    tc = _make_sync_client(app, api_key)
    p = Pluginfer(http_client=tc)
    yield p
    p.close()
    tc.close()


def test_sdk_status(client):
    s = client.status()
    assert s.status == "ok"
    assert s.version


def test_sdk_version(client):
    v = client.version()
    assert v["version"] and v["api"] == "v1"


def test_sdk_jobs_submit_and_wait_for_completion(client):
    j = client.jobs.submit(
        kind="compute.echo",
        payload={"x": 1},
        cost_ceiling_usd=0.01,
        latency_ceiling_ms=5_000,
    )
    assert j.job_id
    final = client.jobs.wait_for(
        j.job_id, timeout_sec=5.0, poll_interval_sec=0.05,
    )
    assert final.state.state == "completed", final
    res = client.jobs.result(j.job_id)
    assert res.state.state == "completed"
    decoded = client.jobs.decode_result(res)
    assert decoded == b"sdk-fake-result"


def test_sdk_jobs_get_unknown_raises_404(client):
    with pytest.raises(JobNotFoundError):
        client.jobs.get("not-a-real-id")


def test_sdk_unauthorised_raises_401():
    auction = Auction()
    auction.register(_FakeProvider())
    app = build_app(auction=auction, rate_limit_capacity=100,
                    rate_limit_refill_per_sec=100)
    tc = _make_sync_client(app)  # no api key
    p = Pluginfer(http_client=tc)
    try:
        with pytest.raises(AuthenticationError):
            p.jobs.submit(kind="x")
    finally:
        p.close()
        tc.close()


def test_sdk_rate_limit_raises_typed_error():
    auction = Auction()
    auction.register(_FakeProvider())
    app = build_app(
        auction=auction,
        rate_limit_capacity=1.0,
        rate_limit_refill_per_sec=0.0001,
    )
    tc = _make_sync_client(app)
    p = Pluginfer(http_client=tc)
    try:
        # First request consumes the only token.
        p.version()
        with pytest.raises(RateLimitError) as ei:
            p.version()
        assert ei.value.retry_after_sec >= 1.0
    finally:
        p.close()
        tc.close()


def test_sdk_providers_list(client):
    ps = client.providers.list()
    assert any(p.pubkey == "fake-sdk-provider" for p in ps)


def test_sdk_request_id_propagated_through_error(client):
    # Issue a 404 and confirm we got the X-Request-ID from the server
    # so the user can grep server logs with it.
    try:
        client.jobs.get("nope")
    except JobNotFoundError as e:
        assert e.request_id and len(e.request_id) >= 8
        return
    assert False, "expected JobNotFoundError"
