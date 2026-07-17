"""W46 — end-to-end round-trip for the gateway-side browser-provider
endpoints (`/v1/providers/{register,heartbeat,bid,open_jobs,deliver}`).

What's pinned here:

  1. A browser tab POSTs ``/v1/providers/register`` and shows up in the
     auction's provider list.
  2. A buyer submits a job (POST /v1/jobs). The gateway runs the
     auction; the browser tab is the only provider, so its
     ``HttpBrowserProvider.bid()`` wins.
  3. The auction executor (running in the executor thread) calls
     ``HttpBrowserProvider.execute()`` which queues the job_id on the
     provider's open_pickups list and blocks on a threading.Event
     waiting for a delivery.
  4. The browser tab polls ``/v1/providers/open_jobs`` and sees the
     pending job.
  5. The tab POSTs ``/v1/providers/deliver`` with a signed result.
  6. The event sets; the executor thread unblocks; JobsService writes
     the result onto the JobRecord. The buyer's polled state
     transitions to "completed" with the delivered payload.

That round trip is the W46 acceptance criterion — without it the in-tab
provider in ``v2/ui/browser_provider/`` would dead-end. Once green, every
Chromium tab on the planet is a candidate supply-side node.

The test drives the full FastAPI app via httpx.ASGITransport (no socket,
no flake) and uses concurrent coroutines so the buyer's polling loop and
the browser tab's poll/deliver loop run together.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import httpx  # noqa: E402

from api.main import build_app  # noqa: E402
from core.providers import Auction  # noqa: E402


# A throwaway P-256 pubkey PEM. The PEM body is opaque to the gateway —
# only the routing handle is derived from it. The browser tab's real
# pubkey is generated client-side by WebCrypto and persisted in
# IndexedDB.
FAKE_PUBKEY_PEM = (
    "-----BEGIN PUBLIC KEY-----\n"
    "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE" + ("A" * 80) + "\n"
    "-----END PUBLIC KEY-----\n"
)


def _make_app():
    auction = Auction()
    app = build_app(
        auction=auction,
        rate_limit_capacity=5_000.0,
        rate_limit_refill_per_sec=10_000.0,
    )
    api_key = app.state.auth_backend.issue_api_key("buyer-1")
    return app, api_key


def _client(app, api_key=None):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(
        transport=transport, base_url="http://testserver", headers=headers,
    )


_DEFAULT_TEMPLATE = {
    "provider_pubkey_pem": FAKE_PUBKEY_PEM,
    "hardware_class": "browser-webgpu",
    "price_per_1k_tok_usd": 0.0001,
    "base_eta_ms": 100,
    "base_quality": 0.9,
    "privacy_grade": "public",
}


# ---------------------------------------------------------------------------
# 1. registration semantics
# ---------------------------------------------------------------------------

def test_register_adds_provider_to_auction():
    app, _ = _make_app()
    assert len(app.state.jobs.auction.providers) == 0

    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/providers/register", json=_DEFAULT_TEMPLATE)
            assert r.status_code == 201, r.text
            body = r.json()
            assert body["provider_id"].startswith("br_")
            assert body["auction_size"] == 1
            assert len(app.state.jobs.auction.providers) == 1
    asyncio.run(_run())


def test_register_is_idempotent_on_repeat():
    """A tab refresh re-posts /register; the registry should refresh the
    template without duplicating the auction entry."""
    app, _ = _make_app()

    async def _run():
        async with _client(app) as c:
            r1 = await c.post("/v1/providers/register", json=_DEFAULT_TEMPLATE)
            r2 = await c.post("/v1/providers/register", json=_DEFAULT_TEMPLATE)
            assert r1.status_code == 201
            assert r2.status_code == 201
            assert r1.json()["provider_id"] == r2.json()["provider_id"]
            assert len(app.state.jobs.auction.providers) == 1
    asyncio.run(_run())


def test_bid_is_register_alias():
    """`/providers/bid` is a documented alias the browser SDK uses. The
    gateway treats it as a register/refresh — same semantics, 200 not
    201."""
    app, _ = _make_app()

    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/providers/bid", json=_DEFAULT_TEMPLATE)
            assert r.status_code == 200, r.text
            assert r.json()["auction_size"] == 1
    asyncio.run(_run())


def test_heartbeat_404_for_unregistered():
    app, _ = _make_app()

    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/providers/heartbeat", json={
                "provider_pubkey_pem": FAKE_PUBKEY_PEM,
            })
            assert r.status_code == 404
            assert r.json()["detail"] == "provider_not_registered"
    asyncio.run(_run())


def test_open_jobs_auto_registers_first_time_tab():
    """A tab can poll /open_jobs before /register and get auto-registered
    with conservative defaults — this is the zero-config on-ramp."""
    app, _ = _make_app()

    async def _run():
        async with _client(app) as c:
            r = await c.get(
                "/v1/providers/open_jobs",
                params={"provider_pubkey": FAKE_PUBKEY_PEM},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["provider_id"].startswith("br_")
            assert body["jobs"] == []
            assert len(app.state.jobs.auction.providers) == 1
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2. the round trip
# ---------------------------------------------------------------------------

def _state_of(info):
    s = info.get("state")
    return s["state"] if isinstance(s, dict) else s


def test_browser_provider_end_to_end_round_trip():
    """register -> buyer submits -> tab polls -> tab delivers ->
    buyer's job completes with the delivered payload."""
    app, buyer_key = _make_app()
    result_payload = b"the lazy dog received it"
    result_hash = hashlib.sha256(result_payload).hexdigest()
    result_b64 = base64.b64encode(result_payload).decode()

    async def _run():
        async with _client(app, buyer_key) as buyer, _client(app) as tab:
            r = await tab.post("/v1/providers/register", json=_DEFAULT_TEMPLATE)
            assert r.status_code == 201, r.text
            provider_id = r.json()["provider_id"]

            r = await buyer.post("/v1/jobs", json={
                "kind": "compute.test",
                "payload": {"x": 1},
                "cost_ceiling_usd": 0.05,
                "latency_ceiling_ms": 10_000,
                "privacy_class": "public",
                "quality_floor": 0.5,
            })
            assert r.status_code == 202, r.text
            job_id = r.json()["job_id"]

            async def tab_poll_until_seen():
                # JobsService dispatches execute() onto a thread inside
                # asyncio.create_task; there's a small window between
                # submit() returning and HttpBrowserProvider.execute()
                # appending to open_pickups.
                for _ in range(100):
                    r = await tab.get(
                        "/v1/providers/open_jobs",
                        params={"provider_pubkey": FAKE_PUBKEY_PEM, "limit": 4},
                    )
                    assert r.status_code == 200, r.text
                    jobs = r.json()["jobs"]
                    if jobs:
                        return jobs[0]
                    await asyncio.sleep(0.05)
                raise AssertionError("tab never saw the job in open_jobs")

            async def buyer_wait_terminal():
                for _ in range(400):
                    r = await buyer.get(f"/v1/jobs/{job_id}")
                    if r.status_code == 200:
                        info = r.json()
                        if _state_of(info) in (
                            "completed", "failed", "timeout", "cancelled",
                        ):
                            return info
                    await asyncio.sleep(0.05)
                raise AssertionError("buyer never saw terminal state")

            buyer_task = asyncio.create_task(buyer_wait_terminal())

            picked = await tab_poll_until_seen()
            assert picked["job_id"] == job_id
            d = await tab.post("/v1/providers/deliver", json={
                "job_id": picked["job_id"],
                "provider_pubkey_pem": FAKE_PUBKEY_PEM,
                "result_bytes": result_b64,
                "result_hash": result_hash,
                "provider_sig": "AAAA",
                "execution_ms": 17,
            })
            assert d.status_code == 200, d.text
            assert d.json()["accepted"] is True

            terminal_info = await buyer_task
            assert _state_of(terminal_info) == "completed", terminal_info
            assert terminal_info["matched_provider_pubkey"] == provider_id

            # W47: jobs_service normalisation lifts result_bytes onto
            # rec.result_b64 so the SDK + devserver see a canonical key.
            r = await buyer.get(f"/v1/jobs/{job_id}/result")
            assert r.status_code == 200
            res = r.json()
            assert res["result_b64"] == result_b64
            assert res["result_hash_hex"] == result_hash

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. delivery edge cases
# ---------------------------------------------------------------------------

def test_deliver_410_when_job_unknown():
    """A racing tab can deliver after the job has been popped (timeout or
    duplicate). The gateway returns 410 so the tab drops the result."""
    app, _ = _make_app()

    async def _run():
        async with _client(app) as c:
            await c.post("/v1/providers/register", json=_DEFAULT_TEMPLATE)
            r = await c.post("/v1/providers/deliver", json={
                "job_id": "no-such-job",
                "provider_pubkey_pem": FAKE_PUBKEY_PEM,
                "result_bytes": "AA==",
                "result_hash": "0" * 64,
                "provider_sig": "AAAA",
            })
            assert r.status_code == 410
            assert r.json()["detail"] == "job_no_longer_pending"
    asyncio.run(_run())


def test_deliver_404_when_provider_unknown():
    app, _ = _make_app()

    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/providers/deliver", json={
                "job_id": "x",
                "provider_pubkey_pem": FAKE_PUBKEY_PEM,
                "result_bytes": "AA==",
                "result_hash": "0" * 64,
                "provider_sig": "AAAA",
            })
            assert r.status_code == 404
    asyncio.run(_run())
