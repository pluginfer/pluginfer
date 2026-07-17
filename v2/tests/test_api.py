"""End-to-end tests for the Pluginfer REST API.

We drive the FastAPI app via httpx.ASGITransport so the entire stack
(routers, middleware, auth, rate limit) runs in-process — no socket, no
flake. The auction is wired with a tiny FakeProvider that bids cheap
and returns a stable result so the JobsService transitions all the way
from queued -> matched -> running -> completed.
"""

from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import httpx  # noqa: E402

from api.main import build_app  # noqa: E402
from core.providers import (  # noqa: E402
    Auction,
    Bid,
    JobSpec,
    Provider,
    PRIVACY_PUBLIC,
)


class _FakeProvider(Provider):
    provider_id = "fake-provider-001"
    privacy_grade = PRIVACY_PUBLIC
    kind = "compute"

    def __init__(self) -> None:
        self.executed = []

    def bid(self, job: JobSpec) -> Bid:
        return Bid(
            provider_id=self.provider_id,
            price_usd=0.001,
            eta_ms=50,
            expected_quality=0.9,
            privacy_grade=PRIVACY_PUBLIC,
            evidence={"src": "fake"},
        )

    def execute(self, job: JobSpec, bid: Bid) -> dict:
        self.executed.append(job.job_id)
        out = b"hello-from-fake"
        import hashlib
        return {
            "status": "executed",
            "result_bytes_b64": base64.b64encode(out).decode(),
            "result_hash": hashlib.sha256(out).hexdigest(),
            "provider_sig": base64.b64encode(b"fake-sig").decode(),
        }


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_and_key():
    auction = Auction()
    auction.register(_FakeProvider())
    app = build_app(
        auction=auction,
        rate_limit_capacity=5_000.0,   # generous in tests; one test overrides
        rate_limit_refill_per_sec=10_000.0,
    )
    api_key = app.state.auth_backend.issue_api_key("test-user-1")
    return app, api_key


def _client(app, api_key=None):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(
        transport=transport, base_url="http://testserver", headers=headers,
    )


# ---------------------------------------------------------------------------
# status + version
# ---------------------------------------------------------------------------


def test_version_endpoint_returns_semver(app_and_key):
    app, _ = app_and_key

    async def _run():
        async with _client(app) as c:
            r = await c.get("/v1/version")
            assert r.status_code == 200
            d = r.json()
            assert "version" in d and "git_sha" in d and d["api"] == "v1"
            assert "X-Request-ID" in r.headers
    asyncio.run(_run())


def test_status_endpoint_returns_ok(app_and_key):
    app, _ = app_and_key

    async def _run():
        async with _client(app) as c:
            r = await c.get("/v1/status")
            assert r.status_code == 200
            d = r.json()
            assert d["status"] == "ok"
            assert d["chain_height"] >= 0
            assert d["uptime_seconds"] >= 0
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# auth: jobs require auth, 401 without
# ---------------------------------------------------------------------------


def test_jobs_post_unauth_rejected(app_and_key):
    app, _ = app_and_key

    async def _run():
        async with _client(app) as c:   # no api key
            r = await c.post("/v1/jobs", json={"kind": "x"})
            assert r.status_code == 401
            assert r.headers.get("WWW-Authenticate") == "Bearer"
    asyncio.run(_run())


def test_jobs_post_auth_accepted(app_and_key):
    app, key = app_and_key

    async def _run():
        async with _client(app, key) as c:
            r = await c.post("/v1/jobs", json={
                "kind": "compute.test",
                "payload": {"x": 1},
                "cost_ceiling_usd": 0.01,
                "latency_ceiling_ms": 5_000,
            })
            assert r.status_code == 202, r.text
            d = r.json()
            assert d["job_id"]
            assert d["state"]["state"] in ("queued", "matched", "running", "completed")
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# auth: API key issuance + revocation round-trip
# ---------------------------------------------------------------------------


def test_revoked_api_key_rejected(app_and_key):
    app, key = app_and_key
    # Revoke and confirm requests now 401.
    app.state.auth_backend.revoke_api_key(key)

    async def _run():
        async with _client(app, key) as c:
            r = await c.post("/v1/jobs", json={"kind": "x"})
            assert r.status_code == 401
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# job lifecycle: submit -> get -> result
# ---------------------------------------------------------------------------


def test_job_lifecycle_submit_to_completed(app_and_key):
    app, key = app_and_key

    async def _run():
        async with _client(app, key) as c:
            r = await c.post("/v1/jobs", json={
                "kind": "compute.echo",
                "payload": {"x": 42},
                "cost_ceiling_usd": 0.01,
            })
            assert r.status_code == 202
            jid = r.json()["job_id"]

            # Wait up to 5s for state to become terminal.
            for _ in range(50):
                r2 = await c.get(f"/v1/jobs/{jid}")
                state = r2.json()["state"]["state"]
                if state in ("completed", "failed", "timeout", "cancelled"):
                    break
                await asyncio.sleep(0.1)
            assert state == "completed", r2.json()

            r3 = await c.get(f"/v1/jobs/{jid}/result")
            d = r3.json()
            assert d["state"]["state"] == "completed"
            assert d["result_b64"]
            assert d["result_hash_hex"]
            assert d["provider_signature_b64"]
    asyncio.run(_run())


def test_get_job_unknown_returns_404(app_and_key):
    app, key = app_and_key

    async def _run():
        async with _client(app, key) as c:
            r = await c.get("/v1/jobs/does-not-exist")
            assert r.status_code == 404
    asyncio.run(_run())


def test_get_other_users_job_returns_403(app_and_key):
    app, key = app_and_key
    # Issue a second key for a different identity, submit job under it,
    # then try to read it with the first key.
    other_key = app.state.auth_backend.issue_api_key("test-user-2")

    async def _run():
        async with _client(app, other_key) as c:
            r = await c.post("/v1/jobs", json={"kind": "compute.test"})
            jid = r.json()["job_id"]
        async with _client(app, key) as c2:
            r = await c2.get(f"/v1/jobs/{jid}")
            assert r.status_code == 403
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# auth: wallet challenge / verify with real ECDSA
# ---------------------------------------------------------------------------


def test_wallet_challenge_response_round_trip(app_and_key):
    """Issue a challenge, sign it with a fresh secp256k1 key, verify."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    app, _ = app_and_key
    priv = ec.generate_private_key(ec.SECP256K1())
    pub = priv.public_key()
    pub_pem = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()

    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/auth/challenge")
            assert r.status_code == 200
            ch = r.json()
            body = f"{ch['nonce']}|{ch['audience']}|{ch['expires_at_unix']}".encode()
            sig = priv.sign(body, ec.ECDSA(hashes.SHA256()))
            sig_b64 = base64.b64encode(sig).decode()
            r2 = await c.post("/v1/auth/verify", json={
                "nonce": ch["nonce"],
                "pubkey_pem": pub_pem,
                "signature_b64": sig_b64,
            })
            assert r2.status_code == 200, r2.text
            assert r2.json()["session_id"]

            # Re-using the same nonce must fail (one-shot).
            r3 = await c.post("/v1/auth/verify", json={
                "nonce": ch["nonce"],
                "pubkey_pem": pub_pem,
                "signature_b64": sig_b64,
            })
            assert r3.status_code == 401
    asyncio.run(_run())


def test_wallet_verify_rejects_bad_signature(app_and_key):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    app, _ = app_and_key
    priv = ec.generate_private_key(ec.SECP256K1())
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()

    async def _run():
        async with _client(app) as c:
            ch = (await c.post("/v1/auth/challenge")).json()
            r = await c.post("/v1/auth/verify", json={
                "nonce": ch["nonce"],
                "pubkey_pem": pub_pem,
                "signature_b64": base64.b64encode(b"x" * 64).decode(),
            })
            assert r.status_code == 401
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# providers + wallet
# ---------------------------------------------------------------------------


def test_providers_list_returns_registered(app_and_key):
    app, key = app_and_key

    async def _run():
        async with _client(app, key) as c:
            r = await c.get("/v1/providers")
            assert r.status_code == 200
            d = r.json()
            assert any(p["pubkey"] == "fake-provider-001" for p in d)
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# request-id propagation
# ---------------------------------------------------------------------------


def test_request_id_echo(app_and_key):
    app, _ = app_and_key

    async def _run():
        async with _client(app) as c:
            r = await c.get("/v1/version", headers={"X-Request-ID": "rid-test-123"})
            assert r.headers.get("X-Request-ID") == "rid-test-123"
    asyncio.run(_run())


def test_request_id_generated_when_absent(app_and_key):
    app, _ = app_and_key

    async def _run():
        async with _client(app) as c:
            r = await c.get("/v1/version")
            rid = r.headers.get("X-Request-ID")
            assert rid and len(rid) >= 16
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# rate limit (build a separate app with a tight bucket)
# ---------------------------------------------------------------------------


def test_rate_limit_returns_429_when_exhausted():
    auction = Auction()
    auction.register(_FakeProvider())
    app = build_app(
        auction=auction,
        rate_limit_capacity=2.0,
        rate_limit_refill_per_sec=0.0001,   # effectively no refill
    )

    async def _run():
        async with _client(app) as c:
            r1 = await c.get("/v1/version")
            r2 = await c.get("/v1/version")
            r3 = await c.get("/v1/version")
            assert r1.status_code == 200
            assert r2.status_code == 200
            assert r3.status_code == 429
            assert "Retry-After" in r3.headers
    asyncio.run(_run())
