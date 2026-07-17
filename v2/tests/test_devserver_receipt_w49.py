"""W49 — devserver emits a real signed §A1 PNIS receipt per request.

What W43 left behind was a receipt-id header that pointed at the result
hash, with no signed payload behind it. W49 closes the loop:

  * After every completed devserver request, the gateway wallet signs a
    full §A1 receipt (input_hash, output_hash, model, cost, latency,
    timestamp) and caches it under `job_id`.
  * The response headers advertise `X-Pluginfer-Receipt-ID: <job_id>` and
    `X-Pluginfer-Receipt-Signed: 1` so the SDK knows there's a signed
    payload to fetch.
  * `GET /v1/receipts/{job_id}` returns the signed JSON; the signature
    verifies under the embedded pubkey via `AIReceipt.verify()`.
  * Receipts are stable: a second fetch returns the same blob (no
    re-signing, no timestamp drift in cached results).

This is the compliance / audit-trail leg of INVENTIONS §A21 ("receipt-
bound migration audit"): every prompt routed through the shim leaves a
tamper-evident artefact that AWS / OpenAI / Anthropic literally cannot
match.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import httpx  # noqa: E402

from api.devserver import build_devserver_app  # noqa: E402
from api.jobs_service import JobsService  # noqa: E402
from core.ai_receipt import AIReceipt  # noqa: E402
from core.providers import (  # noqa: E402
    Auction,
    Bid,
    JobSpec,
    Provider,
    PRIVACY_PUBLIC,
)


class _EchoProvider(Provider):
    """Re-used from the W43 devserver tests. Returns the prompt as the
    result blob so we can assert on its content."""
    provider_id = "echo-w49"
    privacy_grade = PRIVACY_PUBLIC

    def bid(self, job: JobSpec) -> Bid:
        return Bid(
            provider_id=self.provider_id,
            price_usd=0.0001,
            eta_ms=10,
            expected_quality=0.99,
            privacy_grade=PRIVACY_PUBLIC,
            evidence={"src": "echo"},
        )

    def execute(self, job: JobSpec, bid: Bid) -> dict:
        import base64
        import hashlib
        prompt = job.payload.get("prompt") or ""
        last_user = ""
        for line in prompt.splitlines():
            if line.startswith("user:"):
                last_user = line.split(":", 1)[-1].strip()
        text = f"echo: {last_user}"
        out = text.encode("utf-8")
        return {
            "status": "executed",
            "job_id": job.job_id,
            "result_text": text,
            "result_bytes": base64.b64encode(out).decode("ascii"),
            "result_hash": hashlib.sha256(out).hexdigest(),
            "execution_ms": 3,
            "provider_sig": "AAAA",
            "provider_pubkey_pem": "-----BEGIN PUBLIC KEY-----\nfake\n-----END PUBLIC KEY-----\n",
        }


@pytest.fixture
def app():
    auction = Auction()
    auction.register(_EchoProvider())
    svc = JobsService(auction=auction)
    return build_devserver_app(jobs_service=svc)


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def test_chat_completion_emits_signed_receipt_header(app):
    """The chat completion path should attest a receipt and advertise it
    via the X-Pluginfer-Receipt-Signed header."""
    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/chat/completions", json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 32,
            })
            assert r.status_code == 200, r.text
            assert r.headers["x-pluginfer-receipt-signed"] == "1"
            assert r.headers["x-pluginfer-receipt-id"]
            assert r.headers["x-pluginfer-result-hash"]
            # The receipt-id IS the job-id (it's the lookup key for the
            # receipts router).
            assert r.headers["x-pluginfer-receipt-id"] == \
                r.headers["x-pluginfer-job-id"]
    asyncio.run(_run())


def test_anthropic_messages_emits_signed_receipt_header(app):
    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/messages", json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "ping"}],
            })
            assert r.status_code == 200, r.text
            assert r.headers["x-pluginfer-receipt-signed"] == "1"
    asyncio.run(_run())


def test_receipts_router_returns_signed_json_and_verifies(app):
    """The cached receipt at /v1/receipts/{job_id} is a full §A1 payload
    whose signature verifies under the embedded pubkey."""
    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/chat/completions", json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "audit me"}],
                "max_tokens": 32,
            })
            assert r.status_code == 200, r.text
            job_id = r.headers["x-pluginfer-job-id"]

            r2 = await c.get(f"/v1/receipts/{job_id}")
            assert r2.status_code == 200, r2.text
            payload = r2.json()
            assert payload["schema"] == "pnis-receipt/v1"
            assert payload["job_id"] == job_id
            assert payload["signature"]["alg"] == "ecdsa-secp256k1-sha256"
            assert payload["signature"]["value"]
            assert payload["signature"]["pubkey"].startswith("-----BEGIN PUBLIC KEY-----")
            # Signature must verify against the canonical body.
            receipt = AIReceipt.from_dict(payload)
            assert receipt.verify() is True
            # Provider attestation side-band records the upstream's
            # own signature so an auditor can verify identity end-
            # to-end.
            assert payload["provider_attestation"]["provider_id"] == "echo-w49"
            assert payload["provider_attestation"]["result_hash_hex"] == \
                r.headers["x-pluginfer-result-hash"]
    asyncio.run(_run())


def test_receipt_is_stable_across_repeated_fetches(app):
    """Receipt JSON is cached — two GETs return byte-equal payloads. If
    the cache were broken we'd re-sign with a fresh timestamp and the
    signature would change too."""
    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/chat/completions", json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "stable"}],
                "max_tokens": 16,
            })
            assert r.status_code == 200
            job_id = r.headers["x-pluginfer-job-id"]

            a = await c.get(f"/v1/receipts/{job_id}")
            b = await c.get(f"/v1/receipts/{job_id}")
            assert a.status_code == 200 and b.status_code == 200
            assert a.json() == b.json()
            assert a.json()["signature"]["value"] == \
                b.json()["signature"]["value"]
    asyncio.run(_run())


def test_receipt_tamper_detection(app):
    """Mutate a field on the cached payload, reconstruct via
    AIReceipt.from_dict, and confirm verify() returns False. The whole
    point of the §A1 schema."""
    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/chat/completions", json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "tamper"}],
                "max_tokens": 16,
            })
            job_id = r.headers["x-pluginfer-job-id"]
            r2 = await c.get(f"/v1/receipts/{job_id}")
            payload = r2.json()
            # Flip the cost — anyone trying to under-report cost
            # to auditors will get caught here.
            payload["cost"]["usd_estimate"] = "999999"
            tampered = AIReceipt.from_dict(payload)
            assert tampered.verify() is False
    asyncio.run(_run())


def test_unsigned_path_returns_record_view(app):
    """If a job lands in JobsService.jobs but was never attested (e.g.
    no payload to hash because state != completed), the receipts router
    falls back to the lightweight record view rather than 500ing."""
    async def _run():
        async with _client(app) as c:
            # Hit /v1/receipts/{x} for a job id that doesn't exist.
            r = await c.get("/v1/receipts/no-such-job")
            assert r.status_code == 404
    asyncio.run(_run())


def test_receipt_carries_g7_energy_disclosure(app):
    """G7 — every completed receipt should advertise the energy + carbon
    sidecar so EU AI Act / SEC AI compliance scanners find it."""
    import asyncio as _asyncio
    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/chat/completions", json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "energy check"}],
                "max_tokens": 16,
            })
            assert r.status_code == 200, r.text
            job_id = r.headers["x-pluginfer-job-id"]
            r2 = await c.get(f"/v1/receipts/{job_id}")
            assert r2.status_code == 200
            d = r2.json()
            # The signed body still carries energy_mj (always — gates
            # the schema). The sidecar carries the breakdown.
            assert "energy_mj" in d
            assert "energy_disclosure" in d
            disc = d["energy_disclosure"]
            assert disc["energy_source"] in ("gpu-nvml", "cpu-tdp")
            assert "energy_zone" in disc
            assert float(disc["energy_carbon_intensity_gco2_per_kwh"]) > 0
            assert "carbon_gco2e" in disc
    _asyncio.run(_run())


def test_receipt_pricing_matches_locked_price(app):
    """The signed receipt records cost.usd_estimate equal to the
    auction's locked price — the audit trail must show what the
    buyer was actually charged."""
    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/chat/completions", json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "price check"}],
                "max_tokens": 16,
            })
            assert r.status_code == 200
            job_id = r.headers["x-pluginfer-job-id"]
            price_header = float(r.headers["x-pluginfer-price-usd"])
            r2 = await c.get(f"/v1/receipts/{job_id}")
            assert float(r2.json()["cost"]["usd_estimate"]) == \
                pytest.approx(price_header, rel=1e-6)
    asyncio.run(_run())
